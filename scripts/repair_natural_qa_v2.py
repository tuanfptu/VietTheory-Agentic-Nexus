"""Repair reviewed Natural QA v2 records with bounded, resumable Gemini calls."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from viettheory.benchmark import (
    Answerability,
    BenchmarkSplit,
    ChapterScope,
    Difficulty,
    ExpectedBehavior,
    GenerationMetadata,
    GoldEvidenceGroup,
    QuestionType,
    ReasoningScope,
)
from viettheory.benchmark_generation import (
    BenchmarkCandidate,
    is_substantive_chunk,
    load_gemini_key,
    normalized_question,
    reject_unknown_evidence,
)
from viettheory.natural_benchmark import BenchmarkCategory, NaturalQuestionV2, NegativeType
from viettheory.schema import Chunk


class RepairItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidate: BenchmarkCandidate


class RepairBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repairs: tuple[RepairItem, ...]


CATEGORY_OVERRIDES = {
    "mln111_0016": "direct",
    "vnr202_0020": "explanation",
    "vnr202_0021": "explanation",
    "vnr202_0027": "explanation",
    "vnr202_0028": "explanation",
}

FORCE_REGENERATE: set[str] = set()


def _curated_negative_overrides() -> dict[str, BenchmarkCandidate]:
    def candidate(question: str, reason: str) -> BenchmarkCandidate:
        return BenchmarkCandidate(
            question=question,
            primary_category=BenchmarkCategory.NEGATIVE,
            answerability=Answerability.WRONG_SUBJECT,
            unanswerable_reason=reason,
            expected_behavior=ExpectedBehavior.ROUTE_TO_CORRECT_SUBJECT,
            question_types=(QuestionType.OUT_OF_SCOPE, QuestionType.TEMPORAL),
            reasoning_scope=ReasoningScope.SINGLE_CHUNK,
            chapter_scope=ChapterScope.SINGLE_CHAPTER,
            difficulty=Difficulty.EASY,
            gold_answer=None,
            required_concepts=(),
            forbidden_claims=(),
            evidence_groups=(),
            negative_type=NegativeType.WRONG_SUBJECT,
        )

    return {
        "mln111_0064": candidate(
            "Hiệp định Paris về chấm dứt chiến tranh ở Việt Nam được ký vào năm nào?",
            (
                "Đây là câu hỏi lịch sử thuộc VNR202, không thuộc phạm vi Triết học "
                "Mác - Lênin MLN111."
            ),
        ),
        "mln122_0057": candidate(
            "Hồ Chí Minh đưa ra định nghĩa toàn diện về văn hóa vào thời gian nào?",
            (
                "Đây là câu hỏi thuộc học phần Tư tưởng Hồ Chí Minh HCM202, không thuộc "
                "phạm vi Kinh tế chính trị Mác - Lênin MLN122."
            ),
        ),
    }


def _normalized_reasoning_scope(candidate: BenchmarkCandidate) -> ReasoningScope:
    if candidate.reasoning_scope is not ReasoningScope.CROSS_SUBJECT:
        return candidate.reasoning_scope
    if candidate.primary_category.value == "multi_hop_cross_chapter":
        return ReasoningScope.MULTI_HOP
    if len(candidate.evidence_groups) > 1:
        return ReasoningScope.MULTI_CHUNK
    return ReasoningScope.SINGLE_CHUNK


def _tokens(text: str) -> frozenset[str]:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    text_only = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return frozenset(re.findall(r"[a-z0-9]+", text_only))


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "child_id": chunk.chunk_id,
        "parent_id": chunk.parent_chunk_id,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "pages": sorted({span.pdf_page for span in chunk.source_spans}),
        "text": chunk.text,
    }


def _select_revision_evidence(
    question: NaturalQuestionV2,
    notes: str,
    chunks: tuple[Chunk, ...],
    by_id: dict[str, Chunk],
    *,
    limit: int,
) -> tuple[Chunk, ...]:
    current_ids = {
        child_id
        for group in question.required_evidence_groups
        for child_id in (*group.primary_child_ids, *group.acceptable_child_ids)
    }
    selected = [by_id[child_id] for child_id in current_ids if child_id in by_id]
    query_tokens = _tokens(f"{question.question} {notes}")
    ranked = sorted(
        (chunk for chunk in chunks if is_substantive_chunk(chunk)),
        key=lambda chunk: (
            len(query_tokens & _tokens(chunk.text)) / max(1, len(query_tokens)),
            len(chunk.text),
        ),
        reverse=True,
    )
    seen = {chunk.chunk_id for chunk in selected}
    for chunk in ranked:
        if chunk.chunk_id not in seen:
            selected.append(chunk)
            seen.add(chunk.chunk_id)
        if len(selected) >= limit:
            break
    return tuple(selected[:limit])


def _replacement_pool(
    chunks: tuple[Chunk, ...],
    parent_use: Counter[str],
    *,
    category: str,
    offset: int,
    limit: int,
) -> tuple[Chunk, ...]:
    buckets: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if is_substantive_chunk(chunk):
            buckets[chunk.chapter or "unassigned"].append(chunk)
    ordered: list[Chunk] = []
    for chapter in sorted(buckets):
        ranked = sorted(
            buckets[chapter],
            key=lambda chunk: (
                parent_use[chunk.parent_chunk_id or ""],
                chunk.source_spans[0].pdf_page if chunk.source_spans else 0,
            ),
        )
        ordered.extend(ranked[: max(2, limit // max(1, len(buckets)))])
    ordered = sorted(
        {chunk.chunk_id: chunk for chunk in ordered}.values(),
        key=lambda chunk: (
            parent_use[chunk.parent_chunk_id or ""],
            chunk.chapter or "",
            chunk.chunk_id,
        ),
    )
    if not ordered:
        raise ValueError("no substantive replacement evidence available")
    if category in {"multi_chunk", "synthesis"}:
        by_parent: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            if is_substantive_chunk(chunk) and chunk.parent_chunk_id:
                by_parent[chunk.parent_chunk_id].append(chunk)
        candidates = sorted(
            (items for items in by_parent.values() if len(items) >= 2),
            key=lambda items: (
                parent_use[items[0].parent_chunk_id or ""],
                items[0].chapter or "",
                items[0].chunk_id,
            ),
        )
        if candidates:
            family = candidates[offset % len(candidates)]
            return tuple(family[:limit])
    if category == "multi_hop_cross_chapter":
        seed = ordered[offset % len(ordered)]
        seed_tokens = _tokens(seed.text)
        cross_chapter = sorted(
            (
                chunk
                for chunk in chunks
                if is_substantive_chunk(chunk) and chunk.chapter != seed.chapter
            ),
            key=lambda chunk: (
                len(seed_tokens & _tokens(chunk.text)),
                -parent_use[chunk.parent_chunk_id or ""],
            ),
            reverse=True,
        )
        return tuple([seed, *cross_chapter[: limit - 1]])
    start = offset % len(ordered)
    rotated = ordered[start:] + ordered[:start]
    return tuple(rotated[:limit])


def _semantic_candidate_valid(candidate: BenchmarkCandidate) -> bool:
    child_ids = {child_id for group in candidate.evidence_groups for child_id in group.child_ids}
    category = candidate.primary_category.value
    if category in {"multi_chunk", "synthesis", "multi_hop_cross_chapter"}:
        if len(child_ids) < 2:
            return False
    if category == "multi_hop_cross_chapter":
        if len(candidate.evidence_groups) < 2:
            return False
        if candidate.reasoning_scope is not ReasoningScope.MULTI_HOP:
            return False
        if candidate.chapter_scope.value != "multi_chapter":
            return False
    if category == "comparison_relationship":
        if "comparison" not in {item.value for item in candidate.question_types}:
            return False
    return True


def _call_gemini(
    *,
    api_key: str,
    model: str,
    tasks: list[dict[str, Any]],
    evidence: dict[str, list[dict[str, Any]]],
    timeout: float,
    max_retries: int,
) -> tuple[RepairItem, ...]:
    prompt = (
        "Bạn đang sửa benchmark Natural QA tiếng Việt dựa CHỈ trên evidence được cung cấp. "
        "Mỗi task phải trả đúng một repair với task_id giữ nguyên. Nếu required_category "
        "khác null thì primary_category bắt buộc bằng required_category. Với action=replace, "
        "primary_category phải đúng target_category. Với action=revise, reviewer_notes "
        "là yêu cầu bắt buộc và category có thể đổi nếu answerability phải đổi; "
        "có thể viết lại question/gold/evidence hoàn toàn nếu cần để câu tự nhiên và "
        "đúng category. Với action=replace, tạo câu hoàn toàn mới, không nối hai fact "
        "không có quan hệ trong giáo trình. Câu answerable phải có gold_answer được "
        "evidence hỗ trợ và dùng đúng child_id. Câu negative phải thật sự "
        "false_premise/insufficient/out_of_scope, không được gắn unanswerable nếu "
        "evidence thực ra trả lời được. Không bịa fact, ID, trang hoặc quan hệ nhân quả. "
        "Với negative replacement, câu phải rõ ràng ngoài phạm vi môn hoặc có tiền đề sai "
        "được evidence mâu thuẫn trực tiếp; tuyệt đối không hỏi một fact có trong evidence. "
        "Replacement phải là câu mới thật sự, không được dùng lại hoặc paraphrase câu reject cũ; "
        "false-premise phải là một mệnh đề tự nhiên duy nhất, không viết dạng lựa chọn A hay B. "
        "Nếu required_category là category không phải negative thì bắt buộc "
        "answerability=answerable, expected_behavior=answer, negative_type=null và phải có "
        "gold_answer cùng evidence_groups. "
        "multi_chunk và synthesis phải dùng ít nhất 2 child_id; multi_hop_cross_chapter "
        "phải có ít nhất 2 evidence group từ các chương khác nhau, reasoning_scope=multi_hop "
        "và chapter_scope=multi_chapter; comparison_relationship phải có question_type=comparison. "
        "Mọi output vẫn là draft "
        "chờ người kiểm tra.\nTASKS:\n"
        f"{json.dumps(tasks, ensure_ascii=False)}\nEVIDENCE_BY_TASK:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": RepairBatch.model_json_schema(),
            "temperature": 0.35,
        },
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='')}:generateContent"
    )
    expected = {task["task_id"]: task.get("required_category") for task in tasks}
    allowed = {
        task_id: frozenset(item["child_id"] for item in payload)
        for task_id, payload in evidence.items()
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
            repairs = RepairBatch.model_validate(raw).repairs
            if {item.task_id for item in repairs} != set(expected):
                raise ValueError("response task IDs do not match request")
            checked: list[RepairItem] = []
            for item in repairs:
                if (
                    expected[item.task_id] is not None
                    and item.candidate.primary_category.value != expected[item.task_id]
                ):
                    raise ValueError(f"{item.task_id}: category changed")
                if not _semantic_candidate_valid(item.candidate):
                    raise ValueError(f"{item.task_id}: category semantics are invalid")
                task = next(task for task in tasks if task["task_id"] == item.task_id)
                if task["action"] == "replace":
                    original_key = normalized_question(task["original_question"])
                    replacement_key = normalized_question(item.candidate.question)
                    original_tokens = frozenset(original_key.split())
                    replacement_tokens = frozenset(replacement_key.split())
                    similarity = len(original_tokens & replacement_tokens) / max(
                        1, len(original_tokens | replacement_tokens)
                    )
                    if original_key == replacement_key or similarity >= 0.7:
                        raise ValueError(f"{item.task_id}: replacement is too similar")
                reject_unknown_evidence((item.candidate,), allowed[item.task_id])
                if item.candidate.primary_category.value == "multi_hop_cross_chapter":
                    chapter_by_child = {
                        evidence_item["child_id"]: evidence_item["chapter"]
                        for evidence_item in evidence[item.task_id]
                    }
                    used_chapters = {
                        chapter_by_child[child_id]
                        for group in item.candidate.evidence_groups
                        for child_id in group.child_ids
                    }
                    if len(used_chapters) < 2:
                        raise ValueError(f"{item.task_id}: multi-hop evidence is not cross-chapter")
                checked.append(item)
            return tuple(checked)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 and not 500 <= exc.code < 600:
                raise RuntimeError(f"Gemini request failed with HTTP {exc.code}") from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            ValidationError,
            ValueError,
        ) as exc:
            last_error = exc
        if attempt < max_retries:
            delay = min(30.0, 5.0 * 2**attempt)
            print(f"retrying invalid/rate-limited batch in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError("Gemini repeatedly failed to return a valid repair batch") from last_error


def _materialize(
    original: NaturalQuestionV2,
    candidate: BenchmarkCandidate,
    chunks: dict[str, Chunk],
    *,
    model: str,
    action: str,
    reviewer_notes: str,
) -> NaturalQuestionV2:
    cited: list[Chunk] = []
    groups: list[GoldEvidenceGroup] = []
    for index, group in enumerate(candidate.evidence_groups, start=1):
        group_chunks = [chunks[child_id] for child_id in group.child_ids]
        cited.extend(group_chunks)
        groups.append(
            GoldEvidenceGroup(
                group_id=f"g{index}",
                subject_code=original.subject_code,
                role=group.role,
                required=group.required,
                primary_child_ids=group.child_ids,
                gold_parent_ids=tuple(
                    dict.fromkeys(
                        chunk.parent_chunk_id for chunk in group_chunks if chunk.parent_chunk_id
                    )
                ),
                gold_pdf_pages=tuple(
                    sorted({span.pdf_page for chunk in group_chunks for span in chunk.source_spans})
                ),
            )
        )
    return NaturalQuestionV2(
        benchmark_version="natural_qa_v2_500_followup_draft",
        id=original.id,
        subject_code=original.subject_code,
        chapter_labels=tuple(dict.fromkeys(chunk.chapter for chunk in cited if chunk.chapter)),
        section_labels=tuple(dict.fromkeys(chunk.section for chunk in cited if chunk.section)),
        question=candidate.question,
        question_types=candidate.question_types,
        primary_category=candidate.primary_category,
        difficulty=candidate.difficulty,
        reasoning_scope=_normalized_reasoning_scope(candidate),
        chapter_scope=candidate.chapter_scope,
        answerability=candidate.answerability,
        unanswerable_reason=candidate.unanswerable_reason,
        negative_type=candidate.negative_type,
        expected_behavior=candidate.expected_behavior,
        gold_answer=candidate.gold_answer,
        required_evidence_groups=tuple(groups),
        required_concepts=candidate.required_concepts,
        forbidden_claims=candidate.forbidden_claims,
        split=BenchmarkSplit.DEVELOPMENT,
        generation=GenerationMetadata(
            method="llm_assisted",
            model=model,
            prompt_version="natural_qa_v2_human_review_repair_v1",
        ),
        artifact_manifest_ids=original.artifact_manifest_ids,
        review_notes=(
            f"{action} after human review; pending recheck. Original reviewer notes: "
            f"{reviewer_notes}"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--requests-per-minute", type=float, default=4.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.requests_per_minute <= 0:
        parser.error("batch-size and requests-per-minute must be positive")
    api_key = load_gemini_key(args.dotenv)
    if not api_key:
        parser.error("Gemini key is missing from environment/dotenv")

    originals = {
        item.id: item
        for line in args.reviewed.read_text(encoding="utf-8").splitlines()
        if (item := NaturalQuestionV2.model_validate_json(line))
    }
    with args.review_csv.open(encoding="utf-8-sig", newline="") as handle:
        review_rows = {
            row["id"]: row
            for row in csv.DictReader(handle)
            if row["decision"].strip().casefold() != "approve"
        }
    subjects = sorted({originals[item_id].subject_code for item_id in review_rows})
    chunks_by_subject: dict[str, tuple[Chunk, ...]] = {}
    chunks_by_id: dict[str, dict[str, Chunk]] = {}
    for subject in subjects:
        path = args.data_root / subject / "structured_v1" / "children.jsonl"
        chunks = tuple(
            Chunk.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        chunks_by_subject[subject] = chunks
        chunks_by_id[subject] = {chunk.chunk_id: chunk for chunk in chunks}

    parent_use: Counter[str] = Counter(
        parent_id
        for question in originals.values()
        if question.review_status.value == "verified"
        for group in question.required_evidence_groups
        for parent_id in group.gold_parent_ids
    )
    completed: dict[str, BenchmarkCandidate] = {}
    if args.checkpoint.exists():
        for line in args.checkpoint.read_text(encoding="utf-8").splitlines():
            repair = RepairItem.model_validate_json(line)
            if repair.task_id in FORCE_REGENERATE:
                continue
            original = originals[repair.task_id]
            used_chunks = [
                chunks_by_id[original.subject_code][child_id]
                for group in repair.candidate.evidence_groups
                for child_id in group.child_ids
            ]
            actual_chapters = {chunk.chapter for chunk in used_chunks if chunk.chapter}
            required_category = CATEGORY_OVERRIDES.get(repair.task_id)
            if not _semantic_candidate_valid(repair.candidate):
                continue
            if required_category and repair.candidate.primary_category.value != required_category:
                continue
            if (
                repair.candidate.primary_category.value == "multi_hop_cross_chapter"
                and len(actual_chapters) < 2
            ):
                continue
            completed[repair.task_id] = repair.candidate
    completed.update(_curated_negative_overrides())
    pending_ids = [item_id for item_id in review_rows if item_id not in completed]
    interval = 60.0 / args.requests_per_minute
    for start in range(0, len(pending_ids), args.batch_size):
        ids = pending_ids[start : start + args.batch_size]
        tasks: list[dict[str, Any]] = []
        evidence: dict[str, list[dict[str, Any]]] = {}
        for offset, item_id in enumerate(ids, start=start):
            original = originals[item_id]
            row = review_rows[item_id]
            action = "replace" if row["decision"].strip().casefold() == "reject" else "revise"
            if action == "revise":
                selected = _select_revision_evidence(
                    original,
                    row["reviewer_notes"],
                    chunks_by_subject[original.subject_code],
                    chunks_by_id[original.subject_code],
                    limit=10,
                )
            else:
                selected = _replacement_pool(
                    chunks_by_subject[original.subject_code],
                    parent_use,
                    category=original.primary_category.value,
                    offset=offset * 7,
                    limit=10,
                )
            tasks.append(
                {
                    "task_id": item_id,
                    "subject": original.subject_code,
                    "action": action,
                    "target_category": original.primary_category.value,
                    "required_category": (
                        original.primary_category.value
                        if action == "replace"
                        else CATEGORY_OVERRIDES.get(item_id)
                    ),
                    "original_question": original.question,
                    "original_gold_answer": original.gold_answer,
                    "reviewer_notes": row["reviewer_notes"],
                }
            )
            evidence[item_id] = [_chunk_payload(chunk) for chunk in selected]
        repairs = _call_gemini(
            api_key=api_key,
            model=args.model,
            tasks=tasks,
            evidence=evidence,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        completed.update({item.task_id: item.candidate for item in repairs})
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(
            "".join(
                RepairItem(task_id=task_id, candidate=candidate).model_dump_json() + "\n"
                for task_id, candidate in sorted(completed.items())
            ),
            encoding="utf-8",
        )
        print(f"checkpoint={len(completed)}/{len(review_rows)}")
        if start + args.batch_size < len(pending_ids):
            time.sleep(interval)

    repaired = dict(originals)
    for item_id, candidate in completed.items():
        row = review_rows[item_id]
        action = "replacement" if row["decision"].strip().casefold() == "reject" else "revision"
        subject = originals[item_id].subject_code
        repaired[item_id] = _materialize(
            originals[item_id],
            candidate,
            chunks_by_id[subject],
            model=args.model,
            action=action,
            reviewer_notes=row["reviewer_notes"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(repaired[item_id].model_dump_json() + "\n" for item_id in originals),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
