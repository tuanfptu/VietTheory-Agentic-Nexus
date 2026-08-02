"""Import the owner's MLN111 development review and apply explicit corrections."""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from viettheory.benchmark import (
    BenchmarkQuestion,
    BenchmarkReview,
    GoldEvidenceGroup,
    QuestionType,
    ReasoningScope,
    ReviewDecision,
    ReviewStatus,
)
from viettheory.schema import Chunk

QUESTION_REVISIONS = {
    "mln111_0032": "Nội dung thứ nhất trong định nghĩa vật chất của V.I. Lênin là gì?",
    "mln111_0033": (
        "Theo Ph. Ăngghen, lao động sản xuất có vai trò như thế nào đối với "
        "sự hình thành và phát triển của con người?"
    ),
    "mln111_0034": (
        "Trình bày khái quát ba hình thức cơ bản của chủ nghĩa duy vật trong lịch sử triết học."
    ),
    "mln111_0040": (
        "Leucippus và Democritos sống vào khoảng thời gian nào và quan niệm vật chất là gì?"
    ),
    "mln111_0052": "Các định nghĩa về triết học thường bao hàm những nội dung chủ yếu nào?",
    "mln111_0061": (
        "Những lập trường giai cấp khác nhau đã tạo ra các biến thể chủ nghĩa "
        "xã hội nào, và sự xuất hiện của giai cấp vô sản cách mạng tạo cơ sở "
        "xã hội cho lý luận mới ra sao?"
    ),
    "mln111_0068": (
        "Yếu tố nào quyết định năng suất lao động xã hội và vì sao lực lượng "
        "sản xuất có tính khách quan?"
    ),
    "mln111_0071": (
        "Theo giáo trình, mức độ thất lạc các tác phẩm của Plato và Aristotle "
        "được mô tả như thế nào?"
    ),
    "mln111_0085": (
        "Việc C. Mác và Ph. Ăngghen đưa thực tiễn vào triết học đã giúp giải "
        "quyết những vấn đề nào và đóng góp gì cho chủ nghĩa duy vật lịch sử?"
    ),
    "mln111_0088": (
        "Sự phát triển của lực lượng sản xuất được biểu hiện ở những mặt và "
        "trình độ nào, và khoa học hiện đại có vai trò gì?"
    ),
    "mln111_0090": (
        "Triết học cổ điển Đức có vai trò gì đối với sự ra đời của triết học "
        "Mác, và C. Mác đã kế thừa phép biện chứng Hegel như thế nào?"
    ),
}

GOLD_REVISIONS = {
    "mln111_0021": (
        "Nguồn gốc tự nhiên của ý thức gồm bộ óc người và sự tác động của thế "
        "giới khách quan lên bộ óc; nguồn gốc xã hội trực tiếp là lao động và "
        "ngôn ngữ trong hoạt động thực tiễn xã hội."
    ),
    "mln111_0040": (
        "Leucippus sống khoảng năm 500-440 trước Công nguyên, Democritos sống "
        "khoảng năm 460-370 trước Công nguyên; cả hai quan niệm vật chất là "
        "nguyên tử."
    ),
    "mln111_0052": (
        "Triết học là một hình thái ý thức xã hội; khách thể khám phá là thế "
        "giới trong tính chỉnh thể toàn vẹn; mục đích là tìm ra các quy luật "
        "phổ biến nhất chi phối thế giới, con người và tư duy."
    ),
    "mln111_0093": (
        "Thế giới quan đúng đắn là tiền đề xác lập phương thức tư duy hợp lý "
        "và nhân sinh quan tích cực; thế giới quan tôn giáo đặt niềm tin vào "
        "tín điều, coi tín ngưỡng cao hơn lý trí và phủ nhận tính khách quan "
        "của tri thức khoa học nên có thể dẫn đến sai lầm, tiêu cực trong "
        "thực tiễn."
    ),
}

ADD_REQUIRED = {
    "mln111_0001": ("child_3a1b37684f4863253f72",),
    "mln111_0003": ("child_e58f4ca608285d002b03",),
    "mln111_0011": ("child_023574d3cce3cdceb452",),
    "mln111_0012": ("child_ab75b8ccaa7dad1c6164",),
    "mln111_0016": ("child_3a1b37684f4863253f72",),
    "mln111_0047": ("child_b59cb3e0374be1500e4c",),
    "mln111_0050": ("child_909a2db6a1f07876a73c", "child_0c05da130e40b36fbc58"),
    "mln111_0051": ("child_cb5219fa9d025bf49e04", "child_ba345258f2acdd2e6271"),
    "mln111_0054": ("child_a048fc0e426982342757",),
    "mln111_0057": ("child_9113296711fa4ffdc8f3",),
    "mln111_0059": ("child_89b15f42e455477da967",),
    "mln111_0066": ("child_be0cab0e97657e2d7b86",),
    "mln111_0080": ("child_83f301e41110a9b2d812",),
    "mln111_0083": ("child_d8e0ecec5be5a0743b85",),
    "mln111_0088": ("child_871508fab9230d2ad5e5",),
    "mln111_0090": ("child_f105dcdd9e4794cdbdce",),
}

REPLACE_FIRST_GROUP = {
    "mln111_0006": ("child_be0cab0e97657e2d7b86",),
    "mln111_0018": ("child_be0cab0e97657e2d7b86",),
}


def _parse_review(text: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    blocks = re.split(r"\n---\s*\n", text)
    for block in blocks:
        match = re.search(r"^## (mln111_\d{4})\s+—", block, flags=re.MULTILINE)
        if not match:
            continue
        question_id = match.group(1)
        decision_match = re.search(
            r"- Decision: `\[(?P<approve>[ xX])\] approve` "
            r"`\[(?P<revise>[ xX])\] revise` `\[(?P<reject>[ xX])\] reject`",
            block,
        )
        notes_match = re.search(r"- Reviewer notes:\s*(.+)", block, flags=re.DOTALL)
        if not decision_match or not notes_match:
            raise ValueError(f"incomplete review block: {question_id}")
        checked = [
            name
            for name in ("approve", "revise", "reject")
            if decision_match.group(name).strip().lower() == "x"
        ]
        if len(checked) != 1:
            raise ValueError(f"review needs exactly one decision: {question_id}")
        records[question_id] = {
            "decision": checked[0],
            "notes": notes_match.group(1).strip(),
        }
    return records


def _group_from_children(
    group_id: str,
    role: str,
    child_ids: tuple[str, ...],
    children: dict[str, Chunk],
) -> GoldEvidenceGroup:
    chunks = [children[child_id] for child_id in child_ids]
    return GoldEvidenceGroup(
        group_id=group_id,
        subject_code="MLN111",
        role=role,
        primary_child_ids=child_ids,
        gold_parent_ids=tuple(
            sorted({chunk.parent_chunk_id for chunk in chunks if chunk.parent_chunk_id})
        ),
        gold_pdf_pages=tuple(
            sorted({span.pdf_page for chunk in chunks for span in chunk.source_spans})
        ),
        gold_printed_pages=tuple(
            sorted(
                {
                    span.printed_page
                    for chunk in chunks
                    for span in chunk.source_spans
                    if span.printed_page
                }
            )
        ),
    )


def _apply_corrections(
    question: BenchmarkQuestion,
    review: dict[str, str],
    children: dict[str, Chunk],
) -> BenchmarkQuestion:
    updates: dict[str, object] = {
        "review_status": (
            ReviewStatus.CHECKED if review["decision"] == "approve" else ReviewStatus.DRAFT
        ),
        "notes": review["notes"],
    }
    if question.id in QUESTION_REVISIONS:
        updates["question"] = QUESTION_REVISIONS[question.id]
    if question.id in GOLD_REVISIONS:
        updates["gold_answer"] = GOLD_REVISIONS[question.id]

    groups = list(question.gold_evidence_groups)
    if question.id in REPLACE_FIRST_GROUP:
        groups[0] = _group_from_children(
            "g1",
            groups[0].role,
            REPLACE_FIRST_GROUP[question.id],
            children,
        )
    if question.id == "mln111_0021":
        groups[1] = _group_from_children(
            "g2",
            "social_origin",
            ("child_ab75b8ccaa7dad1c6164",),
            children,
        )
    if question.id in ADD_REQUIRED:
        new_id = f"g{len(groups) + 1}"
        groups.append(
            _group_from_children(
                new_id,
                "supplemental_evidence",
                ADD_REQUIRED[question.id],
                children,
            )
        )
    if question.id == "mln111_0022":
        groups = [groups[0]]
        updates["reasoning_scope"] = ReasoningScope.SINGLE_CHUNK
    if question.id in {"mln111_0027", "mln111_0028"}:
        updates["question_types"] = (QuestionType.OUT_OF_SCOPE,)
    if question.id == "mln111_0011":
        updates["chapter_labels"] = ("Chương 2: CHỦ NGHĨA DUY VẬT BIỆN CHỨNG",)
    if question.id == "mln111_0085":
        groups = [
            group for group in groups if "child_05e1460a1749a92cbfd7" in group.primary_child_ids
        ]
        groups[0] = groups[0].model_copy(update={"group_id": "g1"})
        updates["reasoning_scope"] = ReasoningScope.SINGLE_CHUNK
    updates["gold_evidence_groups"] = tuple(groups)
    return question.model_copy(update=updates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--children", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    review_text = args.review.read_text(encoding="utf-8")
    reviews = _parse_review(review_text)
    decisions = [record["decision"] for record in reviews.values()]
    if len(reviews) != 70 or decisions.count("approve") != 38 or decisions.count("revise") != 32:
        raise ValueError("review totals do not match 70/38/32")
    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if {question.id for question in questions} != set(reviews):
        raise ValueError("review question IDs do not exactly match development benchmark")
    children = {
        chunk.chunk_id: chunk
        for line in args.children.read_text(encoding="utf-8").splitlines()
        if (chunk := Chunk.model_validate_json(line))
    }

    updated = [
        _apply_corrections(question, reviews[question.id], children) for question in questions
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(question.model_dump_json() + "\n" for question in updated),
        encoding="utf-8",
    )
    reviewed_at = datetime(2026, 7, 24, tzinfo=UTC)
    audit = []
    for question in questions:
        record = reviews[question.id]
        decision = (
            ReviewDecision.CHECKED if record["decision"] == "approve" else ReviewDecision.REVISE
        )
        digest = hashlib.sha256(question.model_dump_json().encode("utf-8")).hexdigest()
        audit.append(
            BenchmarkReview(
                question_id=question.id,
                decision=decision,
                reviewer_id="project_owner",
                reviewed_at=reviewed_at,
                benchmark_record_sha256=digest,
                notes=record["notes"],
            )
        )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        "".join(item.model_dump_json() + "\n" for item in audit),
        encoding="utf-8",
    )
    print("imported=70 checked=38 revised=32 rejected=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
