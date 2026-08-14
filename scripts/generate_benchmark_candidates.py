"""Generate schema-constrained benchmark drafts from exported corpus evidence.

This command loads only the Gemini credential from the environment or selected
dotenv file. It never logs or persists the credential.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from viettheory.benchmark_generation import (
    BenchmarkCandidate,
    CandidateBatch,
    deduplicate_candidates,
    is_substantive_chunk,
    load_gemini_key,
    quota_category_plan,
    reject_unknown_evidence,
)
from viettheory.natural_benchmark import BenchmarkCategory, PortfolioPlan
from viettheory.schema import Chunk

DEFAULT_CATEGORY_PLAN = (
    "direct,direct,explanation,explanation,comparison_relationship,"
    "comparison_relationship,multi_chunk,multi_chunk,synthesis,"
    "multi_hop_cross_chapter"
)


def _chapter_key(chunk: Chunk) -> str:
    return chunk.chapter or "unassigned"


def _stratified_batches(
    chunks: tuple[Chunk, ...],
    *,
    batch_size: int,
    max_chunks: int,
) -> tuple[tuple[Chunk, ...], ...]:
    buckets: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if is_substantive_chunk(chunk):
            buckets[_chapter_key(chunk)].append(chunk)
    ordered: list[Chunk] = []
    while len(ordered) < max_chunks and any(buckets.values()):
        for chapter in sorted(buckets):
            if buckets[chapter] and len(ordered) < max_chunks:
                ordered.append(buckets[chapter].pop(0))
    return tuple(
        tuple(ordered[start : start + batch_size]) for start in range(0, len(ordered), batch_size)
    )


def _evidence_payload(batch: tuple[Chunk, ...]) -> list[dict[str, Any]]:
    return [
        {
            "child_id": chunk.chunk_id,
            "parent_id": chunk.parent_chunk_id,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "pages": sorted({span.pdf_page for span in chunk.source_spans}),
            "text": chunk.text,
        }
        for chunk in batch
    ]


def _call_gemini(
    *,
    api_key: str,
    model: str,
    subject: str,
    batch: tuple[Chunk, ...],
    candidates_per_batch: int,
    timeout: float,
    max_retries: int,
    desired_categories: tuple[BenchmarkCategory, ...],
) -> tuple[BenchmarkCandidate, ...]:
    prompt = (
        "Bạn đang tạo bản nháp benchmark retrieval tiếng Việt dựa CHỈ trên evidence JSON. "
        f"Hãy tạo đúng {candidates_per_batch} câu tự nhiên cho môn {subject}. "
        f"Gán primary_category theo đúng thứ tự: {list(desired_categories)}. Phân phối dạng "
        "paraphrase, nguyên nhân-kết quả, so sánh, thời gian/thực thể, tổng hợp; không chỉ tạo "
        "câu định nghĩa. Với câu answerable, mỗi evidence group phải dùng nguyên văn child_id "
        "được cung cấp. QUY TẮC BẮT BUỘC: với mọi category KHÁC negative, phải đặt "
        "answerability='answerable', expected_behavior='answer', negative_type=null, "
        "unanswerable_reason=null, đồng thời phải có gold_answer và evidence_groups. "
        "Chỉ với category negative mới được tạo câu không thể trả lời đầy đủ từ evidence; "
        "khi đó gold_answer=null và answerability, expected_behavior, negative_type, "
        "unanswerable_reason phải nhất quán. Primary category của từng câu phải khớp chính xác "
        "vị trí tương ứng trong danh sách yêu cầu. Không bịa ID, trang, kiến thức hoặc đáp án. "
        "Câu nhiều ý phải có nhiều evidence_groups. "
        "Đây chỉ là draft chờ con người xác nhận.\nEvidence JSON:\n"
        f"{json.dumps(_evidence_payload(batch), ensure_ascii=False)}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": CandidateBatch.model_json_schema(),
            "temperature": 0.6,
        },
    }
    quoted_model = urllib.parse.quote(model, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quoted_model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    allowed = frozenset(chunk.chunk_id for chunk in batch)
    expected_categories = tuple(category.value for category in desired_categories)
    last_schema_error: Exception | None = None
    for schema_attempt in range(max_retries + 1):
        payload: dict[str, Any] | None = None
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= max_retries:
                    raise RuntimeError(f"Gemini request failed with HTTP {exc.code}") from exc
                retry_after = exc.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 2**attempt * 5.0
                )
                print(f"rate/server limit: retrying in {delay:.0f}s")
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= max_retries:
                    raise RuntimeError("Gemini network request failed") from exc
                delay = 2**attempt * 5.0
                print(f"network retry in {delay:.0f}s")
                time.sleep(delay)
        if payload is None:
            raise RuntimeError("Gemini returned no response")
        try:
            raw = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
            generated = CandidateBatch.model_validate(raw).candidates
            actual_categories = tuple(item.primary_category.value for item in generated)
            if len(generated) != candidates_per_batch:
                raise ValueError(
                    f"expected {candidates_per_batch} candidates, got {len(generated)}"
                )
            if actual_categories != expected_categories:
                raise ValueError(
                    f"category mismatch: expected {expected_categories}, got {actual_categories}"
                )
            return reject_unknown_evidence(generated, allowed)
        except (
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValidationError,
            ValueError,
        ) as exc:
            last_schema_error = exc
            if schema_attempt >= max_retries:
                break
            delay = 8.0
            print(
                f"structured response rejected: retrying in {delay:.0f}s "
                f"({schema_attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
    raise RuntimeError(
        "Gemini repeatedly returned invalid structured output"
    ) from last_schema_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--children", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--candidates-per-batch", type=int, default=5)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--requests-per-minute", type=float, default=10.0)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--category-plan", default=DEFAULT_CATEGORY_PLAN)
    parser.add_argument(
        "--portfolio-plan",
        type=Path,
        help="Optional portfolio JSON; remaining categories follow its subject quota.",
    )
    args = parser.parse_args()
    raw_category_plan = tuple(
        item.strip() for item in args.category_plan.split(",") if item.strip()
    )
    if (
        args.target < 1
        or args.batch_size < 1
        or args.candidates_per_batch < 1
        or args.requests_per_minute <= 0
        or args.max_requests < 1
        or args.max_retries < 0
    ):
        parser.error("target and batch sizes must be positive")
    api_key = load_gemini_key(args.dotenv)
    if not api_key:
        parser.error(
            "GEMINI_API_KEY/LLM_API_KEY is missing from the environment and selected dotenv file"
        )

    chunks = tuple(
        Chunk.model_validate_json(line)
        for line in args.children.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    max_batches = (args.target + args.candidates_per_batch - 1) // args.candidates_per_batch
    batches = _stratified_batches(
        chunks,
        batch_size=args.batch_size,
        max_chunks=max_batches * args.batch_size,
    )
    existing: list[BenchmarkCandidate] = []
    if args.output.exists():
        existing = [
            BenchmarkCandidate.model_validate_json(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    candidates = list(deduplicate_candidates(existing))
    if args.portfolio_plan:
        portfolio = PortfolioPlan.model_validate_json(
            args.portfolio_plan.read_text(encoding="utf-8")
        )
        subject_plans = [
            batch for batch in portfolio.pilot_batches if batch.subject_code == args.subject
        ]
        if len(subject_plans) != 1:
            parser.error(f"portfolio plan has no unique batch for {args.subject}")
        subject_plan = subject_plans[0]
        if subject_plan.batch_size != args.target:
            parser.error(f"portfolio batch size {subject_plan.batch_size} != target {args.target}")
        targets = {quota.category: quota.target for quota in subject_plan.quotas}
        category_plan = quota_category_plan(candidates, targets)
    else:
        try:
            base_plan = tuple(BenchmarkCategory(item) for item in raw_category_plan)
        except ValueError as exc:
            parser.error(str(exc))
        if not base_plan:
            parser.error("category-plan must not be empty")
        remaining_categories = max(0, args.target - len(candidates))
        category_plan = tuple(
            base_plan[index % len(base_plan)] for index in range(remaining_categories)
        )
    request_count = 0
    last_request_started: float | None = None
    minimum_interval = 60.0 / args.requests_per_minute
    completed_batches = len(candidates) // args.candidates_per_batch
    for batch in batches[completed_batches:]:
        if len(deduplicate_candidates(candidates)) >= args.target:
            break
        if request_count >= args.max_requests:
            print(f"request budget reached: {request_count}/{args.max_requests}")
            break
        if last_request_started is not None:
            remaining = minimum_interval - (time.monotonic() - last_request_started)
            if remaining > 0:
                time.sleep(remaining)
        last_request_started = time.monotonic()
        request_count += 1
        remaining_needed = args.target - len(candidates)
        desired_categories = category_plan[: min(args.candidates_per_batch, remaining_needed)]
        if not desired_categories:
            break
        candidates.extend(
            _call_gemini(
                api_key=api_key,
                model=args.model,
                subject=args.subject,
                batch=batch,
                candidates_per_batch=args.candidates_per_batch,
                timeout=args.timeout,
                max_retries=args.max_retries,
                desired_categories=desired_categories,
            )
        )
        unique = deduplicate_candidates(candidates)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(item.model_dump_json() + "\n" for item in unique),
            encoding="utf-8",
        )
        candidates = list(unique)
        category_plan = category_plan[len(desired_categories) :]
        print(
            f"checkpoint: {len(candidates)} unique candidates; "
            f"requests={request_count}/{args.max_requests}"
        )
    return 0 if len(candidates) >= args.target else 2


if __name__ == "__main__":
    raise SystemExit(main())
