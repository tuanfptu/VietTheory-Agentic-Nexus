"""Apply the owner's second-round corrections to 32 MLN111 development items."""

from __future__ import annotations

import argparse
import hashlib
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

APPROVED = {
    "mln111_0006",
    "mln111_0016",
    "mln111_0018",
    "mln111_0022",
    "mln111_0027",
    "mln111_0028",
    "mln111_0032",
    "mln111_0033",
    "mln111_0040",
    "mln111_0068",
    "mln111_0093",
}
METADATA_ONLY = {
    "mln111_0011",
    "mln111_0012",
    "mln111_0047",
    "mln111_0052",
    "mln111_0057",
    "mln111_0061",
    "mln111_0066",
    "mln111_0085",
}
SIGNIFICANT = {
    "mln111_0001",
    "mln111_0003",
    "mln111_0021",
    "mln111_0034",
    "mln111_0050",
    "mln111_0051",
    "mln111_0054",
    "mln111_0059",
    "mln111_0071",
    "mln111_0080",
    "mln111_0083",
    "mln111_0088",
    "mln111_0090",
}

SINGLE_CHILD = {
    "mln111_0001": "child_6330c351dfe91e952c48",
    "mln111_0003": "child_dc42628f01b95412a354",
    "mln111_0050": "child_909a2db6a1f07876a73c",
    "mln111_0051": "child_cb5219fa9d025bf49e04",
    "mln111_0054": "child_6301929332b6398f2db0",
    "mln111_0059": "child_5c60c014ace4c2ebd7a1",
    "mln111_0071": "child_c758118f1c88c805eb53",
    "mln111_0083": "child_722916f2afdcbf97e864",
    "mln111_0090": "child_9da66e90b8a2faa12dae",
}

GOLD_REVISIONS = {
    "mln111_0054": (
        "Lực lượng sản xuất là nội dung của quá trình sản xuất, có tính năng "
        "động, cách mạng và thường xuyên phát triển; quan hệ sản xuất là hình "
        "thức xã hội của quá trình sản xuất, có tính ổn định tương đối và tác "
        "động trở lại lực lượng sản xuất."
    ),
    "mln111_0071": (
        "Đa số tài liệu triết học thành văn thời cổ đại Hy Lạp đã mất hoặc "
        "không còn nguyên vẹn; tài liệu thời tiền cổ đại chủ yếu chỉ còn một "
        "số câu trích, chú giải và bản ghi tóm lược do tác giả đời sau viết lại."
    ),
    "mln111_0088": (
        "Sự phát triển của lực lượng sản xuất thể hiện ở cả tính chất và trình "
        "độ, gồm trình độ công cụ lao động, tổ chức và phân công lao động xã "
        "hội, ứng dụng khoa học, kinh nghiệm và kỹ năng của người lao động. "
        "Khoa học hiện đại đã trở thành lực lượng sản xuất trực tiếp, rút ngắn "
        "khoảng cách từ phát minh đến ứng dụng, thâm nhập vào các yếu tố sản "
        "xuất và thúc đẩy năng suất lao động."
    ),
}


def _group(
    child_id: str,
    *,
    group_id: str,
    role: str,
    children: dict[str, Chunk],
) -> GoldEvidenceGroup:
    child = children[child_id]
    return GoldEvidenceGroup(
        group_id=group_id,
        subject_code="MLN111",
        role=role,
        primary_child_ids=(child_id,),
        gold_parent_ids=(child.parent_chunk_id,) if child.parent_chunk_id else (),
        gold_pdf_pages=tuple(sorted({span.pdf_page for span in child.source_spans})),
        gold_printed_pages=tuple(
            sorted({span.printed_page for span in child.source_spans if span.printed_page})
        ),
    )


def _keep_child_groups(
    question: BenchmarkQuestion,
    child_ids: tuple[str, ...],
) -> tuple[GoldEvidenceGroup, ...]:
    groups = [
        group
        for group in question.gold_evidence_groups
        if any(child_id in group.primary_child_ids for child_id in child_ids)
    ]
    return tuple(
        group.model_copy(update={"group_id": f"g{index}"})
        for index, group in enumerate(groups, start=1)
    )


def _apply(
    question: BenchmarkQuestion,
    children: dict[str, Chunk],
) -> BenchmarkQuestion:
    updates: dict[str, object] = {}
    if question.id in APPROVED:
        updates["review_status"] = ReviewStatus.CHECKED
    elif question.id in METADATA_ONLY:
        updates["review_status"] = ReviewStatus.CHECKED
    elif question.id in SIGNIFICANT:
        updates["review_status"] = ReviewStatus.DRAFT
    else:
        return question

    groups = question.gold_evidence_groups
    if question.id in SINGLE_CHILD:
        groups = (
            _group(
                SINGLE_CHILD[question.id],
                group_id="g1",
                role="direct_answer",
                children=children,
            ),
        )
        updates["reasoning_scope"] = ReasoningScope.SINGLE_CHUNK
    if question.id == "mln111_0011":
        groups = (question.gold_evidence_groups[1].model_copy(update={"group_id": "g1"}),)
        updates["reasoning_scope"] = ReasoningScope.SINGLE_CHUNK
    if question.id in {
        "mln111_0012",
        "mln111_0047",
        "mln111_0057",
        "mln111_0066",
        "mln111_0088",
    }:
        updates["reasoning_scope"] = ReasoningScope.MULTI_CHUNK
    if question.id == "mln111_0052":
        groups = _keep_child_groups(question, ("child_f9bf8160cb4080ddc7e5",))
        updates["reasoning_scope"] = ReasoningScope.SINGLE_CHUNK
    if question.id == "mln111_0061":
        updates["question_types"] = (QuestionType.SYNTHESIS,)
    if question.id == "mln111_0021":
        updates["reasoning_scope"] = ReasoningScope.MULTI_CHUNK
    if question.id == "mln111_0050":
        updates["reasoning_scope"] = ReasoningScope.SINGLE_CHUNK
    if question.id == "mln111_0080":
        groups = _keep_child_groups(
            question,
            ("child_f7e5fa1588575689bc1d", "child_d8d7cb4635dae1726a07"),
        )
        updates["reasoning_scope"] = ReasoningScope.MULTI_CHUNK
    if question.id == "mln111_0085":
        updates["reasoning_scope"] = ReasoningScope.SINGLE_CHUNK
    if question.id == "mln111_0071":
        updates["question"] = (
            "Theo giáo trình, các tài liệu triết học thành văn thời cổ đại Hy "
            "Lạp còn được bảo tồn ở mức độ nào?"
        )
        updates["required_concepts"] = (
            "tài liệu triết học thành văn",
            "thất lạc",
            "không còn nguyên vẹn",
        )
        updates["forbidden_claims"] = ()
    if question.id in GOLD_REVISIONS:
        updates["gold_answer"] = GOLD_REVISIONS[question.id]
    updates["gold_evidence_groups"] = groups
    return BenchmarkQuestion.model_validate(
        {**question.model_dump(), **updates},
        strict=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--children", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    draft_ids = {
        question.id for question in questions if question.review_status is ReviewStatus.DRAFT
    }
    expected = APPROVED | METADATA_ONLY | SIGNIFICANT
    if draft_ids != expected or len(expected) != 32:
        raise ValueError("round-2 IDs do not match the 32 draft questions")
    children = {
        chunk.chunk_id: chunk
        for line in args.children.read_text(encoding="utf-8").splitlines()
        if (chunk := Chunk.model_validate_json(line))
    }
    updated = [_apply(question, children) for question in questions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(question.model_dump_json() + "\n" for question in updated),
        encoding="utf-8",
    )

    reviewed_at = datetime(2026, 7, 24, tzinfo=UTC)
    audit: list[BenchmarkReview] = []
    for question in questions:
        if question.id not in expected:
            continue
        digest = hashlib.sha256(question.model_dump_json().encode("utf-8")).hexdigest()
        decision = (
            ReviewDecision.CHECKED
            if question.id in APPROVED | METADATA_ONLY
            else ReviewDecision.REVISE
        )
        audit.append(
            BenchmarkReview(
                question_id=question.id,
                decision=decision,
                reviewer_id="project_owner",
                reviewed_at=reviewed_at,
                benchmark_record_sha256=digest,
                notes="Imported from MLN111 human review round 2.",
            )
        )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        "".join(item.model_dump_json() + "\n" for item in audit),
        encoding="utf-8",
    )
    print("round2=32 checked=19 revised=13 rejected=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
