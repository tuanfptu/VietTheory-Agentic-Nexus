from viettheory.pipeline.evidence_gate import (
    GateAction,
    GateThresholds,
    calibrate_sufficient_threshold,
    decide_evidence,
)
from viettheory.pipeline.pre_router import QuestionType, route_question, route_subject
from viettheory.schema import Chunk, RetrievedEvidence, SourceSpan


def _evidence(score: float) -> RetrievedEvidence:
    text = "Phép biện chứng duy vật"
    chunk = Chunk(
        chunk_id="c1",
        document_id="doc",
        subject_code="MLN111",
        text=text,
        token_count=4,
        source_spans=(SourceSpan(page_id="p1", pdf_page=1, bbox=(0.0, 0.0, 1.0, 1.0), text=text),),
    )
    return RetrievedEvidence(
        evidence_id="S1", chunk=chunk, score=score, rank=1, retrieval_method="dense"
    )


def test_router_detects_question_type() -> None:
    route = route_question("Khái niệm phép biện chứng duy vật là gì?")
    assert route.question_type is QuestionType.DEFINITION
    assert not route.obvious_out_of_scope


def test_router_rejects_only_obvious_out_of_scope() -> None:
    route = route_question("Thời tiết hôm nay thế nào?")
    assert route.obvious_out_of_scope


def test_router_detects_comparison() -> None:
    route = route_question("So sánh chủ nghĩa duy vật trước Mác và triết học Mác-Lênin.")

    assert route.question_type is QuestionType.COMPARISON


def test_subject_router_sends_history_of_party_question_to_vnr202() -> None:
    subject = route_subject(
        "Tóm tắt ba giai đoạn lớn trong lịch sử Đảng.",
        frozenset({"MLN111", "MLN122", "MLN131", "HCM202", "VNR202"}),
    )
    assert subject == "VNR202"


def test_subject_router_keeps_ambiguous_question_global() -> None:
    subject = route_subject(
        "Hãy giải thích nội dung này.",
        frozenset({"MLN111", "MLN122", "MLN131", "HCM202", "VNR202"}),
    )
    assert subject is None


def test_gate_allows_only_one_rewrite() -> None:
    thresholds = GateThresholds(sufficient_score=0.7, related_score=0.3)
    assert decide_evidence((_evidence(0.5),), thresholds).action is GateAction.REWRITE
    assert (
        decide_evidence((_evidence(0.5),), thresholds, already_retried=True).action
        is GateAction.REFUSE_INSUFFICIENT
    )


def test_gate_accepts_reranker_logit_thresholds() -> None:
    thresholds = GateThresholds(sufficient_score=-1.0, related_score=-5.0)

    assert thresholds.sufficient_score == -1.0
    assert thresholds.related_score == -5.0


def test_calibration_uses_labeled_dev_scores() -> None:
    threshold = calibrate_sufficient_threshold(
        ((0.1, False), (0.2, False), (0.8, True), (0.9, True))
    )
    assert threshold == 0.8
