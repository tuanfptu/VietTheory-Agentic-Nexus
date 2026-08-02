"""Question-answering pipeline components."""

from viettheory.pipeline.citation_verifier import VerificationResult, verify_citations
from viettheory.pipeline.evidence_gate import GateAction, GateThresholds, decide_evidence
from viettheory.pipeline.generator import GeminiGenerator, GeneratorAdapter
from viettheory.pipeline.orchestrator import RagPipeline
from viettheory.pipeline.pre_router import RouteDecision, route_question

__all__ = [
    "GateAction",
    "GateThresholds",
    "GeminiGenerator",
    "GeneratorAdapter",
    "RagPipeline",
    "RouteDecision",
    "VerificationResult",
    "decide_evidence",
    "route_question",
    "verify_citations",
]
