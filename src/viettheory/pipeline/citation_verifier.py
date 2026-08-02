"""Deterministic citation validity and completeness checks."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from viettheory.schema import Answer, RetrievedEvidence


class VerificationResult(BaseModel):
    """Machine-actionable result used before optional entailment verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    errors: tuple[str, ...]
    unsupported_claim_ids: tuple[str, ...]


def verify_citations(answer: Answer, evidence: tuple[RetrievedEvidence, ...]) -> VerificationResult:
    """Check citation existence, provenance, non-empty spans, and claim completeness."""
    evidence_by_id = {item.evidence_id: item for item in evidence}
    errors: list[str] = []
    valid_citation_ids: set[str] = set()
    for citation in answer.citations:
        retrieved = evidence_by_id.get(citation.evidence_id)
        if retrieved is None:
            errors.append(f"{citation.citation_id}: evidence was not retrieved")
            continue
        if citation.source_span not in retrieved.chunk.source_spans:
            errors.append(f"{citation.citation_id}: source span is not part of evidence")
            continue
        valid_citation_ids.add(citation.citation_id)

    unsupported = tuple(
        claim.claim_id
        for claim in answer.claims
        if not claim.citation_ids
        or any(citation_id not in valid_citation_ids for citation_id in claim.citation_ids)
    )
    if not answer.refused and not answer.claims:
        errors.append("non-refused answer has no claims")
    if unsupported:
        errors.append("one or more claims lack valid citations")
    return VerificationResult(
        valid=not errors,
        errors=tuple(errors),
        unsupported_claim_ids=unsupported,
    )
