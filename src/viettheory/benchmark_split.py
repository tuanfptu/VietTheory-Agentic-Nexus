"""Leakage-safe deterministic splitting for Natural QA benchmark releases."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

from viettheory.natural_benchmark import NaturalQuestionV2


@dataclass(frozen=True)
class SplitResult:
    development: tuple[NaturalQuestionV2, ...]
    hidden: tuple[NaturalQuestionV2, ...]
    component_sizes: tuple[int, ...]


def evidence_parent_ids(record: NaturalQuestionV2) -> frozenset[str]:
    """Return every gold parent referenced by a question."""
    return frozenset(
        parent_id
        for group in record.required_evidence_groups
        for parent_id in group.gold_parent_ids
    )


def semantic_components(
    records: Iterable[NaturalQuestionV2],
) -> tuple[tuple[NaturalQuestionV2, ...], ...]:
    """Group records connected by shared evidence parents.

    Unanswerable records without evidence remain independent components.
    """
    ordered = tuple(sorted(records, key=lambda record: record.id))
    by_id = {record.id: record for record in ordered}
    parent_to_ids: dict[str, set[str]] = defaultdict(set)
    for record in ordered:
        for parent_id in evidence_parent_ids(record):
            parent_to_ids[parent_id].add(record.id)

    adjacency = {record.id: set[str]() for record in ordered}
    for ids in parent_to_ids.values():
        for question_id in ids:
            adjacency[question_id].update(ids - {question_id})

    seen: set[str] = set()
    components: list[tuple[NaturalQuestionV2, ...]] = []
    for question_id in sorted(adjacency):
        if question_id in seen:
            continue
        stack = [question_id]
        seen.add(question_id)
        members: list[NaturalQuestionV2] = []
        while stack:
            current = stack.pop()
            members.append(by_id[current])
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(members, key=lambda record: record.id)))
    return tuple(components)


def _component_key(component: tuple[NaturalQuestionV2, ...], seed: str) -> str:
    ids = "|".join(record.id for record in component)
    return sha256(f"{seed}|{ids}".encode()).hexdigest()


def _distribution(records: Iterable[NaturalQuestionV2]) -> Counter[str]:
    result: Counter[str] = Counter()
    for record in records:
        result[f"category:{record.primary_category.value}"] += 1
        result[f"difficulty:{record.difficulty.value}"] += 1
        result[f"answerability:{record.answerability.value}"] += 1
    return result


def split_subject(
    records: Iterable[NaturalQuestionV2],
    *,
    hidden_size: int,
    seed: str,
    beam_width: int = 4096,
) -> SplitResult:
    """Select an exact hidden size without splitting shared-parent components."""
    ordered = tuple(sorted(records, key=lambda record: record.id))
    if not 0 < hidden_size < len(ordered):
        raise ValueError("hidden_size must be between zero and the subject size")
    subjects = {record.subject_code for record in ordered}
    if len(subjects) != 1:
        raise ValueError("split_subject requires records from exactly one subject")

    components = tuple(
        sorted(semantic_components(ordered), key=lambda item: _component_key(item, seed))
    )
    target_full = _distribution(ordered)
    target = {key: value * hidden_size / len(ordered) for key, value in target_full.items()}

    # Beam states are (component indexes, distribution). Keeping a bounded beam makes the
    # constrained subset selection deterministic and inexpensive for large semantic families.
    states: dict[int, list[tuple[tuple[int, ...], Counter[str]]]] = {0: [((), Counter())]}
    for index, component in enumerate(components):
        size = len(component)
        component_distribution = _distribution(component)
        updated = {count: list(items) for count, items in states.items()}
        for count, candidates in states.items():
            new_count = count + size
            if new_count > hidden_size:
                continue
            bucket = updated.setdefault(new_count, [])
            for indexes, distribution in candidates:
                bucket.append(((*indexes, index), distribution + component_distribution))
        for count, candidates in updated.items():
            scale = count / hidden_size if hidden_size else 0.0
            candidates.sort(
                key=lambda item: (
                    sum(
                        abs(item[1].get(key, 0) - expected * scale)
                        for key, expected in target.items()
                    ),
                    item[0],
                )
            )
            updated[count] = candidates[:beam_width]
        states = updated

    if hidden_size not in states:
        sizes = sorted(len(component) for component in components)
        raise ValueError(f"cannot form exact hidden size {hidden_size} from components {sizes}")
    candidates = states[hidden_size]
    candidates.sort(
        key=lambda item: (
            sum(abs(item[1].get(key, 0) - expected) for key, expected in target.items()),
            item[0],
        )
    )
    selected_indexes = frozenset(candidates[0][0])
    hidden_ids = {
        record.id
        for index, component in enumerate(components)
        if index in selected_indexes
        for record in component
    }
    hidden = tuple(record for record in ordered if record.id in hidden_ids)
    development = tuple(record for record in ordered if record.id not in hidden_ids)
    return SplitResult(
        development=development,
        hidden=hidden,
        component_sizes=tuple(sorted((len(component) for component in components), reverse=True)),
    )


def assert_no_parent_leakage(result: SplitResult) -> None:
    """Raise when a gold evidence parent appears in both splits."""
    development_parents = set().union(
        *(evidence_parent_ids(record) for record in result.development)
    )
    hidden_parents = set().union(*(evidence_parent_ids(record) for record in result.hidden))
    overlap = development_parents & hidden_parents
    if overlap:
        raise ValueError(f"evidence-parent leakage detected: {sorted(overlap)}")
