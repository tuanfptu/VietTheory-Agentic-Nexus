"""Run checkpointed Gemini J1 evaluation on Evidence Sufficiency development."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from viettheory.benchmark_generation import load_gemini_key
from viettheory.evidence_judge import JudgeBatch, JudgeDecision
from viettheory.evidence_sufficiency import EvidenceSufficiencyCase, SufficiencyLabel

PROMPT_VERSION = "evidence_judge_v1"


def _request(
    cases: tuple[EvidenceSufficiencyCase, ...],
    *,
    api_key: str,
    model: str,
    timeout: float,
) -> JudgeBatch:
    payload_cases = [
        {
            "case_id": case.case_id,
            "question": case.question,
            "contexts": [
                {"context_id": context.parent_id, "text": context.text}
                for context in case.provided_contexts
            ],
        }
        for case in cases
    ]
    prompt = (
        "You are an evidence-sufficiency judge for Vietnamese educational QA. "
        "Infer independently supportable answer aspects from each question, then inspect only "
        "the supplied contexts. Never use outside knowledge. Labels: SUFFICIENT when every "
        "required aspect is directly supported; PARTIAL when at least one aspect is supported "
        "and at least one is unsupported; MISSING when no context is supplied; WRONG_ASPECT "
        "when context exists but supports none of the required answer aspects; CONTRADICTED "
        "only for a direct conflict. Similar topic or repeated question wording is not support. "
        "Return one decision for every case_id, preserving IDs exactly. Write aspect descriptions "
        "and rationale in Vietnamese. Gold labels and gold evidence are not provided.\n"
        f"Cases JSON:\n{json.dumps(payload_cases, ensure_ascii=False)}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": JudgeBatch.model_json_schema(),
            "temperature": 0.0,
        },
    }
    encoded_model = urllib.parse.quote(model, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{encoded_model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    raw = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
    batch = JudgeBatch.model_validate(raw, strict=False)
    expected = {case.case_id for case in cases}
    actual = {decision.case_id for decision in batch.decisions}
    if actual != expected or len(actual) != len(batch.decisions):
        raise ValueError("Gemini Judge returned missing, duplicate, or unknown case IDs")
    return batch


def _call_with_retry(
    cases: tuple[EvidenceSufficiencyCase, ...],
    *,
    api_key: str,
    model: str,
    timeout: float,
    max_attempts: int,
) -> JudgeBatch:
    for attempt in range(1, max_attempts + 1):
        try:
            return _request(cases, api_key=api_key, model=model, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValidationError) as exc:
            if attempt == max_attempts:
                raise RuntimeError("Gemini Judge failed after bounded retries") from exc
            delay = min(60.0, 5.0 * (2 ** (attempt - 1)))
            print(f"judge retry attempt={attempt} delay={delay:.1f}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _evaluate(
    cases: tuple[EvidenceSufficiencyCase, ...], decisions: dict[str, JudgeDecision]
) -> dict[str, Any]:
    labels = tuple(SufficiencyLabel)
    confusion = {gold.value: {pred.value: 0 for pred in labels} for gold in labels}
    rows: list[dict[str, Any]] = []
    for case in cases:
        decision = decisions[case.case_id]
        confusion[case.expected_label.value][decision.label.value] += 1
        rows.append(
            {
                "case_id": case.case_id,
                "source_question_id": case.source_question_id,
                "gold": case.expected_label.value,
                "predicted": decision.label.value,
                "correct": decision.label is case.expected_label,
                "required_aspects": list(decision.required_aspects),
                "covered_aspects": list(decision.covered_aspects),
                "missing_aspects": list(decision.missing_aspects),
                "rationale": decision.rationale,
            }
        )
    f1: dict[str, float] = {}
    for label in labels:
        name = label.value
        tp = confusion[name][name]
        fp = sum(confusion[other.value][name] for other in labels if other is not label)
        fn = sum(confusion[name][other.value] for other in labels if other is not label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    active = [label.value for label in labels if any(confusion[label.value].values())]
    return {
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "macro_f1": sum(f1[label] for label in active) / len(active),
        "f1_by_label": f1,
        "confusion": confusion,
        "predicted_distribution": dict(Counter(row["predicted"] for row in rows)),
        "per_case": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--request-interval", type=float, default=5.5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    api_key = load_gemini_key(args.dotenv)
    if not api_key:
        raise RuntimeError("Gemini API key is not configured")
    cases = tuple(
        EvidenceSufficiencyCase.model_validate_json(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    split_values = {case.split for case in cases}
    if len(split_values) != 1:
        raise ValueError(f"Judge input mixes splits: {sorted(split_values)}")
    evaluation_split = next(iter(split_values))
    decisions: dict[str, JudgeDecision] = {}
    if args.checkpoint.exists():
        decisions = {
            decision.case_id: decision
            for line in args.checkpoint.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for decision in (JudgeDecision.model_validate_json(line),)
        }
    pending_by_source: dict[str, list[EvidenceSufficiencyCase]] = {}
    for case in cases:
        if case.case_id not in decisions:
            pending_by_source.setdefault(case.source_question_id, []).append(case)
    batches = [tuple(source_cases) for _, source_cases in sorted(pending_by_source.items())]
    if any(len(batch) > args.batch_size for batch in batches):
        raise ValueError("batch size must fit all perturbations for one source question")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for index, batch_cases in enumerate(batches, 1):
        batch = _call_with_retry(
            batch_cases,
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
        )
        decisions.update({decision.case_id: decision for decision in batch.decisions})
        args.checkpoint.write_text(
            "".join(decisions[case_id].model_dump_json() + "\n" for case_id in sorted(decisions)),
            encoding="utf-8",
        )
        print(f"completed judge batch {index}/{len(batches)}", flush=True)
        if index < len(batches):
            time.sleep(args.request_interval)
    if decisions.keys() != {case.case_id for case in cases}:
        raise ValueError("Judge checkpoint is incomplete")
    report: dict[str, Any] = {
        "benchmark_version": cases[0].benchmark_version,
        "split": evaluation_split,
        "judge": "J1_gemini",
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "case_count": len(cases),
        "gold_labels_not_sent_to_model": True,
        **_evaluate(cases, decisions),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}))


if __name__ == "__main__":
    main()
