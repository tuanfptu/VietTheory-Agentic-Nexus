"""Run checkpointed Gemini evidence judging and targeted planning on public B0 dev contexts."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from viettheory.benchmark_generation import load_gemini_key
from viettheory.corpus import SearchMode, UnifiedCorpusCatalog
from viettheory.natural_benchmark import NaturalQuestionV2
from viettheory.recovery_v2 import RecoveryPlan, RecoveryPlanBatch
from viettheory.schema import Chunk

PROMPT_VERSION = "recovery_v2_evidence_guided_v1"


def _request(
    cases: list[dict[str, Any]], *, api_key: str, model: str, timeout: float
) -> RecoveryPlanBatch:
    prompt = (
        "You are the bounded Evidence Judge and Recovery Planner for Vietnamese educational "
        "RAG. Use only each question and its current retrieved contexts; never answer from "
        "outside knowledge. Infer independently supportable required aspects. Set activate=false "
        "when all aspects are directly supported, the question is contradicted, or recovery is "
        "not justified. Set activate=true only when at least one precise answer aspect is absent. "
        "When active, write one targeted Vietnamese corpus-search query per missing aspect, at "
        "most two queries. A query must be self-contained, concise, retain exact named entities, "
        "dates and technical concepts from the question, and search only the missing proposition; "
        "do not merely repeat the whole question. Preserve request_id exactly. Gold answers, gold "
        "concepts and gold evidence are not supplied.\nCases JSON:\n"
        + json.dumps(cases, ensure_ascii=False)
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": RecoveryPlanBatch.model_json_schema(),
            "temperature": 0.0,
        },
    }
    encoded = urllib.parse.quote(model, safe="")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{encoded}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    raw = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
    raw["schema_version"] = "1.0"
    for plan in raw.get("plans", []):
        plan["schema_version"] = "1.0"
    return RecoveryPlanBatch.model_validate(raw, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--b0-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--request-interval", type=float, default=5.5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 5:
        raise ValueError("batch-size must be in [1, 5] to bound context payloads")
    api_key = load_gemini_key(args.dotenv)
    if not api_key:
        raise RuntimeError("Gemini API key is not configured")
    questions = {
        q.id: q
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for q in (NaturalQuestionV2.model_validate_json(line),)
    }
    b0 = json.loads(args.b0_report.read_text(encoding="utf-8"))
    rows = b0["variants"]["within_subject_parent_aware_b0"]["per_query"]
    catalog = UnifiedCorpusCatalog(Path.cwd())
    parents = {
        chunk.chunk_id: chunk
        for corpus in catalog.resolve(SearchMode.GLOBAL)
        for line in corpus.parents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for chunk in (Chunk.model_validate_json(line),)
    }
    plans: dict[str, RecoveryPlan] = {}
    if args.checkpoint.exists():
        plans = {
            plan.request_id: plan
            for line in args.checkpoint.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for plan in (RecoveryPlan.model_validate_json(line),)
        }
    pending = [row for row in rows if row["question_id"] not in plans]
    batches = [pending[i : i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(batches, 1):
        cases = []
        for row in batch:
            question = questions[row["question_id"]]
            cases.append(
                {
                    "request_id": question.id,
                    "question": question.question,
                    "contexts": [
                        {"context_id": parent_id, "text": parents[parent_id].text}
                        for parent_id in row["retrieved_ids"][:5]
                    ],
                }
            )
        for attempt in range(1, args.max_attempts + 1):
            try:
                result = _request(cases, api_key=api_key, model=args.model, timeout=args.timeout)
                expected = {case["request_id"] for case in cases}
                actual = {plan.request_id for plan in result.plans}
                if actual != expected or len(actual) != len(result.plans):
                    raise ValueError("planner returned missing, duplicate, or unknown IDs")
                plans.update({plan.request_id: plan for plan in result.plans})
                break
            except (urllib.error.URLError, TimeoutError, ValidationError, ValueError) as exc:
                if attempt == args.max_attempts:
                    raise RuntimeError("Recovery V2 planner exhausted bounded retries") from exc
                time.sleep(min(60.0, 5.0 * 2 ** (attempt - 1)))
        args.checkpoint.write_text(
            "".join(plans[key].model_dump_json() + "\n" for key in sorted(plans)),
            encoding="utf-8",
        )
        print(f"completed planner batch {index}/{len(batches)}", flush=True)
        if index < len(batches):
            time.sleep(args.request_interval)
    print(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "planned": len(plans),
                "activated": sum(plan.activate for plan in plans.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
