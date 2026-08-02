"""Render a human-review Markdown pack from benchmark questions and child chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from viettheory.benchmark import BenchmarkQuestion
from viettheory.schema import Chunk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--children", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status",
        action="append",
        help="Include only these review statuses; may be passed more than once.",
    )
    args = parser.parse_args()
    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.status:
        allowed_statuses = set(args.status)
        questions = [
            question for question in questions if question.review_status.value in allowed_statuses
        ]
    children = {
        chunk.chunk_id: chunk
        for line in args.children.read_text(encoding="utf-8").splitlines()
        if (chunk := Chunk.model_validate_json(line))
    }
    lines = [
        "# Benchmark human review pack",
        "",
        "Đánh dấu đúng một quyết định cho từng câu. Không đổi `id` hoặc child ID.",
        "",
    ]
    for question in questions:
        lines.extend(
            [
                f"## {question.id} — {question.question}",
                "",
                f"- Split: `{question.split.value}`",
                f"- Difficulty: `{question.difficulty.value}`",
                "- Types: " + ", ".join(f"`{kind.value}`" for kind in question.question_types),
                f"- Reasoning: `{question.reasoning_scope.value}`",
                "- Chapter: " + ("; ".join(question.chapter_labels) or "(chưa gán)"),
                f"- Expected behavior: `{question.expected_behavior.value}`",
                f"- Gold answer: {question.gold_answer or '(không áp dụng)'}",
                "",
            ]
        )
        for group in question.gold_evidence_groups:
            lines.append(f"### Evidence {group.group_id} — {group.role}")
            lines.append("")
            lines.append("- PDF pages: " + ", ".join(map(str, group.gold_pdf_pages)))
            for child_id in group.primary_child_ids:
                child = children[child_id]
                excerpt = " ".join(child.text.split())
                lines.extend(
                    [
                        f"- Child: `{child_id}`",
                        "",
                        f"> {excerpt}",
                        "",
                    ]
                )
        lines.extend(
            [
                "- Decision: `[ ] approve` `[ ] revise` `[ ] reject`",
                "- Gold evidence correct: `[ ] yes` `[ ] no`",
                "- Missing acceptable chunk IDs:",
                "- Reviewer notes:",
                "",
                "---",
                "",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"review_items={len(questions)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
