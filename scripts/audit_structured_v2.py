"""Compare deterministic structured artifacts with selective Gemini structure hints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

HEADING_TYPES = frozenset({"chapter", "division", "section", "subsection"})
AUDITABLE_PAGE_ROLES = frozenset({"body", "chapter_opening"})


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _artifact_stats(base: Path) -> dict[str, Any]:
    headings = _jsonl(base / "headings.jsonl")
    parents = _jsonl(base / "parents.jsonl")
    children = _jsonl(base / "children.jsonl")
    covered_pages = sorted(
        {span["pdf_page"] for child in children for span in child["source_spans"]}
    )
    return {
        "headings": len(headings),
        "heading_levels": dict(sorted(Counter(row["level"] for row in headings).items())),
        "parents": len(parents),
        "children": len(children),
        "chaptered_children": sum(row["chapter"] is not None for row in children),
        "unassigned_children": sum(row["chapter"] is None for row in children),
        "covered_page_count": len(covered_pages),
        "covered_page_min": min(covered_pages),
        "covered_page_max": max(covered_pages),
    }


def _gemini_alignment(
    structured_dir: Path, batch_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deterministic = {
        (heading["pdf_page"], heading["block_id"]): heading
        for heading in _jsonl(structured_dir / "headings.jsonl")
    }
    matches = Counter[str]()
    disagreements: list[dict[str, Any]] = []
    for path in sorted(batch_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for page in payload["pages"]:
            if page["page_role"] not in AUDITABLE_PAGE_ROLES:
                continue
            pdf_page = page["pdf_page"]
            for element in page["elements"]:
                if element["element_type"] not in HEADING_TYPES:
                    continue
                candidates = [
                    deterministic[(pdf_page, block_id)]
                    for block_id in element["source_block_ids"]
                    if (pdf_page, block_id) in deterministic
                ]
                if not candidates:
                    matches["unmatched"] += 1
                    disagreements.append(
                        {
                            "pdf_page": pdf_page,
                            "gemini_type": element["element_type"],
                            "gemini_level": element["level"],
                            "text": element["text"],
                            "source_block_ids": element["source_block_ids"],
                            "reason": "no_deterministic_heading_on_anchor",
                        }
                    )
                    continue
                if any(candidate["level"] == element["level"] for candidate in candidates):
                    matches["level_match"] += 1
                else:
                    matches["level_mismatch"] += 1
                    disagreements.append(
                        {
                            "pdf_page": pdf_page,
                            "gemini_type": element["element_type"],
                            "gemini_level": element["level"],
                            "deterministic_levels": sorted(
                                {candidate["level"] for candidate in candidates}
                            ),
                            "text": element["text"],
                            "source_block_ids": element["source_block_ids"],
                            "reason": "hierarchy_level_mismatch",
                        }
                    )
    total = sum(matches.values())
    return (
        {
            "gemini_heading_count": total,
            "level_match": matches["level_match"],
            "level_mismatch": matches["level_mismatch"],
            "unmatched": matches["unmatched"],
            "anchor_match_rate": (
                round((total - matches["unmatched"]) / total, 4) if total else None
            ),
            "exact_level_rate": round(matches["level_match"] / total, 4) if total else None,
        },
        disagreements,
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Structured v2 Audit",
        "",
        "The frozen `structured_v1` artifacts were not modified. `structured_v2` is a",
        "candidate namespace and is not used by production retrieval yet.",
        "",
        "| Subject | v1 children | v2 children | v1 pages | v2 pages | Gemini exact level |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for subject, row in report["subjects"].items():
        lines.append(
            f"| {subject} | {row['v1']['children']} | {row['v2']['children']} | "
            f"{row['v1']['covered_page_count']} | {row['v2']['covered_page_count']} | "
            f"{row['gemini_alignment']['exact_level_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A Gemini disagreement is an audit candidate, not an automatic correction. Any",
            "unmatched anchor or hierarchy mismatch must be reviewed before v2 promotion.",
            "Dense indexes must only be rebuilt after this promotion gate.",
            "",
        ]
    )
    for subject, row in report["subjects"].items():
        alignment = row["gemini_alignment"]
        lines.extend(
            [
                f"### {subject}",
                "",
                f"- Gemini headings: {alignment['gemini_heading_count']}",
                f"- Exact anchor and level matches: {alignment['level_match']}",
                f"- Level mismatches: {alignment['level_mismatch']}",
                f"- Unmatched anchors: {alignment['unmatched']}",
                f"- Review candidates: {len(row['disagreements'])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["VNR202", "MLN131"])
    parser.add_argument(
        "--gemini-root",
        type=Path,
        default=Path("data/processed/selective_structure_audit_final_v2"),
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/structured_v2_audit.json")
    )
    parser.add_argument("--output-md", type=Path, default=Path("reports/structured_v2_audit.md"))
    args = parser.parse_args()

    report: dict[str, Any] = {"schema_version": "1.0", "subjects": {}}
    for subject in args.subjects:
        corpus = Path("data/processed") / subject
        alignment, disagreements = _gemini_alignment(
            corpus / "structured_v2", args.gemini_root / subject / "batches"
        )
        report["subjects"][subject] = {
            "v1": _artifact_stats(corpus / "structured_v1"),
            "v2": _artifact_stats(corpus / "structured_v2"),
            "gemini_alignment": alignment,
            "disagreements": disagreements,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
