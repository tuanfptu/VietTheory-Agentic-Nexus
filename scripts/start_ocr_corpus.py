"""Run one background-window OCR process per scanned textbook and wait."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subjects",
        nargs="*",
        default=["HCM202", "MLN131", "VNR202"],
        choices=("HCM202", "MLN131", "VNR202"),
    )
    args = parser.parse_args()
    root = Path.cwd()
    log_dir = root / "tmp" / "ocr_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    launched: list[dict[str, int | str]] = []
    processes: list[tuple[str, subprocess.Popen[bytes]]] = []

    for subject in args.subjects:
        pdf_path = next((root / "Tài liệu").glob(f"*{subject}.pdf"))
        output = root / "data" / "processed" / subject / "pages.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        stdout = (log_dir / f"{subject}.out.log").open("w", encoding="utf-8")
        stderr = (log_dir / f"{subject}.err.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-u",
            "-m",
            "viettheory.extraction.ocr_cli",
            str(pdf_path),
            "--subject",
            subject,
            "--output",
            str(output),
            "--tesseract",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            "--tessdata",
            str(root / "models" / "tesseract"),
        ]
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True,
        )
        stdout.close()
        stderr.close()
        launched.append({"subject": subject, "pid": process.pid})
        processes.append((subject, process))

    state_path = log_dir / "processes.json"
    state_path.write_text(json.dumps(launched, indent=2), encoding="utf-8")
    print(json.dumps(launched))
    failed: list[tuple[str, int]] = []
    for subject, process in processes:
        return_code = process.wait()
        print(f"{subject} exited with {return_code}", flush=True)
        if return_code:
            failed.append((subject, return_code))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
