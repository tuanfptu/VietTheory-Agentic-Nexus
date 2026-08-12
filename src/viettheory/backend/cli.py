"""Start the GPU-backed MLN111 Assistant API."""

from __future__ import annotations

import os

import uvicorn

from viettheory.backend.app import create_app
from viettheory.runtime import build_pipeline


def main() -> None:
    pipeline = build_pipeline()
    uvicorn.run(
        create_app(pipeline),
        host=os.getenv("VIETTHEORY_HOST", "127.0.0.1"),
        port=int(os.getenv("VIETTHEORY_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
