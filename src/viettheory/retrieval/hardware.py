"""GPU-aware runtime selection with conservative VRAM headroom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeviceRequest = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class RetrievalRuntimeProfile:
    """Resolved device and measured-safe batching policy."""

    device: Literal["cpu", "cuda"]
    document_batch_size: int
    query_batch_size: int
    gpu_name: str | None
    total_vram_mb: int | None


def recommend_runtime(
    requested: DeviceRequest,
    *,
    cuda_available: bool,
    gpu_name: str | None = None,
    total_vram_mb: int | None = None,
) -> RetrievalRuntimeProfile:
    """Choose a profile while leaving room for a resident reranker."""
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    use_cuda = cuda_available and requested != "cpu"
    if not use_cuda:
        return RetrievalRuntimeProfile(
            device="cpu",
            document_batch_size=4,
            query_batch_size=1,
            gpu_name=None,
            total_vram_mb=None,
        )
    if total_vram_mb is None or total_vram_mb <= 8 * 1024:
        document_batch_size = 8
    elif total_vram_mb <= 16 * 1024:
        document_batch_size = 16
    else:
        document_batch_size = 32
    return RetrievalRuntimeProfile(
        device="cuda",
        document_batch_size=document_batch_size,
        query_batch_size=1,
        gpu_name=gpu_name,
        total_vram_mb=total_vram_mb,
    )


def detect_runtime(requested: DeviceRequest = "auto") -> RetrievalRuntimeProfile:
    """Inspect PyTorch CUDA state and return the matching policy."""
    available = torch.cuda.is_available()
    if not available:
        return recommend_runtime(requested, cuda_available=False)
    properties = torch.cuda.get_device_properties(0)
    return recommend_runtime(
        requested,
        cuda_available=True,
        gpu_name=torch.cuda.get_device_name(0),
        total_vram_mb=round(properties.total_memory / 1024**2),
    )
