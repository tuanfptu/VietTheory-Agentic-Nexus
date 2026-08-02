import pytest

from viettheory.retrieval.hardware import recommend_runtime


def test_auto_uses_safe_batch_for_six_gb_gpu() -> None:
    profile = recommend_runtime(
        "auto",
        cuda_available=True,
        gpu_name="NVIDIA GeForce GTX 1660 Ti",
        total_vram_mb=6144,
    )
    assert profile.device == "cuda"
    assert profile.document_batch_size == 8
    assert profile.query_batch_size == 1


def test_cpu_request_never_uses_cuda() -> None:
    profile = recommend_runtime(
        "cpu",
        cuda_available=True,
        gpu_name="GPU",
        total_vram_mb=24_576,
    )
    assert profile.device == "cpu"
    assert profile.document_batch_size == 4


def test_explicit_cuda_fails_if_unavailable() -> None:
    with pytest.raises(RuntimeError, match="not available"):
        recommend_runtime("cuda", cuda_available=False)
