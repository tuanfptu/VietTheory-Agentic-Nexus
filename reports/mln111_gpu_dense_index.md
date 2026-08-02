# MLN111 GPU Dense Index Gate

## Environment

- GPU: NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM.
- NVIDIA driver: 610.62.
- PyTorch: 2.11.0+cu128.
- Embedding model: Qwen/Qwen3-Embedding-0.6B.
- Model revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

## CUDA verification

- `torch.cuda.is_available()`: true.
- CUDA matrix multiplication: finite 2048 x 2048 output.
- Qwen query embedding: shape `(2, 1024)`, all values finite.
- Qwen smoke-test peak allocation: approximately 1.16 GB VRAM.

## Index artifact

- Index type: FAISS `IndexFlatIP`.
- Normalized vectors: yes.
- Vector count: 527.
- Dimension: 1024.
- Batch size: 16.
- GPU indexing time: approximately 2 minutes 15 seconds.
- Chunk artifact SHA-256:
  `6aad4db0f99fe1c66dc28f16335b68a1ac90d1bb3dba64f6c8b29778f0b90532`.
- Index SHA-256:
  `14116079fe02ce8ea6c6acb9be98a2edc06a48079e0ca6248ce6de1521ddc587`.
- Mapping SHA-256:
  `f6fa7f9a7a46305a7dd13e086b4455fd0a2a1cb3f0bbf7cfcb93592aeb3b9dc8`.

## Retrieval smoke test

Three Vietnamese queries were encoded on GPU and searched against the persisted
index:

- “Vật chất là gì?”: rank 1 points to PDF page 73 and contains Lenin's definition.
- “Hai nguyên lý cơ bản của phép biện chứng duy vật là gì?”: top results point to
  PDF pages 106–107.
- “Ý thức có nguồn gốc tự nhiên và xã hội như thế nào?”: top results point to PDF
  pages 88–90.

The retriever loaded and validated the index, mapping and chunk checksums before
search. All returned chunk IDs resolved to source spans.

## Gate result

**PASS for GPU runtime, persisted dense index integrity and qualitative retrieval
smoke.** Quantitative retrieval quality remains pending human review of the
benchmark and dev-set evaluation.
