# Architecture — MLN111 Assistant

## Production boundary

Production is intentionally restricted to subject `MLN111`. `runtime.py` loads one structured
corpus, one dense index, one embedding model and one reranker. Subject filters are enforced by
the runtime and retriever adapters; artifacts for other subjects cannot enter an answer.

## Offline corpus pipeline

1. **Extraction:** PyMuPDF preserves page number, text blocks, lines, bounding boxes and font
   metadata. Low-quality pages can be routed to the Tesseract OCR fallback.
2. **Structure parsing:** block roles and headings are inferred to preserve chapter/section
   boundaries instead of treating the PDF as a flat character stream.
3. **Parent/child chunking:** small child chunks optimize retrieval precision; bounded parent
   chunks retain complete arguments and become the generation/citation context.
4. **Dense indexing:** Qwen3 embeddings are normalized and stored in FAISS. A manifest pins the
   model identity/revision and vector-to-chunk mapping.
5. **Benchmark validation:** every gold child ID and page is checked against the frozen corpus
   manifest; stale or missing evidence fails validation.

## Online question path

1. The pre-router rejects only explicit non-MLN111 requests. Academic questions proceed to
   retrieval so a brittle keyword list does not cause false refusals.
2. Elliptical follow-ups use only the immediately preceding user/assistant turn. This resolves
   pronouns such as “chúng” while preventing older topics from contaminating retrieval.
3. BM25 and dense retrieval independently produce candidates. Reciprocal Rank Fusion combines
   ranks and deduplicates by stable chunk ID.
4. Comparison questions generate variants for both compared concepts, then round-robin their
   candidates before reranking.
5. Qwen3-Reranker scores question–passage pairs on CUDA. Parent expansion deduplicates sibling
   children and supplies full passages.
6. The evidence gate can accept, rewrite once, refuse as unrelated or refuse as insufficient.
7. Gemini receives only retrieved evidence and an `Answer` JSON schema. Provider output is
   parsed at the untrusted boundary.
8. Citation spans are replaced with canonical retrieved spans, duplicate sources are merged,
   and a deterministic verifier ensures every claim points to retrieved evidence.

## Identity and conversation isolation

- Users register with a unique case-insensitive username.
- Passwords use `hashlib.scrypt` (`N=2^14`, `r=8`, `p=1`) and a random 16-byte salt.
- Login produces a 32-byte URL-safe opaque token; only its SHA-256 digest is stored.
- Sessions expire after seven days and can be revoked by logout.
- Every conversation row has an owner `user_id`. List, message, chat and delete operations query
  by both `conversation_id` and `user_id`; cross-account access returns 404.
- Pre-migration conversations receive owner `legacy` and are invisible to newly registered users.

## Deployment model

The supported demo topology is one CUDA-backed FastAPI process, one Streamlit process and an
optional Cloudflare Quick Tunnel that exposes only Streamlit. The API listens on loopback. A
production deployment should add managed TLS, rate limiting, PostgreSQL, backups, monitoring,
secret management and an authenticated stable tunnel or reverse proxy.
