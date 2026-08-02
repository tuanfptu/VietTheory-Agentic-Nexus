# VietTheory-RAG

VietTheory-RAG là hệ thống hỏi đáp có dẫn nguồn cho năm giáo trình lý luận chính trị.
Mục tiêu là truy xuất đúng nội dung, chỉ trả lời dựa trên bằng chứng và liên kết từng
nhận định tới đúng trang cùng vùng văn bản trong PDF.

## Kiến trúc

Luồng mục tiêu:

`PDF → extraction + bbox → parent/child chunks → BM25 + dense → RRF → reranker →
evidence gate → structured generation → citation verification → API/UI`

Mã Python dùng `src` layout dưới package `viettheory`. PDF, model, index, dữ liệu xử lý
và secrets nằm ngoài package. Quyền sử dụng corpus được ghi tại
[`docs/data-license.md`](docs/data-license.md); giấy phép code và dữ liệu được quản lý
riêng biệt.

## Trạng thái

Hai tài liệu native-text MLN111 và MLN122 hiện có:

- schema v1 nghiêm ngặt và stable IDs;
- audit PDF, native-text extraction, bbox validation và baseline chunking 400/50;
- benchmark MLN111 schema v1 với 70 câu development đã human-review đầy đủ;
- FAISS dense indexing/retrieval với manifest và checksum;
- GPU runtime đã xác minh trên GTX 1660 Ti;
- cấu hình GPU tự động: embedding batch 8, resident FP16 reranker batch 4;
- structure parser MLN111 với 231 headings và parent–child chunks 175/603;
- pipeline MLN122 với 262/262 trang, 183 headings, parent–child chunks 148/348
  và dense index 348 vectors;
- BM25 tiếng Việt, Reciprocal Rank Fusion và lọc theo môn;
- pre-router, evidence gate có threshold calibrate từ dev set;
- Gemini structured adapter, citation verifier cấp code và SQLite metadata store.

HCM202, MLN131 và VNR202 đã được OCR đủ 774/774 trang bằng Tesseract tiếng Việt,
kiểm tra bbox trực quan, tạo parent–child chunks và dense index GPU. Cả năm giáo
trình hiện đã có extraction artifact và Qwen dense index có checksum. MLN111 có thêm
30 câu hidden-test dạng draft trong khu vực local bị Git ignore; bốn môn còn lại chưa
có benchmark hỏi–đáp. Bước tiếp theo là review hidden test MLN111, freeze benchmark
v1.0, đánh giá hybrid retrieval/reranker và mở rộng benchmark sang toàn corpus.

## Cài đặt phát triển

Yêu cầu Python 3.11 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,retrieval]"
```

Sao chép `.env.example` thành `.env` rồi tự điền credential. Không đặt API key trong
source code hoặc commit `.env`.

## Kiểm tra chất lượng

```powershell
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src tests
python -m pytest
```

## Trích xuất và chunk PDF

Chỉ số trang là zero-based; `--end-page` là exclusive.

```powershell
viettheory-extract "Tài liệu/Giáo trình MLN111.pdf" `
  --subject MLN111 --start-page 0 --end-page 10 `
  --output data/interim/MLN111_pages.jsonl

viettheory-chunk data/interim/MLN111_pages.jsonl `
  --output data/processed/MLN111/chunks.jsonl
```

Extraction và indexing đều sinh manifest có checksum để xác minh khả năng tái lập.

## Chạy API và UI local

API factory không tự load model khi import. Cho tới khi runtime artifact được cấu hình,
`/health` trả `pipeline_ready=false` và `/ask` chủ động trả HTTP 503.

```powershell
uvicorn viettheory.backend.app:app --host 127.0.0.1 --port 8000
streamlit run src/viettheory/frontend/app.py
```

Các endpoint API:

- `GET /health`: trạng thái process và pipeline.
- `POST /ask`: nhận `{ "question": "..." }`, trả structured `Answer`.
- `POST /feedback`: lưu đánh giá local vào SQLite ignored bởi Git.
