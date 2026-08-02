from pathlib import Path

from viettheory.data.metadata_store import MetadataStore
from viettheory.schema import Chunk, SourceSpan


def test_round_trips_chunk(tmp_path: Path) -> None:
    span = SourceSpan(page_id="page-1", pdf_page=0, bbox=(0.0, 0.0, 10.0, 10.0), text="Văn bản")
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        subject_code="MLN111",
        text="Văn bản",
        token_count=2,
        source_spans=(span,),
    )
    with MetadataStore(tmp_path / "metadata.sqlite3") as store:
        store.upsert_chunks((chunk,))
        assert store.get_chunk(chunk.chunk_id) == chunk
