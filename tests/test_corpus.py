from pathlib import Path

import pytest

from viettheory.corpus import SearchMode, UnifiedCorpusCatalog


def test_catalog_resolves_within_subject_and_global_paths() -> None:
    catalog = UnifiedCorpusCatalog(Path("."))

    selected = catalog.resolve(SearchMode.WITHIN_SUBJECT, "MLN122")
    global_corpora = catalog.resolve(SearchMode.GLOBAL)

    assert selected[0].subject_code == "MLN122"
    assert len(global_corpora) == 5


def test_catalog_rejects_ambiguous_search_contracts() -> None:
    catalog = UnifiedCorpusCatalog(Path("."))
    with pytest.raises(ValueError, match="requires subject_code"):
        catalog.resolve(SearchMode.WITHIN_SUBJECT)
    with pytest.raises(ValueError, match="must not pre-filter"):
        catalog.resolve(SearchMode.GLOBAL, "MLN111")


def test_global_logical_corpus_preserves_subject_provenance() -> None:
    catalog = UnifiedCorpusCatalog(Path("."))
    chunks = catalog.load_children(SearchMode.GLOBAL)

    assert len(chunks) == 2103
    assert {chunk.subject_code for chunk in chunks} == set(catalog.subject_codes)
