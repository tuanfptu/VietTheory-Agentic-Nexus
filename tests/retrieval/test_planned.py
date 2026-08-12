from viettheory.retrieval.planned import comparison_query_variants


def test_comparison_query_variants_cover_both_sides() -> None:
    variants = comparison_query_variants(
        "So sánh quan điểm duy vật trước Mác và quan điểm triết học Mác-Lênin."
    )

    assert variants == (
        "quan điểm duy vật trước Mác",
        "quan điểm triết học Mác-Lênin",
    )


def test_non_comparison_keeps_original_query() -> None:
    question = "Vật chất theo V.I. Lênin là gì?"

    assert comparison_query_variants(question) == (question,)
