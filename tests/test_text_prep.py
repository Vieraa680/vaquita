from vaquita.core.text_prep import normalize_text, preprocess_query


def test_normalize_text_strips_accents_and_punctuation() -> None:
    assert normalize_text("  ¿Qué onda, Vaquita?! ") == "que onda vaquita"
    assert normalize_text("Me decís la RAM?") == "me decis la ram"


def test_preprocess_query_segments_clauses_without_hardcoded_fillers() -> None:
    result = preprocess_query("que onda? me decís que onda mi ram?")
    assert result.normalized == "que onda me decis que onda mi ram"
    assert result.clauses == ("que onda", "me decis que onda mi ram")
    assert result.core_query == "que onda me decis que onda mi ram"


def test_preprocess_query_strips_injected_fillers() -> None:
    result = preprocess_query(
        "que onda? me decís que onda mi ram?",
        filler_clauses=frozenset({"que onda"}),
        leading_prefixes=("me decis", "que onda"),
    )
    assert result.core_query == "mi ram"

    greeting = preprocess_query(
        "Hola Vaquita!",
        filler_clauses=frozenset({"hola vaquita"}),
        leading_prefixes=("hola",),
    )
    assert greeting.core_query == "hola vaquita"

    empty = preprocess_query("   ?! ")
    assert empty.normalized == ""
    assert empty.clauses == ()
    assert empty.core_query == ""
