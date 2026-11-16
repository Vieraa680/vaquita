from dataclasses import dataclass
import re
import unicodedata

_CLAUSE_SPLIT_RE = re.compile(r"[?!.,;¿¡]+")


def normalize_text(text: str) -> str:
    """Lowercase, strip diacritics, and collapse non-alphanumeric separators."""
    decomposed = unicodedata.normalize("NFD", text.strip().lower())
    without_accents = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    cleaned = re.sub(r"[^a-z0-9\s]", " ", without_accents)
    return " ".join(cleaned.split())


def _strip_leading_prefixes(clause: str, prefixes: tuple[str, ...]) -> str:
    if not prefixes:
        return clause

    current = clause
    while current:
        matched = False
        for prefix in prefixes:
            if current.startswith(prefix + " "):
                remainder = current[len(prefix) + 1 :].strip()
                if remainder:
                    current = remainder
                    matched = True
                    break
        if not matched:
            break
    return current


@dataclass(frozen=True)
class QueryPreprocessResult:
    original: str
    normalized: str
    clauses: tuple[str, ...]
    core_query: str


def preprocess_query(
    text: str,
    *,
    filler_clauses: frozenset[str] = frozenset(),
    leading_prefixes: tuple[str, ...] = (),
) -> QueryPreprocessResult:
    """Segment a query into normalized clauses and optionally strip configured fillers."""
    normalized = normalize_text(text)
    if not normalized:
        return QueryPreprocessResult(
            original=text,
            normalized="",
            clauses=(),
            core_query="",
        )

    raw_clauses = [normalize_text(part) for part in _CLAUSE_SPLIT_RE.split(text)]
    clauses = tuple(c for c in raw_clauses if c)
    if not clauses:
        clauses = (normalized,)

    operative = [
        _strip_leading_prefixes(c, leading_prefixes)
        for c in clauses
        if c not in filler_clauses
    ]
    operative = [c for c in operative if c]

    core_query = " ".join(operative) if operative else normalized

    return QueryPreprocessResult(
        original=text,
        normalized=normalized,
        clauses=clauses,
        core_query=core_query,
    )
