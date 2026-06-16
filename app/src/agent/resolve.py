from __future__ import annotations

import logging

from agent.session import _nlp

logger = logging.getLogger(__name__)


################################################################################
# Internal helpers
################################################################################

def _lemmatize_phrase(text: str) -> str:
    """Lemmatise a (possibly multi-word) phrase via spaCy."""
    doc = _nlp()(text.lower())
    return " ".join(token.lemma_ for token in doc)


def _lemmatize_word(word: str) -> str:
    """Lemmatise a single token."""
    doc = _nlp()(word.lower())
    return doc[0].lemma_ if doc else word.lower()


################################################################################
# Public API
################################################################################

# Leading/trailing words that carry no semantic content and block exact matches.
_STRIP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "some", "many", "much", "lot", "lots",
    "so", "very", "quite", "really", "more", "most",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
})


def _strip_determiners(text: str) -> str:
    """Remove leading/trailing determiners and articles from a phrase."""
    tokens = text.split()
    while tokens and tokens[0] in _STRIP_WORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _STRIP_WORDS:
        tokens = tokens[:-1]
    return " ".join(tokens)


def resolve_concept(
    concept: str,
    kw_set: set[str],
    *,
    return_method: bool = False,
) -> "list[str] | tuple[list[str], str]":
    """Map a natural-language concept to one or more ARASAAC keyword strings.

    Resolution steps (first match wins):
        0. strip      — remove leading/trailing determiners, then retry steps 1-4
        1. exact      — lowercase exact match
        2. lemma      — full-phrase spaCy lemma
        3. hyphen     — space ↔ hyphen normalisation
        3b. lemma_alt — lemma of the normalised form
        4. token      — individual lemmatised tokens (len > 2), reusing the doc
                        from step 2 — no extra spaCy calls per token
        none          — no match found

    Set return_method=True to get (keywords, method_label) for eval tracing.
    """
    c = concept.strip().lower()
    if not c:
        return ([], "none") if return_method else []

    def _lookup(phrase: str) -> "list[str] | None":
        if not phrase:
            return None

        # Step 1: exact match
        if phrase in kw_set:
            return [phrase]

        # Step 2: full-phrase lemma — run spaCy once and keep the doc for step 4
        doc   = _nlp()(phrase)
        lemma = " ".join(t.lemma_ for t in doc)
        if lemma != phrase and lemma in kw_set:
            return [lemma]

        # Step 3: space ↔ hyphen, then lemma of that form (different string → own call)
        alt = phrase.replace(" ", "-") if " " in phrase else phrase.replace("-", " ")
        if alt in kw_set:
            return [alt]
        lemma_alt = " ".join(t.lemma_ for t in _nlp()(alt))
        if lemma_alt not in (phrase, lemma, alt) and lemma_alt in kw_set:
            return [lemma_alt]

        # Step 4: individual token lemmas — reuse doc from step 2, no extra spaCy calls
        token_lemmas = [t.lemma_ for t in doc if len(t.text) > 2]
        found        = [t for t in dict.fromkeys(token_lemmas) if t in kw_set]
        if found:
            return found

        return None

    # First attempt on the raw concept
    result = _lookup(c)
    if result is not None:
        method = _resolve_method_label(c, result, kw_set)
        return (result, method) if return_method else result

    # Retry after stripping determiners (e.g. "a banana" → "banana")
    stripped = _strip_determiners(c)
    if stripped and stripped != c:
        result = _lookup(stripped)
        if result is not None:
            method = _resolve_method_label(stripped, result, kw_set)
            return (result, method) if return_method else result

    return ([], "none") if return_method else []


def _resolve_method_label(phrase: str, matches: list[str], kw_set: set[str]) -> str:
    """Re-derive the method label for a phrase already known to match."""
    if phrase in kw_set:
        return "exact"
    # Reuse one doc object for both lemma and token checks
    doc   = _nlp()(phrase)
    lemma = " ".join(t.lemma_ for t in doc)
    if lemma in kw_set:
        return "lemma"
    alt = phrase.replace(" ", "-") if " " in phrase else phrase.replace("-", " ")
    if alt in kw_set:
        return "hyphen"
    if " ".join(t.lemma_ for t in _nlp()(alt)) in kw_set:
        return "lemma_alt"
    return "token"
