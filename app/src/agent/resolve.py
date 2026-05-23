"""resolve.py — concept-to-keyword resolution for ARASAAC search.

Maps a natural-language concept (e.g. "going out", "water bottle") to one or
more keyword strings that exist in the ARASAAC keyword index.

Resolution order (first match wins for single results):
  1. Exact match
  2. Full-phrase lemma via spaCy
  3. Space ↔ hyphen normalisation (+ lemma of the normalised form)
  4. Fallback: individual lemmatised tokens that exist in the index

spaCy and the en_core_web_sm model are optional. If unavailable the module
falls back to lowercasing only, which still handles steps 1 and 3 correctly.

Install:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── spaCy — optional, graceful fallback ──────────────────────────────────────

try:
    import spacy as _spacy
    _nlp = _spacy.load("en_core_web_sm", disable=["parser", "ner"])
    _SPACY_OK = True
    logger.info("resolve: spaCy en_core_web_sm loaded — lemmatisation active.")
except ImportError:
    _SPACY_OK = False
    logger.warning(
        "resolve: spaCy not installed — lemmatisation disabled. "
        "Run: pip install spacy && python -m spacy download en_core_web_sm"
    )
except OSError:
    _SPACY_OK = False
    logger.warning(
        "resolve: en_core_web_sm model not found — lemmatisation disabled. "
        "Run: python -m spacy download en_core_web_sm"
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _lemmatize_phrase(text: str) -> str:
    """Return the lemmatised form of a (possibly multi-word) phrase.

    Example: "going out" → "go out"  |  "brushing teeth" → "brush tooth"
    Falls back to lowercasing when spaCy is unavailable.
    """
    if not _SPACY_OK:
        return text.lower()
    doc = _nlp(text.lower())
    return " ".join(token.lemma_ for token in doc)


def _lemmatize_word(word: str) -> str:
    """Return the lemma of a single word token.

    Example: "eating" → "eat"  |  "coats" → "coat"
    Falls back to lowercasing when spaCy is unavailable.
    """
    if not _SPACY_OK:
        return word.lower()
    doc = _nlp(word.lower())
    return doc[0].lemma_ if doc else word.lower()


# ── Public API ────────────────────────────────────────────────────────────────

# Resolve method labels — used for eval tracing.
RESOLVE_METHODS = (
    "exact",       # step 1: lowercase exact match
    "lemma",       # step 2: full-phrase spaCy lemma
    "hyphen",      # step 3a: space ↔ hyphen normalisation
    "lemma_alt",   # step 3b: lemma of the normalised form
    "token",       # step 4: individual token fallback
    "none",        # no match at any step
)


def resolve_concept(
    concept: str,
    kw_set: set[str],
    *,
    return_method: bool = False,
) -> "list[str] | tuple[list[str], str]":
    """Map a natural-language concept to ARASAAC keyword string(s).

    Parameters
    ----------
    concept       : Natural-language concept (e.g. "going out", "water bottle").
    kw_set        : Set of all keyword strings in the ARASAAC keyword index.
    return_method : If True, return (keywords, method_label) instead of just
                    keywords. method_label is one of RESOLVE_METHODS and is
                    useful for eval tracing without changing normal call sites.

    Returns
    -------
    list[str]  (return_method=False, default)
        Query strings to pass to search_pictograms(). Empty list = no match.
    tuple[list[str], str]  (return_method=True)
        (queries, method_label)  — same queries plus the resolution step name.

    Resolution steps
    ----------------
    1. exact      — lowercase exact match
    2. lemma      — full-phrase spaCy lemma ("going out" → "go out")
    3. hyphen     — space ↔ hyphen normalisation ("water bottle" → "water-bottle")
    3b. lemma_alt — lemma of the normalised form
    4. token      — individual lemmatised tokens of length > 2
    none          — no match found
    """
    c = concept.strip().lower()
    if not c:
        return ([], "none") if return_method else []

    # Step 1 — exact
    if c in kw_set:
        return ([c], "exact") if return_method else [c]

    # Step 2 — full-phrase lemma
    lemma = _lemmatize_phrase(c)
    if lemma != c and lemma in kw_set:
        return ([lemma], "lemma") if return_method else [lemma]

    # Step 3 — space ↔ hyphen  (+ lemma of that form)
    alt = c.replace(" ", "-") if " " in c else c.replace("-", " ")
    if alt in kw_set:
        return ([alt], "hyphen") if return_method else [alt]
    lemma_alt = _lemmatize_phrase(alt)
    if lemma_alt not in (c, lemma, alt) and lemma_alt in kw_set:
        return ([lemma_alt], "lemma_alt") if return_method else [lemma_alt]

    # Step 4 — individual tokens (lemmatised), filtered to len > 2
    tokens = [_lemmatize_word(t) for t in c.split() if len(t) > 2]
    found  = [t for t in dict.fromkeys(tokens) if t in kw_set]  # dedup, preserve order
    if found:
        return (found, "token") if return_method else found

    return ([], "none") if return_method else []
