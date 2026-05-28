from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from functools import lru_cache

import spacy


@lru_cache(maxsize=1)
def _nlp() -> spacy.language.Language:
    """Load the spaCy model once (lazy, cached)."""
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


#### Helpers ####################################################################

def _lemmatize_phrase(text: str) -> str:
    """
    Return the lemmatised form of a (possibly multi-word) phrase.

    Example:    "going out" -> "go out"  
                "brushing teeth" -> "brush tooth"
    """
    doc = _nlp()(text.lower())
    return " ".join(token.lemma_ for token in doc)


def _lemmatize_word(word: str) -> str:
    """
    Return the lemma of a single word token.

    Example:    "eating" -> "eat"  
                "coats" -> "coat"
    """
    doc = _nlp()(word.lower())
    return doc[0].lemma_ if doc else word.lower()


#### Publc functions ############################################################

# Resolve method labels, used for eval tracing.
RESOLVE_METHODS = (
    "exact",       # step 1: lowercase exact match
    "lemma",       # step 2: full-phrase spaCy lemma
    "hyphen",      # step 3a: space <-> hyphen normalisation
    "lemma_alt",   # step 3b: lemma of the normalised form
    "token",       # step 4: individual token fallback
    "none",        # no match at any step
)

# Determiners/articles stripped before any lookup (step 0).
# These add no semantic content and prevent exact/lemma matches.
_STRIP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "some", "many", "much", "lot", "lots",
    "so", "very", "quite", "really", "more", "most",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
})


def _strip_determiners(text: str) -> str:
    """
    Remove leading/trailing/isolated determiners and articles from a phrase.

    "a banana"          -> "banana"
    "the local mall"    -> "local mall"
    "so much lightning" -> "lightning"
    "their families"    -> "families"
    """
    tokens = text.split()
    # strip from both ends, then remove isolated stop-words in the middle
    # (keep middle content words even if they happen to be in _STRIP_WORDS)
    while tokens and tokens[0] in _STRIP_WORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _STRIP_WORDS:
        tokens = tokens[:-1]
    return " ".join(tokens)


# The actual mapping function used by the agent before looking for keywords
def resolve_concept(
    concept: str,
    kw_set: set[str],
    *,
    return_method: bool = False,
) -> "list[str] | tuple[list[str], str]":
    """
    Map a natural-language concept to ARASAAC keyword string.

    - concept: natural-language concept (e.g. "going out", "water bottle").
    - kw_set: all keyword strings in the ARASAAC keyword index.
    - return_method: if True, return (keywords, method_label) instead of just
        keywords. Useful for eval tracing without changing normal call sites.

    Resolution steps (first match wins):
        0. strip      -> remove leading/trailing articles & determiners, then
                         re-run steps 1-4 on the stripped form if different
        1. exact      -> lowercase exact match
        2. lemma      -> full-phrase spaCy lemma ("going out" becomes "go out")
        3. hyphen     -> space/hyphen normalisation ("water bottle" becomes "water-bottle")
        3b. lemma_alt -> lemma of the normalised form
        4. token      -> individual lemmatised tokens of length > 2
        none          -> no match found
    """
    c = concept.strip().lower()
    if not c:
        return ([], "none") if return_method else []

    def _lookup(phrase: str) -> "list[str] | None":
        """Run steps 1-4 on a single phrase. Returns matches or None."""
        if not phrase:
            return None

        # Step 1: exact
        if phrase in kw_set:
            return [phrase]

        # Step 2: full-phrase lemma
        lemma = _lemmatize_phrase(phrase)
        if lemma != phrase and lemma in kw_set:
            return [lemma]

        # Step 3: space/hyphen normalisation (+ lemma of that form)
        alt = phrase.replace(" ", "-") if " " in phrase else phrase.replace("-", " ")
        if alt in kw_set:
            return [alt]
        lemma_alt = _lemmatize_phrase(alt)
        if lemma_alt not in (phrase, lemma, alt) and lemma_alt in kw_set:
            return [lemma_alt]

        # Step 4: individual tokens (lemmatised), filtered to len > 2
        tokens = [_lemmatize_word(t) for t in phrase.split() if len(t) > 2]
        found = [t for t in dict.fromkeys(tokens) if t in kw_set]
        if found:
            return found

        return None

    # Step 0: try original form first
    result = _lookup(c)
    if result is not None:
        # Determine which sub-step matched for the method label
        method = _resolve_method_label(c, result, kw_set)
        return (result, method) if return_method else result

    # Step 0b: try determiner-stripped form if different
    stripped = _strip_determiners(c)
    if stripped and stripped != c:
        result = _lookup(stripped)
        if result is not None:
            method = _resolve_method_label(stripped, result, kw_set)
            return (result, method) if return_method else result

    return ([], "none") if return_method else []


def _resolve_method_label(phrase: str, matches: list[str], kw_set: set[str]) -> str:
    """Re-derive the method label for a phrase that is known to match."""
    if phrase in kw_set:
        return "exact"
    lemma = _lemmatize_phrase(phrase)
    if lemma in kw_set:
        return "lemma"
    alt = phrase.replace(" ", "-") if " " in phrase else phrase.replace("-", " ")
    if alt in kw_set:
        return "hyphen"
    if _lemmatize_phrase(alt) in kw_set:
        return "lemma_alt"
    return "token"