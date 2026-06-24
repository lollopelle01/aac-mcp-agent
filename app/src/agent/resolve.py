from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from agent.session import _nlp
from config import DATASETS_DIR

logger = logging.getLogger(__name__)


################################################################################
# Embedding index — lazy singleton, one per language
################################################################################

class _EmbeddingIndex:
    """
    Precomputed keyword embeddings for one language.

    Backed by datasets/{lang}/keyword_embeddings.npz, which must be built
    with build_keyword_embeddings.py before the first use.  The model is
    loaded lazily on the first call to nearest() so startup time is not
    affected when all concepts resolve via exact/lemma/hyphen/token steps.

    Layout of the .npz archive:
        keywords  — 1-D array of UTF-8 keyword strings, shape (N,)
        embeddings — 2-D float32 matrix, shape (N, D), L2-normalised rows
    """

    # Class-level registry: lang -> _EmbeddingIndex (None if build file absent)
    _instances: dict[str, Optional["_EmbeddingIndex"]] = {}

    def __init__(self, lang: str) -> None:
        self.lang  = lang
        self._kws: np.ndarray  = np.array([], dtype=object)
        self._mat: np.ndarray  = np.empty((0, 0), dtype=np.float32)
        self._model            = None   # sentence_transformers.SentenceTransformer, loaded lazily

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, lang: str) -> Optional["_EmbeddingIndex"]:
        """Return the index for lang, or None if the .npz file is missing."""
        if lang not in cls._instances:
            npz_path = DATASETS_DIR / lang / "keyword_embeddings.npz"
            if not npz_path.exists():
                logger.debug(
                    "Embedding index not found for lang=%r (%s). "
                    "Run datasets/build_keyword_embeddings.py to build it.",
                    lang, npz_path,
                )
                cls._instances[lang] = None
                return None
            idx = cls(lang)
            try:
                data = np.load(str(npz_path), allow_pickle=True)
                idx._kws = data["keywords"]     # shape (N,), dtype object (str)
                idx._mat = data["embeddings"]   # shape (N, D), float32, L2-normed
                logger.info(
                    "Loaded keyword embeddings for lang=%r: %d keywords, dim=%d.",
                    lang, len(idx._kws), idx._mat.shape[1],
                )
                cls._instances[lang] = idx
            except Exception as exc:
                logger.warning("Failed to load embedding index for lang=%r: %s", lang, exc)
                cls._instances[lang] = None
        return cls._instances[lang]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the SentenceTransformer model on first use (deferred)."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded SentenceTransformer model 'all-MiniLM-L6-v2' for embedding lookup.")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embedding-based concept resolution. "
                "Install it with: pip install sentence-transformers"
            ) from exc

    def nearest(self, concept: str, threshold: float = 0.75) -> Optional[str]:
        """
        Return the keyword with the highest cosine similarity to concept,
        or None if the best score is below threshold.

        The embedding matrix rows are already L2-normalised, so cosine
        similarity reduces to a plain dot product.
        """
        if self._mat.shape[0] == 0:
            return None
        self._load_model()
        vec = self._model.encode([concept], normalize_embeddings=True).astype(np.float32)
        scores = self._mat @ vec[0]          # shape (N,)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= threshold:
            kw = str(self._kws[best_idx])
            logger.debug(
                "Embedding nearest(%r) → %r  score=%.3f  threshold=%.2f",
                concept, kw, best_score, threshold,
            )
            return kw
        logger.debug(
            "Embedding nearest(%r) — best score %.3f below threshold %.2f, no match.",
            concept, best_score, threshold,
        )
        return None



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
    lang: str = "en",
    embedding_threshold: float = 0.75,
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
        5. embedding  — nearest-neighbour cosine search against precomputed
                        keyword embeddings (lazy-loaded, no-op if .npz absent)
        none          — no match found

    Set return_method=True to get (keywords, method_label) for eval tracing.
    The lang parameter selects the embedding index built by
    build_keyword_embeddings.py (datasets/{lang}/keyword_embeddings.npz).
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

    # Step 5: embedding nearest-neighbour fallback (lazy, no-op if .npz absent)
    # Use the stripped form if available, otherwise the original lowercased concept.
    probe = stripped if stripped else c
    emb_index = _EmbeddingIndex.get(lang)
    if emb_index is not None:
        nearest_kw = emb_index.nearest(probe, threshold=embedding_threshold)
        if nearest_kw is not None:
            return ([nearest_kw], "embedding") if return_method else [nearest_kw]

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
