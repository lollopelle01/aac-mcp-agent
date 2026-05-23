# mcp_server/dataset_cache.py — lazy in-memory loader for local ARASAAC datasets.
#
# Reads from DATASETS_DIR/{lang}/*.json on first access, keeps in memory for the
# process lifetime. All load_* methods return None on missing/malformed files so
# callers can fall back to the live API transparently.
#
# Call invalidate() to force a reload (e.g. after running update_datasets.py in
# the same process).
#
# Layout:
#   app/datasets/{lang}/_meta.json            Build timestamps and counts per file.
#   app/datasets/{lang}/keywords.json         Full keyword list.
#   app/datasets/{lang}/pictograms.json       { id_str → PictogramRecord }
#   app/datasets/{lang}/keyword_index.json    { keyword → [id, ...] }
#   app/datasets/{lang}/synset_index.json     { synset_id → [id, ...] }
#   app/datasets/pictograms/{id}.png          Language-independent images.

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import ARASAAC_IMG_PATTERN, ARASAAC_TIMEOUT, DATASETS_DIR

logger = logging.getLogger(__name__)


class _DatasetCache:
    # Per-language caches
    _keywords:      dict[str, list[str]]             = {}
    _pictograms:    dict[str, dict[str, dict]]       = {}  # lang → { id_str → record }
    _keyword_index: dict[str, dict[str, list[str]]]  = {}  # lang → { keyword → [id, ...] }
    _synset_index:  dict[str, dict[str, list[str]]]  = {}  # lang → { synset_id → [id, ...] }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def _lang_dir(cls, lang: str) -> Path:
        return DATASETS_DIR / lang

    @classmethod
    def _load_json(cls, path: Path) -> object | None:
        """Read and parse a JSON file; return None on any failure."""
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            logger.debug(
                "Dataset file not found: %s. Run datasets/update_datasets.py to build it.",
                path,
            )
            return None
        except json.JSONDecodeError as exc:
            logger.error("Dataset file %s is not valid JSON: %s", path, exc)
            return None

    # ── Meta ──────────────────────────────────────────────────────────────────

    @classmethod
    def load_meta(cls, lang: str) -> dict | None:
        """
        Return the _meta.json dict for lang, or None if unavailable.

        _meta.json structure:
          {
            "pictograms":    { "built_at": "<ISO8601>", "count": 13780, "added": 0, "updated": 3 },
            "keywords":      { "built_at": "<ISO8601>", "count": 18400 },
            "keyword_index": { "built_at": "<ISO8601>", "count": 18400 },
            "synset_index":  { "built_at": "<ISO8601>", "count": 3211 }
          }
        """
        data = cls._load_json(cls._lang_dir(lang) / "_meta.json")
        return data if isinstance(data, dict) else None

    # ── Public loaders ────────────────────────────────────────────────────────

    @classmethod
    def load_keywords(cls, lang: str) -> list[str] | None:
        """Return the keyword list for lang, or None if unavailable."""
        if lang not in cls._keywords:
            data = cls._load_json(cls._lang_dir(lang) / "keywords.json")
            if data is None or not isinstance(data, list):
                return None
            cls._keywords[lang] = [str(k) for k in data if isinstance(k, str)]
            logger.info("Loaded %d keywords for lang='%s'.", len(cls._keywords[lang]), lang)
        return cls._keywords.get(lang)

    @classmethod
    def load_pictograms(cls, lang: str) -> dict[str, dict] | None:
        """Return { id_str → PictogramRecord } for lang, or None if unavailable."""
        if lang not in cls._pictograms:
            data = cls._load_json(cls._lang_dir(lang) / "pictograms.json")
            if data is None or not isinstance(data, dict):
                return None
            cls._pictograms[lang] = data
            logger.info("Loaded %d pictograms for lang='%s'.", len(data), lang)
        return cls._pictograms.get(lang)

    @classmethod
    def load_keyword_index(cls, lang: str) -> dict[str, list[str]] | None:
        """Return { keyword → [id_str, ...] } for lang, or None if unavailable."""
        if lang not in cls._keyword_index:
            data = cls._load_json(cls._lang_dir(lang) / "keyword_index.json")
            if data is None or not isinstance(data, dict):
                return None
            cls._keyword_index[lang] = data
            logger.info("Loaded keyword_index for lang='%s' (%d entries).", lang, len(data))
        return cls._keyword_index.get(lang)

    @classmethod
    def load_synset_index(cls, lang: str) -> dict[str, list[str]] | None:
        """Return { synset_id → [id_str, ...] } for lang, or None if unavailable."""
        if lang not in cls._synset_index:
            data = cls._load_json(cls._lang_dir(lang) / "synset_index.json")
            if data is None or not isinstance(data, dict):
                return None
            cls._synset_index[lang] = data
            logger.info("Loaded synset_index for lang='%s' (%d synsets).", lang, len(data))
        return cls._synset_index.get(lang)

    @classmethod
    def get_pictogram_image(cls, pictogram_id: int) -> Path:
        """
        Return the local path to {pictogram_id}.png, downloading it from ARASAAC
        if not already cached. Images are language-independent and shared across
        all languages in datasets/pictograms/.
        """
        import requests

        path = DATASETS_DIR / "pictograms" / f"{pictogram_id}.png"
        if not path.exists():
            url = ARASAAC_IMG_PATTERN.format(id=pictogram_id)
            r   = requests.get(url, timeout=ARASAAC_TIMEOUT)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            logger.info("Downloaded pictogram %d → %s", pictogram_id, path)
        return path

    @classmethod
    def invalidate(cls) -> None:
        """Clear all in-memory caches so the next access re-reads from disk."""
        cls._keywords.clear()
        cls._pictograms.clear()
        cls._keyword_index.clear()
        cls._synset_index.clear()
        logger.debug("_DatasetCache: invalidated.")
