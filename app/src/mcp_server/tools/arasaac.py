from __future__ import annotations

import logging
from typing import Optional

import requests
from urllib.parse import quote

from mcp_server.server import mcp
from mcp_server.models import Keyword, Pictogram
from mcp_server.dataset_cache import _DatasetCache
from config import (
    LANG,
    ARASAAC_API_BASE,
    ARASAAC_IMG_PATTERN,
    ARASAAC_TIMEOUT,
    USE_LOCAL_DATASETS,
)

logger = logging.getLogger(__name__)

####### Utils #######################################################################


def _image_url(pictogram_id: int) -> str:
    """
    Return the image URL for the given pictogram ID.

    When USE_LOCAL_DATASETS is True the URL points to the local proxy endpoint
    (/api/images/{id}). The Vite dev proxy rewrites /api/* -> backend :8000,
    so the browser resolves this correctly without ever hitting the CDN.
    This makes the app fully functional offline.

    When USE_LOCAL_DATASETS is False the CDN URL is returned directly.
    """
    if USE_LOCAL_DATASETS:
        return f"/api/images/{pictogram_id}"
    return ARASAAC_IMG_PATTERN.format(id=pictogram_id)


def _raw_to_pictogram(raw: dict) -> Pictogram:
    """
    Convert a raw dict into a Pictogram, regardless of origin (live API or local dataset).
    """
    pid = raw.get("_id") or raw.get("id")

    keywords = [Keyword.model_validate(kw) for kw in raw.get("keywords", [])]

    return Pictogram(
        id           = int(pid),
        image_url    = _image_url(int(pid)),
        keywords     = keywords,
        categories   = [c for c in raw.get("categories", []) if isinstance(c, str)],
        synsets      = [s for s in raw.get("synsets",    []) if isinstance(s, str)],
        tags         = [t for t in raw.get("tags",       []) if isinstance(t, str)],
        sex          = bool(raw.get("sex",        False)),
        violence     = bool(raw.get("violence",   False)),
        schematic    = bool(raw.get("schematic",  False)),
        aac          = bool(raw.get("aac",        False)),
        aac_color    = bool(raw.get("aacColor",   False) or raw.get("aac_color", False)),
        skin         = bool(raw.get("skin",       False)),
        hair         = bool(raw.get("hair",       False)),
        created      = raw.get("created"),
        last_updated = raw.get("lastUpdated") or raw.get("last_updated"),
    )


def _get_json(url: str, context: str) -> object:
    """Perform a GET request and return the parsed JSON body."""
    try:
        response = requests.get(url, timeout=ARASAAC_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        logger.error("ARASAAC connection failed [%s]: %s", context, exc)
        raise ConnectionError(f"Cannot reach the ARASAAC API: {exc}") from exc
    except requests.exceptions.Timeout:
        logger.error("ARASAAC timeout [%s]", context)
        raise TimeoutError(f"ARASAAC request timed out ({context})")
    return response.json()


def _extract_strings_from_payload(payload: object) -> list[str]:
    """
    Best-effort extraction of keyword strings from an arbitrary JSON payload.
    Handles the observed /keywords/{lang} response shapes.
    """
    if isinstance(payload, list):
        result: list[str] = []
        for item in payload:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                kw = item.get("keyword") or item.get("name") or item.get("word")
                if kw and isinstance(kw, str):
                    result.append(kw)
        return result

    if isinstance(payload, dict):
        for key in ("data", "keywords", "items", "results"):
            if key in payload and isinstance(payload[key], list):
                return _extract_strings_from_payload(payload[key])
        result = []
        for key, value in payload.items():
            if isinstance(key, str) and not isinstance(value, (dict, list)):
                result.append(key)
            elif isinstance(value, list):
                result.extend(_extract_strings_from_payload(value))
        return result

    logger.warning("_extract_strings_from_payload: unexpected type %s", type(payload))
    return []


def _search_raw(search_text: str, lang: str, max_results: int, context: str) -> list[dict]:
    """Call GET /pictograms/{lang}/search/{searchText} and return raw result dicts."""
    url = f"{ARASAAC_API_BASE}/pictograms/{lang}/search/{quote(search_text)}"

    try:
        raw_list: list[dict] = _get_json(url, context)
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 404:
            logger.info("No pictogram found for %s (%s)", context, lang)
            return []
        raise

    return raw_list[:max_results]


def _safe_parse(raw: dict) -> "Pictogram | None":
    """Convert a raw dict to a Pictogram, returning None on failure."""
    try:
        return _raw_to_pictogram(raw)
    except Exception as exc:
        logger.debug("Unparseable pictogram (id=%s): %s", raw.get("_id") or raw.get("id"), exc)
        return None


def _parse_raw_list(raw_list: list[dict]) -> list[Pictogram]:
    """Parse a list of raw API dicts into Pictogram objects, skipping failures."""
    return [p for raw in raw_list if (p := _safe_parse(raw)) is not None]


def _ids_to_results(
    ids: list[str],
    pictograms: dict[str, dict],
    max_results: int,
) -> list[Pictogram]:
    """Look up pictogram records by ID list and return Pictogram objects."""
    results: list[Pictogram] = []
    for pid_str in ids[:max_results]:
        rec = pictograms.get(pid_str)
        if rec is None:
            logger.debug("Local pictogram id=%s not found in dataset -- skipping.", pid_str)
            continue
        if "id" not in rec and "_id" not in rec:
            rec = {"id": int(pid_str), **rec}
        if (p := _safe_parse(rec)) is not None:
            results.append(p)
    return results


####### Helper ######################################################################

def get_pictogram_image(pictogram_id: int) -> bytes:
    """
    Return the raw PNG bytes for a pictogram.

    When USE_LOCAL_DATASETS is True the image is read from the local dataset
    (downloading it on first access if missing, like all other dataset files).
    When USE_LOCAL_DATASETS is False the image is fetched from the ARASAAC CDN.
    """
    if USE_LOCAL_DATASETS:
        path = _DatasetCache.get_pictogram_image(pictogram_id)
        return path.read_bytes()

    url = ARASAAC_IMG_PATTERN.format(id=pictogram_id)
    r = requests.get(url, timeout=ARASAAC_TIMEOUT)
    r.raise_for_status()
    return r.content


####### MCP tools ###################################################################

@mcp.tool()
def search_pictograms(
    keyword: str,
    lang: str = LANG,
    max_results: int = 5,
) -> dict:
    """
    Search ARASAAC pictograms by a single keyword and return the best matches.

    This is a literal keyword match against ARASAAC's catalogue, not a
    semantic search, the keyword must match (or closely resemble) the exact
    string stored as a pictogram label in the chosen language. Call
    list_keywords() first to discover the exact strings available.
    """
    if USE_LOCAL_DATASETS:
        kw_index   = _DatasetCache.load_keyword_index(lang)
        pictograms = _DatasetCache.load_pictograms(lang)

        if kw_index is not None and pictograms is not None:
            ids     = kw_index.get(keyword) or kw_index.get(keyword.lower()) or []
            results = _ids_to_results(ids, pictograms, max_results)
            logger.info(
                "search_pictograms('%s', lang='%s'): %d results from local dataset.",
                keyword, lang, len(results),
            )
            return {"results": [r.model_dump() for r in results]}

        logger.info(
            "search_pictograms('%s', lang='%s'): local dataset unavailable, calling API.",
            keyword, lang,
        )

    raw_list = _search_raw(keyword, lang, max_results, f"keyword '{keyword}'")
    results  = _parse_raw_list(raw_list)
    logger.info(
        "search_pictograms('%s', lang='%s'): %d results from API.",
        keyword, lang, len(results),
    )
    return {"results": [r.model_dump() for r in results]}


@mcp.tool()
def get_pictogram_metadata(
    pictogram_id: int,
    lang: str = LANG,
) -> dict:
    """
    Fetch the complete metadata for a single pictogram by its numeric ID.

    Uses the local dataset when available; falls back to the live API.
    Returns an error dict (instead of raising) for unknown IDs.
    """
    pid_str = str(pictogram_id)

    if USE_LOCAL_DATASETS:
        pictograms = _DatasetCache.load_pictograms(lang)
        if pictograms is not None:
            rec = pictograms.get(pid_str)
            if rec is not None:
                if "id" not in rec and "_id" not in rec:
                    rec = {"id": pictogram_id, **rec}
                logger.info("get_pictogram_metadata(id=%d): OK from local dataset.", pictogram_id)
                return _raw_to_pictogram(rec).model_dump()
            logger.debug(
                "get_pictogram_metadata(id=%d): not in local dataset, calling API.",
                pictogram_id,
            )

    url = f"{ARASAAC_API_BASE}/pictograms/{lang}/{pictogram_id}"

    try:
        raw: dict = _get_json(url, f"id={pictogram_id}")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 404:
            logger.info("get_pictogram_metadata(id=%d): not found on API.", pictogram_id)
            return {"error": f"Pictogram id={pictogram_id} not found on ARASAAC"}
        raise

    logger.info("get_pictogram_metadata(id=%d): OK from API.", pictogram_id)
    return _raw_to_pictogram(raw).model_dump()


@mcp.tool()
def list_keywords(lang: str = LANG) -> dict:
    """
    Return all keyword strings available in the ARASAAC catalogue for a language.

    Call this before search_pictograms() to discover the exact strings the
    catalogue recognises. Reads from datasets/{lang}/keywords.json when
    USE_LOCAL_DATASETS is True, it falls back to the live /keywords/{lang} endpoint.
    """
    if USE_LOCAL_DATASETS:
        local_kws = _DatasetCache.load_keywords(lang)
        if local_kws is not None:
            logger.info(
                "list_keywords(lang='%s'): %d keywords from local dataset.",
                lang, len(local_kws),
            )
            return {"keywords": local_kws}

    logger.info("list_keywords(lang='%s'): local dataset unavailable, calling API.", lang)
    url = f"{ARASAAC_API_BASE}/keywords/{lang}"

    try:
        payload = _get_json(url, f"keywords lang={lang}")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 404:
            logger.info("No keywords found for lang='%s'.", lang)
            return {"keywords": []}
        raise

    keywords = _extract_strings_from_payload(payload)
    logger.info("list_keywords(lang='%s'): %d keywords from API.", lang, len(keywords))
    return {"keywords": keywords}


@mcp.tool()
def search_pictograms_by_synset(
    synset_id: str,
    wordnet: str = "3.1",
    lang: str = LANG,
) -> dict:
    """
    Fetch all ARASAAC pictograms linked to a Princeton WordNet synset.

    The most semantically precise search available, retrieves exactly the
    pictograms representing a concept without relying on keyword matching.

    Synset IDs are 8-digit zero-padded integers followed by a POS tag.
    es: "00854425-v" (eat/consume) or "04341686-n" (school).
    """
    if USE_LOCAL_DATASETS:
        synset_index = _DatasetCache.load_synset_index(lang)
        pictograms   = _DatasetCache.load_pictograms(lang)

        if synset_index is not None and pictograms is not None:
            ids     = synset_index.get(synset_id, [])
            results = _ids_to_results(ids, pictograms, max_results=50)
            logger.info(
                "search_pictograms_by_synset(synset=%s, lang=%s): %d results from local dataset.",
                synset_id, lang, len(results),
            )
            return {"results": [r.model_dump() for r in results]}

        logger.info(
            "search_pictograms_by_synset(synset=%s): local dataset unavailable, calling API.",
            synset_id,
        )

    url = f"{ARASAAC_API_BASE}/pictograms/{lang}/wordnet/{wordnet}/id/{synset_id}"

    try:
        raw_list: list[dict] = _get_json(url, f"synset={synset_id} wn={wordnet}")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 404:
            logger.info(
                "No pictograms for synset=%s (wn=%s, lang=%s)", synset_id, wordnet, lang
            )
            return {"results": []}
        raise

    results = _parse_raw_list(raw_list)
    logger.info(
        "search_pictograms_by_synset(synset=%s, wn=%s, lang=%s): %d results from API.",
        synset_id, wordnet, lang, len(results),
    )
    return {"results": [r.model_dump() for r in results]}
