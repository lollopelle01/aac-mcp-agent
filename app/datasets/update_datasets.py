#!/usr/bin/env python3
"""
datasets/update_datasets.py

Builds and incrementally refreshes the local ARASAAC metadata used by the MCP
server when config.USE_LOCAL_DATASETS is True.

Dataset layout
--------------
datasets/
  {lang}/
    _meta.json            Per-file build timestamps and record counts.
    keywords.json         Full keyword list for that language (list of strings).
    pictograms.json       { id_str → PictogramRecord } for all pictograms.
    keyword_index.json    { keyword → [id, ...] } for local search_pictograms().
    synset_index.json     { synset_id → [id, ...] } for local search_pictograms_by_synset().
  pictograms/
    {id}.png              Language-independent pictogram images (optional).

_meta.json structure (per language)
-------------------------------------
{
  "keywords":      { "built_at": "<ISO8601>", "count": 7021 },
  "pictograms":    { "built_at": "<ISO8601>", "count": 13780, "added": 0, "updated": 0 },
  "keyword_index": { "built_at": "<ISO8601>", "count": 18400 },
  "synset_index":  { "built_at": "<ISO8601>", "count": 3211 }
}

All "built_at" values reflect the UTC timestamp of the run that wrote that file.
For keyword_index and synset_index (derived from pictogram records, no upstream
timestamp of their own) built_at is the timestamp of the run that rebuilt them,
which happens on every run where at least one pictogram was added or updated.

PictogramRecord fields
----------------------
  id           int               Unique ARASAAC pictogram ID.
  keywords     list[dict]        All keyword objects: {type, keyword, plural, meaning}.
                                 No artificial primary/synonym distinction — all terms
                                 are stored as equals, matching the Keyword model.
  categories   list[str]         Thematic categories (e.g. "food & drink").
  synsets      list[str]         WordNet synset IDs (e.g. "00854425-v").
  tags         list[str]         Free-form editorial labels.
  sex          bool              Content safety flag.
  violence     bool              Content safety flag.
  schematic    bool              Simplified/schematic variant.
  aac          bool              Suitable for AAC use.
  aacColor     bool              Colour AAC variant available.
  skin         bool              Has skin-tone variants.
  hair         bool              Has hair-colour variants.
  created      str|None          ISO 8601 creation timestamp.
  lastUpdated  str|None          ISO 8601 update timestamp — used for incremental updates.

How population works
--------------------
GET /pictograms/all/{lang} returns all ~13 800 pictograms in a single request,
replacing the old approach of searching every keyword individually (~7 000 HTTP
requests, ~20 minutes per language). The full build now completes in seconds.

The keyword list is derived directly from the merged pictogram records rather than
from a separate /keywords/{lang} call, keeping it automatically in sync.

Incremental update logic
------------------------
On re-runs, a record is updated only when the API's lastUpdated timestamp is
strictly newer than the stored one. Pass --force to skip this check.

Images
------
Language-independent PNGs in datasets/pictograms/{id}.png.
Pass --download-images to pre-download all images.

Usage
-----
    python datasets/update_datasets.py                     # all langs in config.DATASET_LANGS
    python datasets/update_datasets.py --langs en it       # specific languages
    python datasets/update_datasets.py --force             # re-fetch all, ignore lastUpdated
    python datasets/update_datasets.py --download-images   # also save {id}.png files
    python datasets/update_datasets.py --verbose           # DEBUG logging
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

####################################################################################################
# Path setup — this file lives in app/datasets/, project root is two levels up.
####################################################################################################

_DATASETS_DIR = Path(__file__).resolve().parent   # app/datasets/
_APP  = _DATASETS_DIR.parent                      # app/
_ROOT = _APP.parent                               # <project_root>/
_SRC  = _APP / "src"                              # app/src/
for _p in (_SRC, _APP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import (
    ARASAAC_API_BASE,
    ARASAAC_IMG_PATTERN,
    ARASAAC_TIMEOUT,
    DATASETS_DIR,
    DATASET_LANGS,
)

####################################################################################################
# Constants
####################################################################################################

# Polite delay between requests (seconds). Only used for image downloads.
REQUEST_DELAY = 0.15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


####################################################################################################
# Utilities
####################################################################################################

def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(url: str) -> object | None:
    """
    GET url and return the parsed JSON body, or None on 404 / network errors.
    Raises requests.HTTPError for other non-2xx status codes.
    """
    try:
        r = requests.get(url, timeout=ARASAAC_TIMEOUT)
        if r.status_code == 404:
            log.debug("404: %s", url)
            return None
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        log.warning("Request failed for %s: %s", url, exc)
        return None


def _read_json(path: Path) -> object | None:
    """Read and parse a JSON file; return None if missing or malformed."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        log.warning("Malformed JSON at %s: %s", path, exc)
        return None


def _write_json(path: Path, data: object, label: str) -> None:
    """Write data as JSON atomically (write to .tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
    log.info("Wrote %-45s  (%s)", label, path.name)


def _read_meta(lang_dir: Path) -> dict:
    """Load _meta.json for a language, returning an empty dict if absent."""
    data = _read_json(lang_dir / "_meta.json")
    return data if isinstance(data, dict) else {}


def _write_meta(lang_dir: Path, meta: dict) -> None:
    """Persist _meta.json for a language."""
    _write_json(lang_dir / "_meta.json", meta, "_meta")


####################################################################################################
# Pictogram record extraction
####################################################################################################

def _extract_record(raw: dict) -> dict | None:
    """
    Convert a raw /pictograms/all/{lang} item to a PictogramRecord.

    All keyword entries are stored as equal peers — no artificial split into
    a primary "keyword" and secondary "synonyms". Each entry preserves the
    type, keyword, plural, and meaning fields from the API so that the
    pipeline can use full Keyword objects without information loss.

    Raw API fields used:
      _id, keywords, categories, synsets, tags,
      sex, violence, schematic, aac, aacColor, skin, hair,
      created, lastUpdated
    """
    pid = raw.get("_id") or raw.get("id")
    if pid is None:
        return None

    keywords = [
        {
            "type":    kw.get("type", 2),           # default 2 = Common_Names
            "keyword": kw.get("keyword", "").strip(),
            "plural":  kw.get("plural"),
            "meaning": kw.get("meaning"),
        }
        for kw in raw.get("keywords", [])
        if isinstance(kw, dict) and kw.get("keyword")
    ]

    return {
        "id":          int(pid),
        "keywords":    keywords,
        "categories":  [c for c in raw.get("categories", []) if isinstance(c, str)],
        "synsets":     [s for s in raw.get("synsets",    []) if isinstance(s, str)],
        "tags":        [t for t in raw.get("tags",       []) if isinstance(t, str)],
        "sex":         bool(raw.get("sex",       False)),
        "violence":    bool(raw.get("violence",  False)),
        "schematic":   bool(raw.get("schematic", False)),
        "aac":         bool(raw.get("aac",       False)),
        "aac_color":   bool(raw.get("aacColor",  False)),  # normalised from API name
        "skin":        bool(raw.get("skin",       False)),
        "hair":        bool(raw.get("hair",       False)),
        "created":     raw.get("created"),
        "last_updated": raw.get("lastUpdated"),             # normalised from API name
    }


def _is_newer(api_ts: str | None, stored_ts: str | None) -> bool:
    """
    Return True if api_ts is strictly newer than stored_ts.
    Comparison is lexicographic, which is correct for ISO 8601.
    """
    if api_ts is None:
        return False
    if stored_ts is None:
        return True
    return api_ts > stored_ts


####################################################################################################
# Per-language dataset build
####################################################################################################

def build_lang_dataset(
    lang: str,
    datasets_dir: Path,
    force: bool,
    download_images: bool,
) -> bool:
    """
    Build or incrementally update the dataset for one language.

    Steps:
      1. Fetch all pictograms via GET /pictograms/all/{lang} (single request).
      2. Merge with existing pictograms.json using lastUpdated for incremental mode.
      3. Derive keywords.json from the merged pictogram records (all keyword strings).
      4. Rebuild keyword_index.json and synset_index.json from the merged records.
      5. Write _meta.json with built_at timestamps and counts for every file.
      6. Optionally download pictogram images.

    Returns True on success, False on failure.
    """
    lang_dir = datasets_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    meta   = _read_meta(lang_dir)
    run_ts = _now_iso()

    # -- 1. Fetch all pictograms ----------------------------------------------
    log.info("[%s] Fetching all pictograms via /pictograms/all/%s ...", lang, lang)
    url  = f"{ARASAAC_API_BASE}/pictograms/all/{lang}"
    data = _get(url)

    if not isinstance(data, list) or not data:
        log.error("[%s] /pictograms/all/%s returned no data — aborting.", lang, lang)
        return False

    log.info("[%s] Received %d pictogram entries from API.", lang, len(data))

    # -- 2. Merge with existing records ---------------------------------------
    pictograms: dict[str, dict] = {}
    if not force:
        existing = _read_json(lang_dir / "pictograms.json")
        if isinstance(existing, dict):
            pictograms = existing
            log.info(
                "[%s] Loaded %d existing records (incremental mode).",
                lang, len(pictograms),
            )

    added = updated = skipped = 0

    for raw in data:
        rec = _extract_record(raw)
        if rec is None:
            continue

        pid_str   = str(rec["id"])
        stored    = pictograms.get(pid_str)
        stored_ts = stored.get("last_updated") if stored else None
        api_ts    = rec["last_updated"]

        if stored is None:
            pictograms[pid_str] = rec
            added += 1
        elif force or _is_newer(api_ts, stored_ts):
            pictograms[pid_str] = rec
            updated += 1
        else:
            skipped += 1

    log.info(
        "[%s] Pictograms merged — %d total  (+%d new, ~%d updated, =%d unchanged)",
        lang, len(pictograms), added, updated, skipped,
    )

    _write_json(
        lang_dir / "pictograms.json",
        pictograms,
        f"pictograms[{lang}] — {len(pictograms)} records",
    )
    meta["pictograms"] = {
        "built_at": run_ts,
        "count":    len(pictograms),
        "added":    added,
        "updated":  updated,
    }

    # -- 3. Derive keywords from pictogram records ----------------------------
    # All keyword strings across all pictogram records, with no hierarchy.
    # This stays in sync with the pictogram data automatically — no separate
    # /keywords/{lang} call required.
    # Keywords are lowercased here so that keywords.json (and therefore the
    # kw_set loaded at runtime) uses the same case as resolve_concept(), which
    # always lowercases the input concept before matching.
    keyword_set: set[str] = set()
    for rec in pictograms.values():
        for kw in rec.get("keywords", []):
            if isinstance(kw, dict) and kw.get("keyword"):
                keyword_set.add(kw["keyword"].lower())

    keywords = sorted(keyword_set)
    _write_json(
        lang_dir / "keywords.json",
        keywords,
        f"keywords[{lang}] — {len(keywords)} entries",
    )
    meta["keywords"] = {"built_at": run_ts, "count": len(keywords)}

    # -- 4a. Rebuild keyword_index --------------------------------------------
    # Keys are lowercased to match keywords.json and the runtime kw_set.
    keyword_index: dict[str, list[str]] = {}
    for pid_str, rec in pictograms.items():
        for kw in rec.get("keywords", []):
            term = kw.get("keyword").lower() if isinstance(kw, dict) and kw.get("keyword") else None
            if term:
                keyword_index.setdefault(term, [])
                if pid_str not in keyword_index[term]:
                    keyword_index[term].append(pid_str)

    keyword_index = {k: sorted(v) for k, v in sorted(keyword_index.items())}
    _write_json(
        lang_dir / "keyword_index.json",
        keyword_index,
        f"keyword_index[{lang}] — {len(keyword_index)} entries",
    )
    meta["keyword_index"] = {"built_at": run_ts, "count": len(keyword_index)}

    # -- 4b. Rebuild synset_index ---------------------------------------------
    synset_index: dict[str, list[str]] = {}
    for pid_str, rec in pictograms.items():
        for syn in rec.get("synsets", []):
            synset_index.setdefault(syn, [])
            if pid_str not in synset_index[syn]:
                synset_index[syn].append(pid_str)

    synset_index = {k: sorted(v) for k, v in sorted(synset_index.items())}
    _write_json(
        lang_dir / "synset_index.json",
        synset_index,
        f"synset_index[{lang}] — {len(synset_index)} synsets",
    )
    meta["synset_index"] = {"built_at": run_ts, "count": len(synset_index)}

    # -- 5. Write _meta.json --------------------------------------------------
    _write_meta(lang_dir, meta)

    log.info(
        "[%s] Done — %d pictograms, %d keywords, %d keyword_index entries, %d synsets.",
        lang, len(pictograms), len(keywords), len(keyword_index), len(synset_index),
    )

    # -- 6. Optionally download images ----------------------------------------
    if download_images:
        _download_images(list(pictograms.keys()), datasets_dir)

    return True


####################################################################################################
# Image download (language-independent)
####################################################################################################

def _download_images(pid_strs: list[str], datasets_dir: Path) -> None:
    """
    Download {id}.png for every pictogram ID not already cached locally.
    Images live in datasets/pictograms/{id}.png and are shared across all languages.
    """
    images_dir = datasets_dir / "pictograms"
    images_dir.mkdir(parents=True, exist_ok=True)

    missing = [p for p in pid_strs if not (images_dir / f"{p}.png").exists()]
    log.info("Downloading %d missing pictogram images ...", len(missing))

    downloaded = failed = 0
    for i, pid_str in enumerate(missing, 1):
        path = images_dir / f"{pid_str}.png"
        url  = ARASAAC_IMG_PATTERN.format(id=pid_str)
        try:
            r = requests.get(url, timeout=ARASAAC_TIMEOUT)
            r.raise_for_status()
            path.write_bytes(r.content)
            downloaded += 1
        except requests.exceptions.RequestException as exc:
            log.warning("  Image download failed for id=%s: %s", pid_str, exc)
            failed += 1

        if i % 200 == 0 or i == len(missing):
            log.info(
                "  %d / %d images — %d ok, %d failed", i, len(missing), downloaded, failed
            )

        time.sleep(REQUEST_DELAY)

    log.info("Image download complete — %d downloaded, %d failed.", downloaded, failed)


####################################################################################################
# Orchestration
####################################################################################################

def run(
    langs: list[str],
    force: bool,
    download_images: bool,
    datasets_dir: Path,
) -> bool:
    """Build or update datasets for all requested languages. Returns True on full success."""
    success = True
    for lang in langs:
        ok = build_lang_dataset(lang, datasets_dir, force, download_images)
        if not ok:
            success = False
    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or incrementally refresh the local ARASAAC metadata datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=DATASET_LANGS,
        metavar="LANG",
        help=f"Language codes to process (default from config: {DATASET_LANGS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch and overwrite all pictogram records, ignoring lastUpdated.",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download {id}.png for all pictograms into datasets/pictograms/.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATASETS_DIR),
        metavar="DIR",
        help=f"Root directory to write dataset files (default: {DATASETS_DIR})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    datasets_dir = Path(args.output_dir)
    log.info(
        "update_datasets  langs=%s  force=%s  download_images=%s  dir=%s",
        args.langs, args.force, args.download_images, datasets_dir,
    )

    ok = run(
        langs=args.langs,
        force=args.force,
        download_images=args.download_images,
        datasets_dir=datasets_dir,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
