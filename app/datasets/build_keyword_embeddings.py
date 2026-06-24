#!/usr/bin/env python3
"""
datasets/build_keyword_embeddings.py

Pre-compute L2-normalised sentence embeddings for every keyword in
datasets/{lang}/keywords.json and save them to
datasets/{lang}/keyword_embeddings.npz.

The .npz archive is consumed at runtime by _EmbeddingIndex in
src/agent/resolve.py (step 5 of resolve_concept).  It is the only way to
enable embedding-based fallback; the file is NOT created automatically by
update_datasets.py so that the 80 MB model download is opt-in.

Usage
-----
    # All languages defined in config.DATASET_LANGS
    python datasets/build_keyword_embeddings.py

    # Specific languages
    python datasets/build_keyword_embeddings.py --langs en it

    # Force rebuild even if .npz already exists
    python datasets/build_keyword_embeddings.py --force

    # Different model (must be compatible with sentence-transformers)
    python datasets/build_keyword_embeddings.py --model paraphrase-MiniLM-L6-v2

Dependencies
------------
    pip install sentence-transformers

The default model is all-MiniLM-L6-v2 (22 MB, 384-dimensional embeddings).
It is downloaded once by sentence-transformers and cached in
~/.cache/torch/sentence_transformers/.

Archive layout
--------------
    keywords   : np.ndarray, shape (N,),    dtype object  — keyword strings
    embeddings : np.ndarray, shape (N, 384), dtype float32 — L2-normalised rows
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

####################################################################################################
# Path setup — same as update_datasets.py
####################################################################################################

_DATASETS_DIR = Path(__file__).resolve().parent   # app/datasets/
_APP          = _DATASETS_DIR.parent              # app/
_ROOT         = _APP.parent                       # <project_root>/
_SRC          = _APP / "src"                      # app/src/
for _p in (_SRC, _APP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import DATASETS_DIR, DATASET_LANGS

####################################################################################################
# Logging
####################################################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

####################################################################################################
# Core
####################################################################################################

DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE    = 512   # keywords per encode() call


def build_for_lang(
    lang: str,
    datasets_dir: Path,
    model_name: str,
    force: bool,
    model=None,  # re-use across languages if already loaded
):
    """
    Build keyword_embeddings.npz for one language.

    Returns the loaded SentenceTransformer model so the caller can pass it
    back in for the next language without reloading.
    """
    lang_dir  = datasets_dir / lang
    npz_path  = lang_dir / "keyword_embeddings.npz"
    kw_path   = lang_dir / "keywords.json"

    if not kw_path.exists():
        log.error("[%s] keywords.json not found at %s — run update_datasets.py first.", lang, kw_path)
        return model

    if npz_path.exists() and not force:
        log.info("[%s] %s already exists — skipping (use --force to rebuild).", lang, npz_path.name)
        return model

    # Load keywords
    with open(kw_path, encoding="utf-8") as fh:
        keywords: list[str] = json.load(fh)

    if not keywords:
        log.warning("[%s] keywords.json is empty — nothing to embed.", lang)
        return model

    log.info("[%s] Embedding %d keywords with model '%s' ...", lang, len(keywords), model_name)

    # Lazy model load (shared across languages)
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.error(
                "sentence-transformers is not installed.\n"
                "Install it with:  pip install sentence-transformers"
            )
            sys.exit(1)
        log.info("Loading model '%s' ...", model_name)
        model = SentenceTransformer(model_name)
        log.info("Model loaded.")

    # Encode in batches with a progress indicator
    all_embeddings = []
    for start in range(0, len(keywords), BATCH_SIZE):
        batch = keywords[start : start + BATCH_SIZE]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(vecs.astype(np.float32))
        log.info(
            "[%s]   encoded %d / %d",
            lang, min(start + BATCH_SIZE, len(keywords)), len(keywords),
        )

    embeddings = np.vstack(all_embeddings)   # shape (N, D)

    # Sanity-check L2 norms (should all be ~1.0 after normalise_embeddings=True)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        log.warning("[%s] Unexpected norm range: min=%.4f max=%.4f", lang, norms.min(), norms.max())

    kw_array = np.array(keywords, dtype=object)

    # Write atomically
    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp_path), keywords=kw_array, embeddings=embeddings)
    tmp_path.replace(npz_path)

    log.info(
        "[%s] Saved %s  (%d keywords, dim=%d, %.1f MB).",
        lang, npz_path.name, len(keywords), embeddings.shape[1],
        npz_path.stat().st_size / 1024 / 1024,
    )
    return model


####################################################################################################
# CLI
####################################################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute keyword embeddings for resolve_concept() step 5.",
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
        "--model",
        default=DEFAULT_MODEL,
        help=f"sentence-transformers model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild .npz even if it already exists.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATASETS_DIR),
        metavar="DIR",
        help=f"Root datasets directory (default: {DATASETS_DIR})",
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
        "build_keyword_embeddings  langs=%s  model=%s  force=%s  dir=%s",
        args.langs, args.model, args.force, datasets_dir,
    )

    model = None
    for lang in args.langs:
        model = build_for_lang(lang, datasets_dir, args.model, args.force, model)

    log.info("Done.")


if __name__ == "__main__":
    main()
