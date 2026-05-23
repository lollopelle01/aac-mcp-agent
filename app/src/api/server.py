"""
src/api/server.py — FastAPI backend for the AAC pictogram selection app.

Wraps AACAgent in a stateful HTTP server. One agent instance per process
(single-user, local app). Session lives in-process memory.

Run from the app/ directory:
    cd app
    uvicorn src.api.server:app --reload --port 8000

Endpoints
---------
    POST /run           {"text": str}         → PictogramList
    POST /select        {"pictogram_id": int} → {"ok": true}
    POST /reset                               → {"ok": true}
    GET  /session                             → SessionHistory
    GET  /settings                            → dict
    PATCH /settings     {key: value, ...}     → {"ok": true}
    GET  /health                              → HealthStatus
    GET  /images/{id}       (PNG proxy)         → image/png
    GET  /datasets/status                       → DatasetStatus
    POST /datasets/update  {langs?, force?, images?} → SSE stream of log lines
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
# This file lives at app/src/api/server.py.
# app/src/ must be on sys.path for all package imports (config, agent, mcp_server…).
_SRC = Path(__file__).resolve().parent.parent   # app/src/
_APP = _SRC.parent                              # app/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))               # allows `from logs.logging_config import …`

# ── Logging setup (before any other local import) ────────────────────────────
try:
    from logs.logging_config import setup_logging
    setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)

# ── FastAPI imports ───────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# ── Domain imports ────────────────────────────────────────────────────────────
from config import AGENT_DEFAULT_MODEL, DATASETS_DIR, DATASET_LANGS
from settings import settings
from agent.agent import AACAgent
from agent.prompts import build_planner_prompt, build_planner_message
from agent.backends import LlamaCppBackend
from mcp_server.tools.arasaac import (
    _image_url,
    get_pictogram_image,
    get_pictogram_metadata,
)

# ── Warmup state ─────────────────────────────────────────────────────────────
# Tracks whether the model is loaded and ready for inference.
# warming_up=True while the GGUF is being read from disk (first load).
# Set to False when _ensure_loaded() completes; errors are logged but non-fatal.
_warmup_done  = threading.Event()   # set when model is loaded
_warmup_error: Optional[str] = None # set if warmup failed


def _warmup_agent() -> None:
    """Load the agent and its LLM backend in a background thread.

    Called once at startup via the lifespan hook, and again whenever the model
    setting changes (triggered by PATCH /settings). Safe to call concurrently —
    the second call is a no-op if the agent is already loaded for the requested
    model.
    """
    global _warmup_error
    _warmup_done.clear()
    _warmup_error = None
    t0 = time.perf_counter()
    try:
        agent = _get_agent()          # creates AACAgent + LlamaCppBackend
        if agent.backend is not None:
            agent.backend._ensure_loaded()   # load GGUF into RAM now
            # Run a dummy inference with the REAL planner prompt so llama.cpp
            # caches the system prompt prefix. A different prompt at warmup means
            # the first real /run still pays full prefill cost — defeating warmup.
            try:
                agent.backend.chat(
                    system = build_planner_prompt(full=False),
                    user   = build_planner_message("he wants water"),
                )
                logger.info("[WARMUP] dummy inference complete")
            except Exception as exc:
                logger.warning("[WARMUP] dummy inference failed (non-fatal): %s", exc)
        elapsed = time.perf_counter() - t0
        logger.info("[WARMUP] model ready in %.1fs", elapsed)
    except Exception as exc:
        _warmup_error = str(exc)
        logger.error("[WARMUP] failed: %s", exc)
    finally:
        _warmup_done.set()


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan: start model warmup in background, then serve."""
    thread = threading.Thread(target=_warmup_agent, name="warmup", daemon=True)
    thread.start()
    logger.info("[WARMUP] background model load started")
    yield
    # Nothing to clean up — threads are daemon, process exit handles the rest.


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="AAC Pictogram Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Single shared agent instance (one user, local app) ───────────────────────
_agent: Optional[AACAgent] = None


def _get_agent() -> AACAgent:
    """Lazily create the agent; rebuild if the model setting changed."""
    global _agent
    current_model = settings.agent_default_model
    if _agent is None or _agent.model != current_model:
        backend = None
        if settings.agent_use_llamacpp:
            gguf_models = settings.gguf_models
            gguf_rel    = gguf_models.get(current_model)
            if gguf_rel:
                gguf_path = str(_APP / gguf_rel)
                logger.info("Using LlamaCppBackend: %s", gguf_path)
                backend = LlamaCppBackend(
                    model_path = gguf_path,
                    n_ctx      = 512,
                    n_threads  = 4,
                    verbose    = False,
                )
            else:
                logger.warning(
                    "agent_use_llamacpp=True but no GGUF found for model %r — "
                    "falling back to Ollama.", current_model
                )
        logger.info("Creating AACAgent model=%r  backend=%s",
                    current_model, type(backend).__name__ if backend else "OllamaHTTP")
        _agent = AACAgent(model=current_model, backend=backend)
    return _agent


# ── Request / response schemas ────────────────────────────────────────────────

class RunRequest(BaseModel):
    text: str

class SelectRequest(BaseModel):
    pictogram_id: int

class PatchSettingsRequest(BaseModel):
    updates: dict[str, Any]

class PictogramOut(BaseModel):
    id:         int
    image_url:  str
    label:      str
    categories: list[str]
    aac:        bool

class RunResponse(BaseModel):
    pictograms:   list[PictogramOut]
    turn:         int
    tools_called: bool  # True if the planner invoked get_time / get_schedule

class SessionTurnOut(BaseModel):
    turn_id:     int
    time_of_day: Optional[str]
    raw_input:   str
    pictograms:  list[PictogramOut]

class SessionResponse(BaseModel):
    turns: list[SessionTurnOut]

class HealthResponse(BaseModel):
    ok:          bool
    model:       str
    ollama:      bool
    warming_up:  bool   # True while GGUF is still loading from disk
    warmup_error: Optional[str] = None  # set if warmup failed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pic_to_out(pic) -> PictogramOut:
    label = pic.keywords[0].keyword if pic.keywords else str(pic.id)
    return PictogramOut(
        id         = pic.id,
        image_url  = _image_url(pic.id),
        label      = label,
        categories = pic.categories,
        aac        = pic.aac,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    """Execute one agent turn: caregiver input → pictogram grid.

    agent.run() calls the LLM backend (llama.cpp or Ollama) which is a
    blocking, CPU-bound operation that can take 10–60 s.  Running it in a
    plain sync endpoint would block uvicorn's asyncio event loop for the
    entire duration, causing the Vite proxy to receive "socket hang up"
    before any response arrives.

    asyncio.to_thread() offloads the blocking call to a ThreadPoolExecutor
    thread so the event loop stays alive, keep-alives are sent, and the
    response is returned as soon as inference finishes.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    agent = _get_agent()

    try:
        result = await asyncio.to_thread(agent.run, req.text)
    except Exception as exc:
        logger.exception("/run agent.run() raised: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    turn = agent.memory.turn_count
    logger.info("/run turn=%d  input=%r  results=%d  tools_called=%s",
                turn, req.text, len(result), agent.last_call_tools)
    return RunResponse(
        pictograms   = [_pic_to_out(p) for p in result],
        turn         = turn,
        tools_called = agent.last_call_tools,
    )


@app.post("/select")
def select(req: SelectRequest) -> dict:
    """Register a pictogram selection into session memory.

    In the UI the subject taps a card; this call records it so the next
    /run turn has accurate history. The agent's last turn is updated in-place:
    we replace its pictogram list with the single selected pictogram so that
    prompt_summary reflects the actual user choice.
    """
    agent = _get_agent()
    if not agent.memory.turns:
        raise HTTPException(status_code=400, detail="No active turn to record selection for")

    pid = req.pictogram_id
    # Find the pictogram object in the last turn's result set
    last = agent.memory.turns[-1]
    chosen = next((p for p in last.pictograms if p.id == pid), None)

    if chosen is None:
        # Pictogram not in last window — fetch metadata and inject anyway
        # (covers the case where user navigates manually outside the grid)
        try:
            meta    = get_pictogram_metadata(pid)
            from mcp_server.models import Pictogram
            chosen  = Pictogram.model_validate(meta)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Pictogram {pid} not found: {exc}")

    # Narrow pictograms to the single selection so prompt_summary reflects
    # what was actually chosen. `presented` is preserved unchanged so that
    # recently_presented_ids() keeps excluding the full previous window.
    last.pictograms = [chosen]
    last.topics     = agent.memory.extract_topics([chosen])
    # Update frequency counter
    for t in last.topics:
        agent.memory.topic_frequency[t] = agent.memory.topic_frequency.get(t, 0) + 1

    logger.info("/select  pid=%d  label=%r", pid, chosen.keywords[0].keyword if chosen.keywords else "?")
    return {"ok": True}


@app.post("/reset")
def reset() -> dict:
    """Clear the session: all history and topic memory."""
    _get_agent().reset_session()
    logger.info("/reset — session cleared")
    return {"ok": True}


@app.get("/session", response_model=SessionResponse)
def session() -> SessionResponse:
    """Return the full session history (for the sidebar)."""
    agent = _get_agent()
    turns = [
        SessionTurnOut(
            turn_id     = t.turn_id,
            time_of_day = t.time_of_day,
            raw_input   = t.raw_input,
            pictograms  = [_pic_to_out(p) for p in t.pictograms],
        )
        for t in agent.memory.turns
    ]
    return SessionResponse(turns=turns)


@app.get("/settings")
def get_settings() -> dict:
    """Return all user settings (excludes sensitive credentials)."""
    return settings.all()


@app.patch("/settings")
def patch_settings(req: PatchSettingsRequest) -> dict:
    """Update one or more user settings and persist to disk.

    If agent_default_model changes, triggers a background warmup of the new
    model so it's ready before the user sends the first request.
    """
    old_model = settings.agent_default_model
    settings.update(req.updates)
    new_model = settings.agent_default_model
    logger.info("/settings PATCH  keys=%s", list(req.updates.keys()))

    if new_model != old_model:
        # Invalidate the cached agent so _get_agent() rebuilds for new_model.
        global _agent
        _agent = None
        thread = threading.Thread(target=_warmup_agent, name="warmup-switch", daemon=True)
        thread.start()
        logger.info("[WARMUP] model switch %r → %r — background warmup started", old_model, new_model)

    return {"ok": True}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Simple health check: model name + Ollama reachability + warmup status."""
    model = settings.agent_default_model
    try:
        import ollama as _ol
        _ol.list()
        ollama_ok = True
    except Exception:
        ollama_ok = False
    return HealthResponse(
        ok           = True,
        model        = model,
        ollama       = ollama_ok,
        warming_up   = not _warmup_done.is_set(),
        warmup_error = _warmup_error,
    )


@app.get("/images/{pictogram_id}")
def get_image(pictogram_id: int) -> Response:
    """Serve pictogram PNG from local dataset or ARASAAC CDN."""
    try:
        png_bytes = get_pictogram_image(pictogram_id)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── Category browser ────────────────────────────────────────────────────────
#
# Two-level category hierarchy for manual pictogram search.
# Level 0: ~15 macro-categories (defined here)
# Level 1: original ARASAAC category names within each macro
# Level 2: pictograms in a specific category (via /by_category)
#
# ARASAAC categories not listed in any macro go into the catch-all "Other".

MACRO_CATEGORIES: list[dict] = [
    {
        "name": "Actions",
        "emoji": "🏃",
        "categories": [
            "verb", "usual verbs", "routine", "body position",
            "locomotion verb", "communication verb", "movement",
            "daily life activity", "action", "activity",
        ],
    },
    {
        "name": "People & Body",
        "emoji": "👤",
        "categories": [
            "family", "human anatomy", "child", "adult", "elderly",
            "personal care", "body part", "human body", "person",
            "social role", "gender",
        ],
    },
    {
        "name": "Feelings",
        "emoji": "😊",
        "categories": [
            "feeling", "human response", "disruptive behavior",
            "expression", "mood", "emotion", "behavior",
        ],
    },
    {
        "name": "Animals",
        "emoji": "🐾",
        "categories": [
            "terrestrial animal", "marine animal", "bird", "insect",
            "domestic animal", "pet", "farm animal", "wild animal",
            "reptile", "amphibian", "animal",
            # biological trait categories that ARASAAC applies to animals
            # and would otherwise overflow into "Other"
            "mammal", "viviparous", "herbivorous", "omnivorous",
            "carnivorous", "oviparous", "invertebrate", "arachnid",
        ],
    },
    {
        "name": "Food & Drink",
        "emoji": "🍎",
        "categories": [
            "food", "beverage", "fruit", "vegetable", "gastronomy",
            "baking", "meal", "snack", "dairy product", "meat", "fish",
            "mineral rich food", "legume", "cereal", "sweet", "dessert",
            "condiment", "spice",
        ],
    },
    {
        "name": "Places",
        "emoji": "🏠",
        "categories": [
            "residential building", "commercial building", "building room",
            "educational space", "public space", "outdoor space",
            "room", "city", "country", "continent", "space", "place",
            "environment",
        ],
    },
    {
        "name": "Objects",
        "emoji": "🔧",
        "categories": [
            "work tool", "utensil", "electrical appliance", "toy",
            "educational material", "kitchen", "container", "furniture",
            "household item", "electronic device", "object",
            "instrument", "material",
        ],
    },
    {
        "name": "Clothes",
        "emoji": "👕",
        "categories": [
            "clothes", "footwear", "accessories", "clothing",
        ],
    },
    {
        "name": "Health",
        "emoji": "🏥",
        "categories": [
            "symptom", "disease", "medicament", "medical procedure",
            "hygiene product", "hospital", "medicine", "body care",
            "health", "medical",
        ],
    },
    {
        "name": "School & Work",
        "emoji": "📚",
        "categories": [
            "educational task", "educational material", "subject",
            "professional", "school", "job", "work", "office",
            "study",
        ],
    },
    {
        "name": "Transport",
        "emoji": "🚗",
        "categories": [
            "land transport", "aerial transport", "water transport",
            "vehicle component", "transport",
        ],
    },
    {
        "name": "Nature",
        "emoji": "🌿",
        "categories": [
            "atmospheric phenomena", "landform", "plant", "flower",
            "tree", "weather", "season", "geography", "natural element",
            "nature", "landscape",
        ],
    },
    {
        "name": "Time & Numbers",
        "emoji": "🕐",
        "categories": [
            "number", "day hours", "unit of time", "month", "day",
            "time", "date", "year", "calendar",
        ],
    },
    {
        "name": "Communication",
        "emoji": "💬",
        "categories": [
            "core vocabulary-communication", "mass media", "computing",
            "social interaction", "language", "communication",
            "symbol", "sign",
        ],
    },
]


@app.get("/categories")
def get_categories(lang: str = "en") -> dict:
    """Return the macro-category tree with sub-categories, counts and representative IDs.

    Counts reflect *unique* pictograms per category / macro (a pictogram that
    belongs to multiple categories is counted only once per macro, and only once
    in the macro-level total). The "Other" macro contains only pictograms that do
    not appear in any of the named macro-categories.

    Response structure::

        {
          "macros": [
            {
              "name": "Food & Drink",
              "emoji": "🍎",
              "count": 830,            # unique pictograms in this macro
              "representative_id": 2248,
              "categories": [
                {"name": "food", "count": 64, "representative_id": 1234},
                ...
              ]
            },
            ...
          ]
        }
    """
    from mcp_server.dataset_cache import _DatasetCache
    pics = _DatasetCache.load_pictograms(lang)
    if not pics:
        raise HTTPException(
            status_code=503,
            detail=f"Dataset not available for lang='{lang}'. Run /datasets/update first.",
        )

    # ── Step 1: build per-category sets of IDs (unique per category) ─────────
    # cat_ids: category_name → set of pictogram IDs that belong to it
    # cat_rep: category_name → (representative_id, has_aac_rep)
    cat_ids: dict[str, set[int]] = {}
    cat_rep: dict[str, tuple[int, bool]] = {}
    for id_str, rec in pics.items():
        pid = int(id_str)
        aac = rec.get("aac", False)
        for cat in rec.get("categories", []):
            if cat not in cat_ids:
                cat_ids[cat] = set()
                cat_rep[cat] = (pid, False)
            cat_ids[cat].add(pid)
            rep_id, has_aac = cat_rep[cat]
            if aac and not has_aac:
                cat_rep[cat] = (pid, True)

    # ── Step 2: build macro-categories ────────────────────────────────────────
    # A pictogram is counted at most once per macro even if it belongs to several
    # sub-categories of the same macro.
    covered_cats: set[str] = set()   # ARASAAC category names assigned to a macro
    covered_pids: set[int] = set()   # pictogram IDs assigned to any named macro
    macros: list[dict] = []

    for mc in MACRO_CATEGORIES:
        sub_cats: list[dict] = []
        macro_pids: set[int] = set()
        macro_rep_id: int | None = None

        for cat_name in mc["categories"]:
            if cat_name not in cat_ids:
                continue
            ids = cat_ids[cat_name]
            rep_id, _ = cat_rep[cat_name]
            sub_cats.append({
                "name":              cat_name,
                "count":             len(ids),
                "representative_id": rep_id,
            })
            macro_pids.update(ids)
            covered_cats.add(cat_name)
            if macro_rep_id is None:
                macro_rep_id = rep_id

        if not sub_cats:
            continue

        covered_pids.update(macro_pids)
        # Sort sub-categories by count descending
        sub_cats.sort(key=lambda x: -x["count"])
        macros.append({
            "name":              mc["name"],
            "emoji":             mc["emoji"],
            "count":             len(macro_pids),  # unique IDs in this macro
            "representative_id": macro_rep_id,
            "categories":        sub_cats,
        })

    # ── Step 3: "Other" — only pictograms not covered by any named macro ──────
    # Collect uncovered pictograms, grouped by their first uncovered category.
    # We avoid inflating the count by tracking unique IDs.
    other_cat_ids: dict[str, set[int]] = {}
    other_cat_rep: dict[str, tuple[int, bool]] = {}
    for id_str, rec in pics.items():
        pid = int(id_str)
        if pid in covered_pids:
            continue
        aac  = rec.get("aac", False)
        cats = rec.get("categories", [])
        # Assign this pictogram to its first category (or a synthetic "uncategorised")
        bucket = cats[0] if cats else "uncategorised"
        if bucket not in other_cat_ids:
            other_cat_ids[bucket] = set()
            other_cat_rep[bucket] = (pid, False)
        other_cat_ids[bucket].add(pid)
        rep_id, has_aac = other_cat_rep[bucket]
        if aac and not has_aac:
            other_cat_rep[bucket] = (pid, True)

    if other_cat_ids:
        other_sub: list[dict] = []
        all_other_pids: set[int] = set()
        other_rep_id: int | None = None
        for cat_name, ids in sorted(other_cat_ids.items(), key=lambda x: -len(x[1])):
            rep_id, _ = other_cat_rep[cat_name]
            other_sub.append({
                "name":              cat_name,
                "count":             len(ids),
                "representative_id": rep_id,
            })
            all_other_pids.update(ids)
            if other_rep_id is None:
                other_rep_id = rep_id
        macros.append({
            "name":              "Other",
            "emoji":             "📦",
            "count":             len(all_other_pids),
            "representative_id": other_rep_id,
            "categories":        other_sub,
        })

    return {"macros": macros}


@app.get("/by_category")
def get_by_category(
    category: str,
    lang: str = "en",
    max_results: int = 50,
) -> list:
    """Return pictograms that belong to a given ARASAAC category.

    Results are sorted: AAC=True first, then alphabetically by label.
    Limit is enforced by ``max_results`` (default 50).
    """
    from mcp_server.dataset_cache import _DatasetCache
    pics = _DatasetCache.load_pictograms(lang)
    if not pics:
        raise HTTPException(
            status_code=503,
            detail=f"Dataset not available for lang='{lang}'.",
        )

    results: list[dict] = []
    for id_str, rec in pics.items():
        if category not in rec.get("categories", []):
            continue
        pid      = int(id_str)
        keywords = rec.get("keywords", [])
        label    = keywords[0].get("keyword", str(pid)) if keywords else str(pid)
        results.append({
            "id":         pid,
            "label":      label,
            "image_url":  _image_url(pid),
            "categories": rec.get("categories", []),
            "aac":        rec.get("aac", False),
        })

    results.sort(key=lambda r: (not r["aac"], r["label"].lower()))
    logger.debug("/by_category category=%r lang=%r → %d results", category, lang, len(results))
    return results[:max_results]


# ── Dataset update ────────────────────────────────────────────────────────────

# One concurrent update job at a time; lock prevents double-runs.
_update_lock = threading.Lock()


class DatasetUpdateRequest(BaseModel):
    langs:           Optional[list[str]] = None   # None = all configured langs
    force:           bool = False
    download_images: bool = False


@app.get("/datasets/status")
def datasets_status() -> dict:
    """Return metadata for each configured language dataset."""
    import json
    result: dict[str, Any] = {}
    for lang in DATASET_LANGS:
        meta_path = DATASETS_DIR / lang / "_meta.json"
        try:
            with open(meta_path, encoding="utf-8") as f:
                result[lang] = json.load(f)
        except FileNotFoundError:
            result[lang] = None
    images_dir = DATASETS_DIR / "pictograms"
    result["images"] = {
        "count": len(list(images_dir.glob("*.png"))) if images_dir.is_dir() else 0
    }
    return result


@app.post("/datasets/update")
def datasets_update(req: DatasetUpdateRequest) -> StreamingResponse:
    """Stream a dataset update as Server-Sent Events.

    Runs update_datasets.build_lang_dataset() in a background thread so the
    HTTP connection isn't blocked. Log output is forwarded line-by-line as
    SSE ``data:`` messages so the UI can show a live progress feed.

    SSE event format::

        data: {"type": "log",  "msg": "..."}
        data: {"type": "done", "ok": true|false}
    """
    if _update_lock.locked():
        raise HTTPException(status_code=409, detail="Dataset update already in progress")

    langs = req.langs or DATASET_LANGS
    # Resolve datasets/ dir relative to update_datasets.py which now lives there
    datasets_dir = DATASETS_DIR

    async def _event_stream():
        import json as _json
        import queue as _queue

        loop = asyncio.get_event_loop()
        q: _queue.Queue = _queue.Queue()

        # ── Custom log handler that pushes lines into the queue ────────────
        class _QueueHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                q.put_nowait(self.format(record))

        handler = _QueueHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

        # Attach to the root logger used inside update_datasets
        _ud_logger = logging.getLogger()  # root
        _ud_logger.addHandler(handler)

        # ── Run the update in a thread ─────────────────────────────────────
        result_container: list[bool] = []

        def _run():
            with _update_lock:
                try:
                    # Load update_datasets.py from its absolute path to avoid
                    # collision with the 'datasets' PyPI package (HuggingFace).
                    import importlib.util as _ilu
                    _ud_path = datasets_dir / "update_datasets.py"
                    _spec = _ilu.spec_from_file_location("_update_datasets", _ud_path)
                    _ud_mod = _ilu.module_from_spec(_spec)  # type: ignore
                    _spec.loader.exec_module(_ud_mod)  # type: ignore
                    build_lang_dataset = _ud_mod.build_lang_dataset
                    ok = True
                    for lang in langs:
                        if not build_lang_dataset(
                            lang=lang,
                            datasets_dir=datasets_dir,
                            force=req.force,
                            download_images=req.download_images,
                        ):
                            ok = False
                    result_container.append(ok)
                except Exception as exc:
                    logger.exception("Dataset update failed: %s", exc)
                    result_container.append(False)
                finally:
                    q.put_nowait(None)   # sentinel

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # ── Forward queue items as SSE ────────────────────────────────────
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        loop.run_in_executor(None, q.get, True, 0.2),
                        timeout=0.5,
                    )
                except (asyncio.TimeoutError, Exception):
                    item = _queue.Empty  # keep polling

                if item is None:         # sentinel — thread finished
                    break
                if item is _queue.Empty:
                    continue

                payload = _json.dumps({"type": "log", "msg": str(item)})
                yield f"data: {payload}\n\n"
        finally:
            _ud_logger.removeHandler(handler)
            thread.join(timeout=5)

        ok = result_container[0] if result_container else False
        yield f"data: {_json.dumps({'type': 'done', 'ok': ok})}\n\n"
        # Invalidate the dataset cache so the next query uses fresh data
        try:
            from mcp_server.dataset_cache import _DatasetCache
            _DatasetCache.invalidate()
            logger.info("Dataset cache invalidated after update.")
        except Exception as exc:
            logger.warning("Could not invalidate dataset cache: %s", exc)

    return StreamingResponse(_event_stream(), media_type="text/event-stream")
