"""
These are the values the frontend Settings panel can read and write.
Persisted to app/user_settings.json (gitignored) at the app root.
On first run, user_settings.json is created automatically with the defaults below.

Not to be confused with config.py, which holds:
  - pure app constants and is never user-settable
  - sensitive credentials, loaded from app/.env via python-dotenv
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SRC           = Path(__file__).resolve().parent
_APP           = _SRC.parent
_SETTINGS_FILE = _APP / "user_settings.json"

####### Defaults #########################################################

_DEFAULTS: dict[str, Any] = {
    # Locale / time
    "timezone":         "Europe/Rome",
    "lang":             "en",

    # Calendar
    "calendar_provider": "apple",
    "subscribed_ics_urls": {
        # UNIBO calendar — Ethics lectures only
        "UNIBO": (
            "https://calendar.students.cs.unibo.it/cal/9063/2"
            "?curr=000-000&subjects=91257_1,91257_2"
        ),
    },

    # Ollama models available locally.
    # num_predict: cap on generated tokens for the planner call. The expected
    #   JSON output is ~50 tokens; 150 is a safe ceiling that avoids runaway
    #   generation while leaving room for slightly verbose models.
    # num_ctx: context window. The planner prompt is well under 512 tokens in
    #   both full and short variants; 512 reduces KV-cache on CPU vs Ollama default.
    # NOTE: the -h suffix on granite4:3b-h is the hybrid mamba-2 architecture,
    #   NOT a reasoning/thinking mode. None of these models activate reasoning
    #   by default.
    "models": {
        "granite4:3b-h": {"size_gb": 2.1, "num_predict": 150, "num_ctx": 512},
        "qwen2.5:3b":    {"size_gb": 2.0, "num_predict": 150, "num_ctx": 512},
        "llama3.2:3b":   {"size_gb": 1.9, "num_predict": 150, "num_ctx": 512},
        "mistral:7b":    {"size_gb": 4.1, "num_predict": 150, "num_ctx": 512},
    },

    # GGUF model paths for llama-cpp-python backend.
    # Keys mirror the Ollama aliases for easy cross-reference.
    # Paths are relative to the app/ directory.
    "gguf_models": {
        "qwen2.5:3b":    "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "llama3.2:3b":   "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "granite4:3b-h": "models/ibm-granite_granite-4.1-3b-Q4_K_M.gguf",
        "mistral:7b":    "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    },

    # Agent behaviour
    "agent_default_model":       "qwen2.5:3b",
    "agent_use_llamacpp":        True,   # if True, use LlamaCppBackend instead of Ollama
    "agent_max_results":         25,     # R22: 24->25 (eval window size experiment)
    "agent_candidates_per_term": 10,
    "agent_memory_turns":        3,
    "agent_fetch_schedule":      True,
    "agent_synset_expand":       True,
    "agent_synset_expand_max":   8,

    # Dataset
    # en_eval is the merged eval dataset (local + HF clean) used by run_eval_hf.py.
    # It is never touched by update_datasets.py -- treat it as a frozen snapshot.
    "use_local_datasets": True,
    "dataset_langs":      ["en", "en_eval", "it", "es"],
}


####### Manager ##########################################################

class _SettingsManager:
    """Singleton settings manager -- import as `from settings import settings`."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = dict(_DEFAULTS)
        self._load()

    ####### Persistence ###################################################

    def _load(self) -> None:
        if _SETTINGS_FILE.exists():
            try:
                stored     = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                self._data = {**_DEFAULTS, **stored}
                logger.debug("Settings loaded from %s", _SETTINGS_FILE)
            except Exception as exc:
                logger.warning(
                    "Could not read %s: %s -- using defaults.", _SETTINGS_FILE, exc
                )
        else:
            self._save()
            logger.info("Created %s with default settings.", _SETTINGS_FILE)

    def _save(self) -> None:
        try:
            _SETTINGS_FILE.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Could not write %s: %s", _SETTINGS_FILE, exc)

    ####### Public API ####################################################

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def update(self, updates: dict[str, Any]) -> None:
        """Update one or more settings and persist to disk immediately."""
        changed = False
        for k, v in updates.items():
            if k in _DEFAULTS:
                self._data[k] = v
                changed = True
            else:
                logger.warning("Unknown setting key %r -- ignored.", k)
        if changed:
            self._save()

    def reload(self) -> None:
        """Re-read user_settings.json from disk (e.g. after external edit)."""
        self._load()

    def all(self) -> dict[str, Any]:
        """Return a copy of all current settings (safe to serialise to JSON)."""
        return dict(self._data)

    ####### Getters #######################################################

    @property
    def timezone(self) -> str:
        return str(self._data["timezone"])

    @property
    def lang(self) -> str:
        return str(self._data["lang"])

    @property
    def calendar_provider(self) -> str:
        return str(self._data["calendar_provider"])

    @property
    def subscribed_ics_urls(self) -> dict[str, str]:
        return dict(self._data["subscribed_ics_urls"])

    @property
    def models(self) -> dict[str, dict]:
        return dict(self._data["models"])

    @property
    def agent_default_model(self) -> str:
        return str(self._data["agent_default_model"])

    @property
    def agent_use_llamacpp(self) -> bool:
        return bool(self._data["agent_use_llamacpp"])

    @property
    def agent_max_results(self) -> int:
        return int(self._data["agent_max_results"])

    @property
    def agent_candidates_per_term(self) -> int:
        return int(self._data["agent_candidates_per_term"])

    @property
    def agent_memory_turns(self) -> int:
        return int(self._data["agent_memory_turns"])

    @property
    def agent_fetch_schedule(self) -> bool:
        return bool(self._data["agent_fetch_schedule"])

    @property
    def agent_synset_expand(self) -> bool:
        return bool(self._data["agent_synset_expand"])

    @property
    def agent_synset_expand_max(self) -> int:
        return int(self._data["agent_synset_expand_max"])

    @property
    def gguf_models(self) -> dict[str, str]:
        return dict(self._data["gguf_models"])

    @property
    def use_local_datasets(self) -> bool:
        return bool(self._data["use_local_datasets"])

    @property
    def dataset_langs(self) -> list[str]:
        return list(self._data["dataset_langs"])


####### Singleton ########################################################

settings = _SettingsManager()
