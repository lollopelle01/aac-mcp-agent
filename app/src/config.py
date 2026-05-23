from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from settings import settings as _s

_SRC  = Path(__file__).resolve().parent
_APP  = _SRC.parent                    
_ROOT = _APP.parent                    

# Load app/.env before accessing os.environ (no-op if file doesn't exist)
load_dotenv(_APP / ".env")


######## From settings (used defined) ###########################################
MODELS                    = _s.models
TIMEZONE                  = _s.timezone
CALENDAR_PROVIDER         = _s.calendar_provider
LANG                      = _s.lang
USE_LOCAL_DATASETS        = _s.use_local_datasets
DATASET_LANGS             = _s.dataset_langs
SUBSCRIBED_ICS_URLS       = _s.subscribed_ics_urls
AGENT_DEFAULT_MODEL       = _s.agent_default_model
AGENT_MAX_RESULTS         = _s.agent_max_results
AGENT_CANDIDATES_PER_TERM = _s.agent_candidates_per_term
AGENT_MEMORY_TURNS        = _s.agent_memory_turns
AGENT_FETCH_SCHEDULE      = _s.agent_fetch_schedule
AGENT_SYNSET_EXPAND       = _s.agent_synset_expand
AGENT_SYNSET_EXPAND_MAX   = _s.agent_synset_expand_max


######## Sensitive credentials from .env ###########################################

# Google Calendar — OAuth2 credentials (files stored in app/credentials/)
_CREDENTIALS_DIR        = _APP / "credentials"
GOOGLE_CREDENTIALS_PATH = str(_CREDENTIALS_DIR / "credentials.json")
GOOGLE_TOKEN_PATH       = str(_CREDENTIALS_DIR / "token.pickle")
GOOGLE_CALENDAR_ID      = os.environ.get("GOOGLE_CALENDAR_ID", "primary")

# Apple iCloud — CalDAV + app-specific password
APPLE_CALDAV_URL   = os.environ.get("APPLE_CALDAV_URL", "https://caldav.icloud.com/")
APPLE_USERNAME     = os.environ.get("APPLE_USERNAME", "")
APPLE_APP_PASSWORD = os.environ.get("APPLE_APP_PASSWORD", "")

# HuggingFace (used only for cluster evaluation — eval/cluster_work/run_eval_hf.py)
HF_DATASET = os.environ.get("HF_DATASET", "")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")


######## App constants ##############################################################

OLLAMA_BASE_URL = "http://localhost:11434"

# Time-of-day slots
DAY_TIMES:           list[str] = ["morning", "afternoon", "evening", "night"]
DAY_TIME_THRESHOLDS: list[int] = [5, 13, 18, 21]

# ARASAAC public API
ARASAAC_API_BASE    = "https://api.arasaac.org/v1"
ARASAAC_IMG_PATTERN = "https://static.arasaac.org/pictograms/{id}/{id}_2500.png"
ARASAAC_TIMEOUT     = 10   # seconds

# Local dataset lives inside app/
DATASETS_DIR = _APP / "datasets"


