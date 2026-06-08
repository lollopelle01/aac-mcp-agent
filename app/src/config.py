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


####### From settings (user defined) #####################################

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


####### Sensitive credentials from .env ##################################

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


####### App constants ####################################################

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


####### Category browser #################################################
# Two-level category hierarchy for the /categories and /by_category endpoints.
# Keys in each "categories" list are the original ARASAAC category strings.
# ARASAAC categories not listed in any macro go into the catch-all "Other"
# computed at runtime by api/server.py.

MACRO_CATEGORIES: list[dict] = [
    {
        "name": "Actions",
        "categories": [
            "verb", "usual verbs", "routine", "body position",
            "locomotion verb", "communication verb", "movement",
            "daily life activity", "action", "activity",
        ],
    },
    {
        "name": "People & Body",
        "categories": [
            "family", "human anatomy", "child", "adult", "elderly",
            "personal care", "body part", "human body", "person",
            "social role", "gender",
        ],
    },
    {
        "name": "Feelings",
        "categories": [
            "feeling", "human response", "disruptive behavior",
            "expression", "mood", "emotion", "behavior",
        ],
    },
    {
        "name": "Animals",
        "categories": [
            "terrestrial animal", "marine animal", "bird", "insect",
            "domestic animal", "pet", "farm animal", "wild animal",
            "reptile", "amphibian", "animal",
            "mammal", "viviparous", "herbivorous", "omnivorous",
            "carnivorous", "oviparous", "invertebrate", "arachnid",
        ],
    },
    {
        "name": "Food & Drink",
        "categories": [
            "food", "beverage", "fruit", "vegetable", "gastronomy",
            "baking", "meal", "snack", "dairy product", "meat", "fish",
            "mineral rich food", "legume", "cereal", "sweet", "dessert",
            "condiment", "spice",
        ],
    },
    {
        "name": "Places",
        "categories": [
            "residential building", "commercial building", "building room",
            "educational space", "public space", "outdoor space",
            "room", "city", "country", "continent", "space", "place",
            "environment",
        ],
    },
    {
        "name": "Objects",
        "categories": [
            "work tool", "utensil", "electrical appliance", "toy",
            "educational material", "kitchen", "container", "furniture",
            "household item", "electronic device", "object",
            "instrument", "material",
        ],
    },
    {
        "name": "Clothes",
        "categories": [
            "clothes", "footwear", "accessories", "clothing",
        ],
    },
    {
        "name": "Health",
        "categories": [
            "symptom", "disease", "medicament", "medical procedure",
            "hygiene product", "hospital", "medicine", "body care",
            "health", "medical",
        ],
    },
    {
        "name": "School & Work",
        "categories": [
            "educational task", "educational material", "subject",
            "professional", "school", "job", "work", "office",
            "study",
        ],
    },
    {
        "name": "Transport",
        "categories": [
            "land transport", "aerial transport", "water transport",
            "vehicle component", "transport",
        ],
    },
    {
        "name": "Nature",
        "categories": [
            "atmospheric phenomena", "landform", "plant", "flower",
            "tree", "weather", "season", "geography", "natural element",
            "nature", "landscape",
        ],
    },
    {
        "name": "Time & Numbers",
        "categories": [
            "number", "day hours", "unit of time", "month", "day",
            "time", "date", "year", "calendar",
        ],
    },
    {
        "name": "Communication",
        "categories": [
            "core vocabulary-communication", "mass media", "computing",
            "social interaction", "language", "communication",
            "symbol", "sign",
        ],
    },
]
