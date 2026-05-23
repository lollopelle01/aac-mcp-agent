# logging_config.py — configurazione centralizzata del logging per aac-mcp-agent.
#
# Uso: chiama setup_logging() prima possibile. È idempotente.
#
#   from logs.logging_config import setup_logging; setup_logging()
#
# Output:
#   logs/app.log     — tutto (DEBUG+), rotazione 2 MB × 5 file
#   logs/errors.log  — solo WARNING+, rotazione 1 MB × 3 file
#   logs/agent.log   — trace leggibile del flusso agent (INFO+): input LLM,
#                      tool calls, output LLM, risultato finale
#   stderr           — WARNING+ senza timestamp (leggibile nel terminale, catturato da pytest -s)
#
# Formato file:   "2025-04-15 14:23:01,042 WARNING  mcp_server.tools.arasaac: <msg>"
# Formato console:"WARNING  mcp_server.tools.arasaac: <msg>"
# Formato agent:  messaggio grezzo (nessun prefisso — già formattato dall'agent)
#
# Livello file: APP_LOG_LEVEL (default DEBUG). Es.: APP_LOG_LEVEL=INFO python ...

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

# ── Costanti ──────────────────────────────────────────────────────────────────

# Il file è dentro logs/, quindi Path(__file__).parent è già la cartella logs/.
_LOGS_DIR = Path(__file__).parent
_FILE_LEVEL = os.environ.get("APP_LOG_LEVEL", "DEBUG").upper()

_FMT_FILE    = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_FMT_CONSOLE = "%(levelname)-8s %(name)s: %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

_APP_LOG_MAX_BYTES    = 2 * 1024 * 1024   # 2 MB
_APP_LOG_BACKUP_COUNT = 5
_ERR_LOG_MAX_BYTES    = 1 * 1024 * 1024   # 1 MB
_ERR_LOG_BACKUP_COUNT = 3

_CONFIGURED = False


# ── Funzione pubblica ─────────────────────────────────────────────────────────

def setup_logging() -> None:
    """Configura il sistema di logging globale. Idempotente."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    config: dict = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "file_fmt":    {"format": _FMT_FILE,    "datefmt": _DATE_FMT},
            "console_fmt": {"format": _FMT_CONSOLE, "datefmt": _DATE_FMT},
            # Nessun prefisso: l'agent.py formatta già ogni riga leggibilmente
            "agent_fmt":   {"format": "%(message)s"},
        },

        "handlers": {
            "app_file": {
                "class":       "logging.handlers.RotatingFileHandler",
                "filename":    str(_LOGS_DIR / "app.log"),
                "maxBytes":    _APP_LOG_MAX_BYTES,
                "backupCount": _APP_LOG_BACKUP_COUNT,
                "encoding":    "utf-8",
                "formatter":   "file_fmt",
                "level":       _FILE_LEVEL,
            },
            "error_file": {
                "class":       "logging.handlers.RotatingFileHandler",
                "filename":    str(_LOGS_DIR / "errors.log"),
                "maxBytes":    _ERR_LOG_MAX_BYTES,
                "backupCount": _ERR_LOG_BACKUP_COUNT,
                "encoding":    "utf-8",
                "formatter":   "file_fmt",
                "level":       "WARNING",
            },
            # Trace leggibile del flusso agent: input LLM → tool → output
            "agent_file": {
                "class":       "logging.handlers.RotatingFileHandler",
                "filename":    str(_LOGS_DIR / "agent.log"),
                "maxBytes":    _APP_LOG_MAX_BYTES,
                "backupCount": _APP_LOG_BACKUP_COUNT,
                "encoding":    "utf-8",
                "formatter":   "agent_fmt",
                "level":       "INFO",
            },
            "console": {
                "class":     "logging.StreamHandler",
                "stream":    "ext://sys.stderr",
                "formatter": "console_fmt",
                "level":     "WARNING",
            },
        },

        "root": {
            "level":    "DEBUG",
            "handlers": ["app_file", "error_file", "console"],
        },

        "loggers": {
            # Logger dedicato al trace agent — non propaga al root per non
            # duplicare le righe in app.log con il formato sbagliato
            "agent.run": {
                "level":     "INFO",
                "handlers":  ["agent_file"],
                "propagate": False,
            },
            # Silenzia librerie verbose
            "urllib3":            {"level": "WARNING", "propagate": True},
            "requests":           {"level": "WARNING", "propagate": True},
            "charset_normalizer": {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(config)
    _CONFIGURED = True

    logging.getLogger(__name__).info(
        "Logging inizializzato. app.log=%s, errors.log=%s, agent.log=%s, file_level=%s",
        _LOGS_DIR / "app.log",
        _LOGS_DIR / "errors.log",
        _LOGS_DIR / "agent.log",
        _FILE_LEVEL,
    )
