# logging_config.py -- centralised logging configuration for aac-mcp-agent.
#
# Usage: call setup_logging() as early as possible. It is idempotent.
#
#   from logs.logging_config import setup_logging; setup_logging()
#
# Output:
#   logs/app.log     -- everything (DEBUG+), rotation 2 MB x 5 files
#   logs/errors.log  -- WARNING+ only, rotation 1 MB x 3 files
#   logs/agent.log   -- human-readable trace of the agent flow (INFO+): LLM input,
#                       tool calls, LLM output, final result
#   stderr           -- WARNING+ without timestamp (readable in terminal, captured by pytest -s)
#
# File format:    "2025-04-15 14:23:01,042 WARNING  mcp_server.tools.arasaac: <msg>"
# Console format: "WARNING  mcp_server.tools.arasaac: <msg>"
# Agent format:   raw message (no prefix -- already formatted by the agent)
#
# File level: APP_LOG_LEVEL (default DEBUG). E.g.: APP_LOG_LEVEL=INFO python ...

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

##############################################################################
## Constants #################################################################
##############################################################################

# This file lives inside logs/, so Path(__file__).parent is already the logs/ folder.
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


##############################################################################
## Public function ###########################################################
##############################################################################

def setup_logging() -> None:
    """Configure the global logging system. Idempotent."""
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
            # No prefix: agent.py already formats each line in a readable way
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
            # Human-readable trace of the agent flow: LLM input -> tool -> output
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
            # Dedicated logger for agent trace -- does not propagate to root to avoid
            # duplicating lines in app.log with the wrong format
            "agent.run": {
                "level":     "INFO",
                "handlers":  ["agent_file"],
                "propagate": False,
            },
            # Silence verbose libraries
            "urllib3":            {"level": "WARNING", "propagate": True},
            "requests":           {"level": "WARNING", "propagate": True},
            "charset_normalizer": {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(config)
    _CONFIGURED = True

    logging.getLogger(__name__).info(
        "Logging initialised. app.log=%s, errors.log=%s, agent.log=%s, file_level=%s",
        _LOGS_DIR / "app.log",
        _LOGS_DIR / "errors.log",
        _LOGS_DIR / "agent.log",
        _FILE_LEVEL,
    )
