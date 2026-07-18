import json
import logging
import os
from datetime import datetime

from mcp_server.server import mcp
from mcp_server.models import TimeInfo
from config import DAY_TIMES, DAY_TIME_THRESHOLDS

logger = logging.getLogger(__name__)

####### Utils ############################################################

def _resolve_time_of_day(hour: int) -> str:
    """
    Map an integer hour (0-23) to the corresponding
    time-of-day label defined in config.DAY_TIMES.
    """
    for i in range(len(DAY_TIME_THRESHOLDS) - 1, -1, -1):
        if hour >= DAY_TIME_THRESHOLDS[i]:
            return DAY_TIMES[i]
    # hour < DAY_TIME_THRESHOLDS[0] (i.e. before 05:00) -> "morning" (safe fallback)
    return DAY_TIMES[0]


####### MCP tool #########################################################

@mcp.tool()
def get_time() -> dict:
    """
    Return the current local datetime and the time-of-day slot.

    Use this tool when the caregiver's input is vague about the time of
    day or the date and temporal context is needed. You can use this with
    the tool get_schedule in order to infer the actual plans and enrich the
    content of the caregiver.

    Returns
    -------
    dict with fields:
        current_dt  : str  -- ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS)
        time_of_day : str  -- slot label (e.g. "morning", "afternoon", etc.)
    """
    # Demo/eval override: when MOCK_TIME_INFO is set in the environment, return
    # it as-is instead of the real system clock. Mirrors the MOCK_SCHEDULE_EVENTS
    # override in schedule_tool.py -- used together so a notebook/script can pin
    # get_time and get_schedule to the same fictional moment (e.g. a row from the
    # evaluation dataset) instead of mixing today's real clock with mock events
    # written for a different time of day. The call still goes through the tool
    # and the MCP protocol like any other call; only the data source changes.
    # Unset by default: does not affect the app.
    mock_env = os.environ.get("MOCK_TIME_INFO")
    if mock_env:
        try:
            mock_payload = json.loads(mock_env)
            result = TimeInfo.model_validate(mock_payload)
            logger.info("get_time: mocked -> %s | %s (MOCK_TIME_INFO set).",
                        result.current_dt, result.time_of_day)
            return result.model_dump(mode="json")
        except Exception as exc:
            logger.warning("MOCK_TIME_INFO set but invalid (%s) -- falling back to system clock.", exc)

    now = datetime.now()
    time_of_day = _resolve_time_of_day(now.hour)

    result = TimeInfo(current_dt=now, time_of_day=time_of_day)

    logger.debug("get_time() -> %s | %s", now.strftime("%Y-%m-%d %H:%M"), time_of_day)

    return result.model_dump(mode="json")
