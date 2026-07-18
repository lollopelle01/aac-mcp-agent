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
    day or the date and temporal context is needed. Pair it with
    get_schedule to infer the actual plans and enrich the context.
    """
    # Demo/eval override: when MOCK_TIME_INFO is set in the environment, return
    # it as-is instead of the real system clock.
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
