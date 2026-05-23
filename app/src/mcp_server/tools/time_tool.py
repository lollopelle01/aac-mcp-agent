import logging
from datetime import datetime

from mcp_server.server import mcp
from mcp_server.models import TimeInfo
from config import DAY_TIMES, DAY_TIME_THRESHOLDS

logger = logging.getLogger(__name__)

################################################################################################
## Utils #######################################################################################
################################################################################################

def _resolve_time_of_day(hour: int) -> str:
    """
    Map an integer hour (0–23) to the corresponding 
    time-of-day label efined in config.DAY_TIMES.
    """
    for i in range(len(DAY_TIME_THRESHOLDS) - 1, -1, -1):
        if hour >= DAY_TIME_THRESHOLDS[i]:
            return DAY_TIMES[i]
    return DAY_TIMES[-1]  


################################################################################################
## MCP tool ####################################################################################
################################################################################################

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
        current_dt  : str  — ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS)
        time_of_day : str  — slot label (e.g. "morning", "afternoon", etc.)
    """
    now = datetime.now()
    time_of_day = _resolve_time_of_day(now.hour)

    result = TimeInfo(current_dt=now, time_of_day=time_of_day)

    logger.debug("get_time() → %s | %s", now.strftime("%Y-%m-%d %H:%M"), time_of_day)

    return result.model_dump(mode="json")
