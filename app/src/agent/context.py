from __future__ import annotations

import logging
import re
from typing import Optional

from mcp_server.models import ScheduleEvent, TimeInfo
from mcp_server.tools.schedule_tool import get_schedule
from mcp_server.tools.time_tool import get_time

logger = logging.getLogger(__name__)

# Maps time-of-day slot to a representative hour for proximity scoring.
_TOD_TO_HOUR: dict[str, int] = {
    "morning": 9, "afternoon": 14, "evening": 19, "night": 22,
}


########################################################################
# Public API
########################################################################

def collect_context(
    eval_ctx,
    fetch_schedule: bool,
) -> tuple[Optional[str], list[ScheduleEvent], Optional[TimeInfo]]:
    """Fetch (or mock) time and schedule for the current turn.

    When eval_ctx is provided, real MCP calls are replaced by the mock
    values injected by the eval harness.

    Returns (time_of_day, schedule, time_info). time_info is the full
    TimeInfo object (includes the exact current_dt) -- callers that only
    need the coarse slot can ignore it, but build_context_block needs the
    exact time to surface it to the planner (e.g. "14:30", not just
    "afternoon").
    """
    time_info:   Optional[TimeInfo]  = None
    schedule:    list[ScheduleEvent] = []
    time_of_day: Optional[str]       = None

    # get_time — real call or eval mock.
    # NOTE: no per-tool log line here on purpose. The single [CTX] line in
    # agent.py already reports tool_calls / time_of_day / events / context_block
    # for the whole phase — logging here too would just repeat the same
    # numbers a second time right above it.
    if eval_ctx is not None:
        if eval_ctx.mock_time is not None:
            try:
                time_info   = TimeInfo.model_validate(eval_ctx.mock_time)
                time_of_day = time_info.time_of_day
                eval_ctx.tool_calls.append("get_time")
            except Exception as exc:
                logger.warning("EvalContext mock_time invalid: %s", exc)
    else:
        try:
            time_info   = TimeInfo.model_validate(get_time())
            time_of_day = time_info.time_of_day
        except Exception as exc:
            logger.warning("get_time() failed: %s", exc)

    # get_schedule — real call or eval mock
    if eval_ctx is not None:
        raw_sched = eval_ctx.mock_schedule or []
        try:
            schedule = [ScheduleEvent.model_validate(e) for e in raw_sched]
            if schedule:
                eval_ctx.tool_calls.append("get_schedule")
        except Exception as exc:
            logger.warning("EvalContext mock_schedule invalid: %s", exc)
    elif fetch_schedule:
        try:
            schedule = [ScheduleEvent.model_validate(e) for e in get_schedule()]
        except Exception as exc:
            logger.warning("get_schedule() failed: %s", exc)

    return time_of_day, schedule, time_info


def filter_schedule_by_time(
    events: list[ScheduleEvent],
    time_of_day: Optional[str],
) -> list[ScheduleEvent]:
    """Return the single event closest to the current time-of-day slot."""
    if not events:
        return events
    target_h = _TOD_TO_HOUR.get(time_of_day or "", 12)
    return [min(events, key=lambda e: abs(_event_hour(e) - target_h))]


def terms_from_schedule(events: list[ScheduleEvent]) -> list[str]:
    """Extract deduplicated search terms from event text fields.

    Bigrams are emitted before unigrams so multi-word phrases get priority.
    """
    seen:  set[str]  = set()
    terms: list[str] = []

    def _add(t: str) -> None:
        if t not in seen:
            seen.add(t)
            terms.append(t)

    for ev in events[:5]:
        text  = " ".join(filter(None, [ev.title, ev.description or "", ev.location or ""]))
        words = re.findall(r"[a-zA-Z\u00C0-\u024F]{3,}", text.lower())
        # Emit bigrams first so multi-word phrases take priority over single words
        for i in range(len(words) - 1):
            _add(f"{words[i]} {words[i + 1]}")
        for w in words:
            _add(w)

    return terms


def build_context_block(
    time_of_day: Optional[str],
    schedule_events: list[ScheduleEvent],
    exact_time: Optional[str] = None,
) -> str:
    """Format time and schedule data into a single compact line for the planner prompt.

    The output is intentionally flat (no newlines) so that:
    - the log line shows the full context without escaped \\n sequences;
    - the LLM receives a clean, unambiguous inline string.

    exact_time is the precise current time (e.g. "14:30"), distinct from the
    coarse time_of_day slot (e.g. "afternoon"). Without it the planner only
    ever sees the slot label, even though the exact time is what the caregiver's
    schedule is actually anchored to.

    Example output:
        "time=afternoon (14:30) | schedule: Math class at 14:30:00 @ school"
    """
    parts: list[str] = []
    if time_of_day:
        if exact_time:
            parts.append(f"time={time_of_day} ({exact_time})")
        else:
            parts.append(f"time={time_of_day}")
    if schedule_events:
        event_strs: list[str] = []
        for ev in schedule_events[:5]:
            loc  = f" @ {ev.location}"    if ev.location    else ""
            desc = f" ({ev.description})" if ev.description else ""
            event_strs.append(f"{ev.title} at {ev.start_time}{loc}{desc}")
        parts.append("schedule: " + "; ".join(event_strs))
    return " | ".join(parts)


########################################################################
# Internal helpers
########################################################################

def _event_hour(e: ScheduleEvent) -> int:
    """Parse the hour from an event's start_time; returns 12 on failure."""
    try:
        return int(str(e.start_time).split(":")[0])
    except Exception:
        return 12

def _sched_detail(events: list[ScheduleEvent]) -> str:
    """Compact inline suffix listing the first few events — used by the
    single [CTX] log line built in agent.py."""
    if not events:
        return ""
    return "  [" + ", ".join(f"{e.title}@{e.start_time}" for e in events[:5]) + "]"
