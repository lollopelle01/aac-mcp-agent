from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from mcp_server.server import mcp
from mcp_server.models import ScheduleEvent
from config import (
    CALENDAR_PROVIDER,
    TIMEZONE,
    # Google
    GOOGLE_CREDENTIALS_PATH,
    GOOGLE_TOKEN_PATH,
    GOOGLE_CALENDAR_ID,
    # Apple
    APPLE_CALDAV_URL,
    APPLE_USERNAME,
    APPLE_APP_PASSWORD,
)

logger = logging.getLogger(__name__)

##################################################################################
### MCP Tool #####################################################################
##################################################################################

@mcp.tool()
def get_schedule(date_str: Optional[str] = None) -> list[dict]:
    """
    Return today's (or a given date's) calendar events.

    Parameters
    ----------
    date_str : str | None
        ISO 8601 date string — either YYYY-MM-DD or a full datetime (e.g. the
        current_dt field returned by get_time). Defaults to today.

    Returns
    -------
    list[dict]  — Serialised ScheduleEvent objects:
                  {title, start_time, location, description}
    """
    if date_str:
        # Accept both YYYY-MM-DD and full ISO 8601 datetimes (e.g. from get_time)
        target = datetime.fromisoformat(date_str).date()
    else:
        target = date.today()

    if CALENDAR_PROVIDER == "google":
        events = _fetch_google(target)
    elif CALENDAR_PROVIDER == "apple":
        events = _fetch_apple(target)
    else:
        raise ValueError(f"Unknown CALENDAR_PROVIDER '{CALENDAR_PROVIDER}'. Use 'google' or 'apple'.")

    return [e.model_dump() for e in events]


##################################################################################
### Utils ########################################################################
##################################################################################

def _day_window(target: date) -> tuple[datetime, datetime]:
    """Return the start and end datetimes (timezone-aware)."""
    tz = ZoneInfo(TIMEZONE)
    start = datetime(target.year, target.month, target.day, tzinfo=tz)
    end = start + timedelta(days=1)
    return start, end


def _to_hhmmss(dt: datetime) -> str:
    """Format a datetime as HH:MM:SS in the configured timezone."""
    return dt.astimezone(ZoneInfo(TIMEZONE)).strftime("%H:%M:%S")


##################################################################################
### Google Calendar ##############################################################
##################################################################################

# Auth: OAuth2 via credentials.json (first run) ==> token stored in token.pickle.
def _fetch_google(target: date) -> list[ScheduleEvent]:
    import pickle
    import os
    from google.oauth2.credentials import Credentials
    from google.auth.exceptions import RefreshError
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

    creds = None
    # Load previously saved token to avoid re-authenticating every run.
    if os.path.exists(GOOGLE_TOKEN_PATH):
        with open(GOOGLE_TOKEN_PATH, "rb") as fh:
            creds = pickle.load(fh)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                # Token expired but refresh_token is still valid ==> renew silently, no browser.
                creds.refresh(Request())
            except RefreshError:
                # Refresh token revoked or expired ==> delete the stale token and re-authenticate.
                logger.warning("Google token revoked/expired (invalid_grant). Deleting '%s' and re-authenticating.", GOOGLE_TOKEN_PATH)
                os.remove(GOOGLE_TOKEN_PATH)
                creds = None
        if not creds or not creds.valid:
            # No token yet (or just deleted above) ==> open browser for interactive OAuth login.
            flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)   # NOTE: port=0 lets the OS pick any free port on localhost for the redirect.
        # Persist the (new or refreshed) token for next run.
        with open(GOOGLE_TOKEN_PATH, "wb") as fh:
            pickle.dump(creds, fh)

    # Build the API client wrapper for Google Calendar v3.
    service = build("calendar", "v3", credentials=creds)

    start, end = _day_window(target)
    result = (
        service.events()
        .list(
            calendarId=GOOGLE_CALENDAR_ID,    
            timeMin=start.isoformat(),        
            timeMax=end.isoformat(),          
            singleEvents=True,                # recurrent events represented as single ones
            orderBy="startTime",              
            maxResults=50,                    # safety cap
        )
        .execute()
    )

    events: list[ScheduleEvent] = []
    for item in result.get("items", []):
        # Google distinguishes timed events ("dateTime") from all-day 
        # events ("date"), the T tells the difference between them.
        raw = item["start"].get("dateTime") or item["start"].get("date")
        hhmm = _to_hhmmss(datetime.fromisoformat(raw)) if "T" in raw else "00:00:00"

        events.append(ScheduleEvent(
            title=item.get("summary", "(no title)"),
            start_time=hhmm,
            location=item.get("location"),
            description=item.get("description"),
        ))

    logger.info("Google Calendar: %d events for %s", len(events), target)
    return events


##################################################################################
### ICloud Calendar ##############################################################
##################################################################################

# Auth: Apple app-specific password (generated at appleid.apple.com).
def _fetch_apple(target: date) -> list[ScheduleEvent]:
    import caldav

    # Connect to iCloud's CalDAV endpoint and resolve the user's principal (account root).
    # The principal is the entry point from which all calendars are listed.
    client = caldav.DAVClient(
        url=APPLE_CALDAV_URL,
        username=APPLE_USERNAME,
        password=APPLE_APP_PASSWORD,
    )
    principal = client.principal()

    # Build the [00:00, 24:00) window for the target day (timezone-aware).
    start, end = _day_window(target)
    tz = ZoneInfo(TIMEZONE) 

    events: list[ScheduleEvent] = []
    for calendar in principal.calendars():
        try:
            # expand=True as in Google
            raw_events = calendar.date_search(start=start, end=end, expand=True)
        except Exception as exc:
            # Some calendars (e.g. subscribed .ics feeds) may not support date_search
            logger.debug("Calendar '%s' skipped: %s", calendar.name, exc)
            continue

        for vevent_obj in raw_events:
            try:
                cal = vevent_obj.icalendar_instance
                for component in cal.walk("VEVENT"):
                    dtstart = component.get("DTSTART")
                    if dtstart is None:
                        continue
                    dtstart = dtstart.dt   # to python types

                    if isinstance(dtstart, datetime):
                        # Timed event: convert to configured timezone and format as HH:MM:SS.
                        if dtstart.tzinfo is None:
                            # Naive datetime (no tz in the .ics) ==> assume configured timezone.
                            dtstart = dtstart.replace(tzinfo=tz)
                        hhmm = _to_hhmmss(dtstart)
                    else:
                        # All-day event: DTSTART is a plain date object with no time component.
                        hhmm = "00:00:00"

                    events.append(ScheduleEvent(
                        title=str(component.get("SUMMARY", "(no title)")),
                        start_time=hhmm,
                        location=str(component.get("LOCATION"))    if component.get("LOCATION")    else None,
                        description=str(component.get("DESCRIPTION")) if component.get("DESCRIPTION") else None,
                    ))
            except Exception as exc:
                logger.debug("Unparseable event: %s", exc)

    # CalDAV does not guarantee order like Google, so sort here.
    events.sort(key=lambda e: e.start_time)
    logger.info("Apple CalDAV: %d events for %s", len(events), target)
    return events
