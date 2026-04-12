"""
Tests for mcp_server/models.py

Run with: python -m pytest tests/models_tests.py -v -s
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from pydantic import ValidationError
from typing import get_args

from config import DAYS, DAY_TIMES, SEASONS
from mcp_server.models import (
    DayType, TimeOfDay, SeasonType,
    Keyword, Pictogram, PictogramResult,
    TimeInfo, ScheduleEvent, ContextBundle,
)


# =============================================================================
# 0. Dynamic Literal mechanism
# =============================================================================

class TestDynamicLiterals:
    """Verify the Literal.__class_getitem__ trick produces the right types."""

    def test_day_type_values_match_config(self):
        assert set(get_args(DayType)) == set(DAYS)

    def test_time_of_day_values_match_config(self):
        assert set(get_args(TimeOfDay)) == set(DAY_TIMES)

    def test_season_type_values_match_config(self):
        assert set(get_args(SeasonType)) == set(SEASONS)

    def test_literals_not_empty(self):
        assert len(get_args(DayType))    > 0
        assert len(get_args(TimeOfDay))  > 0
        assert len(get_args(SeasonType)) > 0


# =============================================================================
# 1. Keyword
# =============================================================================

class TestKeyword:
    def test_minimal_valid(self):
        k = Keyword(type=2, keyword="library")
        assert k.keyword == "library"
        assert k.plural is None

    def test_full_valid(self):
        k = Keyword(type=3, keyword="read", plural="reads", idKeyword=42, meaning="to read a book")
        assert k.meaning == "to read a book"

    def test_missing_keyword_raises(self):
        with pytest.raises(ValidationError):
            Keyword(type=2)

    def test_missing_type_raises(self):
        with pytest.raises(ValidationError):
            Keyword(keyword="run")


# =============================================================================
# 2. Pictogram
# =============================================================================

class TestPictogram:
    BASE = {"_id": 1234}

    def test_minimal_valid(self):
        p = Pictogram(**self.BASE)
        assert p.id == 1234
        assert p.keywords == []
        assert p.violence is False
        assert p.sex is False

    def test_alias_id(self):
        """Must be constructed with _id (alias), not id."""
        p = Pictogram(**{"_id": 99})
        assert p.id == 99

    def test_with_keywords(self):
        p = Pictogram(**self.BASE, keywords=[{"type": 2, "keyword": "book"}])
        assert p.keywords[0].keyword == "book"

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            Pictogram()


# =============================================================================
# 3. PictogramResult
# =============================================================================

class TestPictogramResult:
    VALID = {"id": 1, "keyword": "school"}

    def test_minimal_valid(self):
        pr = PictogramResult(**self.VALID)
        assert pr.relevance_score == 0.0
        assert pr.tags == []

    def test_relevance_score_bounds(self):
        PictogramResult(**self.VALID, relevance_score=0.0)
        PictogramResult(**self.VALID, relevance_score=1.0)
        PictogramResult(**self.VALID, relevance_score=0.75)

    def test_relevance_score_below_zero_raises(self):
        with pytest.raises(ValidationError):
            PictogramResult(**self.VALID, relevance_score=-0.1)

    def test_relevance_score_above_one_raises(self):
        with pytest.raises(ValidationError):
            PictogramResult(**self.VALID, relevance_score=1.01)

    def test_safety_flags(self):
        pr = PictogramResult(**self.VALID, violence=True, sex=True)
        assert pr.violence is True
        assert pr.sex is True

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            PictogramResult(keyword="school")

    def test_missing_keyword_raises(self):
        with pytest.raises(ValidationError):
            PictogramResult(id=1)


# =============================================================================
# 4. TimeInfo
# =============================================================================

VALID_TIME = {
    "hour": 9,
    "min": 30,
    "time_of_day": DAY_TIMES[0],   # e.g. "morning"
    "day_of_week": DAYS[0],        # e.g. "monday"
    "season": SEASONS[0],          # e.g. "spring"
}

class TestTimeInfo:
    def test_valid(self):
        t = TimeInfo(**VALID_TIME)
        assert t.hour == 9
        assert t.min == 30

    def test_hour_bounds(self):
        TimeInfo(**{**VALID_TIME, "hour": 0})
        TimeInfo(**{**VALID_TIME, "hour": 23})
        with pytest.raises(ValidationError):
            TimeInfo(**{**VALID_TIME, "hour": 24})
        with pytest.raises(ValidationError):
            TimeInfo(**{**VALID_TIME, "hour": -1})

    def test_min_bounds(self):
        TimeInfo(**{**VALID_TIME, "min": 0})
        TimeInfo(**{**VALID_TIME, "min": 59})
        with pytest.raises(ValidationError):
            TimeInfo(**{**VALID_TIME, "min": 60})
        with pytest.raises(ValidationError):
            TimeInfo(**{**VALID_TIME, "min": -1})

    def test_invalid_time_of_day_raises(self):
        with pytest.raises(ValidationError):
            TimeInfo(**{**VALID_TIME, "time_of_day": "midnight"})

    def test_invalid_day_of_week_raises(self):
        with pytest.raises(ValidationError):
            TimeInfo(**{**VALID_TIME, "day_of_week": "funday"})

    def test_invalid_season_raises(self):
        with pytest.raises(ValidationError):
            TimeInfo(**{**VALID_TIME, "season": "monsoon"})

    def test_all_time_of_day_values_accepted(self):
        for tod in DAY_TIMES:
            TimeInfo(**{**VALID_TIME, "time_of_day": tod})

    def test_all_days_accepted(self):
        for day in DAYS:
            TimeInfo(**{**VALID_TIME, "day_of_week": day})

    def test_all_seasons_accepted(self):
        for season in SEASONS:
            TimeInfo(**{**VALID_TIME, "season": season})


# =============================================================================
# 5. ScheduleEvent
# =============================================================================

class TestScheduleEvent:
    def test_minimal_valid(self):
        e = ScheduleEvent(title="School", start_time="08:30")
        assert e.location is None
        assert e.description is None

    def test_full_valid(self):
        e = ScheduleEvent(
            title="Therapy",
            start_time="10:00",
            location="Room 3",
            description="Speech therapy session",
        )
        assert e.location == "Room 3"

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            ScheduleEvent(start_time="08:00")

    def test_missing_start_time_raises(self):
        with pytest.raises(ValidationError):
            ScheduleEvent(title="School")


# =============================================================================
# 6. ContextBundle + to_prompt_text
# =============================================================================

class TestContextBundle:
    def test_minimal_valid(self):
        cb = ContextBundle(raw_input="We are at the library")
        assert cb.time_info is None
        assert cb.schedule == []

    def test_with_time_info(self):
        cb = ContextBundle(raw_input="Morning walk", time_info=TimeInfo(**VALID_TIME))
        assert cb.time_info is not None

    def test_with_schedule(self):
        cb = ContextBundle(
            raw_input="School day",
            schedule=[ScheduleEvent(title="Math", start_time="09:00")],
        )
        assert len(cb.schedule) == 1


class TestToPromptText:
    def test_only_raw_input(self):
        cb = ContextBundle(raw_input="We are at the library")
        text = cb.to_prompt_text()
        assert "library" in text
        assert "Time:" not in text
        assert "Plans" not in text

    def test_with_time_info(self):
        cb = ContextBundle(raw_input="Morning", time_info=TimeInfo(**VALID_TIME))
        text = cb.to_prompt_text()
        assert "Time:" in text
        assert DAY_TIMES[0] in text   # time_of_day
        assert DAYS[0] in text        # day_of_week

    def test_with_schedule(self):
        cb = ContextBundle(
            raw_input="School day",
            schedule=[
                ScheduleEvent(title="Math", start_time="09:00"),
                ScheduleEvent(title="Lunch", start_time="12:30"),
            ],
        )
        text = cb.to_prompt_text()
        assert "Math" in text
        assert "Lunch" in text
        assert "Plans for the day" in text

    def test_minute_zero_padding(self):
        t = TimeInfo(**{**VALID_TIME, "hour": 9, "min": 5})
        cb = ContextBundle(raw_input="test", time_info=t)
        assert "9:05" in cb.to_prompt_text()

    def test_full_bundle(self):
        cb = ContextBundle(
            raw_input="Going to school",
            time_info=TimeInfo(**VALID_TIME),
            schedule=[ScheduleEvent(title="PE", start_time="10:00")],
        )
        text = cb.to_prompt_text()
        print()
        print(text)
        assert "Going to school" in text
        assert "Time:" in text
        assert "PE" in text