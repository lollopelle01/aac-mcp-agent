from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime

from config import DAY_TIMES

# Time-of-day slot type derived from DAY_TIMES in config.py.
# LANG is fixed to "en" — do not change.
TimeOfDay = Literal[*DAY_TIMES]

####### ARASAAC API models ###############################################

class Keyword(BaseModel):
    # type = part of speech: 1=Proper_Names 2=Common_Names 3=Verbs 4=Descriptives 5=Social 6=Misc
    # ref: https://github.com/Arasaac/public-api/blob/main/src/api/models/Pictograms.js
    type: int               = Field()
    keyword: str            = Field()               # The keyword itself
    plural: Optional[str]   = Field(default=None)   # The plural form of the keyword
    meaning: Optional[str]  = Field(default=None)   # The dense meaning of the keyword


class Pictogram(BaseModel):
    # NOTE: the API field "_id" is mapped to "id" in _raw_to_pictogram, not via an alias here.
    
    id: int                     = Field(             )
    image_url: str              = Field(default=""   )
    keywords: List[Keyword]     = Field(default=[]   )  # All associated terms
    categories: List[str]       = Field(default=[]   )  # ARASAAC thematic categories
    synsets: List[str]          = Field(default=[]   )  # WordNet synset IDs
    tags: List[str]             = Field(default=[]   )  # Free-form semantic tags
    sex: bool                   = Field(default=False)
    violence: bool              = Field(default=False)
    schematic: bool             = Field(default=False)  # Simplified/schematic variant
    aac: bool                   = Field(default=False)  # Suitable for AAC use
    aac_color: bool             = Field(default=False)  # Colour AAC variant available
    skin: bool                  = Field(default=False)  # Has skin-tone variants
    hair: bool                  = Field(default=False)  # Has hair-colour variants
    created: Optional[str]      = Field(default=None )  # ISO 8601
    last_updated: Optional[str] = Field(default=None )  # ISO 8601 — used for incremental updates

####### Pipeline models ##################################################

# ScoredPictogram (Tuple[Pictogram, float]) removed in Fix 5 — the float
# score was always 0.0 after the LLM filter stage was dropped in R3.
# All internal arasaac.py helpers now work directly with Pictogram.


class TimeInfo(BaseModel):
    """
    Temporal context collected by the get_time() MCP tool.
    """

    current_dt: datetime   = Field(description="Current local datetime.")          # YYYY-MM-DD HH:mm:ss
    time_of_day: TimeOfDay = Field(description="Human-readable time slot.")        # NOTE: thresholds defined in config.py


class ScheduleEvent(BaseModel):
    """
    A single calendar event collected by the get_schedule() MCP tool.
    """

    title: str                  = Field(               description="Event title"                                 )
    start_time: str             = Field(               description="Start time as HH:MM string"                  )
    location: Optional[str]     = Field(default=None,  description="Physical or symbolic location"               )
    description: Optional[str]  = Field(default=None,  description="Optional free-text description of the event" )
