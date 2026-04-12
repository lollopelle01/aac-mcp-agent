# The models used in the agent, the ones referring to the
# pictograms and metadata are inspired by the Models in
# https://arasaac.org/developers/api

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from config import DAY_TIMES, SEASONS, DAYS

# Dynamic Literal types — change ARASAAC_LANG in config.py to switch language
DayType     = Literal[*DAYS]      
TimeOfDay   = Literal[*DAY_TIMES] 
SeasonType  = Literal[*SEASONS]   


#################################################################################################################################################################################
##### MODELS RETURNED BY ARASAAC ################################################################################################################################################
#################################################################################################################################################################################

class Keyword(BaseModel):
    """
    Keyword structure as returned by ARASAAC API.
    Each pictogram can have multiple keywords (main term + synonyms).
    """

    type: int = Field()
    # From the source code of the github of the ARASAAC team
    # https://github.com/Arasaac/public-api/blob/main/src/api/models/Pictograms.js
    # "type" stands for part of speech:
    #   1=Proper_Names, 2=Common_Names, 3=Verbs,
    #   4=Descriptives (adjective/adverb), 5=Social_content, 6=Miscellaneous

    idKeyword: Optional[int]    = Field(default=None)   # Internal ARASAAC keyword ID
    keyword: str                = Field(            )   # The keyword itself
    plural: Optional[str]       = Field(default=None)   # The plural version of the keyword
    idLocution: Optional[str]   = Field(default=None)   # If it is a locution for something else
    meaning: Optional[str]      = Field(default=None)   # The dense meaning of the keyword
    lse: Optional[int]          = Field(default=None)   # Spanish Sign Language (LSE).  # TODO: toglierlo?

class Pictogram(BaseModel):
    """
    Raw pictogram structure as returned by ARASAAC API.
    """

    id: int                     = Field(alias="_id"  )  # Unique ARASAAC pictogram ID.
    keywords: List[Keyword]     = Field(default=[]   )  # NOTE: first entry is the primary label, others are synonyms.
    schematic: bool             = Field(default=False)  # If this is a schematic/simplified version of the pictogram
    sex: bool                   = Field(default=False)  # For safe filtering if inappropriate
    violence: bool              = Field(default=False)  # For safe filtering if inappropriate
    created: Optional[str]      = Field(default=None )  # ISO 8601 (YYYY-MM-DDTHH:mm:ss.sssZ)
    lastUpdated: Optional[str]  = Field(default=None )  # ISO 8601 (YYYY-MM-DDTHH:mm:ss.sssZ)
    downloads: int              = Field(default=0    )  # Popularity counter
    categories: List[str]       = Field(default=[]   )  # ARASAAC thematic categories
    synsets: List[str]          = Field(default=[]   )  # Set of synonyms
    tags: List[str]             = Field(default=[]   )  # Free-form semantic tags
    desc: Optional[str]         = Field(default=None )  # Description of the pictogram      TODO: capire bene come viene usato, lo vedo spesso vuoto...

#################################################################################################################################################################################
##### MODELS USED IN THE PIPELINE ###############################################################################################################################################
#################################################################################################################################################################################

class PictogramResult(BaseModel):
    """
    What the pipeline produces and the renderer consumes.
    """

    id: int                  = Field(                             description="ARASAAC pictogram ID"                                      )
    keyword: str             = Field(                             description="Primary label, cleaned and stripped."                      )
    meaning: Optional[str]   = Field(default=None,                description="Textual definition for the keyword"                        )
    tags: List[str]          = Field(default=[],                  description="Semantic tags from the raw ARASAAC response"               )
    categories: List[str]    = Field(default=[],                  description="Thematic categories from the raw ARASAAC response"         )
    violence: bool           = Field(default=False,               description="Safety flag — pipeline should filter these out by default" )
    sex: bool                = Field(default=False,               description="Safety flag — pipeline should filter these out by default" )
    relevance_score: float   = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score assigned by the LLM filter (0.0 → 1.0)"    )


class TimeInfo(BaseModel):
    """
    Temporal context collected by the get_time() MCP tool.
    """

    hour: int              = Field(ge=0, le=23, description="Current hour in 24h format.")
    min: int               = Field(ge=0, le=59, description="Current minute.")
    time_of_day: TimeOfDay = Field(             description=(f"Human-readable usual time slots. Thresholds: 6-12 {DAY_TIMES[0]}, 12-18 {DAY_TIMES[1]}, 18-22 {DAY_TIMES[2]}, 22-6 {DAY_TIMES[3]}."))
    day_of_week: DayType   = Field(             description="Weekday name.")
    season: SeasonType     = Field(             description="Current season derived from month.")


class ScheduleEvent(BaseModel):
    """
    A single calendar event collected by the get_schedule() MCP tool.
    """

    title: str                  = Field(               description="Event title"                                  )
    start_time: str             = Field(               description="Start time as HH:MM string"                   )
    location: Optional[str]     = Field(default=None,  description="Physical or symbolic location"                )
    description: Optional[str]  = Field(default=None,  description="Optional free-text description of the event"  )


class ContextBundle(BaseModel):
    """
    Aggregated context passed to the LLM filter.
    Combines the caregiver's raw input with enrichment from MCP tools.
    """

    raw_input: str                  = Field(               description="Original caregiver description"                                                    )
    time_info: Optional[TimeInfo]   = Field(default=None,  description="Temporal context from get_time(). None if the input was already detailed enough."  )
    schedule: List[ScheduleEvent]   = Field(default=[],    description="Today's calendar events from get_schedule(). Empty if no agenda was fetched."      )

    def to_prompt_text(self) -> str:
        """Serialize the bundle into a readable string for the LLM prompt."""
        parts = [f"Context: {self.raw_input}"]

        if self.time_info:
            t = self.time_info
            parts.append(
                f"Time: {t.hour}:{t.min:02d} ({t.time_of_day}), "
                f"{t.day_of_week}, season: {t.season}"
            )

        if self.schedule:
            events = ", ".join(f"{e.title} ({e.start_time})" for e in self.schedule)
            parts.append(f"Plans for the day: {events}")

        return "\n".join(parts)