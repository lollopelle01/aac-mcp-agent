from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Optional

import spacy

from config import AGENT_MEMORY_TURNS
from mcp_server.models import Pictogram


@lru_cache(maxsize=1)
def _nlp() -> spacy.language.Language:
    """Load the spaCy English model once, shared across the whole process."""
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


@dataclass
class Turn:
    """Single conversation turn stored by SessionMemory."""

    turn_id:     int              # 0-based index within the session
    timestamp:   datetime         # wall-clock time of the turn
    raw_input:   str              # original caregiver description
    presented:   list[Pictogram]  # full window shown to the user this turn
    pictograms:  list[Pictogram]  # pictogram(s) actually selected (/select narrows this to [chosen])
    time_of_day: Optional[str]    # slot at turn time (e.g. "morning")


@dataclass
class SessionMemory:
    """Accumulates turns for a single caregiver session."""

    turns: list[Turn] = field(default_factory=list)

    ############################################################################
    # Write
    ############################################################################

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def reset(self) -> None:
        """Clear all state (new user / new conversation)."""
        self.turns.clear()

    ############################################################################
    # Read
    ############################################################################

    @property
    def last_turn(self) -> Optional[Turn]:
        return self.turns[-1] if self.turns else None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def recently_presented_ids(self, n_turns: int = AGENT_MEMORY_TURNS) -> set[int]:
        """IDs of all pictograms shown in the last n_turns (selected + rejected).

        NOTE (R22 — Option A): intentionally NOT used in _rank_and_fill.
        Only selected IDs are excluded from the next window; keeping the full
        pool available is needed for the max_results=25 window size.
        """
        ids: set[int] = set()
        for turn in self.turns[-n_turns:]:
            for p in turn.presented:
                ids.add(p.id)
        return ids

    def recently_selected_ids(self, n_turns: int = AGENT_MEMORY_TURNS) -> set[int]:
        """IDs of pictograms chosen by the user in the last n_turns.

        After /select, ``pictograms`` is narrowed to the single chosen item.
        Selected pictograms must not reappear — it would confuse the subject.
        """
        ids: set[int] = set()
        for turn in self.turns[-n_turns:]:
            for p in turn.pictograms:  # narrowed to [chosen] after /select
                ids.add(p.id)
        return ids

    def prompt_summary(self, n_turns: int = AGENT_MEMORY_TURNS) -> str:
        """LLM-readable summary of the last n_turns. Empty string when no history.

        Format per turn:
            Turn N (time_of_day): "caregiver input" → label1, label2
        time_of_day is omitted when unknown. Labels come from the first keyword
        of each selected pictogram — after /select, pictograms contains only
        the chosen one, so typically one label per turn.
        """
        if not self.turns:
            return ""
        lines = ["Recent conversation history:"]
        for t in self.turns[-n_turns:]:
            slot   = f" ({t.time_of_day})" if t.time_of_day else ""
            labels = ", ".join(
                p.keywords[0].keyword if p.keywords else str(p.id)
                for p in t.pictograms
            )
            chosen = f" → {labels}" if labels else ""
            lines.append(f"  Turn {t.turn_id}{slot}: \"{t.raw_input}\"{chosen}")
        return "\n".join(lines)
