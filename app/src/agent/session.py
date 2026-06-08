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
    """Load the spaCy model once (lazy, cached)."""
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

    turn_id:      int             # 1-based index within the session
    timestamp:    datetime        # wall-clock time of the turn
    raw_input:    str             # original caregiver description
    search_terms: list[str]       # keywords sent to ARASAAC (base + enriched)
    presented:    list[Pictogram] # FULL window shown to the user this turn
    pictograms:   list[Pictogram] # pictogram(s) actually selected by the user
    topics:       list[str]       # keywords extracted from selected pictograms
    time_of_day:  Optional[str]   # slot at turn time (e.g. "morning")


@dataclass
class SessionMemory:
    """Accumulates turns for a single caregiver session.

    Attributes
    ----------
    turns           : Chronological list of completed turns.
    topic_frequency : Session-wide keyword frequency counter used to rank
                      "hot" topics for search enrichment.
    """

    turns:           list[Turn]     = field(default_factory=list)
    topic_frequency: dict[str, int] = field(default_factory=dict)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_turn(self, turn: Turn) -> None:
        """Append a turn and update topic frequencies."""
        self.turns.append(turn)
        for topic in turn.topics:
            self.topic_frequency[topic] = self.topic_frequency.get(topic, 0) + 1

    def reset(self) -> None:
        """Clear all state (new user / new conversation)."""
        self.turns.clear()
        self.topic_frequency.clear()

    # ── Read ──────────────────────────────────────────────────────────────────

    @property
    def last_turn(self) -> Optional[Turn]:
        return self.turns[-1] if self.turns else None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def recent_topics(self, n_turns: int = AGENT_MEMORY_TURNS) -> list[str]:
        """Deduplicated topics from the last n_turns, ranked by session frequency."""
        seen:   set[str]  = set()
        topics: list[str] = []
        for turn in reversed(self.turns[-n_turns:]):
            for t in turn.topics:
                if t not in seen:
                    seen.add(t)
                    topics.append(t)
        return sorted(topics, key=lambda t: self.topic_frequency.get(t, 0), reverse=True)

    def recent_pictogram_ids(self, n_turns: int = AGENT_MEMORY_TURNS) -> list[int]:
        """Deduplicated IDs of SELECTED pictograms from the last n_turns, most-recent first."""
        seen: set[int]  = set()
        ids:  list[int] = []
        for turn in reversed(self.turns[-n_turns:]):
            for p in turn.pictograms:
                if p.id not in seen:
                    seen.add(p.id)
                    ids.append(p.id)
        return ids

    def recently_presented_ids(self, n_turns: int = AGENT_MEMORY_TURNS) -> set[int]:
        """IDs of ALL pictograms shown to the user in the last n_turns.

        This includes both selected and rejected ones.

        NOTE (R22 — Opzione A): this method is intentionally NOT used in
        _rank_and_fill. Only selected pictograms are excluded from the next
        window (recently_selected_ids), not all presented ones. This was a
        deliberate choice to keep the available pool large enough for the
        max_results=25 window. Re-enabling stricter exclusion here would
        require increasing AGENT_CANDIDATES_PER_TERM or the pool size.
        """
        ids: set[int] = set()
        for turn in self.turns[-n_turns:]:
            for p in turn.presented:
                ids.add(p.id)
        return ids

    def recently_selected_ids(self, n_turns: int = AGENT_MEMORY_TURNS) -> set[int]:
        """IDs of pictograms CHOSEN by the user in the last n_turns.

        A selected pictogram is one the subject actually tapped; after /select
        the turn's ``pictograms`` list is narrowed to that single item.
        Selected pictograms must never reappear — not even as stale padding —
        because showing them again would confuse the subject.
        """
        ids: set[int] = set()
        for turn in self.turns[-n_turns:]:
            for p in turn.pictograms:   # narrowed to [chosen] after /select
                ids.add(p.id)
        return ids

    def prompt_summary(self, n_turns: int = AGENT_MEMORY_TURNS) -> str:
        """LLM-readable summary of the last n_turns. Empty string when no history.

        Each selected pictogram is rendered as:
            → keyword (id=NNNN, cat: category[; aac=true])
        so the LLM understands *what* was selected, not just a numeric ID.
        """
        if not self.turns:
            return ""
        lines = ["Recent conversation history:"]
        for t in self.turns[-n_turns:]:
            lines.append(
                f"  Turn {t.turn_id} ({t.time_of_day or '?'}): \"{t.raw_input}\""
            )
            for p in t.pictograms[:4]:
                name = p.keywords[0].keyword if p.keywords else str(p.id)
                cat  = p.categories[0] if p.categories else "—"
                aac  = "; aac=true" if p.aac else ""
                lines.append(f"    → {name} (id={p.id}, cat: {cat}{aac})")
        return "\n".join(lines)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def extract_topics(pictograms: list[Pictogram]) -> list[str]:
        """Extract one primary keyword per pictogram for memory enrichment.

        Keyword type reference (ARASAAC):
          1 = Proper names  2 = Common names (nouns)  3 = Verbs
          4 = Descriptives  5 = Social                6 = Misc

        Strategy: prefer nouns (2) and verbs (3) — they are stable, searchable
        concepts. Descriptives (4) like adjective inflections are poor search
        terms. Fall back to the first keyword only if no noun/verb exists.

        The chosen keyword is then lemmatised via spaCy and filtered against
        spaCy's built-in English stop-word list, so function words are removed
        without any manually maintained list.
        """
        nlp   = _nlp()
        seen:   set[str]  = set()
        topics: list[str] = []
        for pic in pictograms:
            if not pic.keywords:
                continue
            preferred = next(
                (kw for kw in pic.keywords if kw.type in (2, 3)),
                pic.keywords[0],
            )
            raw_word = preferred.keyword.lower().strip()
            if not raw_word:
                continue
            # Lemmatise and filter stop-words via spaCy
            doc = nlp(raw_word)
            if not doc:
                continue
            token = doc[0]
            word  = token.lemma_.lower()
            if word and not token.is_stop and word not in seen:
                seen.add(word)
                topics.append(word)
        return topics
