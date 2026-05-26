from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

try:
    import ollama as _ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _ollama = None  # type: ignore[assignment]
    _OLLAMA_AVAILABLE = False

from config import (
    AGENT_CANDIDATES_PER_TERM,
    AGENT_DEFAULT_MODEL,
    AGENT_FETCH_SCHEDULE,
    AGENT_MAX_RESULTS,
    AGENT_MEMORY_TURNS,
    AGENT_SYNSET_EXPAND,
    AGENT_SYNSET_EXPAND_MAX,
    LANG,
    MODELS,
)
from mcp_server.models import Pictogram, ScheduleEvent, TimeInfo
from mcp_server.tools.arasaac       import list_keywords, search_pictograms, search_pictograms_by_synset
from agent.resolve import resolve_concept, RESOLVE_METHODS
from mcp_server.tools.schedule_tool import get_schedule
from mcp_server.tools.time_tool     import get_time
from agent.prompts import build_planner_prompt, build_planner_message
from agent.session import SessionMemory, Turn, _nlp
from agent.backends import LLMBackend, LlamaCppBackend

logger      = logging.getLogger(__name__)
agent_log   = logging.getLogger("agent.run")

_SEP = "─" * 60


@dataclass
class EvalContext:
    """Frozen tool outputs injected during evaluation to replace live MCP calls."""
    mock_time:     Optional[dict]       = None
    mock_schedule: Optional[list[dict]] = None
    tool_calls:    list[str]            = field(default_factory=list)


def _log_sep(turn_id: int, label: str) -> None:
    agent_log.info("\n%s\n── Turn %d | %s\n%s", _SEP, turn_id, label, _SEP)


class AACAgent:
    """AAC pictogram selection agent with per-session memory.

    Design: one LLM call per turn (planner only).
    Candidate ranking is deterministic — no second LLM call.
    The window is always filled to exactly `max_results` pictograms;
    already-shown pictograms (the full previous window, not just selected ones)
    are excluded from subsequent windows so the user always sees fresh options.
    """

    def __init__(
        self,
        model:          str                    = AGENT_DEFAULT_MODEL,
        lang:           str                    = LANG,
        max_results:    int                    = AGENT_MAX_RESULTS,
        fetch_schedule: bool                   = AGENT_FETCH_SCHEDULE,
        synset_expand:  bool                   = AGENT_SYNSET_EXPAND,
        backend:        Optional[LLMBackend]   = None,
    ) -> None:
        self.model          = model
        self.lang           = lang
        self.max_results    = max_results
        self.fetch_schedule = fetch_schedule
        self.synset_expand  = synset_expand
        self.backend        = backend   # if set, used instead of Ollama in _plan()
        self.memory         = SessionMemory()
        self._eval_ctx:     Optional[EvalContext] = None
        self._kw_set:       set[str]              = self._load_kw_set()
        self._concept_order: dict[int, int]       = {}   # pid → concept index, set per turn
        self.last_candidates: list[Pictogram]     = []
        self.last_call_tools: bool                = False  # exposed to API
        self.last_resolve_info: list[dict]        = []     # exposed to eval: [{concept, queries, method}]
        self.last_plan_method: str                = "llm"  # exposed to eval: "llm" | "fallback_spacy" | "fallback_empty"
        self.last_synset_added: int               = 0      # exposed to eval: pictograms added by synset expansion
        self.last_fresh_count: int                = 0      # exposed to eval: pictograms from fresh pool (not stale padding)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        raw_input: str,
        *,
        eval_ctx: Optional[EvalContext] = None,
    ) -> list[Pictogram]:
        """Execute one agent turn and return the window of pictograms.

        Pipeline (single LLM call):
          1. Planner LLM → call_tools + concepts
          2. Optional: get_time / get_schedule (if call_tools=True)
          3. Keyword search → candidate pool
          4. Synset expansion (if enabled)
          5. Deterministic ranking → exclude already-shown → fill window

        The window is always exactly `max_results` items (or all candidates
        if fewer exist). Only pictograms the user actually SELECTED are excluded
        from subsequent windows; pictograms that were shown but not chosen can
        reappear, since the new context may make them relevant again.
        """
        _t_run_start = time.perf_counter()
        self._eval_ctx = eval_ctx
        turn_id = self.memory.turn_count + 1
        logger.info("── Turn %d ── %r", turn_id, raw_input)
        _log_sep(turn_id, "START")
        agent_log.info("User: %r", raw_input)

        # ── Phase 1: planning (LLM) ───────────────────────────────────────────
        history    = self.memory.prompt_summary(AGENT_MEMORY_TURNS)
        agent_log.info("[TIMING] run→plan: %.2fs", time.perf_counter() - _t_run_start)
        call_tools, concepts = self._plan(raw_input, history, turn_id)
        self.last_call_tools = call_tools  # expose to API / eval

        if not concepts:
            base     = self._extract_terms(raw_input)
            concepts = self._enrich_terms(base)
            self.last_plan_method = "fallback_empty" if not raw_input.strip() else "fallback_spacy"
            agent_log.info("[PLAN]   no concepts from LLM — fallback: %s  [plan_method=%s]", concepts, self.last_plan_method)
        else:
            self.last_plan_method = "llm"

        # ── Phase 2: context enrichment (only when planner says so) ──────────
        if call_tools:
            time_of_day, schedule_events = self._collect_context(raw_input, turn_id)
            if schedule_events:
                sched_terms = self._terms_from_schedule(schedule_events)
                for t in sched_terms:
                    if t not in concepts:
                        concepts.append(t)
                if sched_terms:
                    agent_log.info("[CTX]    schedule terms injected into concepts: %s", sched_terms)
        else:
            time_of_day = None
            agent_log.info("[CTX]    skipped (planner: input is explicit)")

        # ── Phase 3: retrieval ────────────────────────────────────────────────
        candidates = self._search_candidates(concepts, turn_id)
        if self.synset_expand and candidates:
            candidates = self._expand_pool_by_synset(candidates, turn_id)
        self.last_candidates = list(candidates)

        if not candidates:
            logger.warning("Turn %d: no candidates found.", turn_id)
            agent_log.info("[RESULT] no candidates — turn aborted")
            self._eval_ctx = None
            return []

        # ── Phase 4: deterministic ranking + window fill ──────────────────────
        # Opzione A: solo i pittogrammi SELEZIONATI vengono esclusi dalla finestra.
        # Quelli mostrati ma non scelti possono riapparire — il contesto è cambiato
        # e potrebbero essere rilevanti per il concetto corrente.
        selected_ids = self.memory.recently_selected_ids(n_turns=AGENT_MEMORY_TURNS)
        result       = self._rank_and_fill(candidates, set(), selected_ids, turn_id)

        # ── Record turn ───────────────────────────────────────────────────────
        topics = SessionMemory.extract_topics(result)
        self.memory.add_turn(Turn(
            turn_id      = turn_id,
            timestamp    = datetime.now(),
            raw_input    = raw_input,
            search_terms = concepts,
            presented    = list(result),   # full window — never overwritten
            pictograms   = list(result),   # will be narrowed to [chosen] by /select
            topics       = topics,
            time_of_day  = time_of_day,
        ))
        self._eval_ctx = None
        agent_log.info(
            "[RESULT] window=%d  selected_excluded=%d  ids=%s",
            len(result), len(selected_ids), [p.id for p in result],
        )
        logger.info("Turn %d done — %d pictograms", turn_id, len(result))
        return result

    def reset_session(self) -> None:
        self.memory.reset()
        logger.info("Session reset.")

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _load_kw_set(self) -> set[str]:
        try:
            raw = list_keywords(lang=self.lang)
            kws = raw.get("keywords", [])
            logger.info("Loaded %d ARASAAC keywords for lang=%r.", len(kws), self.lang)
            return set(kws)
        except Exception as exc:
            logger.warning("list_keywords(lang=%r) failed: %s", self.lang, exc)
            return set()

    # ── Phase 1: planner ──────────────────────────────────────────────────────

    def _plan(self, raw_input: str, history: str, turn_id: int) -> tuple[bool, list[str]]:
        _t_plan_start = time.perf_counter()
        system_msg = build_planner_prompt(full=False)
        user_msg   = build_planner_message(raw_input, history)

        agent_log.debug(
            "[PLAN IN] model=%s\n--- system ---\n%s\n--- user ---\n%s",
            self.model, system_msg, user_msg,
        )

        try:
            # ── llama.cpp backend (no Ollama) ─────────────────────────────────
            if self.backend is not None:
                model_label = self.backend.model_id
                agent_log.info(
                    "[PLAN CALL] backend=llama_cpp  model=%s  pre_backend=%.2fs",
                    model_label, time.perf_counter() - _t_plan_start,
                )
                _t0      = time.perf_counter()
                raw_text = self.backend.chat(system_msg, user_msg)
                _elapsed = time.perf_counter() - _t0
                agent_log.info("[PLAN OUT] elapsed=%.2fs  raw=%r", _elapsed, raw_text)

            # ── Ollama backend (default) ───────────────────────────────────────
            else:
                if not _OLLAMA_AVAILABLE:
                    raise RuntimeError("ollama not installed — pass a backend= to AACAgent.")

                model_meta  = MODELS.get(self.model, {})
                num_predict = model_meta.get("num_predict", 150)
                num_ctx     = model_meta.get("num_ctx", 512)
                options: dict = {"temperature": 0.0, "num_predict": num_predict, "num_ctx": num_ctx}

                agent_log.info(
                    "[PLAN CALL] model=%s  num_predict=%d  num_ctx=%d  pre_ollama=%.2fs  options=%s",
                    self.model, num_predict, num_ctx,
                    time.perf_counter() - _t_plan_start, options,
                )
                _t0 = time.perf_counter()
                response = _ollama.chat(
                    model    = self.model,
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg},
                    ],
                    options = options,
                )
                _elapsed = time.perf_counter() - _t0
                raw_text = response["message"]["content"].strip()
                agent_log.info("[PLAN OUT] elapsed=%.2fs  raw=%r", _elapsed, raw_text)

            parsed     = self._parse_planner_response(raw_text)
            call_tools = bool(parsed.get("call_tools", True))
            concepts   = [str(c).strip() for c in parsed.get("concepts", []) if c]
            agent_log.info("[PLAN]   call_tools=%s  concepts=%s", call_tools, concepts)
            return call_tools, concepts

        except Exception as exc:
            logger.warning("Planner LLM failed: %s — falling back to regex.", exc)
            return True, []

    @staticmethod
    def _parse_planner_response(text: str) -> dict:
        text = re.sub(r"```(?:json)?", "", text).strip()

        # Pass 1: standard JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Pass 2: enriched regex fallback — try to salvage call_tools + concepts
        result: dict = {}

        ct_match = re.search(r"call_tools[\s:=]+([Tt]rue|[Ff]alse|1|0)", text)
        if ct_match:
            result["call_tools"] = ct_match.group(1).lower() in ("true", "1")

        # Look for a JSON array of strings (e.g. ["eat", "snack"] or ['eat','snack'])
        arr_match = re.search(r"\[\s*[\"'][\w\s]+[\"'](?:\s*,\s*[\"'][\w\s]+[\"'])*\s*\]", text)
        if arr_match:
            try:
                concepts = json.loads(arr_match.group().replace("'", '"'))
                if isinstance(concepts, list):
                    result["concepts"] = [str(c) for c in concepts if c]
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: bare comma-separated words after 'concepts'
        if "concepts" not in result:
            kv_match = re.search(
                r"concepts[\s:=]+([a-zA-Z][\w,\s-]*)", text, re.IGNORECASE
            )
            if kv_match:
                words = [w.strip().strip(',') for w in kv_match.group(1).split(',')]
                concepts = [w for w in words if w and not w.startswith('{')]
                if concepts:
                    result["concepts"] = concepts

        if result:
            logger.warning(
                "[FALLBACK] planner JSON malformed — salvaged from text: %s", result
            )
            return result

        logger.warning("[FALLBACK] Could not parse planner response at all: %r", text)
        return {}

    # ── Phase 2: context ──────────────────────────────────────────────────────

    def _collect_context(self, raw_input: str, turn_id: int) -> tuple[Optional[str], list[ScheduleEvent]]:
        """Call get_time and (optionally) get_schedule.

        Returns
        -------
        time_of_day : str | None
            Current time slot (e.g. "morning") used for recording in the turn.
        schedule    : list[ScheduleEvent]
            Today's calendar events — empty when fetch_schedule is off or fails.
            Callers use this to inject event keywords into the concept list.
        """
        time_info:   Optional[TimeInfo]  = None
        schedule:    list[ScheduleEvent] = []
        time_of_day: Optional[str]       = None
        ctx = self._eval_ctx

        if ctx is not None:
            if ctx.mock_time is not None:
                try:
                    time_info   = TimeInfo.model_validate(ctx.mock_time)
                    time_of_day = time_info.time_of_day
                    ctx.tool_calls.append("get_time")
                except Exception as exc:
                    logger.warning("EvalContext mock_time invalid: %s", exc)
            agent_log.info("[EVAL]   get_time() mocked → time_of_day=%r", time_of_day)
        else:
            try:
                time_raw    = get_time()
                time_info   = TimeInfo.model_validate(time_raw)
                time_of_day = time_info.time_of_day
            except Exception as exc:
                logger.warning("get_time() failed: %s", exc)
            agent_log.info(
                "[TOOL]   get_time() → time_of_day=%r  dt=%s",
                time_of_day,
                time_info.current_dt.isoformat() if time_info else "N/A",
            )

        if ctx is not None:
            raw_sched = ctx.mock_schedule or []
            try:
                schedule = [ScheduleEvent.model_validate(e) for e in raw_sched]
                if schedule:
                    ctx.tool_calls.append("get_schedule")
            except Exception as exc:
                logger.warning("EvalContext mock_schedule invalid: %s", exc)
            agent_log.info("[EVAL]   get_schedule() mocked → %d events", len(schedule))
        elif self.fetch_schedule:
            try:
                schedule = [ScheduleEvent.model_validate(e) for e in get_schedule()]
            except Exception as exc:
                logger.warning("get_schedule() failed: %s", exc)
            agent_log.info("[TOOL]   get_schedule() → %d events", len(schedule))

        return time_of_day, schedule

    # ── Phase 3: search ───────────────────────────────────────────────────────

    def _search_candidates(self, terms: list[str], turn_id: int) -> list[Pictogram]:
        """Resolve each concept and fetch candidates from ARASAAC.

        Candidates are returned in concept order: all pictograms for the first
        concept come before those of the second, etc. This mirrors the planner's
        priority so _rank_and_fill can preserve it via concept_order.
        No boosting of previously shown IDs — exclusion is handled later
        in _rank_and_fill.
        """
        seen:         set[int]        = set()
        seen_queries: set[str]        = set()
        candidates:   list[Pictogram] = []
        concept_order: dict[int, int] = {}   # pid → concept index (0-based)
        resolve_info: list[dict]      = []   # [{concept, queries, method}]

        for concept_idx, term in enumerate(terms):
            queries, method = resolve_concept(term, self._kw_set, return_method=True)
            resolve_info.append({"concept": term, "queries": queries, "method": method})
            if not queries:
                agent_log.info("[RESOLVE] %r → no match in keyword index (skipped)  [method=none]", term)
                continue
            agent_log.info("[RESOLVE] %r → %s  [method=%s]", term, queries, method)

            for query in queries:
                if query in seen_queries:
                    continue
                seen_queries.add(query)
                try:
                    raw     = search_pictograms(keyword=query, lang=self.lang,
                                                max_results=AGENT_CANDIDATES_PER_TERM)
                    results = raw.get("results", [])
                    agent_log.info(
                        "[TOOL]   search_pictograms(keyword=%r) → %d results",
                        query, len(results),
                    )
                    for pic_dict in results:
                        pic = Pictogram.model_validate(pic_dict)
                        if pic.id not in seen:
                            seen.add(pic.id)
                            concept_order[pic.id] = concept_idx
                            candidates.append(pic)
                except Exception as exc:
                    logger.warning("search_pictograms(%r) failed: %s", query, exc)

        self._concept_order = concept_order   # expose to _rank_and_fill
        self.last_resolve_info = resolve_info  # expose to eval
        return candidates

    def _expand_pool_by_synset(self, candidates: list[Pictogram], turn_id: int) -> list[Pictogram]:
        seen_ids:      set[int] = {p.id for p in candidates}
        synsets_tried: set[str] = set()
        expanded       = list(candidates)

        for pic in candidates:
            if len(synsets_tried) >= AGENT_SYNSET_EXPAND_MAX:
                break
            for synset_id in pic.synsets[:2]:
                if synset_id in synsets_tried:
                    continue
                if len(synsets_tried) >= AGENT_SYNSET_EXPAND_MAX:
                    break
                synsets_tried.add(synset_id)
                try:
                    raw = search_pictograms_by_synset(synset_id=synset_id, lang=self.lang)
                    for pic_dict in raw.get("results", []):
                        try:
                            p2 = Pictogram.model_validate(pic_dict)
                        except Exception:
                            continue
                        if p2.id not in seen_ids:
                            seen_ids.add(p2.id)
                            expanded.append(p2)
                except Exception as exc:
                    logger.debug("synset expand(%r) failed: %s", synset_id, exc)

        added = len(expanded) - len(candidates)
        self.last_synset_added = added
        if added:
            agent_log.info(
                "[SYNSET] pool %d → %d (+%d via %d synset(s))",
                len(candidates), len(expanded), added, len(synsets_tried),
            )
        return expanded

    # ── Phase 4: ranking + window fill ────────────────────────────────────────

    def _rank_and_fill(
        self,
        candidates:   list[Pictogram],
        shown_ids:    set[int],
        selected_ids: set[int],
        turn_id:      int,
    ) -> list[Pictogram]:
        """Rank candidates deterministically and fill the window to max_results.

        Strategy:
          1. Hard-exclude pictograms the user SELECTED in recent turns.
             These never reappear because the user already chose them —
             showing them again would be redundant and confusing.
          2. All other candidates (including those previously shown but not chosen)
             are treated as fresh — shown_ids is always empty (Opzione A).
             Rationale: the new context may make a previously-shown pictogram
             relevant again; suppressing it artificially reduces the useful pool.
          3. Rank by (concept_order ASC, quality_score DESC):
             concept_order mirrors the planner's priority (first concept first);
             within the same concept, quality wins: aac_color > aac > no violence/sex.
          4. Take up to max_results from the ranked pool.
        """
        concept_order = getattr(self, "_concept_order", {})
        _MAX_CONCEPT = 9999

        def _sort_key(pic: Pictogram) -> tuple[int, int]:
            cidx  = concept_order.get(pic.id, _MAX_CONCEPT)   # lower = higher priority
            score = 0
            if pic.aac_color: score += 4
            if pic.aac:       score += 2
            if not pic.violence: score += 1
            if not pic.sex:      score += 1
            return (cidx, -score)   # sort by concept first, then best quality first

        # shown_ids is always empty (Opzione A) — only selected_ids are excluded.
        pool = [p for p in candidates if p.id not in selected_ids]
        pool.sort(key=_sort_key)

        window = pool[:self.max_results]
        self.last_fresh_count = len(window)

        agent_log.info(
            "[RANK]   total_candidates=%d  selected_hard_excluded=%d  pool=%d  window=%d",
            len(candidates),
            len(selected_ids & {p.id for p in candidates}),
            len(pool), len(window),
        )
        return window

    # ── Fallback helpers ──────────────────────────────────────────────────────

    def _terms_from_schedule(self, events: list[ScheduleEvent]) -> list[str]:
        """Extract searchable terms from calendar event fields.

        Combines title, description and location for each event, then produces:
        - consecutive bigrams  ("ice cream", "go out", ...)
        - individual words  (>=3 chars)

        Bigrams are inserted first so compound ARASAAC labels ("ice cream")
        are attempted before their constituent tokens ("ice", "cream").
        Words use a Unicode-aware regex so accented characters are preserved.
        Unknown words / bigrams are silently skipped (method=none).
        """
        import re as _re
        seen:  set[str]  = set()
        terms: list[str] = []

        def _add(t: str) -> None:
            if t not in seen:
                seen.add(t)
                terms.append(t)

        for ev in events[:5]:   # cap at 5 events to avoid noise
            text = " ".join(filter(None, [ev.title, ev.description or "", ev.location or ""]))
            words = _re.findall(r"[a-zA-Z\u00C0-\u024F]{3,}", text.lower())

            # Bigrams first — compound labels have priority over single tokens
            for i in range(len(words) - 1):
                _add(f"{words[i]} {words[i + 1]}")

            # Then individual words
            for w in words:
                _add(w)

        return terms

    def _extract_terms(self, raw_input: str) -> list[str]:
        """Regex-free fallback term extractor using spaCy POS tags.

        Keeps only NOUN, VERB, and PROPN tokens (lemmatised, lowercased),
        filtered against spaCy's built-in stop-word list. This replaces the
        old hand-written AGENT_STOPWORDS constant.
        """
        nlp  = _nlp()
        doc  = nlp(raw_input.lower())
        seen: set[str]  = set()
        terms: list[str] = []
        for token in doc:
            if token.pos_ not in ("NOUN", "VERB", "PROPN"):
                continue
            if token.is_stop:
                continue
            word = token.lemma_.lower()
            if len(word) > 2 and word not in seen:
                seen.add(word)
                terms.append(word)
        return terms

    def _enrich_terms(self, base_terms: list[str]) -> list[str]:
        enriched = list(base_terms)
        recent   = self.memory.recent_topics(AGENT_MEMORY_TURNS)
        if len(base_terms) <= 1 and recent:
            for t in recent[:3]:
                if t not in enriched:
                    enriched.append(t)
        elif recent and recent[0] not in enriched:
            enriched.append(recent[0])
        return enriched
