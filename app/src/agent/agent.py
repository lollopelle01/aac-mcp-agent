from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import (
    AGENT_CANDIDATES_PER_TERM,
    AGENT_DEFAULT_MODEL,
    AGENT_FETCH_SCHEDULE,
    AGENT_MAX_RESULTS,
    AGENT_MEMORY_TURNS,
    AGENT_SYNSET_EXPAND,
    AGENT_SYNSET_EXPAND_MAX,
    AGENT_USE_TWO_PHASE,
    LANG,
    MODELS,
)
from mcp_server.models import Pictogram
from mcp_server.tools.arasaac import list_keywords, search_pictograms, search_pictograms_by_synset
from agent.resolve import resolve_concept
from agent.prompts import (
    build_planner_prompt,
    build_planner_message,
    parse_planner_response,
    build_decision_prompt,
    parse_decision_response,
    parse_new_planner_response,
)
from agent.context import (
    collect_context,
    filter_schedule_by_time,
    terms_from_schedule,
    build_context_block,
)
from agent.session import SessionMemory, Turn
from agent.backends import LLMBackend, OllamaBackend
from agent.ranking import rank_and_fill

logger    = logging.getLogger(__name__)
agent_log = logging.getLogger("agent.run")


### Eval context #################################################################
@dataclass
class EvalContext:
    """Frozen tool outputs injected during evaluation to replace live MCP calls."""
    mock_time:          Optional[dict]       = None
    mock_schedule:      Optional[list[dict]] = None
    mock_needs_context: Optional[bool]       = None   # bypass decision LLM in eval
    tool_calls:         list[str]            = field(default_factory=list)


################################################################################################################################################
# AGENT                                                                                                                                        #
################################################################################################################################################

class AACAgent:
    """AAC pictogram selection agent with per-session memory.

    Two-phase pipeline (when use_two_phase=True, the default):
      Phase 1 DECISION  — fast LLM call: does this input need time/schedule context?
      Phase 2 CONTEXT   — MCP tools (get_time, get_schedule) only if needed
      Phase 3 PLANNING  — LLM planner with full context already available
      Phase 4 RETRIEVAL — keyword search on ARASAAC
      Phase 5 SYNSET    — optional pool expansion via synset siblings
      Phase 6 RANKING   — deterministic ranking + window fill

    Legacy mode (use_two_phase=False): single LLM call that returns call_tools + concepts,
    context collected after. Kept for backward compatibility with eval scripts.

    Backend: pass a ``LLMBackend`` instance (LlamaCppBackend, HuggingFaceBackend,
    OllamaBackend) or leave ``backend=None`` to auto-build an OllamaBackend from
    the model alias and its parameters in MODELS.
    Call ``agent.unload()`` between models in a multi-model eval loop.
    """

    def __init__(
        self,
        model:            str                  = AGENT_DEFAULT_MODEL,
        lang:             str                  = LANG,
        max_results:      int                  = AGENT_MAX_RESULTS,
        fetch_schedule:   bool                 = AGENT_FETCH_SCHEDULE,
        synset_expand:    bool                 = AGENT_SYNSET_EXPAND,
        ranking_strategy: str                  = "sequential_blocks",
        use_two_phase:    bool                 = AGENT_USE_TWO_PHASE,
        backend:          Optional[LLMBackend] = None,
    ) -> None:
        self.model            = model
        self.lang             = lang
        self.max_results      = max_results
        self.fetch_schedule   = fetch_schedule
        self.synset_expand    = synset_expand
        self.ranking_strategy = ranking_strategy
        self.use_two_phase    = use_two_phase
        self.memory           = SessionMemory()

        # If no backend is provided, build an OllamaBackend from the model alias.
        # Parameters (num_predict, num_ctx) come from MODELS so the planner stays
        # within the same token budget regardless of which code path is used.
        if backend is not None:
            self.backend: LLMBackend = backend
        else:
            model_meta = MODELS.get(model, {})
            self.backend = OllamaBackend(
                model       = model,
                num_predict = model_meta.get("num_predict", 150),
                num_ctx     = model_meta.get("num_ctx", 2048),
            )

        self._eval_ctx:      Optional[EvalContext] = None
        self._kw_set:        set[str]              = self._load_kw_set()
        self._concept_order: dict[int, int]        = {}  # pid -> concept index, refreshed per turn

        # Diagnostic fields exposed to the API and eval notebooks
        self.last_candidates:   list[Pictogram] = []
        self.last_call_tools:   bool            = False   # backward compat: True ↔ needs_context
        self.last_resolve_info: list[dict]      = []      # [{concept, queries, method}]
        self.last_plan_method:  str             = "llm"   # "llm" | "fallback_spacy" | "fallback_empty"
        self.last_synset_added: int             = 0       # pictograms added by synset expansion
        self.last_pool_ids:     list[int]       = []      # full ranked candidate pool, before window cut
        self.last_needs_context: bool           = False   # result of decision call
        self.last_context_block: str            = ""      # context_block injected into planner
        self.last_tool_calls:   list[str]       = []      # MCP tools actually called this turn

    #######################################################################################################
    # Public API                                                                                          #
    #######################################################################################################

    def run(
        self,
        raw_input: str,
        *,
        eval_ctx: Optional[EvalContext] = None,
    ) -> list[Pictogram]:
        """Execute one turn: decide -> [context tools] -> plan -> search -> [synset expand] -> rank."""

        _t_run_start = time.perf_counter()
        self._eval_ctx = eval_ctx
        turn_id = self.memory.turn_count + 1
        logger.info("── Turn %d ── %r", turn_id, raw_input)
        agent_log.info("User: %r", raw_input)

        history = self.memory.prompt_summary(AGENT_MEMORY_TURNS)

        ####################################################################################################
        # Phase 1: DECISION                                                                                #
        ####################################################################################################

        agent_log.info("[TIMING] run→decide: %.2fs", time.perf_counter() - _t_run_start)

        if eval_ctx is not None and eval_ctx.mock_needs_context is not None:
            # Eval bypass: honor the mock even with empty raw_input (turn_pos > 0
            # in multi-turn eval, where context is assumed to already be in history).
            needs_context = eval_ctx.mock_needs_context
            agent_log.info("[DECISION] mocked  needs_context=%s", needs_context)
        elif self.use_two_phase and raw_input.strip():
            needs_context = self._decide(raw_input, history)
        else:
            # Empty input (turn_pos > 0 in multi-turn eval) or legacy mode with
            # no mock set: don't force context collection — the context is already
            # in session history from turn 0.  Forcing True here caused
            # called_get_time=True / called_get_schedule=False bleed-through in
            # the eval CSV for all subsequent turns.
            needs_context = False

        ####################################################################################################
        # Phase 2: CONTEXT (MCP tools — only if needed)                                                   #
        ####################################################################################################

        context_block:   str            = ""
        time_of_day:     Optional[str]  = None
        schedule_events: list           = []
        tool_calls_done: list[str]      = []

        if needs_context:
            time_of_day, schedule_events = collect_context(self._eval_ctx, self.fetch_schedule)
            tool_calls_done.append("get_time")
            if schedule_events:
                tool_calls_done.append("get_schedule")
                relevant      = filter_schedule_by_time(schedule_events, time_of_day)
                context_block = build_context_block(time_of_day, relevant)
            agent_log.info(
                "[CTX]    tool_calls=%s  events=%d  context_block=%s",
                tool_calls_done,
                len(schedule_events),
                context_block,
            )
        else:
            agent_log.info("[CTX]    skipped — decision: needs_context=False")

        # Update diagnostic fields
        self.last_tool_calls    = tool_calls_done
        self.last_needs_context = needs_context
        self.last_context_block = context_block
        self.last_call_tools    = needs_context   # backward compat for server.py / eval

        ####################################################################################################
        # Phase 3: PLANNING (LLM with context already available)                                          #
        ####################################################################################################

        agent_log.info("[TIMING] run→plan: %.2fs", time.perf_counter() - _t_run_start)

        if self.use_two_phase:
            concepts = self._plan(raw_input, history, context_block=context_block)
        else:
            # Legacy mode: old planner returns (call_tools, concepts)
            _call_tools_legacy, concepts = self._plan_legacy(raw_input, history)
            self.last_call_tools = _call_tools_legacy
            # In legacy mode context was already collected above (needs_context=True always);
            # inject schedule terms as the old flow did
            if _call_tools_legacy and schedule_events:
                relevant    = filter_schedule_by_time(schedule_events, time_of_day)
                sched_terms = terms_from_schedule(relevant)
                for t in sched_terms:
                    if t not in concepts:
                        concepts.append(t)
                if sched_terms:
                    agent_log.info(
                        "[CTX]    schedule terms injected legacy (1/%d events used): %s",
                        len(schedule_events), sched_terms,
                    )

        if not concepts:
            # LLM returned no concepts: fall back to spaCy term extraction
            concepts = self._extract_terms(raw_input)
            self.last_plan_method = "fallback_empty" if not raw_input.strip() else "fallback_spacy"
            agent_log.info(
                "[PLAN]   no concepts from LLM — fallback: %s  [plan_method=%s]",
                concepts, self.last_plan_method,
            )
        else:
            self.last_plan_method = "llm"

        #######################################################################################################################
        # Phase 4: retrieval                                                                                                  #
        #######################################################################################################################

        candidates = self._search_candidates(concepts)

        if self.synset_expand and candidates:
            candidates = self._expand_pool_by_synset(candidates)

        self.last_candidates = list(candidates)

        if not candidates:
            logger.warning("Turn %d: no candidates found.", turn_id)
            agent_log.info("[RESULT] no candidates — turn aborted")
            self._eval_ctx = None
            return []

        #######################################################################################################################
        # Phase 5: deterministic ranking + window fill                                                                        #
        #######################################################################################################################

        # Exclude only the pictograms the user actually selected (Option A, R22)
        selected_ids = self.memory.recently_selected_ids(n_turns=AGENT_MEMORY_TURNS)
        result       = self._rank_and_fill(candidates, selected_ids)

        #######################################################################################################################
        # Record turn in session memory                                                                                       #
        #######################################################################################################################

        self.memory.add_turn(Turn(
            turn_id     = turn_id,
            timestamp   = datetime.now(),
            raw_input   = raw_input,
            presented   = list(result),
            pictograms  = list(result),
            time_of_day = time_of_day,
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
        # Reset diagnostic fields so stale values from the previous session
        # don't bleed into turn_pos > 0 rows when the session is reused.
        self.last_candidates    = []
        self.last_call_tools    = False
        self.last_resolve_info  = []
        self.last_plan_method   = "llm"
        self.last_synset_added  = 0
        self.last_pool_ids      = []
        self.last_needs_context = False
        self.last_context_block = ""
        self.last_tool_calls    = []
        logger.info("Session reset.")

    def unload(self) -> None:
        """Release backend memory (GGUF / GPU). No-op for OllamaBackend. Safe to always call."""
        if hasattr(self.backend, "unload"):
            self.backend.unload()
        logger.info("AACAgent.unload() called — backend=%s", type(self.backend).__name__)

    #########################################################################################################################
    # Init helpers                                                                                                          #
    #########################################################################################################################

    def _load_kw_set(self) -> set[str]:
        """Load the full ARASAAC keyword index into a set for O(1) lookups."""
        try:
            raw = list_keywords(lang=self.lang)
            kws = raw.get("keywords", [])
            logger.info("Loaded %d ARASAAC keywords for lang=%r.", len(kws), self.lang)
            return set(kws)
        except Exception as exc:
            logger.warning("list_keywords(lang=%r) failed: %s", self.lang, exc)
            return set()

    #########################################################################################################################
    # Phase 1: decision                                                                                                     #
    #########################################################################################################################

    def _decide(self, raw_input: str, history: str) -> bool:
        """Phase 1: fast LLM call to determine if time/schedule context is needed.

        Returns True if context is needed, False otherwise.
        The decision prompt no longer requests a "reason" field — the choice is
        objective enough that the extra token cost is not justified.
        """
        # Eval bypass: if mock_needs_context is set, skip the LLM call entirely
        if self._eval_ctx is not None and self._eval_ctx.mock_needs_context is not None:
            needs = self._eval_ctx.mock_needs_context
            agent_log.info("[DECISION] mocked  needs_context=%s", needs)
            return needs

        _t0        = time.perf_counter()
        system_msg = build_decision_prompt()
        user_msg   = build_planner_message(raw_input, history)   # reuses existing builder

        try:
            raw_text = self.backend.chat(system_msg, user_msg)
            elapsed  = time.perf_counter() - _t0
            parsed   = parse_decision_response(raw_text)
            needs    = bool(parsed.get("needs_context", True))
            agent_log.info(
                "[DECISION] needs_context=%s  elapsed=%.2fs  raw=%r",
                needs, elapsed, raw_text,
            )
            return needs
        except Exception as exc:
            logger.warning("Decision LLM failed: %s — defaulting needs_context=True", exc)
            return True

    #########################################################################################################################
    # Phase 3: planner                                                                                                      #
    #########################################################################################################################

    def _plan(
        self,
        raw_input:     str,
        history:       str,
        context_block: str = "",
    ) -> list[str]:
        """Phase 3 (two-phase mode): LLM planner. Returns concepts list only; no call_tools."""
        _t_plan_start = time.perf_counter()
        system_msg = build_planner_prompt(full=False)
        user_msg   = build_planner_message(raw_input, history, context_block=context_block)

        agent_log.debug(
            "[PLAN IN] model=%s\n--- system ---\n%s\n--- user ---\n%s",
            self.backend.model_id, system_msg, user_msg,
        )

        try:
            agent_log.info(
                "[PLAN CALL] backend=%s  model=%s  pre_call=%.2fs",
                type(self.backend).__name__, self.backend.model_id,
                time.perf_counter() - _t_plan_start,
            )
            _t0      = time.perf_counter()
            raw_text = self.backend.chat(system_msg, user_msg)
            _elapsed = time.perf_counter() - _t0
            agent_log.info("[PLAN OUT] elapsed=%.2fs  raw=%r", _elapsed, raw_text)

            parsed   = parse_new_planner_response(raw_text)   # no call_tools expected
            concepts = [str(c).strip() for c in parsed.get("concepts", []) if c]
            agent_log.info("[PLAN]   concepts=%s  elapsed=%.2fs", concepts, _elapsed)
            return concepts

        except Exception as exc:
            logger.warning("Planner LLM failed: %s — falling back to regex.", exc)
            return []

    def _plan_legacy(
        self,
        raw_input: str,
        history:   str,
    ) -> tuple[bool, list[str]]:
        """Legacy planner (use_two_phase=False): returns (call_tools, concepts).

        Kept for backward compatibility with eval scripts that depend on last_call_tools.
        Body is identical to the original _plan(); only the name changed.
        """
        _t_plan_start = time.perf_counter()
        system_msg = build_planner_prompt(full=False)
        user_msg   = build_planner_message(raw_input, history)

        agent_log.debug(
            "[PLAN IN] model=%s\n--- system ---\n%s\n--- user ---\n%s",
            self.backend.model_id, system_msg, user_msg,
        )

        try:
            agent_log.info(
                "[PLAN CALL] backend=%s  model=%s  pre_call=%.2fs",
                type(self.backend).__name__, self.backend.model_id,
                time.perf_counter() - _t_plan_start,
            )
            _t0      = time.perf_counter()
            raw_text = self.backend.chat(system_msg, user_msg)
            _elapsed = time.perf_counter() - _t0
            agent_log.info("[PLAN OUT] elapsed=%.2fs  raw=%r", _elapsed, raw_text)

            parsed     = parse_planner_response(raw_text)   # legacy parser — includes call_tools
            call_tools = bool(parsed.get("call_tools", True))
            concepts   = [str(c).strip() for c in parsed.get("concepts", []) if c]
            agent_log.info("[PLAN]   call_tools=%s  concepts=%s", call_tools, concepts)
            return call_tools, concepts

        except Exception as exc:
            logger.warning("Planner LLM failed: %s — falling back to regex.", exc)
            return True, []

    #########################################################################################################################
    # Phase 4: search                                                                                                       #
    #########################################################################################################################

    def _search_candidates(self, terms: list[str]) -> list[Pictogram]:
        """Resolve each concept to ARASAAC keywords and fetch candidate pictograms."""
        seen:          set[int]        = set()
        seen_queries:  set[str]        = set()
        candidates:    list[Pictogram] = []
        concept_order: dict[int, int]  = {}
        resolve_info:  list[dict]      = []

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
                            concept_order[pic.id] = concept_idx  # track which concept introduced this pictogram
                            candidates.append(pic)
                except Exception as exc:
                    logger.warning("search_pictograms(%r) failed: %s", query, exc)

        self._concept_order    = concept_order
        self.last_resolve_info = resolve_info
        return candidates

    def _expand_pool_by_synset(self, candidates: list[Pictogram]) -> list[Pictogram]:
        """Query ARASAAC for synset siblings of existing candidates (max AGENT_SYNSET_EXPAND_MAX synsets)."""
        seen_ids:      set[int] = {p.id for p in candidates}
        synsets_tried: set[str] = set()
        expanded       = list(candidates)

        for pic in candidates:
            if len(synsets_tried) >= AGENT_SYNSET_EXPAND_MAX:
                break
            for synset_id in pic.synsets[:2]:  # at most 2 synsets per pictogram to limit queries
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

    #########################################################################################################################
    # Phase 5: ranking + window fill                                                                                        #
    #########################################################################################################################

    def _rank_and_fill(
        self,
        candidates:   list[Pictogram],
        selected_ids: set[int],
    ) -> list[Pictogram]:
        """Rank candidates and fill the window. Strategies defined in ranking.py."""
        window, pool_ids = rank_and_fill(
            candidates, selected_ids, self._concept_order,
            self.max_results, self.ranking_strategy,
        )

        self.last_pool_ids = pool_ids

        agent_log.info(
            "[RANK]   strategy=%s  total_candidates=%d  selected_hard_excluded=%d  pool=%d  window=%d",
            self.ranking_strategy,
            len(candidates),
            len(selected_ids & {p.id for p in candidates}),
            len(pool_ids), len(window),
        )
        return window

    #########################################################################################################################
    # Fallback term extraction (spaCy)                                                                                      #
    #########################################################################################################################

    def _extract_terms(self, raw_input: str) -> list[str]:
        """Fallback: extract non-stop NOUNs/VERBs/PROPNs via spaCy when the planner returns nothing."""
        if not raw_input.strip():
            return []
        from agent.session import _nlp
        nlp  = _nlp()
        doc  = nlp(raw_input.lower())
        seen:  set[str]  = set()
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
