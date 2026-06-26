from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


################################################################################
# System prompts
################################################################################

# Decision prompt — Phase 1 of the two-phase pipeline.
# Stripped to bare minimum: no "reason" field (adds tokens for no downstream value),
# 2 examples instead of 5 (enough for few-shot, saves ~80 tokens on every call).
_DECISION_SYSTEM_PROMPT = """\
You are deciding whether temporal or schedule context is needed to understand a caregiver's input.
Return ONLY a JSON object, no explanation, no markdown.
Format: {"needs_context": <bool>}

needs_context is true when the input is vague, references a routine, implies shared knowledge,
or leaves the specific activity or objects unstated.
needs_context is false ONLY when the input explicitly names the activity, timing, and key objects
— nothing left implicit.
Empty input → false.

Examples:
"he keeps covering his ears"          → {"needs_context": true}
"she is doing physiotherapy at 10:45" → {"needs_context": false}
"""

# ~650 tokens — higher quality, slower inference.
_PLANNER_SYSTEM_PROMPT_FULL = """\
You are an AAC (Augmentative and Alternative Communication) planner.
Read the caregiver's input and generate pictogram concepts.

Generate 5 to 10 concepts for the request.
Start with the core concept, then add words a person would naturally associate
with this situation — objects, actions, feelings, or settings.
Use base forms: infinitive for verbs, singular for nouns.
Write "eat", "go out", "coat", not "eating", "going out", "coats".
Leave out function words and filler. No synonyms or variants of words already listed.
If a "Context" block is present in the input, use it to generate more specific
and relevant concepts.
If the input is empty, infer the next concept from session history.

Return ONLY a JSON object. No explanation, no markdown, no extra text.
Format: {"concepts": ["concept1", "concept2", ...]}

Examples:

Caregiver: "he keeps covering his ears"
{"concepts": ["ear", "noise", "loud", "pain", "headphone", "quiet", "stop"]}

Caregiver: "she keeps reaching for the snacks"
{"concepts": ["snack", "hungry", "eat", "food", "want", "more"]}

Caregiver: "he seems upset"
{"concepts": ["upset", "sad", "angry", "pain", "scared", "tired", "help"]}

Caregiver: "coat and shoes, we are going out right now"
{"concepts": ["coat", "shoes", "go out", "door", "bag", "ready", "outside"]}

Caregiver: "she is doing physiotherapy at 10:45 this morning"
{"concepts": ["physiotherapy", "exercise", "arm", "leg", "stretch", "therapist", "pain", "movement"]}

Caregiver: "his usual"
{"concepts": ["routine", "morning", "afternoon", "activity", "favourite"]}

Caregiver: "" (empty, see history)
{"concepts": ["drink", "juice", "water", "cup"]}
"""

# ~330 tokens — shorter, faster inference.
_PLANNER_SYSTEM_PROMPT_SHORT = """\
You are an AAC pictogram planner. Return ONLY a JSON object, no explanation.
Format: {"concepts": ["word1", "word2", ...]}

Generate 5 to 10 concepts. Start with the core concept, then add words a person would naturally
associate with this situation — objects, actions, feelings, settings.
Use base forms: infinitive for verbs, singular for nouns.
No function words, no synonyms, no variants of words already listed.
If a "Context" block is present in the input, use it to generate more specific and relevant concepts.
If the input is empty, infer the next concept from session history.

Examples:

Caregiver: "he keeps covering his ears"
{"concepts": ["ear", "noise", "loud", "pain", "headphone", "quiet", "stop"]}

Caregiver: "she keeps reaching for the snacks"
{"concepts": ["snack", "hungry", "eat", "food", "want", "more"]}

Caregiver: "she is doing physiotherapy at 10:45 this morning"
{"concepts": ["physiotherapy", "exercise", "arm", "leg", "stretch", "therapist", "pain", "movement"]}

Caregiver: "coat and shoes, we are going out"
{"concepts": ["coat", "shoes", "go out", "door", "bag", "ready", "outside"]}
"""


################################################################################
# Public builders
################################################################################

def build_planner_prompt(*, full: bool = False) -> str:
    """Return the system prompt. Use full=True for higher quality at the cost of speed."""
    return _PLANNER_SYSTEM_PROMPT_FULL if full else _PLANNER_SYSTEM_PROMPT_SHORT


def build_planner_message(raw_input: str, history: str = "", context_block: str = "") -> str:
    """Build the user-turn message, prepending session history and context when available.

    At turn 0 (history is empty) a one-line hint is appended to remind the planner
    to emit the grammatical subject of the caregiver's sentence as the first concept.
    The hint is omitted on subsequent turns to avoid interfering with the normal flow.
    """
    parts: list[str] = []
    if history:
        parts.append(history)
        parts.append("")
    if context_block:
        parts.append(f"Context:\n{context_block}")
        parts.append("")
    if raw_input.strip():
        parts.append(f'Caregiver: "{raw_input}"')
    else:
        parts.append('Caregiver: "" (empty, see history)')
    if not history:
        parts.append(
            "\nHint: this is the first turn — no history yet. "
            "Start your concept list with the grammatical subject of the caregiver's sentence "
            '(e.g. "I", "my ...", "we", "you") before listing semantic concepts.'
        )
    return "\n".join(parts)


def build_decision_prompt() -> str:
    """Return the system prompt for the decision (Phase 1) LLM call."""
    return _DECISION_SYSTEM_PROMPT


################################################################################
# Response parsers
################################################################################

def parse_planner_response(text: str) -> dict:
    """Parse raw LLM output into {call_tools, concepts}.

    Pass 1: standard JSON extraction (handles markdown fences).
    Pass 2: regex salvage — recovers partial fields from malformed output
    so the pipeline degrades gracefully instead of returning an empty plan.
    """
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Pass 1: look for a complete JSON object anywhere in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Pass 2: field-by-field salvage for malformed output
    result: dict = {}

    ct_match = re.search(r"call_tools[\s:=]+([Tt]rue|[Ff]alse|1|0)", text)
    if ct_match:
        result["call_tools"] = ct_match.group(1).lower() in ("true", "1")

    # Try a JSON array literal first, then fall back to a bare comma-separated list
    arr_match = re.search(r"\[\s*[\"'][\w\s]+[\"'](?:\s*,\s*[\"'][\w\s]+[\"'])*\s*\]", text)
    if arr_match:
        try:
            concepts = json.loads(arr_match.group().replace("'", '"'))
            if isinstance(concepts, list):
                result["concepts"] = [str(c) for c in concepts if c]
        except (json.JSONDecodeError, ValueError):
            pass

    if "concepts" not in result:
        kv_match = re.search(r"concepts[\s:=]+([a-zA-Z][\w,\s-]*)", text, re.IGNORECASE)
        if kv_match:
            words = [w.strip().strip(",") for w in kv_match.group(1).split(",")]
            concepts = [w for w in words if w and not w.startswith("{")]
            if concepts:
                result["concepts"] = concepts

    if result:
        logger.warning("[FALLBACK] planner JSON malformed — salvaged: %s", result)
        return result

    logger.warning("[FALLBACK] could not parse planner response at all: %r", text)
    return {}


def parse_new_planner_response(text: str) -> dict:
    """Parse planner response for two-phase mode. Expects {concepts: [...]} only.

    call_tools is not expected; if it appears, log a warning and ignore it.
    Pass 1: standard JSON extraction.
    Pass 2: regex salvage on concepts only (no ct_match).
    """
    text = re.sub(r"```(?:json)?", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if "call_tools" in parsed:
                logger.warning(
                    "[PLAN] unexpected 'call_tools' in planner response — ignoring (old prompt cached?)"
                )
                parsed.pop("call_tools", None)
            return parsed
        except json.JSONDecodeError:
            pass

    # Pass 2: concepts-only salvage
    result: dict = {}

    arr_match = re.search(r"\[\s*[\"'][\w\s]+[\"'](?:\s*,\s*[\"'][\w\s]+[\"'])*\s*\]", text)
    if arr_match:
        try:
            concepts = json.loads(arr_match.group().replace("'", '"'))
            if isinstance(concepts, list):
                result["concepts"] = [str(c) for c in concepts if c]
        except (json.JSONDecodeError, ValueError):
            pass

    if "concepts" not in result:
        kv_match = re.search(r"concepts[\s:=]+([a-zA-Z][\w,\s-]*)", text, re.IGNORECASE)
        if kv_match:
            words = [w.strip().strip(",") for w in kv_match.group(1).split(",")]
            concepts = [w for w in words if w and not w.startswith("{")]
            if concepts:
                result["concepts"] = concepts

    # Pass 3: JSON troncato da max_tokens — lista senza ']' finale, oppure con
    # apici singoli non gestiti dai pass precedenti.
    # Cerca la chiave "concepts" o 'concepts', estrae tutti i token quotati
    # (doppi o singoli apici) e deduplica per rimuovere loop di ripetizione.
    if "concepts" not in result:
        trunc_match = re.search(r'["\']concepts["\']\s*:\s*\[([^\]]*)', text)
        if trunc_match:
            tokens = re.findall(r'["\']([^"\']+)["\']', trunc_match.group(1))
            seen: set[str] = set()
            deduped: list[str] = []
            for t in tokens:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            if deduped:
                result["concepts"] = deduped

    if result:
        logger.warning("[FALLBACK] new planner JSON malformed — salvaged: %s", result)
        return result

    logger.warning("[FALLBACK] could not parse new planner response at all: %r", text)
    return {}


def parse_decision_response(text: str) -> dict:
    """Parse raw LLM output into {needs_context: bool}.

    Pass 1: standard JSON extraction.
    Pass 2: regex salvage on needs_context only.
    """
    text = re.sub(r"```(?:json)?", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if "needs_context" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Pass 2: regex salvage
    result: dict = {}
    nc_match = re.search(r"needs_context[\s:=]+([Tt]rue|[Ff]alse|1|0)", text)
    if nc_match:
        result["needs_context"] = nc_match.group(1).lower() in ("true", "1")

    if result:
        logger.warning("[FALLBACK] decision JSON malformed — salvaged: %s", result)
        return result

    logger.warning("[FALLBACK] could not parse decision response at all: %r", text)
    return {}
