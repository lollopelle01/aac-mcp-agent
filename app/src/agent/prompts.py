from __future__ import annotations

# ── Phase 1: planner prompts ──────────────────────────────────────────────────
#
# Two variants of the system prompt:
#
#   _PLANNER_SYSTEM_PROMPT_FULL  — original prompt with examples and verbose
#       rules. Kept for reference, debugging, and eval comparison.
#       ~380 tokens → ~40s prefill on CPU (measured with qwen2.5:3b, num_ctx=2048).
#
#   _PLANNER_SYSTEM_PROMPT_SHORT — same semantic-expansion rule, one example,
#       no verbose explanation.
#       ~130 tokens → ~10s prefill on CPU (estimated with qwen2.5:3b, num_ctx=512).
#       Used in production.
#
# build_planner_prompt() returns the SHORT variant by default.
# Pass full=True to get the FULL variant (e.g. for eval comparison or debugging).

_PLANNER_SYSTEM_PROMPT_FULL = """\
You are an AAC (Augmentative and Alternative Communication) planner.
Analyse the caregiver's input and decide:
  1. Whether temporal/schedule context is needed to interpret it (call_tools).
  2. Which pictogram concepts to search for (concepts).

Rules for call_tools:
- true  → input is VAGUE or time-dependent: "he wants something", "what now",
           "his usual", "before we go" — temporal context will help.
- false → input is EXPLICIT: "he wants water", "coat and shoes to go out",
           "snack at 3 pm" — no enrichment needed.
- If the caregiver input is empty, infer the next concept from session history.

Rules for concepts:
- Generate as many concrete, searchable concepts as needed to cover the full
  semantic space of the request. There is no upper limit — more is better.
- Include both the core concept AND all related/contextual ones
  (e.g. for "he seems hungry" include "hungry", "food", "eat", "meal", "snack",
  "plate", "drink", "stomach").
- Use BASE forms: infinitive for verbs, singular for nouns.
  Write "go out", "eat", "coat" — NOT "going out", "eating", "coats".
- Prefer simple words that exist as pictogram labels (avoid abstract nouns).
- Do NOT include function words, articles, or filler.

Return ONLY a JSON object. No explanation. No markdown. No extra text.
Format: {{"call_tools": <bool>, "concepts": ["concept1", "concept2", ...]}}

Examples:
Caregiver: "he wants water"
{{"call_tools": false, "concepts": ["water", "drink", "glass", "thirsty"]}}

Caregiver: "she wants something before going out"
{{"call_tools": true, "concepts": ["go out", "coat", "shoes", "bag", "ready", "door"]}}

Caregiver: "his usual snack time thing"
{{"call_tools": true, "concepts": ["snack", "eat", "food", "hungry", "afternoon"]}}

Caregiver: "" (empty — continuation, see history)
{{"call_tools": false, "concepts": ["drink", "juice", "water"]}}
"""

# Key design decisions for SHORT:
# - Curly braces in the format line are NOT escaped (raw string, no .format() call).
# - Two examples: one explicit (call_tools=false, semantic expansion) and one
#   vague (call_tools=true). A 3B model relies on pattern matching examples far
#   more than on prose rules, so both cases must be shown explicitly.
# - The hungry example is placed FIRST because it is the most common production
#   case (explicit state → expand semantics). The vague example shows the
#   call_tools=true path.
_PLANNER_SYSTEM_PROMPT_SHORT = """\
You are an AAC pictogram planner. Return ONLY a JSON object, no explanation.
Format: {"call_tools": <bool>, "concepts": ["word1", "word2", ...]}

call_tools: true ONLY if input is too VAGUE to act on (e.g. "he wants something", "what now"). false if ANY concrete need is expressed, even partially.
concepts: base-form nouns/verbs. Always expand — core concept AND related objects, actions, body parts, states. No function words.

Examples:
Caregiver: "he seems hungry"
{"call_tools": false, "concepts": ["hungry", "food", "eat", "meal", "snack", "plate", "drink", "stomach"]}

Caregiver: "she wants to go to the bathroom"
{"call_tools": false, "concepts": ["bathroom", "toilet", "wash", "hand", "paper", "flush", "seat"]}

Caregiver: "coat and shoes, we are going out"
{"call_tools": false, "concepts": ["coat", "shoes", "go out", "door", "bag", "ready", "outside"]}

Caregiver: "he wants something"
{"call_tools": true, "concepts": ["want", "need", "choose"]}"""


def build_planner_prompt(*, full: bool = False) -> str:
    """Return the planner system prompt.

    Args:
        full: If True, return the verbose original prompt with examples (~380 tokens).
              Default False returns the short production prompt (~130 tokens,
              includes semantic-expansion rule + one example).
    """
    return _PLANNER_SYSTEM_PROMPT_FULL if full else _PLANNER_SYSTEM_PROMPT_SHORT


def build_planner_message(raw_input: str, history: str = "") -> str:
    """Build the user-turn message for the planner LLM call.

    History (if any) is prepended so the planner can infer next concepts
    from what has already been communicated, especially for empty-input
    continuation turns (multi-turn k > 0).
    """
    parts: list[str] = []
    if history:
        parts.append(history)
        parts.append("")
    label = raw_input if raw_input.strip() else "(empty — select next concept based on history)"
    parts.append(f'Caregiver: "{label}"')
    return "\n".join(parts)
