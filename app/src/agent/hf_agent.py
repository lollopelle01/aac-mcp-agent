"""hf_agent.py — HuggingFace-backed AACAgent for cluster evaluation.

Drop-in replacement for AACAgent when Ollama is not available.
Overrides only _plan() — all retrieval, resolve, ranking,
and session-memory logic is inherited unchanged.

Usage
-----
    from agent.hf_agent import HFAACAgent

    agent = HFAACAgent(
        model="Qwen/Qwen2.5-3B-Instruct",
        hf_device="auto",           # "auto" | "cuda" | "cpu"
        hf_load_in_8bit=False,      # True → needs bitsandbytes, saves VRAM
    )
    results = agent.run("he wants water")

Supported model families (any model with a chat template):
    Qwen/Qwen2.5-{1.5,3,7}B-Instruct
    meta-llama/Llama-3.2-{1,3}B-Instruct
    meta-llama/Meta-Llama-3.1-8B-Instruct
    mistralai/Mistral-7B-Instruct-v0.3
    ibm-granite/granite-3.1-{2,8}b-instruct
    google/gemma-3-{1,4,12}b-it
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    AGENT_FETCH_SCHEDULE,
    AGENT_MAX_RESULTS,
    AGENT_MEMORY_TURNS,
    AGENT_SYNSET_EXPAND,
    LANG,
)
from mcp_server.models import Pictogram
from agent.agent import AACAgent
from agent.prompts import build_planner_prompt, build_planner_message

logger    = logging.getLogger(__name__)
agent_log = logging.getLogger("agent.run")


class HFAACAgent(AACAgent):
    """AACAgent variant that replaces Ollama with HuggingFace Transformers.

    Only _plan() is overridden; everything else (resolve_concept,
    search_pictograms, synset expansion, ranking, session memory) is
    inherited from AACAgent and runs identically.

    The HF model and tokenizer are loaded lazily on the first inference call so
    that __init__ stays fast (useful when the caller creates many agents in a loop
    for comparison but only needs one at a time).

    Parameters
    ----------
    model              : HuggingFace model name or local path.
    lang               : ARASAAC language code (default: LANG from config).
    max_results        : Max pictograms returned per turn.
    fetch_schedule     : Whether to call get_schedule() (pass False during eval).
    synset_expand      : Whether to expand pool via WordNet synset siblings.
    hf_device          : Placement string ('auto', 'cuda', 'cpu'). 'auto' lets
                         Accelerate choose the best available device.
    hf_load_in_8bit    : Load in INT8 via bitsandbytes. Saves ~50% VRAM at a
                         small accuracy cost. Requires bitsandbytes>=0.43.
    hf_max_new_tokens  : Max tokens the model may generate per call (default 512).
    hf_dtype           : Torch dtype string ('float16', 'bfloat16', 'float32').
                         Default: 'float16' on GPU, 'float32' on CPU.
    """

    def __init__(
        self,
        model:             str  = "Qwen/Qwen2.5-3B-Instruct",
        lang:              str  = LANG,
        max_results:       int  = AGENT_MAX_RESULTS,
        fetch_schedule:    bool = AGENT_FETCH_SCHEDULE,
        synset_expand:     bool = AGENT_SYNSET_EXPAND,
        hf_device:         str  = "auto",
        hf_load_in_8bit:   bool = False,
        hf_max_new_tokens: int  = 512,
        hf_dtype:          str  = "float16",
    ) -> None:
        # Store HF-specific config before super().__init__ so they are available
        # if any inherited method (unlikely) triggers inference at construction time.
        self._hf_device         = hf_device
        self._hf_load_in_8bit   = hf_load_in_8bit
        self._hf_max_new_tokens = hf_max_new_tokens
        self._hf_dtype          = hf_dtype
        self._hf_tokenizer: Optional[AutoTokenizer]           = None
        self._hf_model_obj: Optional[AutoModelForCausalLM]   = None

        # AACAgent.__init__:
        #   - sets self.model, self.lang, self.max_results, …
        #   - calls _load_kw_set() (MCP tool, no LLM needed — safe)
        super().__init__(
            model          = model,
            lang           = lang,
            max_results    = max_results,
            fetch_schedule = fetch_schedule,
            synset_expand  = synset_expand,
        )
        logger.info(
            "HFAACAgent ready — model=%r  device=%r  8bit=%s  dtype=%s",
            model, hf_device, hf_load_in_8bit, hf_dtype,
        )

    # ── Model loader ──────────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Lazy-load tokenizer + model on first call. Thread-unsafe by design
        (eval is single-threaded), but safe to call multiple times."""
        if self._hf_tokenizer is not None:
            return

        logger.info(
            "Loading HF model %r  (device=%r, 8bit=%s, dtype=%s) …",
            self.model, self._hf_device, self._hf_load_in_8bit, self._hf_dtype,
        )

        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code = True,
        )
        # Some models lack a pad token; set it to eos so generation doesn't warn.
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        torch_dtype = dtype_map.get(self._hf_dtype, torch.float16)

        load_kwargs: dict = {"trust_remote_code": True}
        if self._hf_load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"]   = "auto"
        else:
            load_kwargs["torch_dtype"] = torch_dtype
            load_kwargs["device_map"]  = self._hf_device

        self._hf_model_obj = AutoModelForCausalLM.from_pretrained(
            self.model, **load_kwargs
        )
        self._hf_model_obj.eval()
        logger.info("HF model loaded.")

    def unload(self) -> None:
        """Explicitly free GPU memory. Call between models in a multi-model loop."""
        if self._hf_model_obj is not None:
            del self._hf_model_obj
            self._hf_model_obj = None
        if self._hf_tokenizer is not None:
            del self._hf_tokenizer
            self._hf_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HFAACAgent: model unloaded and CUDA cache cleared.")

    # ── Core inference ────────────────────────────────────────────────────────

    def _hf_chat(self, system_msg: str, user_msg: str) -> str:
        """Format messages with the model's chat template and run greedy decoding.

        Falls back to a generic <|system|>/<|user|>/<|assistant|> format if the
        tokenizer has no chat_template (very old or custom models).

        Returns the decoded assistant turn (new tokens only, special tokens stripped).
        """
        self._ensure_model_loaded()
        tok = self._hf_tokenizer
        mdl = self._hf_model_obj

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]

        # Apply the model's official chat template if available.
        if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
            text = tok.apply_chat_template(
                messages,
                tokenize              = False,
                add_generation_prompt = True,
            )
        else:
            # Minimal fallback that works for most instruction-tuned models.
            text = (
                f"<|system|>\n{system_msg}\n"
                f"<|user|>\n{user_msg}\n"
                f"<|assistant|>\n"
            )

        inputs = tok(text, return_tensors="pt").to(mdl.device)
        in_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = mdl.generate(
                **inputs,
                max_new_tokens = self._hf_max_new_tokens,
                do_sample      = False,      # greedy — same as temperature=0 in Ollama
                pad_token_id   = tok.eos_token_id,
            )

        new_ids = output_ids[0, in_len:]
        return tok.decode(new_ids, skip_special_tokens=True).strip()

    # ── Override: _plan ──────────────────────────────────────────────────────

    def _plan(
        self,
        raw_input: str,
        history:   str,
        turn_id:   int,
    ) -> tuple[bool, list[str]]:
        """Phase-1 planning via HuggingFace model.

        Identical logic to AACAgent._plan; only the inference call differs.
        """
        _t_plan_start = time.perf_counter()
        system_msg = build_planner_prompt(full=False)   # explicit: always SHORT (fast)
        user_msg   = build_planner_message(raw_input, history)

        agent_log.debug(
            "[PLAN IN][HF] model=%s\n--- system ---\n%s\n--- user ---\n%s",
            self.model, system_msg, user_msg,
        )

        try:
            agent_log.info(
                "[PLAN CALL][HF] model=%s  pre_hf=%.2fs",
                self.model, time.perf_counter() - _t_plan_start,
            )
            _t0      = time.perf_counter()
            raw_text = self._hf_chat(system_msg, user_msg)
            _elapsed = time.perf_counter() - _t0
            agent_log.info("[PLAN OUT][HF] elapsed=%.2fs  raw=%r", _elapsed, raw_text)

            parsed     = self._parse_planner_response(raw_text)
            call_tools = bool(parsed.get("call_tools", True))
            concepts   = [str(c).strip() for c in parsed.get("concepts", []) if c]
            agent_log.info(
                "[PLAN][HF]  call_tools=%s  concepts=%s", call_tools, concepts
            )
            return call_tools, concepts

        except Exception as exc:
            logger.warning("HF Planner failed: %s — falling back to regex.", exc)
            agent_log.info("[PLAN][HF]  ERROR %s — fallback to empty list", exc)
            return True, []


