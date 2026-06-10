"""
backends.py — LLM backend abstraction for AACAgent.

Three concrete backends:
  OllamaBackend        — local inference via Ollama HTTP API (default for production)
  LlamaCppBackend      — local inference via llama-cpp-python (faster on CPU, no HTTP overhead)
  HuggingFaceBackend   — GPU/cluster inference via HuggingFace Transformers

All backends share the same interface: a single `chat(system, user)` method
that returns the raw assistant string. Parsing and timing stay in AACAgent.

Model selection per backend
---------------------------
OllamaBackend:
    model = Ollama alias, e.g. "qwen2.5:3b"
    Listed in settings.py under "models" -> used by AACAgent today.

LlamaCppBackend:
    model = absolute path to a .gguf file, e.g. "/models/qwen2.5-3b-q4_k_m.gguf"
    GGUF files are downloaded from HuggingFace (bartowski or lmstudio-community
    namespaces have pre-quantized versions of all models in settings.py).
    Recommended quantization for CPU: Q4_K_M (best speed/quality tradeoff).
    The model alias -> path mapping lives in settings.py under "gguf_models"
    (to be added when llama.cpp backend is wired into AACAgent).

HuggingFaceBackend:
    model = HF repo id, e.g. "Qwen/Qwen2.5-3B-Instruct"
    Used for cluster eval only; not intended for CPU production use.
    Equivalent to what HFAACAgent._hf_chat does today -- extracted here so the
    same logic can be reused without subclassing AACAgent.

Future wiring (not yet done):
    agent = AACAgent(backend=OllamaBackend("qwen2.5:3b"))
    agent = AACAgent(backend=LlamaCppBackend("/models/qwen2.5-3b-q4_k_m.gguf"))
    agent = AACAgent(backend=HuggingFaceBackend("Qwen/Qwen2.5-3B-Instruct"))
    # HFAACAgent will become: AACAgent(backend=HuggingFaceBackend(...))
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


####### Base ###############################################################

class LLMBackend(ABC):
    """Abstract base — all LLM backends implement this interface.

    Subclasses must implement `chat()` and `model_id`.
    All planner logic (prompt building, response parsing, timing) stays in AACAgent.
    """

    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        """Run one inference call and return the raw assistant string.

        Args:
            system: System prompt string.
            user:   User message string.

        Returns:
            Raw text from the model (not yet parsed).
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable identifier for logging (model alias or file path)."""
        ...


####### Ollama #############################################################

class OllamaBackend(LLMBackend):
    """Ollama HTTP backend.

    Talks to a locally running Ollama daemon via the `ollama` Python package.
    Model is identified by its Ollama alias (e.g. "qwen2.5:3b").

    Args:
        model:       Ollama model alias.
        num_predict: Max tokens to generate. 150 is safe for the planner JSON.
        num_ctx:     Context window. 512 is sufficient for the planner prompt
                     and reduces KV-cache allocation on CPU vs Ollama default.
        temperature: 0.0 = greedy decoding (recommended for structured JSON output).
    """

    def __init__(
        self,
        model:       str   = "qwen2.5:3b",
        num_predict: int   = 150,
        num_ctx:     int   = 512,
        temperature: float = 0.0,
    ) -> None:
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")

        self._model       = model
        self._num_predict = num_predict
        self._num_ctx     = num_ctx
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model

    def chat(self, system: str, user: str) -> str:
        response = self._ollama.chat(
            model    = self._model,
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options = {
                "temperature": self._temperature,
                "num_predict": self._num_predict,
                "num_ctx":     self._num_ctx,
            },
        )
        return response["message"]["content"].strip()


####### llama.cpp ##########################################################

def _extract_first_json_object(text: str) -> str:
    """Extract the first well-formed {...} JSON object from raw model output.

    llama.cpp Instruct models occasionally produce a valid JSON object followed
    by repeated tokens or extra text before EOS. This function extracts only
    the first complete {...} block so the upstream JSON parser always receives
    clean input.

    Strategy: scan forward tracking brace depth; stop at the first balanced `}`.
    Falls back to the full stripped text when no `{` is found.
    """
    start = text.find("{")
    if start == -1:
        return text.strip()
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unbalanced — return from start to end and let the parser handle it
    return text[start:].strip()


class LlamaCppBackend(LLMBackend):
    """llama-cpp-python backend.

    Loads a GGUF model file directly into RAM — no HTTP daemon, no Ollama
    overhead. Faster prefill on CPU because the model is in-process and n_ctx
    can be set precisely.

    Installation:
        pip install llama-cpp-python
        # For CPU-optimised build (OpenBLAS):
        CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python

    GGUF models:
        https://huggingface.co/bartowski          (all models in settings.py available)
        https://huggingface.co/lmstudio-community
        Recommended quantization: Q4_K_M

    Args:
        model_path:  Absolute path to the .gguf file.
        n_ctx:       Context window. Default 2048 — 512 was too small: by turn 3
                     the prompt (system + session history + user message) exceeds
                     512 tokens, causing llama.cpp to truncate the input and
                     corrupt the generation.
        n_threads:   CPU threads. None = llama.cpp auto-detects (uses all cores).
        temperature: 0.0 = greedy decoding.
        max_tokens:  Max tokens to generate per call.
        verbose:     If False, suppresses llama.cpp's C-level stdout logs.

    Notes on stop tokens
    --------------------
    We use stop=["\n\n"] to block extra prose after the JSON object without
    interfering with the closing brace.

    Previous versions used stop=["\n}", "}\n"], which caused consistent parse
    failures: those sequences match exactly at the point where the model is
    about to emit the final "}" — so generation halted with the concepts array
    still open and the top-level object never closed.

    _extract_first_json_object() handles the remaining failure modes:
    1. Greedy-repetition loops: the brace-depth scanner returns whatever
       balanced fragment is available; if malformed the JSON parser rejects it
       and the spaCy fallback takes over.
    2. Tail junk / second JSON object: the scanner stops at the first balanced
       "}" so any trailing content is safely ignored.
    """

    def __init__(
        self,
        model_path:  str,
        n_ctx:       int            = 2048,
        n_threads:   Optional[int]  = None,
        temperature: float          = 0.0,
        max_tokens:  int            = 300,
        verbose:     bool           = False,
    ) -> None:
        try:
            from llama_cpp import Llama
            self._Llama = Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "Run: pip install llama-cpp-python\n"
                "Or with OpenBLAS: "
                "CMAKE_ARGS='-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS' "
                "pip install llama-cpp-python"
            )

        self._model_path  = model_path
        self._n_ctx       = n_ctx
        self._n_threads   = n_threads
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._verbose     = verbose
        self._llm: Optional[object] = None   # lazy-loaded on first call

    @property
    def model_id(self) -> str:
        return self._model_path

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        kwargs: dict = {
            "model_path": self._model_path,
            "n_ctx":      self._n_ctx,
            "verbose":    self._verbose,
        }
        if self._n_threads is not None:
            kwargs["n_threads"] = self._n_threads
        logger.info("Loading GGUF model from %r (n_ctx=%d) ...", self._model_path, self._n_ctx)
        self._llm = self._Llama(**kwargs)
        logger.info("GGUF model loaded.")

    def chat(self, system: str, user: str) -> str:
        self._ensure_loaded()
        response = self._llm.create_chat_completion(  # type: ignore[union-attr]
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature = self._temperature,
            max_tokens  = self._max_tokens,
            # stop=["\n\n"] blocks extra prose after the JSON object without
            # truncating the closing brace.
            # Previous stop=["\n}", "}\n"] caused consistent parse failures:
            # those sequences match at the point where the model is about to
            # emit the final "}" — so generation halted with the array still
            # open and the top-level object never closed.
            stop=["\n\n"],
        )
        raw = response["choices"][0]["message"]["content"].strip()
        return _extract_first_json_object(raw)

    def unload(self) -> None:
        """Release the GGUF model from RAM.

        Sets _llm to None so the llama.cpp Llama object is garbage-collected.
        Call between models in a multi-model eval loop to avoid OOM.
        """
        if self._llm is not None:
            self._llm = None
            logger.info("LlamaCppBackend: model unloaded from RAM.")


####### HuggingFace ########################################################

class HuggingFaceBackend(LLMBackend):
    """HuggingFace Transformers backend.

    Used for cluster evaluation (GPU). Not intended for CPU production use.
    Extracts the inference logic from HFAACAgent._hf_chat so it can be reused
    without subclassing AACAgent. Once AACAgent accepts a backend argument,
    HFAACAgent becomes simply: AACAgent(backend=HuggingFaceBackend(...)).

    Args:
        model:          HF model name or local path (e.g. "Qwen/Qwen2.5-3B-Instruct").
        device:         "auto" | "cuda" | "cpu". "auto" lets Accelerate choose.
        load_in_8bit:   INT8 quantization via bitsandbytes (~50% VRAM saving).
        max_new_tokens: Max tokens to generate.
        dtype:          "float16" | "bfloat16" | "float32".
    """

    def __init__(
        self,
        model:          str  = "Qwen/Qwen2.5-3B-Instruct",
        device:         str  = "auto",
        load_in_8bit:   bool = False,
        max_new_tokens: int  = 150,
        dtype:          str  = "float16",
    ) -> None:
        self._model_name     = model
        self._device         = device
        self._load_in_8bit   = load_in_8bit
        self._max_new_tokens = max_new_tokens
        self._dtype          = dtype
        self._tokenizer      = None
        self._model_obj      = None

    @property
    def model_id(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch required. Run: pip install transformers torch"
            )

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype, torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict = {"trust_remote_code": True}
        if self._load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"]   = "auto"
        else:
            load_kwargs["torch_dtype"] = torch_dtype
            load_kwargs["device_map"]  = self._device

        self._model_obj = AutoModelForCausalLM.from_pretrained(
            self._model_name, **load_kwargs
        )
        self._model_obj.eval()
        logger.info("HuggingFace model %r loaded.", self._model_name)

    def unload(self) -> None:
        """Free GPU memory. Call between models in a multi-model eval loop."""
        try:
            import torch
            if self._model_obj is not None:
                del self._model_obj
                self._model_obj = None
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("HuggingFaceBackend: model unloaded.")
        except Exception as exc:
            logger.warning("unload() failed: %s", exc)

    def chat(self, system: str, user: str) -> str:
        self._ensure_loaded()
        import torch

        tok = self._tokenizer
        mdl = self._model_obj
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Minimal fallback for models without a chat template.
            text = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

        inputs = tok(text, return_tensors="pt").to(mdl.device)
        in_len  = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = mdl.generate(
                **inputs,
                max_new_tokens = self._max_new_tokens,
                do_sample      = False,       # greedy = temperature 0.0
                pad_token_id   = tok.eos_token_id,
            )
        return tok.decode(output_ids[0, in_len:], skip_special_tokens=True).strip()
