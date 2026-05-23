#!/usr/bin/env python3
"""run_eval_hf.py — AACAgent evaluation for GPU cluster (self-contained).

Uses HFAACAgent (HuggingFace Transformers) instead of Ollama, enabling
execution on SLURM clusters. Supports multiple models evaluated sequentially.
Results are saved to a single CSV with a `model` column for comparison.

This is the single entry point for evaluation — eval/run_eval.py has been
merged into this file and removed.

Usage
-----
    # Single model
    python eval/cluster_work/run_eval_hf.py \\
        --models Qwen/Qwen2.5-3B-Instruct \\
        --split both --n_rows 200

    # Multiple models — sequential, one GPU load at a time
    python eval/cluster_work/run_eval_hf.py \\
        --models Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct \\
        --split both --n_rows 500 --output eval/cluster_work/results/run1.csv

    # Full dataset, 8-bit, resume from checkpoint
    python eval/cluster_work/run_eval_hf.py \\
        --models ibm-granite/granite-3.1-2b-instruct \\
        --split both --n_rows 0 --load_in_8bit --resume

    # Use merged eval dataset (default)
    python eval/cluster_work/run_eval_hf.py \\
        --models Qwen/Qwen2.5-3B-Instruct --lang en_eval

    # Use standard local dataset (ablation)
    python eval/cluster_work/run_eval_hf.py \\
        --models Qwen/Qwen2.5-3B-Instruct --lang en

Arguments
---------
--models        : One or more HuggingFace model names (space-separated).
                  Default: Qwen/Qwen2.5-3B-Instruct
--split         : 'clear' | 'vague' | 'both'  (default: both)
--n_rows        : Rows to sample per model (0 = full dataset, default: 200)
--seed          : Random seed for sampling (default: 42)
--output        : Output CSV path (default: eval/cluster_work/results/eval_hf.csv)
--lang          : Dataset language dir under app/datasets/ (default: en_eval).
                  Use 'en_eval' for the merged eval dataset, 'en' for the stock local dataset.
--hf_device     : 'auto' | 'cuda' | 'cpu'  (default: auto)
--load_in_8bit  : Load models in INT8 with bitsandbytes (saves VRAM)
--max_new_tokens: Max tokens per generation call (default: 512)
--verbose       : Per-turn console output + DEBUG log file
--log_every     : Print progress every N rows (default: 10)
--save_every    : Flush CSV every N rows (default: 10)
--resume        : Skip (row_idx, split, model) triples already in output CSV

Output CSV columns
------------------
model, row_idx, split, turn_pos, n_turns_total, caregiver_input, concept,
gold_id, all_gold_ids, predicted_ids, gold_in_candidates, hit,
n_candidates, window_len, called_get_time, called_get_schedule, overlap_level,
resolve_method, resolve_queries, plan_method, synset_added, fresh_count
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent   # eval/cluster_work/
_PROJECT_ROOT = _HERE.parent.parent               # <project_root>/
_SRC          = _PROJECT_ROOT / "app" / "src"     # app/src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

# ── Base logging (silenced; verbose adds handlers) ────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from config import AGENT_DEFAULT_MODEL
from agent.agent import AACAgent, EvalContext
from agent.hf_agent import HFAACAgent
from agent.session import SessionMemory
from mcp_server.models import Pictogram, Keyword
from mcp_server.tools.arasaac import get_pictogram_metadata

# ── Constants ─────────────────────────────────────────────────────────────────

EVAL_PARQUET = _PROJECT_ROOT / "hf_dataset_annotation" / "eval_final.parquet"

DATE_BASE  = "2025-01-15"
SLOT_TIMES = {
    "morning":   "09:00",
    "afternoon": "15:00",
    "evening":   "19:00",
    "night":     "22:00",
}

OVERLAP_LEVELS = ["synset", "category", "keyword", "tag"]

CSV_COLUMNS = [
    "row_idx", "split", "turn_pos", "n_turns_total", "caregiver_input", "concept",
    "gold_id", "all_gold_ids", "predicted_ids", "gold_in_candidates", "hit",
    "n_candidates", "window_len", "called_get_time", "called_get_schedule",
    "overlap_level",
    "resolve_method",   # one of: exact | lemma | hyphen | lemma_alt | token | none
    "resolve_queries",  # list of keyword strings passed to search_pictograms
    "plan_method",      # one of: llm | fallback_spacy | fallback_empty
    "synset_added",     # int: pictograms added by synset expansion this turn
    "fresh_count",      # int: pictograms in window from fresh pool (not stale padding)
]

CSV_COLUMNS_HF = ["model"] + CSV_COLUMNS

# ── Verbose logger (console) ──────────────────────────────────────────────────
_vlog = logging.getLogger("eval.verbose")
_vlog.propagate = False


def _setup_verbose_logging(output_csv: Path) -> None:
    """Add INFO→console and DEBUG→file handlers on the agent.run logger."""
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    _vlog.setLevel(logging.DEBUG)
    _vlog.addHandler(ch)

    log_dir  = output_csv.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{run_ts}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))

    agent_run_log = logging.getLogger("agent.run")
    agent_run_log.setLevel(logging.DEBUG)
    agent_run_log.addHandler(ch)
    agent_run_log.addHandler(fh)

    _vlog.info(f"Verbose log file: {log_path}")


# ── Gold metadata cache ───────────────────────────────────────────────────────

_gold_cache: dict[int, dict] = {}
_eval_lang: str = "en_eval"   # set by main() from --lang arg


def get_gold_meta(pic_id: int) -> dict:
    k = int(pic_id)
    if k not in _gold_cache:
        try:
            _gold_cache[k] = get_pictogram_metadata(pictogram_id=k, lang=_eval_lang)
        except Exception:
            _gold_cache[k] = {}
    return _gold_cache[k]


def gold_as_pictogram(pic_id: int, concept: str) -> Pictogram:
    meta = get_gold_meta(pic_id)
    if meta:
        try:
            return Pictogram.model_validate(meta)
        except Exception:
            pass
    return Pictogram(id=int(pic_id), keywords=[Keyword(type=2, keyword=concept)])


# ── Semantic overlap ──────────────────────────────────────────────────────────

def _pic_to_dict(p) -> dict:
    if isinstance(p, dict):
        return p
    if hasattr(p, "model_dump"):
        return p.model_dump()
    return vars(p)


def _pic_synsets(d: dict)    -> set: return set(d.get("synsets") or [])
def _pic_categories(d: dict) -> set: return set(d.get("categories") or [])
def _pic_tags(d: dict)       -> set: return set(d.get("tags") or [])


def _pic_keywords(d: dict) -> set:
    kws = d.get("keywords") or []
    out: set[str] = set()
    for kw in kws:
        if isinstance(kw, dict):
            out.add(kw.get("keyword", "").lower())
        elif hasattr(kw, "keyword"):
            out.add(kw.keyword.lower())
    return out - {""}


def semantic_overlap_level(pred, gold_meta: dict) -> Optional[str]:
    p = _pic_to_dict(pred)
    ps, gs = _pic_synsets(p), _pic_synsets(gold_meta)
    if ps and gs and ps & gs:
        return "synset"
    if _pic_categories(p) & _pic_categories(gold_meta):
        return "category"
    if _pic_keywords(p) & _pic_keywords(gold_meta):
        return "keyword"
    if _pic_tags(p) & _pic_tags(gold_meta):
        return "tag"
    return None


def window_best_overlap(window: list, gold_meta: dict) -> Optional[str]:
    best_idx = len(OVERLAP_LEVELS)
    for pic in window:
        level = semantic_overlap_level(pic, gold_meta)
        if level is not None:
            idx = OVERLAP_LEVELS.index(level)
            if idx < best_idx:
                best_idx = idx
            if best_idx == 0:
                break
    return OVERLAP_LEVELS[best_idx] if best_idx < len(OVERLAP_LEVELS) else None


# ── Eval helpers ──────────────────────────────────────────────────────────────

def _make_mock_time(time_of_day: str) -> dict:
    hhmm = SLOT_TIMES.get(str(time_of_day).lower(), "09:00")
    return {"current_dt": f"{DATE_BASE}T{hhmm}:00", "time_of_day": time_of_day}


def _make_mock_schedule(schedule_arr) -> list:
    if schedule_arr is None:
        return []
    out = []
    for ev in schedule_arr:
        if hasattr(ev, "items"):
            out.append(dict(ev))
        elif hasattr(ev, "_asdict"):
            out.append(ev._asdict())
        else:
            out.append(
                {k: getattr(ev, k, None)
                 for k in ("title", "start_time", "location", "description")}
            )
    return out


def _teacher_force(agent: AACAgent, gold_id: int, concept: str) -> None:
    """Replace last memory turn with gold pictogram (teacher forcing)."""
    if not agent.memory.turns:
        return
    last = agent.memory.turns[-1]
    for t in last.topics:
        agent.memory.topic_frequency[t] = max(
            0, agent.memory.topic_frequency.get(t, 0) - 1
        )
    gold_pic    = gold_as_pictogram(gold_id, concept)
    gold_topics = SessionMemory.extract_topics([gold_pic])
    last.pictograms = [gold_pic]
    last.topics     = gold_topics
    for t in gold_topics:
        agent.memory.topic_frequency[t] = agent.memory.topic_frequency.get(t, 0) + 1


def _print_turn(
    *,
    turn_pos:       int,
    n_turns:        int,
    concept:        str,
    gold_id:        int,
    ec:             EvalContext,
    candidates:     list[Pictogram],
    predicted_ids:  list[int],
    hit:            bool,
    gold_in_cands:  bool,
    overlap_level:  Optional[str],
    window:         list[Pictogram],
    resolve_method: str = "?",
    resolve_queries: list[str] = (),
) -> None:
    hit_badge     = "✓ HIT " if hit           else "✗ MISS"
    pool_badge    = "pool=✓" if gold_in_cands else "pool=✗"
    overlap_badge = f"overlap={overlap_level}" if overlap_level else "overlap=none"
    tools_called  = ", ".join(ec.tool_calls) if ec.tool_calls else "—"
    resolve_badge = f"{resolve_method}({', '.join(resolve_queries)})" if resolve_queries else f"{resolve_method}(no match)"
    _vlog.info(
        "  turn %d/%d  concept=%r  gold=%d\n"
        "    tools_called : %s\n"
        "    resolve      : %s\n"
        "    window       : %s\n"
        "  → %s  %s  %s",
        turn_pos + 1, n_turns, concept, gold_id,
        tools_called, resolve_badge, predicted_ids,
        hit_badge, pool_badge, overlap_badge,
    )


def _get_resolve_info(agent: AACAgent, concept: str) -> tuple[str, list[str]]:
    """Extract resolve method and queries for a specific concept from last_resolve_info."""
    for entry in agent.last_resolve_info:
        if entry["concept"] == concept:
            return entry["method"], entry["queries"]
    return "none", []


def run_multi_turn(
    agent:   AACAgent,
    row:     "pd.Series",
    split:   str,
    verbose: bool = False,
) -> list[dict]:
    """Run all turns for one dataset row with teacher forcing."""
    concepts  = list(row["concept"])
    gold_ids  = [int(g) for g in row["best_id"]]
    n_turns   = len(concepts)
    caregiver = row["caregiver_clear"] if split == "clear" else row["caregiver_vague"]

    if split == "vague":
        mock_time     = _make_mock_time(row["time_of_day"])
        mock_schedule = _make_mock_schedule(row.get("schedule"))
    else:
        mock_time, mock_schedule = None, []

    agent.reset_session()
    results: list[dict] = []

    for turn_pos, (concept, gold_id) in enumerate(zip(concepts, gold_ids)):
        ec     = EvalContext(mock_time=mock_time, mock_schedule=mock_schedule)
        raw_in = caregiver if turn_pos == 0 else ""

        window     = agent.run(raw_in, eval_ctx=ec)
        candidates = list(agent.last_candidates)

        predicted_ids = [p.id for p in window]
        gold_in_cands = gold_id in {p.id for p in candidates}
        hit           = gold_id in set(predicted_ids)
        gold_meta     = get_gold_meta(gold_id)
        overlap_level = window_best_overlap(window, gold_meta)

        resolve_method, resolve_queries = _get_resolve_info(agent, concept)

        if verbose:
            _print_turn(
                turn_pos=turn_pos, n_turns=n_turns, concept=concept,
                gold_id=gold_id, ec=ec, candidates=candidates,
                predicted_ids=predicted_ids, hit=hit,
                gold_in_cands=gold_in_cands, overlap_level=overlap_level,
                window=window,
                resolve_method=resolve_method,
                resolve_queries=resolve_queries,
            )

        results.append({
            "row_idx":             row.name,
            "split":               split,
            "turn_pos":            turn_pos,
            "n_turns_total":       n_turns,
            "caregiver_input":     raw_in,
            "concept":             concept,
            "gold_id":             gold_id,
            "all_gold_ids":        gold_ids,
            "predicted_ids":       predicted_ids,
            "gold_in_candidates":  gold_in_cands,
            "hit":                 hit,
            "n_candidates":        len(candidates),
            "window_len":          len(window),
            "called_get_time":     "get_time"     in ec.tool_calls,
            "called_get_schedule": "get_schedule" in ec.tool_calls,
            "overlap_level":       overlap_level,
            "resolve_method":      resolve_method,
            "resolve_queries":     resolve_queries,
            "plan_method":         agent.last_plan_method,
            "synset_added":        agent.last_synset_added,
            "fresh_count":         agent.last_fresh_count,
        })

        _teacher_force(agent, gold_id, concept)

    return results


# ── Progress tracking ─────────────────────────────────────────────────────────

class _ProgressTracker:
    def __init__(self, total: int, splits: list[str]) -> None:
        self.total    = total
        self.splits   = splits
        self.n_done   = 0
        self.n_errors = 0
        self.hits: dict[str, list[bool]] = {s: [] for s in splits}
        self._t0      = time.monotonic()

    def record(self, split: str, row_results: list[dict]) -> None:
        self.n_done += 1
        for r in row_results:
            self.hits[split].append(bool(r["hit"]))

    def record_error(self) -> None:
        self.n_done  += 1
        self.n_errors += 1

    def print_progress(self, split: str) -> None:
        elapsed   = time.monotonic() - self._t0
        avg_secs  = elapsed / max(self.n_done, 1)
        eta_secs  = avg_secs * max(self.total - self.n_done, 0)
        eta_str   = str(timedelta(seconds=int(eta_secs)))
        hit_strs  = [
            f"{s}={sum(h)/len(h):.3f}"
            for s, h in self.hits.items() if h
        ]
        print(
            f"  [{self.n_done:>{len(str(self.total))}}/{self.total}]"
            f"  hit@window: {'  '.join(hit_strs) or '—'}"
            f"  ETA {eta_str}"
            f"  errors={self.n_errors}",
            flush=True,
        )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_done_pairs(csv_path: Path) -> set[tuple[int, str]]:
    """Return set of (row_idx, split) already in the output CSV."""
    if not csv_path.exists():
        return set()
    try:
        existing = pd.read_csv(csv_path, usecols=["row_idx", "split"])
        return set(zip(existing["row_idx"], existing["split"]))
    except Exception as e:
        print(f"[WARN] Could not read checkpoint from {csv_path}: {e}")
        return set()


def _load_done_triples(csv_path: Path) -> set[tuple[int, str, str]]:
    """Return set of (row_idx, split, model) already in the output CSV."""
    if not csv_path.exists():
        return set()
    try:
        existing = pd.read_csv(csv_path, usecols=["row_idx", "split", "model"])
        return set(zip(existing["row_idx"], existing["split"], existing["model"]))
    except Exception as e:
        print(f"[WARN] Could not read checkpoint: {e}")
        return set()


# ── Incremental CSV ───────────────────────────────────────────────────────────

class _IncrementalCSVHF:
    """Appends rows to a CSV; creates header on first write."""

    def __init__(self, path: Path) -> None:
        self.path    = path
        self._is_new = not path.exists()
        self._buffer: list[dict] = []
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, rows: list[dict]) -> None:
        self._buffer.extend(rows)

    def flush(self) -> None:
        if not self._buffer:
            return
        mode = "w" if self._is_new else "a"
        with open(self.path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS_HF, extrasaction="ignore")
            if self._is_new:
                writer.writeheader()
                self._is_new = False
            writer.writerows(self._buffer)
        self._buffer.clear()


# ── Metrics ───────────────────────────────────────────────────────────────────

def _print_metrics_hf(res: pd.DataFrame) -> None:
    W      = 70
    models = sorted(res["model"].unique())

    print("\n" + "=" * W)
    print("HIT@WINDOW  (primary metric)")
    print("=" * W)
    for model_name in models:
        m = res[res["model"] == model_name]
        print(f"  MODEL: {model_name}")
        for split, g in m.groupby("split"):
            hit_all   = g["hit"].mean()
            hit_first = g[g["turn_pos"] == 0]["hit"].mean()
            print(f"    {split:8s}  hit_rate={hit_all:.3f}  hit_first={hit_first:.3f}"
                  f"  n_turns={len(g)}  n_rows={g['row_idx'].nunique()}")

    print("\n" + "=" * W)
    print("FAILURE BREAKDOWN  (retrieval fail / LLM fail / success)")
    print("=" * W)
    for model_name in models:
        m = res[res["model"] == model_name]
        print(f"  MODEL: {model_name}")
        for split, g in m.groupby("split"):
            n              = len(g)
            retrieval_fail = (~g["gold_in_candidates"]).sum()
            llm_fail       = (~g[g["gold_in_candidates"]]["hit"]).sum()
            success        = g["hit"].sum()
            print(f"    {split:8s}  n={n}")
            print(f"      retrieval_fail: {retrieval_fail:4d} ({retrieval_fail/n*100:.1f}%)")
            print(f"      llm_fail:       {llm_fail:4d} ({llm_fail/n*100:.1f}%)")
            print(f"      success (hit):  {success:4d} ({success/n*100:.1f}%)")

    print("\n" + "=" * W)
    print("SEMANTIC OVERLAP LEVELS")
    print("=" * W)
    for model_name in models:
        m = res[res["model"] == model_name]
        print(f"  MODEL: {model_name}")
        for split, g in m.groupby("split"):
            n = len(g)
            print(f"    {split:8s}  semantic_hit={g['overlap_level'].notna().mean():.3f}")
            for lvl in OVERLAP_LEVELS:
                pct = (g["overlap_level"] == lvl).mean()
                print(f"      {lvl:12s}: {pct:.3f}")
            print(f"      none        : {g['overlap_level'].isna().mean():.3f}")

    print("\n" + "=" * W)
    print("RESOLVE METHOD DISTRIBUTION")
    print("=" * W)
    from agent.resolve import RESOLVE_METHODS
    for model_name in models:
        m = res[res["model"] == model_name]
        print(f"  MODEL: {model_name}")
        for split, g in m.groupby("split"):
            n = len(g)
            print(f"    {split:8s}  n={n}")
            for method in RESOLVE_METHODS:
                pct = (g["resolve_method"] == method).mean()
                if pct > 0:
                    hit_r = g[g["resolve_method"] == method]["hit"].mean()
                    print(f"      {method:12s}: {pct:.3f}  hit_rate={hit_r:.3f}")

    print("\n" + "=" * W)
    print("PLAN METHOD DISTRIBUTION")
    print("=" * W)
    for model_name in models:
        m = res[res["model"] == model_name]
        print(f"  MODEL: {model_name}")
        for split, g in m.groupby("split"):
            n = len(g)
            print(f"    {split:8s}  n={n}")
            for method in ("llm", "fallback_spacy", "fallback_empty"):
                pct = (g["plan_method"] == method).mean()
                if pct > 0:
                    hit_r = g[g["plan_method"] == method]["hit"].mean()
                    print(f"      {method:18s}: {pct:.3f}  hit_rate={hit_r:.3f}")

    print("\n" + "=" * W)
    print("SYNSET EXPANSION & RANK SOURCE")
    print("=" * W)
    for model_name in models:
        m = res[res["model"] == model_name]
        print(f"  MODEL: {model_name}")
        for split, g in m.groupby("split"):
            n = len(g)
            expanded_turns = (g["synset_added"] > 0).mean()
            avg_added      = g["synset_added"].mean()
            fresh_pct      = (g["fresh_count"] == g["window_len"]).mean()
            padded_pct     = (g["fresh_count"] < g["window_len"]).mean()
            print(f"    {split:8s}  n={n}")
            print(f"      synset expanded turns : {expanded_turns:.3f}  avg_added={avg_added:.1f}")
            print(f"      window fully fresh    : {fresh_pct:.3f}")
            print(f"      window padded (stale) : {padded_pct:.3f}")

    print("\n" + "=" * W)
    print("TOOL-CALL BEHAVIOUR  (clear→0%  vague→~100%)")
    print("=" * W)
    for model_name in models:
        m = res[res["model"] == model_name]
        print(f"  MODEL: {model_name}")
        for split, g in m.groupby("split"):
            pct_t = g["called_get_time"].mean()     * 100
            pct_s = g["called_get_schedule"].mean() * 100
            print(f"    {split:8s}  get_time={pct_t:.1f}%  get_schedule={pct_s:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AACAgent multi-model HuggingFace evaluation for cluster"
    )
    p.add_argument(
        "--models", nargs="+", default=["Qwen/Qwen2.5-3B-Instruct"],
        metavar="HF_MODEL",
        help="HuggingFace model name(s) — evaluated sequentially",
    )
    p.add_argument("--split",      choices=["clear", "vague", "both"], default="both")
    p.add_argument("--n_rows",     type=int,  default=200,
                   help="Rows to sample per model (0 = full dataset)")
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--output",     type=str,
                   default=str(_HERE / "results" / "eval_hf.csv"))
    p.add_argument("--hf_device",  type=str,  default="auto",
                   help="'auto' | 'cuda' | 'cpu'")
    p.add_argument("--load_in_8bit", action="store_true",
                   help="INT8 quantisation via bitsandbytes")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--verbose",    action="store_true",
                   help="Per-turn console output + DEBUG log file")
    p.add_argument("--log_every",  type=int, default=10)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume",     action="store_true",
                   help="Skip (row_idx, split, model) triples already in output CSV")
    p.add_argument("--lang",       type=str, default="en_eval",
                   help="Dataset language dir to use (default: en_eval)")
    return p.parse_args()


def main() -> None:
    global _eval_lang

    args        = parse_args()
    output_path = Path(args.output)
    splits      = ["clear", "vague"] if args.split == "both" else [args.split]

    # propagate --lang to gold metadata cache and agent
    _eval_lang = args.lang
    print(f"Dataset lang: {_eval_lang}")

    if args.verbose:
        _setup_verbose_logging(output_path)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading dataset from {EVAL_PARQUET} …")
    df_full = pd.read_parquet(EVAL_PARQUET)
    print(f"Full dataset: {len(df_full):,} rows × {df_full.shape[1]} cols")

    n_rows = args.n_rows if args.n_rows > 0 else None
    if n_rows is not None:
        df = df_full.sample(n_rows, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled: {len(df)} rows (seed={args.seed})")
    else:
        df = df_full
        print("Using full dataset")

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    done_triples: set[tuple[int, str, str]] = set()
    if args.resume:
        done_triples = _load_done_triples(output_path)
        print(f"Resume: {len(done_triples)} (row, split, model) triples already done.")

    # ── CSV writer ────────────────────────────────────────────────────────────
    csv_writer = _IncrementalCSVHF(output_path)
    if args.resume and output_path.exists():
        csv_writer._is_new = False

    total_t0 = time.monotonic()

    # ── Model loop ────────────────────────────────────────────────────────────
    for model_name in args.models:
        print(f"\n{'━'*70}")
        print(f"  MODEL: {model_name}")
        print(f"{'━'*70}")

        agent = HFAACAgent(
            model             = model_name,
            hf_device         = args.hf_device,
            hf_load_in_8bit   = args.load_in_8bit,
            hf_max_new_tokens = args.max_new_tokens,
            lang              = args.lang,
        )

        total_work = len(df) * len(splits)
        tracker    = _ProgressTracker(total=total_work, splits=splits)

        for split in splits:
            print(f"\n  ── split: {split.upper()} ──")

            for i, (_, row) in enumerate(df.iterrows()):
                row_idx = int(row.name)

                if args.resume and (row_idx, split, model_name) in done_triples:
                    tracker.n_done += 1
                    continue

                if args.verbose:
                    caregiver = (
                        row["caregiver_clear"] if split == "clear"
                        else row["caregiver_vague"]
                    )
                    _vlog.info(
                        "\n%s\n  ROW %d/%d  model=%s  split=%s\n  %r\n%s",
                        "━"*60, i+1, len(df), model_name, split, caregiver, "━"*60,
                    )

                try:
                    row_results = run_multi_turn(
                        agent, row, split, verbose=args.verbose
                    )
                    for r in row_results:
                        r["model"] = model_name
                    csv_writer.add(row_results)
                    tracker.record(split, row_results)
                except Exception as e:
                    tracker.record_error()
                    print(f"  [ERROR] row={row_idx} model={model_name}: {e}", flush=True)

                if tracker.n_done % args.save_every == 0:
                    csv_writer.flush()

                if tracker.n_done % args.log_every == 0 or tracker.n_done == total_work:
                    tracker.print_progress(split)

        csv_writer.flush()

        agent.unload()
        print(f"\n  Model {model_name!r} done — GPU memory freed.")

    csv_writer.flush()

    elapsed = time.monotonic() - total_t0
    print(f"\n{'═'*70}")
    print(f"  All models done in {timedelta(seconds=int(elapsed))}")
    print(f"  Output: {output_path}")
    print(f"{'═'*70}")

    # ── Metrics summary ───────────────────────────────────────────────────────
    try:
        import ast
        res = pd.read_csv(output_path)
        for col in ("predicted_ids", "all_gold_ids"):
            if col in res.columns:
                res[col] = res[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
        _print_metrics_hf(res)
    except Exception as e:
        print(f"\n[WARN] Could not compute metrics: {e}")


if __name__ == "__main__":
    main()