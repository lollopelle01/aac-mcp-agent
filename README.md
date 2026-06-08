# AAC MCP Agent

Agentic system for context-aware [ARASAAC](https://arasaac.org/) pictogram selection for Augmentative and Alternative Communication (AAC).

The caregiver describes the situation in free text; the agent uses a local LLM as a planner to decide which semantic concepts to search for, queries the local ARASAAC dataset, and returns a window of candidate pictograms. The user selects one; the session tracks choices to avoid repetitions.

---

## Prerequisites

| Dependency | Minimum version | Notes |
|---|---|---|
| Python | 3.11 | |
| Node.js | 18 | for the frontend |
| Ollama | any | local only |
| `granite4:3b-h` (or other) | — | see Settings |

```bash
# Python setup (one-time)
pip install -r app/requirements.txt
python -m spacy download en_core_web_sm

# Local LLM model
ollama pull granite4:3b-h
```

---

## Local startup

```bash
# 1. Backend (FastAPI on :8000)
cd app
uvicorn src.api.server:app --reload --port 8000

# 2. Frontend (another terminal — Vite on :5173)
cd app/frontend
npm install        # first time only
npm run dev
```

Open **http://localhost:5173**. The Vite proxy forwards `/api/*` → `:8000`.

### Environment variables

Copy `app/.env.example` → `app/.env` and fill in the optional credentials (Google Calendar, Apple, HuggingFace). Operational settings (model, number of results, etc.) are managed from the interface → ⚙ Settings or directly in `app/user_settings.json`.

### Pictogram dataset

On first startup the text dataset is included in `app/datasets/en/`. PNG images are downloaded on demand from the ARASAAC CDN (or served locally if already present in `app/datasets/pictograms/`). To pre-download all images before going offline, use the **⬇ Datasets** panel → *Download images*.

---

## Interface usage

```
┌────────────────────────────────────────────────────────────┐
│  AAC Pictogram Agent  [granite4:3b-h ▾]  [⚙ Settings] [↺] │
├────────────────────────────────────────────────────────────┤
│  He wants something to eat before leaving        [→ Send]  │
├────────────────────────────────────────────────────────────┤
│  ┌────────┐  ┌────────┐  ┌────────┐  ←  tap to select     │
│  │ apple  │  │ yogurt │  │ snack  │                        │
└──┴────────┴──┴────────┴──┴────────┴────────────────────────┘
│  Session: [apple] [coat] [shoes]                            │
└────────────────────────────────────────────────────────────┘
```

1. The caregiver types the situation and presses **Send** (or Enter).
2. The agent returns a grid of candidate pictograms.
3. The user taps a pictogram → it is recorded in the session.
4. Repeat for each concept in the sentence.
5. **↺** resets the session.

The model dropdown changes the LLM on the fly (PATCH `/settings` → agent rebuilt on the next turn).

---

## Quick local test

The notebook `test/tools_test.ipynb` tests the MCP tools in isolation (ARASAAC search, time, schedule) without starting the full backend. Useful for checking that the local dataset is intact.

```bash
cd test
jupyter notebook tools_test.ipynb
```

---

## Evaluation (eval)

### Eval dataset

The annotated dataset is located in `hf_dataset_annotation/eval_final.parquet` (~54k rows). **Do not modify it** — it is the product of the upstream annotation phase (completed).

Each row contains: `raw_input` (caregiver text), `concept` (gold concept), `gold_id` (expected pictogram ID), `split` (`clear` or `vague`).

### Local eval (notebook)

```bash
cd eval
jupyter notebook agent_eval.ipynb
```

Useful for manual inspection and debugging on a subset of rows.

### Cluster eval (SLURM + GPU)

Single entry point for formal evaluation:

```bash
cd eval/cluster_work

# Submit SLURM job (edit parameters at the top of the script first)
sbatch run_eval_cluster.sh

# or run directly (CPU, debug)
python run_eval_hf.py \
  --models Qwen/Qwen2.5-3B-Instruct \
  --split both \
  --n_rows 200 \
  --output results/debug.csv \
  --verbose
```

**Main `run_eval_hf.py` parameters:**

| Flag | Default | Description |
|---|---|---|
| `--models` | — | one or more HF models (space-separated) |
| `--split` | `both` | `clear`, `vague`, or `both` |
| `--n_rows` | `0` (all) | rows to evaluate (0 = full dataset) |
| `--seed` | `42` | sampling seed |
| `--output` | `results/eval_hf_<job>.csv` | output CSV |
| `--load_in_8bit` | off | INT8 quantisation (saves VRAM) |
| `--log_every` | `25` | progress every N rows |
| `--save_every` | `10` | CSV checkpoint every N rows |
| `--verbose` | off | turn-by-turn details |

**Main metrics** (CSV + summary at end of run):

| Metric | Description |
|---|---|
| `hit` / `gold_in_window` | gold ID in final window — **primary metric** |
| `gold_in_candidates` | gold ID in pool before ranking |
| `overlap_level` | best semantic overlap (`synset` > `category` > `keyword` > `tag`) |
| `resolve_method` | `resolve_concept` strategy (`exact`, `lemma`, `token`, …) |
| `plan_method` | `llm` / `fallback_spacy` / `fallback_empty` |
| `synset_added` | pictograms added by WordNet expansion |
| `fresh_count` | fresh (non-stale padding) pictograms in the window |

The CSV has one row per turn and a `model` column, so multiple models can share the same file.

**Teacher forcing:** after each turn the eval injects `gold_id` as the selected pictogram to simulate realistic multi-step sequences.

### Configuring models in the SLURM job

Edit `HF_MODELS` in `run_eval_cluster.sh` before `sbatch`:

```bash
HF_MODELS="Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct ibm-granite/granite-3.1-2b-instruct"
```

Models are downloaded automatically to `$HF_HOME` (cluster scratch or `~/.cache/huggingface`). The venv is built on the first run and reused thanks to the sentinel file.

---

## Dataset annotation (`hf_dataset_annotation/`)

> **Do not touch this folder** — the annotation phase is complete.

Contains the notebooks and intermediate parquets (`eval_raw`, `eval_annotated`, `eval_final`) used to build the eval dataset. Documented in `annotated_dataset_eval.ipynb`.

---

## Project structure

```
aac-mcp-agent/
  app/
    src/
      agent/        # AACAgent, HFAACAgent, planner, session, resolve
      mcp_server/   # ARASAAC, time, schedule tools + FastMCP
      api/          # FastAPI (REST endpoints)
    frontend/       # React + Vite
    datasets/en/    # local ARASAAC JSON
    datasets/pictograms/  # cached PNGs (gitignored)
  eval/
    agent_eval.ipynb
    cluster_work/   # run_eval_hf.py, run_eval_cluster.sh, results/
  hf_dataset_annotation/   # DO NOT TOUCH
  test/
    tools_test.ipynb
  docs/
    context_for_next_agent.md   # full context for new LLM sessions
```

For the complete architectural context (design decisions, open questions, implementation table) see [`docs/context_for_next_agent.md`](docs/context_for_next_agent.md).


---
