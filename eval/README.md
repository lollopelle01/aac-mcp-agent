# Evaluation

CPU evaluation experiments for the agent: exploration to pick model + ranking strategy, then extended evaluation of the winning config, on Colab and locally.

## Structure

```
eval/
├── cpu-colab/
│   ├── explore/                     # ranking strategy comparison (3 models, 30 sentences)
│   │   ├── round_robin_weighted/    # commented reference notebook
│   │   └── sequential_blocks/       # same pipeline, different strategy
│   └── best/                        # chosen model, larger dataset (Colab)
└── cpu-local_only_best/             # same "best" run, locally
```

Each experiment folder has 4 files: 

1. `eval_cpu_*.ipynb` runs the agent over the dataset and produces a CSV
2. `eval_cpu_*.csv` is that output, the input to the analysis notebook
3. `metrics_eval_cpu_*.ipynb` computes metrics and plots from the CSV
4. `agent_run.log` is the raw per-turn trace (decide → plan → resolve → rank), useful for debugging a single sentence.

## Running an experiment

Execution notebook, Colab (`cpu-colab/**`): open via the "Open in Colab" badge. The first cell clones the repo into `/content/aac-mcp-agent` and downloads the GGUF models (edit `NB_MODELS` / `NB_N_ROWS` / `NB_SEED` in section 2.1 before running).

Execution notebook, local (`cpu-local_only_best/`): open from Jupyter inside the already-cloned repo, no cloning or downloading needed, more threads, no time limit, same env var section as above. For prerequisites (GGUF model, Python environment) see the [root README § Setup and running](../README.md#setup-and-running).

The output CSV must stay in the experiment's own folder, the analysis notebook expects it right there.

Analysis notebook: just run it after the CSV exists, no config needed.

> **NOTE**: the analysis pipeline (code + markdown) is identical in every experiment, it's the same notebook applied to different runs. Only `cpu-colab/explore/round_robin_weighted/metrics_eval_cpu_rrw.ipynb` has commentary written for its specific results, read the others against that same framework.

## What the analysis measures

End-to-end quality: whether the gold pictogram lands in the final window, or at least in the retrieved pool. Ranking diagnostic: what the ranker loses relative to the retriever's pool. Retriever diagnostic: raw pool hit rate against a random baseline. Resolve and tool usage: the distribution of concept resolution methods, plus `get_time`/`get_schedule` call accuracy. Latency: per-turn, first turn vs. subsequent ones. Sentence difficulty and dataset audit: always-zero sentences, annotation quality, sample generalizability.

## Results

`cpu-colab/explore/`: 30 sentences, 3 models (`qwen2.5:3b`, `llama3.2:3b`, `granite4:3b-h`), identical input/planning/resolve, only the ranking strategy changes. No strategy wins across the board, differences are mostly outside the metrics that matter most. Qwen is the fastest model in both runs. `round_robin_weighted` is slightly better on ranking metrics, so it's the one chosen for the next stage.

`cpu-colab/best/` and `cpu-local_only_best/`: chosen model + strategy, larger sample. Same notebook, same config, the local run is about 4x faster than the Colab one.

None of these experiments go through the MCP protocol, the agent calls its tools in-process, same as in production. For a version of the pipeline running over a real MCP client/server connection, see `../test/mcp_protocol_demo.ipynb`.
