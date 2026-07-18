# `annotation/`, building the evaluation dataset

This folder contains the offline, already-run pipeline that produced the annotated evaluation dataset used by `eval/` (`annotation/eval_final.parquet`: caregiver input + expected concept + "gold" pictogram ID, with a `clear`/`vague` split).

You don't need to re-run any of this to use the app or to evaluate it, `eval_final.parquet` is already committed. Besides `eval/`, it's also what `../test/mcp_protocol_demo.ipynb` samples a row from to build its mocked time/schedule context.

Submission zip note: inside `app/datasets/en_eval/` (the frozen snapshot produced by step 1 below), only `keyword_embeddings.npz` was removed from this archive to fit the 100MB size limit. To regenerate just the embeddings, run `python datasets/build_keyword_embeddings.py --langs en_eval` from `app/`, it works on `en_eval` like any other language, no need to re-run step 1. Re-running step 1 is only needed if you want to rebuild `en_eval` itself.

## Pipeline overview

Three steps, run in this order, each consuming the previous step's output:

```
1. arasaac_vs_hf_vs_eval.ipynb                 (local, no GPU needed)
        │
        ├──> app/datasets/en_eval/             (merged source dataset, frozen snapshot)
        └──> annotation/eval_filtered.parquet
                    │
2. cluster_work/annotate_eval.ipynb            (GPU, run via SLURM: sbatch run_annotate.sh)
        │
        └──> annotation/eval_annotated.parquet
                    │
3. annotation_quality_evaluation.ipynb         (local, no GPU needed)
        │
        └──> annotation/eval_final.parquet     (consumed by eval)
```

## Step 1: `arasaac_vs_hf_vs_eval.ipynb`

Compares three sources to build a fair, self-consistent evaluation base: 

- `df_local`, the local ARASAAC dataset already in `app/datasets/en/`
- `df_hf`, the HuggingFace mirror of the ARASAAC pictogram catalog (`disi-unibo-nlp-students/ARASAAC-Pictograms`)
- `df_eval`, a HuggingFace dataset of caregiver-style sentences (`disi-unibo-nlp-students/aac_database`).

It cleans each source, finds overlaps/mismatches between them, and produces two artifacts: `app/datasets/en_eval/`, `df_local` merged with the pictogram IDs that only exist in `df_hf`, written in the same file layout as `app/datasets/en/` (this exists so the evaluation always has a pictogram record for every ID referenced by the eval sentences, even ones missing from the "production" `en/` dataset); and `annotation/eval_filtered.parquet`, the caregiver sentences from `df_eval`, filtered/deduplicated so that every referenced pictogram ID actually resolves in the merged dataset above.

Requires `HF_TOKEN` in `app/.env`, only needed to pull the two gated/rate limited HF datasets, see `app/.env.example`.

`en_eval` is a frozen snapshot. Once built, it's never touched by `app/datasets/update_datasets.py`. Regenerating `en_eval` means re-running this notebook, not the dataset-update command.

## Step 2: `cluster_work/annotate_eval.ipynb` (GPU, via SLURM)

Takes `eval_filtered.parquet` and asks an LLM (default `Qwen/Qwen2.5-7B-Instruct`, 4-bit) to annotate each sentence with `split` (`clear` / `vague` / `both` / `none`), `time_of_day` (`morning` / `afternoon` / `evening` / `night`), `event_time` (concrete `HH:MM`), `caregiver_clear` (a specific rephrasing of the sentence), `caregiver_vague` (a short, implicit fragment that tests the agent's context-disambiguation phase), and `schedule` (a list of plausible calendar events, used later to test whether the agent picks the right one among distractors).

This step needs a GPU and is meant to run on a SLURM cluster, not locally:

```bash
cd annotation/cluster_work
sbatch run_annotate.sh
```

`run_annotate.sh` creates its own virtualenv on first run, then executes the notebook headlessly via `papermill` and writes the executed copy to `annotate_eval_out.ipynb`, useful to inspect what actually happened, cell by cell, after the job finishes. Logs go to `cluster_work/logs/`.

Output: `annotation/eval_annotated.parquet` plus a running `annotation/annotation_log.jsonl` (one line per annotated row, used by the notebook itself to resume after an interrupted job, safe to delete if you want to force a clean restart).

## Step 3: `annotation_quality_evaluation.ipynb`

Loads `eval_annotated.parquet` and inspects the LLM annotation for formal and semantic correctness (malformed times, empty schedules, duplicated distractors, etc.), "manually" correcting rows where needed. Ends by writing `annotation/eval_final.parquet`, the file actually consumed by `eval/`.
