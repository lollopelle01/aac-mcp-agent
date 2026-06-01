"""
fix_eval_loop_tqdm.py
Sostituisce il loop eval nel notebook con versione tqdm per Colab.
Esegui dalla root: python fix_eval_loop_tqdm.py
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "eval" / "eval.ipynb"

NEW_SOURCE = [
    "from tqdm.auto import tqdm\n",
    "\n",
    "SAVE_EVERY = 10\n",
    "\n",
    "splits = ['clear', 'vague'] if SPLIT == 'both' else [SPLIT]\n",
    "\n",
    "csv_writer = IncrementalCSV(OUTPUT_CSV_PATH)\n",
    "total_t0   = time.monotonic()\n",
    "\n",
    "for model_name in MODELS:\n",
    "    print(f'\\n{\"━\"*70}')\n",
    "    print(f'  MODEL: {model_name}  |  window={EVAL_MAX_RESULTS}')\n",
    "    print(f'{\"━\"*70}', flush=True)\n",
    "\n",
    "    agent = HFAACAgent(\n",
    "        model             = model_name,\n",
    "        hf_device         = HF_DEVICE,\n",
    "        hf_load_in_8bit   = LOAD_8BIT,\n",
    "        hf_max_new_tokens = MAX_NEW_TOKENS,\n",
    "        lang              = LANG_CODE,\n",
    "        max_results       = EVAL_MAX_RESULTS,\n",
    "    )\n",
    "\n",
    "    total_work = len(df) * len(splits)\n",
    "    tracker    = ProgressTracker(total=total_work, splits=splits)\n",
    "\n",
    "    for split in splits:\n",
    "        print(f'\\n  ── split: {split.upper()} ──', flush=True)\n",
    "\n",
    "        pbar = tqdm(df.iterrows(), total=len(df), desc=split, unit='seq', leave=True)\n",
    "        for _, row in pbar:\n",
    "            try:\n",
    "                row_results = run_multi_turn(agent, row, split)\n",
    "                for r in row_results:\n",
    "                    r['model'] = model_name\n",
    "                csv_writer.add(row_results)\n",
    "                tracker.record(split, row_results)\n",
    "            except Exception as exc:\n",
    "                tracker.record_error()\n",
    "                tqdm.write(f'  [ERROR] row={row.name}  model={model_name}: {exc}')\n",
    "\n",
    "            if tracker.n_done % SAVE_EVERY == 0:\n",
    "                csv_writer.flush()\n",
    "\n",
    "            # aggiorna postfix barra con hit rate corrente e ETA\n",
    "            hit_strs = {s: f'{sum(h)/len(h):.3f}' for s, h in tracker.hits.items() if h}\n",
    "            elapsed_s = time.monotonic() - tracker._t0\n",
    "            avg_s = elapsed_s / max(tracker.n_done, 1)\n",
    "            eta = str(timedelta(seconds=int(avg_s * max(tracker.total - tracker.n_done, 0))))\n",
    "            pbar.set_postfix(hit=hit_strs, eta=eta, err=tracker.n_errors)\n",
    "\n",
    "        pbar.close()\n",
    "\n",
    "    csv_writer.flush()\n",
    "    agent.unload()\n",
    "    print(f'  Model {model_name!r} done — GPU memory freed.', flush=True)\n",
    "\n",
    "csv_writer.flush()\n",
    "elapsed = time.monotonic() - total_t0\n",
    "print(f'\\n{\"═\"*70}')\n",
    "print(f'  All models done in {timedelta(seconds=int(elapsed))}')\n",
    "print(f'  Output: {OUTPUT_CSV_PATH}')\n",
    "print(f'{\"═\"*70}')",
]

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

found = False
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "LOG_EVERY" in src and "SAVE_EVERY" in src and "for model_name in MODELS" in src:
        cell["source"] = NEW_SOURCE
        found = True
        print("✓ Cella loop trovata e sostituita con versione tqdm.")
        break

if not found:
    print("✗ ERRORE: cella non trovata nel notebook.")
    exit(1)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ Notebook salvato: {NB_PATH}")
