# AAC MCP Agent

Sistema agentico per la selezione contestuale di pittogrammi [ARASAAC](https://arasaac.org/) per la Comunicazione Aumentativa e Alternativa (CAA/AAC).

Il caregiver descrive la situazione a parole libere; l'agente usa un LLM locale come planner per decidere quali concetti semantici cercare, interroga il dataset ARASAAC locale, e restituisce una finestra di pittogrammi candidati. Il soggetto seleziona; la sessione tiene traccia delle scelte per evitare ripetizioni.

---

## Prerequisiti

| Dipendenza | Versione minima | Note |
|---|---|---|
| Python | 3.11 | |
| Node.js | 18 | per il frontend |
| Ollama | qualsiasi | solo in locale |
| `granite4:3b-h` (o altro) | — | vedi Settings |

```bash
# Setup Python (una tantum)
pip install -r app/requirements.txt
python -m spacy download en_core_web_sm

# Modello LLM locale
ollama pull granite4:3b-h
```

---

## Avvio locale

```bash
# 1. Backend (FastAPI su :8000)
cd app
uvicorn src.api.server:app --reload --port 8000

# 2. Frontend (altro terminale — Vite su :5173)
cd app/frontend
npm install        # prima volta
npm run dev
```

Aprire **http://localhost:5173**. Il proxy Vite inoltra `/api/*` → `:8000`.

### Variabili d'ambiente

Copiare `app/.env.example` → `app/.env` e compilare le credenziali opzionali (Google Calendar, Apple, HuggingFace). Le impostazioni operative (modello, numero risultati, ecc.) si gestiscono dall'interfaccia → ⚙ Settings o direttamente in `app/user_settings.json`.

### Dataset pittogrammi

Al primo avvio il dataset testuale è incluso in `app/datasets/en/`. Le immagini PNG sono scaricate on-demand da ARASAAC CDN (o servite localmente se già presenti in `app/datasets/pictograms/`). Per pre-scaricare tutte le immagini prima di andare offline, usare il pannello **⬇ Datasets** → *Download images*.

---

## Uso dell'interfaccia

```
┌────────────────────────────────────────────────────────────┐
│  AAC Pictogram Agent  [granite4:3b-h ▾]  [⚙ Settings] [↺] │
├────────────────────────────────────────────────────────────┤
│  He wants something to eat before leaving        [→ Send]  │
├────────────────────────────────────────────────────────────┤
│  ┌────────┐  ┌────────┐  ┌────────┐  ←  tap per selezionare│
│  │ apple  │  │ yogurt │  │ snack  │                        │
└──┴────────┴──┴────────┴──┴────────┴────────────────────────┘
│  Sessione: [🍎] [🧥] [👟]                                   │
└────────────────────────────────────────────────────────────┘
```

1. Il caregiver scrive la situazione e preme **Send** (o Enter).
2. L'agente restituisce una griglia di pittogrammi candidati.
3. Il soggetto tocca un pittogramma → viene registrato nella sessione.
4. Ripetere per ogni concetto della frase.
5. **↺** azzera la sessione.

Il dropdown del modello cambia il LLM al volo (PATCH `/settings` → agente ricostruito al turno successivo).

---

## Test locale rapido

Il notebook `test/tools_test.ipynb` verifica i tool MCP in isolamento (ARASAAC search, time, schedule) senza avviare il backend completo. Utile per controllare che il dataset locale sia integro.

```bash
cd test
jupyter notebook tools_test.ipynb
```

---

## Valutazione (eval)

### Dataset di eval

Il dataset annotato si trova in `hf_dataset_annotation/eval_final.parquet` (~54 k righe). **Non modificarlo** — è il prodotto della fase di annotazione a monte (completata).

Ogni riga contiene: `raw_input` (testo caregiver), `concept` (concetto gold), `gold_id` (ID pittogramma atteso), `split` (`clear` o `vague`).

### Eval locale (notebook)

```bash
cd eval
jupyter notebook agent_eval.ipynb
```

Utile per ispezione manuale e debug su un sottoinsieme di righe.

### Eval su cluster (SLURM + GPU)

Unico entry point per la valutazione formale:

```bash
cd eval/cluster_work

# Sottomettere job SLURM (modificare i parametri in cima allo script prima)
sbatch run_eval_cluster.sh

# oppure eseguire direttamente (CPU, debug)
python run_eval_hf.py \
  --models Qwen/Qwen2.5-3B-Instruct \
  --split both \
  --n_rows 200 \
  --output results/debug.csv \
  --verbose
```

**Parametri principali di `run_eval_hf.py`:**

| Flag | Default | Descrizione |
|---|---|---|
| `--models` | — | uno o più modelli HF (space-separated) |
| `--split` | `both` | `clear`, `vague`, o `both` |
| `--n_rows` | `0` (tutti) | righe da valutare (0 = dataset completo) |
| `--seed` | `42` | seed per il campionamento |
| `--output` | `results/eval_hf_<job>.csv` | CSV di output |
| `--load_in_8bit` | off | quantizzazione INT8 (risparmia VRAM) |
| `--log_every` | `25` | progress ogni N righe |
| `--save_every` | `10` | checkpoint CSV ogni N righe |
| `--verbose` | off | dettagli turno per turno |

**Metriche principali** (CSV + summary a fine run):

| Metrica | Descrizione |
|---|---|
| `hit` / `gold_in_window` | gold ID nella finestra finale — **metrica primaria** |
| `gold_in_candidates` | gold ID nel pool prima del ranking |
| `overlap_level` | miglior overlap semantico (`synset` > `category` > `keyword` > `tag`) |
| `resolve_method` | strategia `resolve_concept` (`exact`, `lemma`, `token`, …) |
| `plan_method` | `llm` / `fallback_spacy` / `fallback_empty` |
| `synset_added` | pittogrammi aggiunti dall'espansione WordNet |
| `fresh_count` | pittogrammi fresh (non padding stale) nella finestra |

Il CSV ha una riga per turno e una colonna `model`, così più modelli possono stare nello stesso file.

**Teacher forcing:** dopo ogni turno l'eval inietta `gold_id` come pittogramma selezionato per simulare sequenze multi-step realistiche.

### Configurare i modelli nel job SLURM

Modificare `HF_MODELS` in `run_eval_cluster.sh` prima di `sbatch`:

```bash
HF_MODELS="Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct ibm-granite/granite-3.1-2b-instruct"
```

I modelli vengono scaricati automaticamente in `$HF_HOME` (scratch del cluster o `~/.cache/huggingface`). Il venv viene costruito alla prima esecuzione e riusato grazie al sentinel file.

---

## Annotazione dataset (`hf_dataset_annotation/`)

> **Non toccare questa cartella** — la fase di annotazione è completata.

Contiene i notebook e i parquet intermedi (`eval_raw`, `eval_annotated`, `eval_final`) usati per costruire il dataset di eval. Documentati in `annotated_dataset_eval.ipynb`.

---

## Struttura del progetto

```
aac-mcp-agent/
  app/
    src/
      agent/        # AACAgent, HFAACAgent, planner, session, resolve
      mcp_server/   # tool ARASAAC, time, schedule + FastMCP
      api/          # FastAPI (endpoint REST)
    frontend/       # React + Vite
    datasets/en/    # JSON ARASAAC locale
    datasets/pictograms/  # PNG cachati (gitignored)
  eval/
    agent_eval.ipynb
    cluster_work/   # run_eval_hf.py, run_eval_cluster.sh, results/
  hf_dataset_annotation/   # DO NOT TOUCH
  test/
    tools_test.ipynb
  docs/
    context_for_next_agent.md   # contesto completo per nuove sessioni LLM
```

Per il contesto architetturale completo (decisioni di design, dubbi aperti, tabella implementazioni) vedere [`docs/context_for_next_agent.md`](docs/context_for_next_agent.md).
