# AACAgent — Contesto per il prossimo agente

> Questo documento riassume lo stato del progetto, le decisioni prese e i
> dubbi aperti. Va allegato come contesto a qualsiasi nuova sessione di
> lavoro sul progetto.

---

## 1. Descrizione del progetto

Sistema MCP agentico per la selezione contestuale di pittogrammi ARASAAC per la
Comunicazione Aumentativa e Alternativa (CAA/AAC).

**Flusso di un turno:**

1. Il caregiver fornisce una descrizione testuale breve
   (es. *"vuole qualcosa prima di uscire"*)
2. Il **planner LLM** decide:
   - input vago → `call_tools=true` → il codice chiama `get_time` e
     `get_schedule` per arricchire il contesto
   - input esplicito → `call_tools=false` → nessun tool contestuale
3. Il planner genera una lista di **concetti semantici** da cercare
   (es. `["coat", "shoes", "bag", "go out"]`)
4. `resolve_concept()` mappa ogni concetto su keyword reali ARASAAC
5. `search_pictograms()` costruisce il pool di candidati
6. Espansione opzionale del pool via WordNet synsets (`_expand_pool_by_synset`)
7. Ranking deterministico → `_rank_and_fill` → finestra di `max_results`
   pittogrammi (esclusi quelli già selezionati nei turni recenti)
8. Il soggetto seleziona un pittogramma dalla finestra → `/select` aggiorna la
   memoria di sessione; si riparte dal punto 1 per il concetto successivo

**Vincoli hardware:** il sistema deve girare su CPU su un tablet/iPad.
La leggerezza è un requisito funzionale. Le GPU sono rilevanti solo per la
fase di valutazione sul cluster.

---

## 2. Stack tecnologico

| Componente            | Dettaglio                                                     |
| --------------------- | ------------------------------------------------------------- |
| LLM locale (CPU)      | llama.cpp `LlamaCppBackend` (GGUF Q4_K_M) — backend default  |
| LLM locale (fallback) | Ollama (`granite4:3b-h`, `qwen2.5:3b`, `llama3.2:3b`)       |
| LLM eval cluster/GPU  | HuggingFace Transformers via `HuggingFaceBackend`             |
| Tool MCP              | FastMCP (`app/src/mcp_server/`)                               |
| NLP                   | spaCy `en_core_web_sm` (lemmatizzazione, POS, stop-word)     |
| Backend               | FastAPI + uvicorn su `:8000`                                  |
| Frontend              | React + Vite su `:5173`                                       |
| Dataset pittogrammi   | ARASAAC locale in `app/datasets/en/`                          |
| Dataset eval primario | `annotation/eval_filtered.parquet` (1760 frasi) — vedi §10   |
| Lingua principale     | inglese (`LANG = "en"`)                                       |

---

## 3. Struttura del progetto

```
aac-mcp-agent/
  app/
    src/
      agent/
        agent.py        # AACAgent — pipeline procedurale (un LLM call/turno)
        backends.py     # LlamaCppBackend, OllamaBackend, HuggingFaceBackend
        prompts.py      # build_planner_prompt, build_planner_message
        session.py      # SessionMemory, Turn, _nlp() (spaCy loader, lru_cache)
        resolve.py      # resolve_concept — concept → keyword ARASAAC
      mcp_server/
        tools/
          arasaac.py        # search_pictograms, search_pictograms_by_synset,
                            # get_pictogram_metadata, get_pictogram_image,
                            # list_keywords
          time_tool.py      # get_time()
          schedule_tool.py  # get_schedule()
        models.py           # Pictogram, Keyword, TimeInfo, ScheduleEvent
        dataset_cache.py    # _DatasetCache — loader locale per i JSON
        server.py           # istanza FastMCP
      api/
        server.py           # FastAPI — wrappa AACAgent, espone REST
      config.py             # costanti pure + re-export da settings.py e .env
      settings.py           # SettingsManager — legge/scrive user_settings.json
    frontend/               # React + Vite
    datasets/
      en/                   # dataset produzione — NON TOCCARE
      en_eval/              # snapshot frozen per eval (merge local+HF) — NON TOCCARE
      pictograms/           # PNG cachati
      update_datasets.py    # ricostruisce en/ dall'API ARASAAC
    models/                 # GGUF scaricati (es. Qwen2.5-3B-Instruct-Q4_K_M.gguf)
  annotation/
    arasaac_vs_hf_vs_eval.ipynb        # notebook analisi e produzione dataset (R26)
    annotation_quality_evaluation.ipynb # analisi qualità annotazione LLM (R27)
    eval_filtered.parquet              # dataset eval primario (1760 frasi) — prodotto R26
    eval_annotated.parquet             # dataset annotato da LLM (R27) — con colonne contesto
    eval_final.parquet                 # dataset finale con split clear/vague
    annotation_log.jsonl               # log idempotente annotazione
    cluster_work/
      annotate_eval.ipynb              # notebook annotazione su cluster (R27)
      annotate_eval_out.ipynb          # output annotazione (non modificare)
      run_annotate.sh                  # sbatch script
  eval/
    eval_cpu.ipynb          # eval locale/Colab su CPU con LlamaCppBackend
    eval_gpu.ipynb          # eval Colab/cluster su GPU con HuggingFaceBackend
    results/                # CSV output delle run (gitignored o vuota al push)
  docs/
    context_for_next_agent.md
    consegna.md
  test/
    tools_test.ipynb
```

**Nota struttura eval:** i notebook `eval_cpu.ipynb` e `eval_gpu.ipynb` vivono
direttamente in `eval/`. Il prossimo step organizzativo (già pianificato) è
creare `eval/cpu/` e `eval/gpu/` e spostare lì i rispettivi notebook, CSV di
risultato e script di lancio — tenendo lo stesso file CSV come output condiviso
di riferimento. Non è ancora stato fatto.

---

## 4. Come avviare il sistema

```bash
# Prerequisiti (una tantum)
pip install -r app/requirements.txt
python -m spacy download en_core_web_sm
ollama pull qwen2.5:3b   # solo se si usa il backend Ollama

# Backend
cd app
uvicorn src.api.server:app --reload --port 8000

# Frontend (altra finestra)
cd app/frontend
npm install
npm run dev
```

---

## 5. Backend FastAPI — endpoint

| Metodo  | Path               | Descrizione                                                         |
| ------- | ------------------ | ------------------------------------------------------------------- |
| `POST`  | `/run`             | `{"text": str}` → lista pittogrammi + turn + tools_called          |
| `POST`  | `/select`          | `{"pictogram_id": int}` → aggiorna memoria sessione                |
| `POST`  | `/reset`           | svuota sessione                                                      |
| `GET`   | `/session`         | storia sessione corrente                                             |
| `GET`   | `/settings`        | legge `user_settings.json`                                          |
| `PATCH` | `/settings`        | `{"updates": {...}}` → aggiorna settings + warmup nuovo modello    |
| `GET`   | `/health`          | `{"ok": true, "model": "...", "ollama": bool, "warming_up": bool}` |
| `GET`   | `/images/{id}`     | serve PNG da dataset locale o CDN ARASAAC                           |
| `GET`   | `/datasets/status` | metadata dataset per ogni lingua + conteggio PNG cachati            |
| `POST`  | `/datasets/update` | `{langs?, force?, download_images?}` → SSE stream log lines        |
| `GET`   | `/categories`      | categorie ARASAAC raggruppate in macro-categorie con count          |
| `GET`   | `/by_category`     | pittogrammi di una data categoria                                   |

---

## 6. Metadata di un pittogramma ARASAAC

```json
{
  "id": 2248,
  "keywords": [
    {"type": 2, "keyword": "water", "plural": "waters", "meaning": "..."}
  ],
  "categories": ["beverage", "mineral rich food"],
  "synsets": ["07951744-n", "14869913-n"],
  "tags": ["feeding", "food", "beverage"],
  "aac": false,
  "aac_color": false,
  "violence": false,
  "sex": false
}
```

**Tipi di keyword:** 1=nomi propri, 2=nomi comuni, 3=verbi, 4=aggettivi, 5=sociale, 6=misc.
Circa il 40% dei pittogrammi non ha synsets.

---

## 7. Pipeline agente (un LLM call per turno)

```
input caregiver
  └─► _plan() — LLM planner
        ├─ call_tools=true  → _collect_context() → get_time, get_schedule
        └─ call_tools=false → time_of_day=None

  └─► se planner fallisce o concepts=[] → fallback _extract_terms() [spaCy]

  └─► _search_candidates()
        └─► resolve_concept() → keyword ARASAAC
        └─► search_pictograms() per ogni keyword → pool

  └─► _expand_pool_by_synset() (se AGENT_SYNSET_EXPAND=True)

  └─► _rank_and_fill()
        ├─ escludi solo selected_ids (Opzione A, R22)
        ├─ ordina per (concept_order ASC, quality_score DESC)
        └─ riempi fino a max_results

  └─► add_turn() → memoria sessione aggiornata
```

---

## 8. Parametri funzionali

```python
AGENT_MAX_RESULTS         = 25   # R22: 24→25
AGENT_CANDIDATES_PER_TERM = 10
AGENT_MEMORY_TURNS        = 3
AGENT_DEFAULT_MODEL       = "qwen2.5:3b"
AGENT_SYNSET_EXPAND       = True
AGENT_SYNSET_EXPAND_MAX   = 8
LANG                      = "en"
AGENT_FETCH_SCHEDULE      = True
```

---

## 9. Strategia `resolve_concept` (`app/src/agent/resolve.py`)

| Step | Label       | Esempio                                             |
|------|-------------|-----------------------------------------------------|
| 0    | `strip`     | `"a banana"` → `"banana"` (article stripping, R23) |
| 1    | `exact`     | `"eat"` → `"eat"`                                  |
| 2    | `lemma`     | `"eating"` → `"eat"` via spaCy                    |
| 3    | `hyphen`    | `"go out"` → `"go-out"`                           |
| 3b   | `lemma_alt` | lemma della forma normalizzata                      |
| 4    | `token`     | `"wash hands"` → `["wash", "hand"]`               |
| —    | `none`      | nessun match → concetto saltato                     |

`resolve_concept(concept, kw_set, return_method=True)` ritorna `(queries, method)`.

`_nlp()` è definita una sola volta in `session.py` con `@lru_cache(maxsize=1)` e
importata da `resolve.py` e `agent.py` — un solo caricamento spaCy in RAM.

---

## 10. Dataset per la valutazione — stato attuale (R27)

### Dataset primario: `annotation/eval_filtered.parquet`

Prodotto da `annotation/arasaac_vs_hf_vs_eval.ipynb` (R26). Dataset corretto
da usare per tutte le eval future. Struttura sentence-based.

**Numeri finali:**

| Fase | Frasi |
|---|---|
| Originale (`aac_database/sentences` HF) | 2.461 |
| Dopo filtro gold irrecuperabili (none_ids) | 2.025 |
| Dopo filtro candidati >20% irrecuperabili | 1.787 |
| Dopo rimozione duplicati esatti | **1.760** |

**Struttura di ogni riga:**
```python
{
  "sentence": "I want to go to the park.",
  "concepts": [
    {
      "concept_text": "go",
      "gold_id": 8544,
      "candidate_ids": [8544, 2603, ...]   # sempre 10 candidati, gold sempre incluso
    },
    ...
  ]
}
```

**Proprietà garantite:**
- Gold sempre tra i 10 candidati (100%) — per costruzione del dataset HF
- Nessun gold con metadata irrecuperabile
- Nessun concetto con >20% candidati irrecuperabili
- Nessun duplicato esatto (18 gruppi ambigui tenuti — gold diversi sullo stesso pool)
- Candidati sempre esattamente 10
- Concetti per frase: mean=3.68, std=1.05, min=2, max=7

### Dataset annotato: `annotation/eval_annotated.parquet`

Prodotto da `annotation/cluster_work/annotate_eval.ipynb` (R27).
Modello: `Qwen/Qwen2.5-3B-Instruct` (4-bit NF4), GPU NVIDIA L40, cluster SLURM.
Contiene colonne di contesto sintetico generate dal LLM (es. `time_of_day`,
`input_type`, `schedule`, `distractors`). Ha 1760 righe (Fix A R27 corregge un
drop errato di 20 duplicate).

### Dataset finale: `annotation/eval_final.parquet`

Versione con colonna `split` (`clear`/`vague`) — usare questo per run che
vogliono filtrare per tipo di input.

### Dataset sorgente pittogrammi: `app/datasets/en_eval/`

Snapshot frozen prodotto da R26. **Non toccare.**

| File                 | Contenuto                                  |
|----------------------|--------------------------------------------|
| `pictograms.json`    | 13.804 pittogrammi (`{ id_str → record }`) |
| `keywords.json`      | 15.761 keyword uniche                      |
| `keyword_index.json` | 15.761 entries                             |
| `synset_index.json`  | 8.422 synset                               |
| `_meta.json`         | timestamp e conteggi                       |

### Perché il nuovo dataset è un upgrade rispetto al legacy

| Aspetto | Legacy (`eval/eval_filtered.parquet`) | Nuovo (`annotation/eval_filtered.parquet`) |
|---|---|---|
| Struttura gold | ID singolo per concept, spesso deprecato | Gold sempre tra i 10 candidati (100%) |
| Concetti | Caption fragments (es. "a football match") | Parole/frasi brevi normalizzate (es. "go") |
| Stima resolve=none | ~88% (da R21) | ~12% (stimato) |
| Frasi usabili | ~54k righe ma molto rumore | **1.760 frasi pulite** |

---

## 11. Analisi qualità dataset (R26) — risultati chiave

### `df_local` (13.780 righe)
- Nessun duplicato, nessuna riga None
- `synsets` vuoti: 1.213 righe (8.8%)
- Keywords: 65.9% dei meaning sono None (non bloccante)
- 138 pittogrammi con keyword duplicate interne (non bloccante)
- Type distribution: common noun 61.8%, verb 23.2%, adjective 7.4%

### `df_hf` (originale 12.484 → dopo cleaning 10.657 righe valide)
- 10 ID duplicati (38264–38273): copia con metadata tenuta
- 1.817 righe con keywords/categories/tags tutti None → `none_ids`
- Solo 1 ID su 1.807 `none_ids` presente anche in `df_local`
- Schema keyword diverso da local: `{hasLocution, keyword, meaning}`

### Overlap `df_local` vs `df_hf` valido
- In entrambi: 10.633 (77.0% dell'union)
- Solo in local: 3.147 (22.8%) — Solo in HF: 24 (0.2%)
- Similarity score (keywords×0.6 + categories×0.2 + tags×0.2): mean=0.985, 92% identici
- Merge strategy: local vince sempre (schema più ricco: synsets, type, plural)

---

## 12. Eval — come funziona

### `eval/eval_cpu.ipynb` — CPU locale / Colab CPU

Usa `AACAgent(backend=LlamaCppBackend(...))`, identico all'app in produzione.
Progettato per Colab con runtime CPU o validazione rapida in locale.

**Parametri (env var):**

| Variabile | Default | Note |
|---|---|---|
| `NB_MODELS` | `qwen2.5:3b` | alias separati da spazio, devono essere in `settings.gguf_models` |
| `NB_N_ROWS` | `100` | 0 = tutto il dataset |
| `NB_N_THREADS` | `4` | thread CPU per llama.cpp |
| `NB_N_CTX` | `512` | context window (uguale a produzione) |
| `NB_OUTPUT_CSV` | `eval/results/eval_cpu.csv` | |
| `NB_ANNOTATED_PARQUET` | percorso `eval_annotated.parquet` | |

### `eval/eval_gpu.ipynb` — GPU Colab / cluster SLURM

Usa `AACAgent(backend=HuggingFaceBackend(...))` con modelli HuggingFace.

**Parametri (env var):**

| Variabile | Default | Note |
|---|---|---|
| `NB_MODELS` | `Qwen/Qwen2.5-3B-Instruct` | repo HF separati da spazio |
| `NB_N_ROWS` | `100` | 0 = tutto il dataset |
| `NB_LOAD_8BIT` | `True` | INT8 via bitsandbytes (~50% VRAM) |
| `NB_DTYPE` | `float16` | |
| `NB_SPLIT_FILTER` | `all` | `clear` \| `vague` \| `all` |
| `NB_MAX_NEW_TOKENS` | `150` | |
| `NB_OUTPUT_CSV` | `eval/results/eval_gpu_colab.csv` | |
| `NB_ANNOTATED_PARQUET` | percorso `eval_annotated.parquet` | |

**Lancio su cluster SLURM:**
```bash
NB_MODELS="Qwen/Qwen2.5-3B-Instruct" NB_N_ROWS=500 sbatch eval/cluster_work/run_eval_cluster.sh
```

### Metriche principali

- `gold_in_candidates` — il gold ID era nel pool prima del ranking?
- `hit` (`gold_in_window`) — il gold ID è nella finestra finale?
- `overlap_level` — livello semantico migliore (`synset` > `category` > `keyword` > `tag`)

### Colonne CSV output principali

| Colonna | Descrizione |
|---|---|
| `model` | nome modello |
| `sentence` | frase caregiver |
| `concept` | concetto gold del turno (= `concept_text`) |
| `gold_id` | ID pittogramma gold |
| `hit` | gold nella finestra finale |
| `gold_in_candidates` | gold nel pool pre-ranking |
| `resolve_method` | step usato da resolve_concept |
| `planner_had_gold_concept` | planner ha generato esattamente il gold concept |
| `input_triggered_tools` | True solo a turn_pos==0 |

**Nota:** `resolve_method` è significativo solo dove `planner_had_gold_concept=True`.

---

## 13. Risultati prima eval run (R21) — su dataset LEGACY

⚠️ Questi risultati sono sul dataset legacy, non comparabili con run future.

| Metrica | Valore |
|---|---|
| Hit@window | **22.0%** |
| gold_in_candidates | 23.5% |
| resolve_method=none | **88.3%** |
| planner_had_gold_concept | 13.7% |
| Hit quando planner_had_gold_concept=True | 33.8% |

**Diagnosi:** bottleneck primario = planner genera concetti non allineati al
vocabolario ARASAAC. Il nuovo dataset riduce questo problema strutturalmente.

---

## 14. Piano miglioramenti score

### Strategia ibrida raccomandata (3 strati)

| Strato | Implementazione | Impatto stimato |
|---|---|---|
| **1. CONCEPT_MAP** | Dict `{"groceries":"shopping", ...}` in `resolve.py` step 4b | Top-50 casi OOV frequenti |
| **2. Embedding** | `all-MiniLM-L6-v2` + `.npy` cached, step 5 in resolve | Casi OOV long-tail (~2-3pp) |
| **3. Prompt few-shot** | 10-15 coppie nel `_PLANNER_SYSTEM_PROMPT_SHORT` | Previene OOV a monte |

**P4:** aumentare `AGENT_CANDIDATES_PER_TERM` da 10 a 15-20 (+1-3pp stimati, zero codice).

---

## 15. Dubbi aperti — da chiarire con i tutor

**D1 — Tool-use: LLM decide o codice decide?**

**D2 — Una sola chiamata LLM per turno: sufficiente?**

**D5 — Soglia di overlap per il successo semantico**

**D6 — Teacher forcing: metodologicamente accettabile?**

**D7 — Quante righe sono sufficienti?** Il dataset filtrato ha 1.760 frasi.

**D8 — Gold multipli per concetto?** I 18 gruppi ambigui mostrano che esistono
interpretazioni alternative legittime — estendere a top-3?

---

## 16. Cose da NON fare

- **Non passare `concept` o `sentence` all'agente** durante l'eval
- **Non sovrascrivere parametri di `config.py` nel notebook**
- **Non usare MRR** come metrica primaria
- **Non toccare `hf_dataset_annotation/`** — fase a monte completata
- **Non toccare `app/datasets/en_eval/`** — snapshot frozen per eval
- **Non toccare `app/datasets/en/`** — dataset produzione
- **Non usare `eval/eval_filtered.parquet`** — è il dataset LEGACY (non esiste più
  come file, ma la confusione di nome è ancora possibile)
- **Non mettere logica applicativa fuori da `app/`**

---

## 17. Tabella implementazioni

| ID  | Descrizione | File principali |
|---|---|---|
| R3  | Rimosso filter LLM morto; `_rank_and_fill` usa `recently_presented_ids()` | `agent.py` |
| R4  | `datasets/` → `app/datasets/`; `_image_url` offline-aware | `config.py`, `arasaac.py` |
| R5  | Endpoint SSE dataset update; UI DatasetPanel; tracing resolve | `api/server.py`, `frontend/` |
| R6  | Fix path setup eval; bigrammi schedule; test `return_method` | `test/`, `agent.py` |
| R7  | `CategoryBrowser.jsx` — ricerca per categoria (frontend) | `frontend/` |
| R8  | Fix `/categories` conteggi; UI inglese; `_errorMessage` hook | `api/server.py`, `frontend/` |
| R9  | Aggiunte categorie biologiche animali a `MACRO_CATEGORIES` | `config.py` |
| R10 | Diagnostica latenza; `num_predict`; modello default → `qwen2.5:3b` | `settings.py`, `agent.py` |
| R11 | Fix errata `has_thinking`; prompt SHORT/FULL; `backends.py` | `prompts.py`, `backends.py` |
| R12 | Rimosso cap concepts nel prompt; few-shot espansione semantica | `prompts.py` |
| R14 | InputBar warmup; GGUF scaricati; `LlamaCppBackend` agganciato | `frontend/`, `agent.py` |
| R19 | Fix eval pipeline: `get_resolve_info` 3 valori; prompt HF | `backends.py`, `eval_gpu.ipynb` |
| R20 | Completamento fix eval notebook (celle 10, 18, 19) | `eval/eval_gpu.ipynb` |
| R21 | Prima eval run Colab (200 righe, T4); diagnosi bottleneck resolve | `eval/results/` |
| R22 | Opzione A exclusion + window 25 + `NB_MAX_RESULTS` | `agent.py`, `settings.py` |
| R23 | Fix article stripping in `resolve.py` | `resolve.py` |
| R24 | Analisi embedding 7-punti; strategia ibrida 3-strati | `docs/` |
| R26 | Analisi dataset, cleaning, merge, produzione `en_eval/` e `annotation/eval_filtered.parquet` | `annotation/` |
| R27 | Annotazione LLM dataset (1760 righe); fix drop duplicate; `eval_annotated.parquet` | `annotation/` |
| R28 | `eval_cpu.ipynb` (CPU/llama.cpp); `eval_gpu.ipynb` (GPU/HF); analisi piano pulizia `hf_agent.py` | `eval/` |
| R29 | Fix codebase pre-eval: 9 fix su `resolve.py`, `session.py`, `agent.py`, `api/server.py`, `config.py`, `models.py`, `arasaac.py`, `time_tool.py` — vedi §18 | tutti i file `app/src/` |

---

## 18. Fix applicati in R29 (pre-eval cleanup)

Tutti i fix seguenti sono stati applicati e verificati prima del push.

| Fix | File | Descrizione |
|-----|------|-------------|
| **1** | `resolve.py`, `session.py` | `_nlp()` duplicata rimossa da `resolve.py`; importata da `agent.session`. Un solo caricamento spaCy in RAM via `@lru_cache(maxsize=1)`. |
| **2** | `config.py` | `DATASET_LANGS` aggiunta come re-export da `settings`. Era un bug vivo: `api/server.py` la importava da `config` ma non esisteva. |
| **3** | `mcp_server/models.py` | `ContextBundle` rimosso — era dead code dalla R3. Commento su `ScoredPictogram` aggiornato. |
| **4** | `agent.py`, `session.py` | `shown_ids` rimosso da `_rank_and_fill`. `recently_presented_ids()` ha ora un docstring che spiega esplicitamente la scelta R22/Opzione A. |
| **5** | `mcp_server/tools/arasaac.py`, `mcp_server/models.py` | `ScoredPictogram` eliminato. `_safe_to_scored` → `_safe_parse` (ritorna `Pictogram \| None`); `_parse_raw_list` e `_ids_to_results` ora ritornano `list[Pictogram]`; `_scored_to_dict` eliminata; call site aggiornati a `r.model_dump()`. `Tuple` rimosso dall'import di `models.py`. |
| **6** | `config.py`, `api/server.py` | `MACRO_CATEGORIES` spostato in `config.py`. `server.py` ora la importa. ~80 righe di dati sparite da un file di logica API. |
| **7** | `api/server.py` | `_agent.unload()` aggiunto prima di `_agent = None` in `PATCH /settings`. Evita due GGUF in RAM contemporaneamente durante il warmup del nuovo modello. |
| **8** | tutti i file toccati | Separatori uniformati a `##` — rimossi `####` residui in `resolve.py`. |
| **9** | `mcp_server/tools/time_tool.py` | Commento del fallback `_resolve_time_of_day` corretto: `DAY_TIMES[0]` è `"morning"`, non `"night"`. Il comportamento era già corretto; solo il commento era sbagliato. |

---

## 19. Stato `app/src/agent/` dopo R29

`hf_agent.py` è stato **eliminato** in una sessione precedente. La pulizia
pianificata in R28 (§20 del vecchio contesto) è già completata.

La directory `agent/` contiene ora solo:

| File | Ruolo |
|---|---|
| `agent.py` | `AACAgent` — pipeline principale, backend-agnostico |
| `backends.py` | `LlamaCppBackend`, `HuggingFaceBackend`, `OllamaBackend` |
| `prompts.py` | `build_planner_prompt`, `build_planner_message` |
| `session.py` | `SessionMemory`, `Turn`, `_nlp()` |
| `resolve.py` | `resolve_concept`, `_lemmatize_phrase`, `_lemmatize_word` |

`eval_gpu.ipynb` usa già `AACAgent(backend=HuggingFaceBackend(...))` —
non fa più riferimento a `HFAACAgent`.

---

## 20. Prossimi passi

1. **Prima eval run sul nuovo dataset** — `eval/eval_cpu.ipynb` (locale) o
   `eval/eval_gpu.ipynb` (Colab/cluster) usando `annotation/eval_annotated.parquet`.
   Verificare che `gold_in_candidates` e `resolve_method=none` siano
   significativamente migliorati rispetto ai numeri R21 (~88% none).

2. **Riorganizzare `eval/`** — creare `eval/cpu/` e `eval/gpu/`, spostare
   `eval_cpu.ipynb` e `eval_gpu.ipynb` nelle rispettive sottodirectory insieme
   ai CSV di risultato. Unico file CSV condiviso tra le due come riferimento
   comparativo. (Già pianificato — non ancora eseguito.)

3. **Implementare CONCEPT_MAP** (step 4b in `resolve.py`) con le top-50 coppie
   OOV identificate dall'analisi R21.

4. **Aggiungere colonna `split`** (`clear`/`vague`) alle frasi via LLM in batch,
   se non già presente in `eval_final.parquet`.

5. **Embedding fallback** (`all-MiniLM-L6-v2`) come step 5 in `resolve_concept()`.

---

*Ultimo aggiornamento: R29 — fix pre-eval codebase (9 fix); `hf_agent.py` eliminato;
`eval_cpu.ipynb` e `eval_gpu.ipynb` pronti in `eval/`.*
