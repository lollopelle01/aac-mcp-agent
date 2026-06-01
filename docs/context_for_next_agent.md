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
   pittogrammi (esclusi quelli già mostrati nei turni recenti)
8. Il soggetto seleziona un pittogramma dalla finestra → `/select` aggiorna la
   memoria di sessione; si riparte dal punto 1 per il concetto successivo

**Vincoli hardware:** il sistema deve girare su CPU su un tablet/iPad.
La leggerezza è un requisito funzionale. Le GPU sono rilevanti solo per la
fase di valutazione sul cluster.

---

## 2. Stack tecnologico

| Componente            | Dettaglio                                                     |
| --------------------- | ------------------------------------------------------------- |
| LLM locale            | Ollama (`granite4:3b-h`, `qwen2.5:3b`, `llama3.2:3b`)       |
| LLM eval cluster      | HuggingFace Transformers (`HFAACAgent`)                       |
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
        hf_agent.py     # HFAACAgent — sottoclasse per cluster HuggingFace
        prompts.py      # build_planner_prompt, build_planner_message
        session.py      # SessionMemory, Turn, _nlp() (spaCy loader)
        resolve.py      # resolve_concept — concept → keyword ARASAAC
      mcp_server/
        tools/
          arasaac.py        # search_pictograms, search_pictograms_by_synset,
                            # get_pictogram_metadata, get_pictogram_image,
                            # list_keywords
          time_tool.py      # get_time()
          schedule_tool.py  # get_schedule()
        models.py           # Pictogram, TimeInfo, ScheduleEvent, ...
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
  annotation/
    arasaac_vs_hf_vs_eval.ipynb   # notebook analisi e produzione dataset eval (R26)
    eval_filtered.parquet         # dataset eval primario (1760 frasi) — prodotto da R26
  eval/
    eval.ipynb              # notebook eval principale (unico entry point)
    eval_filtered.parquet   # dataset eval LEGACY (~54k righe) — NON USARE per nuove run
    cluster_work/
      run_eval_cluster.sh
      results/
  hf_dataset_annotation/   # DO NOT TOUCH — fase a monte completata
  docs/
    context_for_next_agent.md
    consegna.md
  test/
    tools_test.ipynb
```

---

## 4. Come avviare il sistema

```bash
# Prerequisiti (una tantum)
pip install -r app/requirements.txt
python -m spacy download en_core_web_sm
ollama pull qwen2.5:3b

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
| `PATCH` | `/settings`        | `{"updates": {...}}` → aggiorna settings                           |
| `GET`   | `/health`          | `{"ok": true, "model": "...", "ollama": bool}`                     |
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
        └─ riempi fino a max_results; se mancano fresh → padding con stale

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

---

## 10. Dataset per la valutazione — stato attuale (R26)

### Dataset primario: `annotation/eval_filtered.parquet`

Prodotto da `annotation/arasaac_vs_hf_vs_eval.ipynb` (R26). È il dataset corretto
da usare per tutte le eval future. Struttura sentence-based: ogni riga è una frase
con lista di `concepts` (concept_text, gold_id, candidate_ids).

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
- Nessun duplicato esatto (18 gruppi ambigui tenuti — gold diversi sullo stesso pool, interpretazioni legittime)
- Gold non ripetuto nella stessa frase (0 casi)
- Candidati sempre esattamente 10
- Concetti per frase: mean=3.68, std=1.05, min=2, max=7

### Dataset sorgente pittogrammi: `app/datasets/en_eval/`

Snapshot frozen prodotto da R26. **Non toccare.**

| File                 | Contenuto                                  |
|----------------------|--------------------------------------------|
| `pictograms.json`    | 13.804 pittogrammi (`{ id_str → record }`) |
| `keywords.json`      | 15.761 keyword uniche                      |
| `keyword_index.json` | 15.761 entries                             |
| `synset_index.json`  | 8.422 synset                               |
| `_meta.json`         | timestamp e conteggi                       |

**Come è stato costruito:**
- Base: `df_local` (13.780 pittogrammi, dataset ARASAAC ufficiale locale)
- Aggiunta: 24 ID presenti solo in `df_hf` valido, normalizzati allo schema local
- I 24 only-HF hanno `synsets=[]`, `type=None`, `plural=None`
- 1.807 ID con metadata None in `df_hf` (`none_ids`) esclusi

### Perché il nuovo dataset è un upgrade rispetto al legacy

| Aspetto | Legacy (`eval/eval_filtered.parquet`) | Nuovo (`annotation/eval_filtered.parquet`) |
|---|---|---|
| Struttura gold | ID singolo per concept, spesso deprecato | Gold sempre tra i 10 candidati (100%) |
| Concetti | Caption fragments (es. "a football match") | Parole/frasi brevi normalizzate (es. "go", "train") |
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
- 10 ID duplicati (38264–38273): ogni duplicato aveva una copia con metadata e una None → tenuta la copia non-None
- 1.817 righe con keywords/categories/tags tutti None → `none_ids` (1.807 dopo drop duplicati)
- Solo 1 ID su 1.807 `none_ids` presente anche in `df_local`
- Schema keyword diverso da local: `{hasLocution, keyword, meaning}` invece di `{type, keyword, plural, meaning}`

### Overlap `df_local` vs `df_hf` valido
- In entrambi: 10.633 (77.0% dell'union)
- Solo in local: 3.147 (22.8%)
- Solo in HF: 24 (0.2%)
- Similarity score (keywords×0.6 + categories×0.2 + tags×0.2): mean=0.985, 92% identici
- Merge strategy: local vince sempre nell'overlap (schema più ricco: synsets, type, plural)

### Copertura eval gold in en_eval
- Gold ids in en_eval: 1.023/1.026 (99.7%)
- I 3 restanti sono only-HF e quindi in en_eval
- 214 candidate IDs irrecuperabili (none_ids) — accettati perché <20% per concetto

---

## 12. Eval — come funziona (`eval/eval.ipynb`)

Notebook self-contained per il cluster. Unico entry point per la valutazione.

Carica `annotation/eval_filtered.parquet`, raggruppa per `sentence` (ogni frase = una sessione multi-turn con N concepts), esegue `agent.run(sentence)` sulla frase completa al primo turno poi usa teacher forcing per i turni successivi.

**Metriche principali:**
- `gold_in_candidates` — il gold ID era nel pool prima del ranking?
- `hit` (`gold_in_window`) — il gold ID è nella finestra finale?
- `overlap_level` — livello semantico migliore (`synset` > `category` > `keyword` > `tag`)

**Colonne CSV di output principali:**

| Colonna | Descrizione |
|---|---|
| `model` | nome modello HF |
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

## 13. Infrastruttura eval: cluster vs Colab

### Cluster (SLURM)

```bash
NB_MODELS="Qwen/Qwen2.5-3B-Instruct" NB_N_ROWS=500 sbatch eval/cluster_work/run_eval_cluster.sh
```

### Colab

```python
os.environ["NB_IS_COLAB"]    = "True"
os.environ["NB_MODELS"]      = "Qwen/Qwen2.5-3B-Instruct"
os.environ["NB_N_ROWS"]      = "200"
os.environ["NB_LANG"]        = "en_eval"
os.environ["NB_MAX_RESULTS"] = "0"   # 0 = usa default settings.py (25)
```

---

## 14. Risultati prima eval run (R21) — su dataset LEGACY

⚠️ Questi risultati sono sul dataset legacy, non comparabili con run future.

| Metrica | Valore |
|---|---|
| Hit@window | **22.0%** |
| gold_in_candidates | 23.5% |
| resolve_method=none | **88.3%** |
| planner_had_gold_concept | 13.7% |
| Hit quando planner_had_gold_concept=True | 33.8% |

**Diagnosi:** bottleneck primario = planner genera concetti non allineati al vocabolario ARASAAC (vocabolario legacy molto distante da ARASAAC). Il nuovo dataset riduce questo problema strutturalmente.

---

## 15. Piano miglioramenti score

### Strategia ibrida raccomandata (3 strati)

| Strato | Implementazione | Impatto stimato |
|---|---|---|
| **1. CONCEPT_MAP** | Dict `{"groceries":"shopping", ...}` in `resolve.py` step 4b | Top-50 casi OOV frequenti |
| **2. Embedding** | `all-MiniLM-L6-v2` + `.npy` cached, step 5 in resolve | Casi OOV long-tail (~2-3pp) |
| **3. Prompt few-shot** | 10-15 coppie nel `_PLANNER_SYSTEM_PROMPT_SHORT` | Previene OOV a monte |

**P4:** aumentare `AGENT_CANDIDATES_PER_TERM` da 10 a 15-20 (+1-3pp stimati, zero codice).

---

## 16. Dubbi aperti — da chiarire con i tutor

**D1 — Tool-use: LLM decide o codice decide?**

**D2 — Una sola chiamata LLM per turno: sufficiente?**

**D5 — Soglia di overlap per il successo semantico**

**D6 — Teacher forcing: metodologicamente accettabile?**

**D7 — Quante righe sono sufficienti?** Il dataset filtrato ha 1.760 frasi.

**D8 — Gold multipli per concetto?** I 18 gruppi ambigui mostrano che esistono interpretazioni alternative legittime — estendere a top-3?

---

## 17. Cose da NON fare

- **Non passare `concept` o `sentence` all'agente** durante l'eval
- **Non sovrascrivere parametri di `config.py` nel notebook**
- **Non usare MRR** come metrica primaria
- **Non toccare `hf_dataset_annotation/`** — fase a monte completata
- **Non toccare `app/datasets/en_eval/`** — snapshot frozen per eval
- **Non toccare `app/datasets/en/`** — dataset produzione
- **Non usare `eval/eval_filtered.parquet`** per nuove run — è il dataset legacy
- **Non mettere logica applicativa fuori da `app/`**

---

## 18. Tabella implementazioni

| ID  | Descrizione | File principali |
|---|---|---|
| R3  | Rimosso filter LLM morto; `_rank_and_fill` usa `recently_presented_ids()` | `agent.py` |
| R4  | `datasets/` → `app/datasets/`; `_image_url` offline-aware | `config.py`, `arasaac.py` |
| R5  | Endpoint SSE dataset update; UI DatasetPanel; tracing resolve | `api/server.py`, `frontend/` |
| R6  | Fix path setup eval; bigrammi schedule; test `return_method` | `test/`, `agent.py` |
| R7  | `CategoryBrowser.jsx` — ricerca per categoria (frontend) | `frontend/` |
| R8  | Fix `/categories` conteggi; UI inglese; `_errorMessage` hook | `api/server.py`, `frontend/` |
| R9  | Aggiunte categorie biologiche animali a `MACRO_CATEGORIES` | `api/server.py` |
| R10 | Diagnostica latenza; `num_predict`; modello default → `qwen2.5:3b` | `settings.py`, `agent.py` |
| R11 | Fix errata `has_thinking`; prompt SHORT/FULL; `backends.py` | `prompts.py`, `backends.py` |
| R12 | Rimosso cap concepts nel prompt; few-shot espansione semantica | `prompts.py` |
| R14 | InputBar warmup; GGUF scaricati; `LlamaCppBackend` agganciato | `frontend/`, `agent.py` |
| R19 | Fix eval pipeline: `get_resolve_info` 3 valori; `hf_agent.py` prompt | `hf_agent.py`, `eval.ipynb` |
| R20 | Completamento fix eval.ipynb (celle 10, 18, 19) | `eval/eval.ipynb` |
| R21 | Prima eval run Colab (200 righe, T4); diagnosi bottleneck resolve | `eval/results/` |
| R22 | Opzione A exclusion + window 25 + `NB_MAX_RESULTS` | `agent.py`, `settings.py`, `eval.ipynb` |
| R23 | Fix article stripping in `resolve.py` | `resolve.py` |
| R24 | Analisi embedding 7-punti; strategia ibrida 3-strati | `docs/` |
| R26 | Analisi dataset, cleaning, merge, produzione `en_eval/` e `annotation/eval_filtered.parquet` | `annotation/` |

---

## 19. Prossimi passi

1. **Prima eval run sul nuovo dataset** — adattare loader in `eval.ipynb` per struttura sentence-based (`annotation/eval_filtered.parquet`): raggruppare per `sentence`, teacher forcing per concept, colonna `concept` = `concept_text`
2. **Implementare CONCEPT_MAP** (step 4b in `resolve.py`) con top-50 coppie da R21 CSV
3. **Aggiungere colonna `split`** (`clear`/`vague`) alle 1.760 frasi via LLM in batch
4. **Embedding fallback** (`all-MiniLM-L6-v2`) come step 5 in `resolve_concept()`
5. **Confronto con R21**: il nuovo dataset dovrebbe ridurre resolve=none da ~88% a ~12%

---

*Ultimo aggiornamento: R26 — analisi completa dataset, cleaning df_hf, merge local+HF, produzione `en_eval/` (13.804 pittogrammi) e `annotation/eval_filtered.parquet` (1.760 frasi). Il dataset legacy `eval/eval_filtered.parquet` non va più usato per nuove run.*
