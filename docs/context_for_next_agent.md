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

| Componente          | Dettaglio                                                   |
| ------------------- | ----------------------------------------------------------- |
| LLM locale          | Ollama (`granite4:3b-h`, `qwen2.5:3b`, `llama3.2:3b`) |
| LLM eval cluster    | HuggingFace Transformers (`HFAACAgent`)                   |
| Tool MCP            | FastMCP (`app/src/mcp_server/`)                           |
| NLP                 | spaCy `en_core_web_sm` (lemmatizzazione, POS, stop-word)  |
| Backend             | FastAPI + uvicorn su `:8000`                              |
| Frontend            | React + Vite su `:5173`                                   |
| Dataset pittogrammi | ARASAAC locale in `app/datasets/en/`                      |
| Dataset eval        | `hf_dataset_annotation/eval_final.parquet` (~54k righe)   |
| Lingua principale   | inglese (`LANG = "en"`)                                   |

---

## 3. Struttura del progetto

```
aac-mcp-agent/
  README.md
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
        __init__.py
      config.py             # costanti pure + re-export da settings.py e .env
      settings.py           # SettingsManager — legge/scrive user_settings.json
    frontend/
      src/
        components/
          PictogramGrid.jsx
          PictogramCard.jsx
          InputBar.jsx
          SessionSidebar.jsx
          SettingsPanel.jsx
          DatasetPanel.jsx   # modale aggiornamento dataset locale (R5)
        hooks/
          useAgent.js        # /run, /select, /reset, /session, /health
          useSettings.js     # GET/PATCH /settings
          useDatasets.js     # GET /datasets/status, POST /datasets/update (SSE) (R5)
        App.jsx
        main.jsx
      index.html
      vite.config.js
      package.json
    logs/
      logging_config.py
      __init__.py
      .gitkeep
      *.log               # gitignored
    credentials/          # gitignored
      credentials.json
      token.pickle
    .env                  # gitignored — credenziali live
    .env.example          # committato — template
    requirements.txt
    user_settings.json    # gitignored — generato al primo avvio
  eval/
    eval.ipynb            # notebook eval principale (unico entry point)
    eval_filtered.parquet # dataset filtrato (gold irrecuperabili rimossi)
    metric_evaluation.ipynb
    arasaac_vs_eval.ipynb
    cluster_work/
      run_eval_cluster.sh
      results/
  hf_dataset_annotation/  # DO NOT TOUCH — fase a monte completata
    eval_final.parquet
    eval_annotated.parquet
    eval_raw.parquet
  app/
    datasets/
      en/                 # dataset produzione (non toccare)
      en_eval/            # snapshot frozen per eval (merge local+HF)
      pictograms/
        {id}.png
      update_datasets.py
  docs/
    context_for_next_agent.md   # questo file
    consegna.md
  test/
    tools_test.ipynb
  .gitignore
  LICENSE
```

**Separazione dei valori di configurazione:**

| File                       | Contenuto                                              | Sensibile?    | Modificabile dal frontend? |
| -------------------------- | ------------------------------------------------------ | ------------- | -------------------------- |
| `app/src/config.py`      | Costanti pure (URL ARASAAC, slot orari, percorsi)      | No            | No                         |
| `app/src/settings.py`    | SettingsManager — legge/scrive `user_settings.json` | No            | **Sì**              |
| `app/user_settings.json` | Valori utente correnti (gitignored)                    | No            | **Sì**              |
| `app/.env`               | Credenziali (Apple, Google, HF)                        | **Sì** | No (manuale)               |

---

## 4. Come avviare il sistema

```bash
# Prerequisiti (una tantum)
pip install -r app/requirements.txt
python -m spacy download en_core_web_sm
ollama pull qwen2.5:3b   # modello default

# Backend
cd app
uvicorn src.api.server:app --reload --port 8000

# Frontend (altra finestra)
cd app/frontend
npm install          # prima volta
npm run dev          # Vite su http://localhost:5173
```

Il proxy Vite inoltra `/api/*` → `http://localhost:8000`.
Aprire `http://localhost:5173` nel browser.

---

## 5. Backend FastAPI — endpoint (`app/src/api/server.py`)

| Metodo    | Path                  | Descrizione                                                         |
| --------- | --------------------- | ------------------------------------------------------------------- |
| `POST`  | `/run`              | `{"text": str}` → lista pittogrammi + turn + tools_called        |
| `POST`  | `/select`           | `{"pictogram_id": int}` → aggiorna memoria sessione              |
| `POST`  | `/reset`            | svuota sessione                                                       |
| `GET`   | `/session`          | storia sessione corrente                                              |
| `GET`   | `/settings`         | legge `user_settings.json`                                          |
| `PATCH` | `/settings`         | `{"updates": {...}}` → aggiorna settings                         |
| `GET`   | `/health`           | `{"ok": true, "model": "...", "ollama": bool}`                  |
| `GET`   | `/images/{id}`      | serve PNG da dataset locale o CDN ARASAAC                            |
| `GET`   | `/datasets/status`  | metadata dataset per ogni lingua configurata + conteggio PNG cachati |
| `POST`  | `/datasets/update`  | `{langs?, force?, download_images?}` → SSE stream log lines        |
| `GET`   | `/categories`       | categorie ARASAAC raggruppate in macro-categorie con count           |
| `GET`   | `/by_category`      | pittogrammi di una data categoria                                    |

**`POST /select`** aggiorna l'ultimo turno in memoria: sostituisce `pictograms`
con il solo pittogramma scelto e ricalcola `topics`. `presented` rimane intatto
(usato da `recently_presented_ids()` per escludere l'intera finestra passata).

**Logica immagini:** `USE_LOCAL_DATASETS=True` → `/api/images/{id}` (proxy Vite → backend).
`USE_LOCAL_DATASETS=False` → URL CDN ARASAAC diretta. Definita in `_image_url()` in
`arasaac.py`, importata anche da `api/server.py`.

---

## 6. Frontend React

- Caregiver scrive → Enter o `→ Search` → spinner → griglia pittogrammi
- Soggetto tocca una card → POST `/select` → card appare nella Session bar → input svuotato
- `[↺]` → POST `/reset` → sessione azzerata
- `⚙ Settings` → modal con tutti i valori di `user_settings.json`
- `⬇ Datasets` → `DatasetPanel.jsx`: status dataset per lingua, SSE stream log update
- `🔍 Browse` → `CategoryBrowser.jsx`: navigazione pittogrammi per macro-categoria (3 livelli)
- Il dropdown modello in header chiama PATCH `/settings` → al turno successivo l'agente usa il nuovo modello

---

## 7. Metadata di un pittogramma ARASAAC

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

## 8. Pipeline agente (un LLM call per turno)

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
        ├─ escludi pittogrammi in recently_presented_ids() (intera finestra)
        ├─ ordina per (concept_order ASC, quality_score DESC)
        └─ riempi fino a max_results; se mancano fresh → padding con stale

  └─► add_turn() → memoria sessione aggiornata
```

**Una sola chiamata LLM per turno.** Nessun secondo LLM call — la selezione è
interamente deterministica dopo la fase 1.

---

## 9. Parametri funzionali

```python
AGENT_MAX_RESULTS         = 24   # dimensione finestra pittogrammi
AGENT_CANDIDATES_PER_TERM = 10   # candidati per keyword ARASAAC
AGENT_MEMORY_TURNS        = 3    # turni di storia in memoria
AGENT_DEFAULT_MODEL       = "qwen2.5:3b"   # default aggiornato in R11
AGENT_SYNSET_EXPAND       = True
AGENT_SYNSET_EXPAND_MAX   = 8
LANG                      = "en"
AGENT_FETCH_SCHEDULE      = True
```

---

## 10. Uso di spaCy nel progetto

spaCy (`en_core_web_sm`) è usato in tre punti:

1. **`resolve.py`** — `resolve_concept()`: lemmatizza il concetto per trovare la keyword ARASAAC.
2. **`session.py`** — `extract_topics()`: lemmatizza keyword dei pittogrammi selezionati, filtra stop-word.
3. **`agent.py`** — `_extract_terms()`: fallback quando il planner LLM fallisce; POS tagging per tenere solo NOUN/VERB/PROPN.

Non esiste più nessuna lista di stop-word hardcoded (`AGENT_STOPWORDS` rimossa in R3).

---

## 11. Tabella implementazioni

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
| R15 | ⚠️ Revisione critica — analisi HF coverage errata | — |
| R16 | Cleaning df_hf; numeri definitivi copertura gold; piano merge | `arasaac_vs_eval.ipynb` |
| R17 | Merge local+HF; creazione `en_eval/`; `--lang` in eval script | notebook, `settings.py` |
| R19 | Fix eval pipeline: `get_resolve_info` 3 valori; `hf_agent.py` prompt | `hf_agent.py`, `eval.ipynb` |
| R20 | Completamento fix eval.ipynb (celle 10, 18, 19) | `eval/eval.ipynb` |

---

## 12. Dubbi aperti — da chiarire con i tutor

**D1 — Tool-use: LLM decide o codice decide?**
Attualmente il planner LLM decide autonomamente se chiamare `get_time` / `get_schedule`.

**D2 — Una sola chiamata LLM per turno: sufficiente?**
Il vecchio design prevedeva un secondo LLM call (filter). Ora è tutto deterministico.

**D4 — Il comportamento tool-call entra nella valutazione formale?**
`tools_called` è esposto nell'API ma non è ancora una metrica di eval.

**D5 — Soglia di overlap per il successo semantico**
Qual è la soglia accettabile per `gold_in_candidates` e `gold_in_window`?

**D6 — Teacher forcing: metodologicamente accettabile?**
L'eval inietta `gold_id` come pittogramma selezionato per simulare turni multi-step.

**D7 — Quante righe sono sufficienti?**
Il dataset ha ~54k righe. Quante servono per risultati statisticamente significativi?

**D8 — Gold multipli per concetto?**
Attualmente ogni riga ha un solo gold ID. Avrebbe senso estendere a top-3?

---

## 13. Cose da NON fare

- **Non passare `concept` o `sentence` all'agente** durante l'eval
- **Non sovrascrivere parametri di `config.py` nel notebook**
- **Non usare MRR** come metrica primaria
- **Non valutare in "single-turn"** — ogni riga ha N pittogrammi gold
- **Non toccare `hf_dataset_annotation/`** — fase a monte completata
- **Non toccare `app/datasets/en_eval/`** — snapshot frozen per eval
- **Non mettere logica applicativa fuori da `app/`**

---

## 14. Strategia `resolve_concept` (`app/src/agent/resolve.py`)

| Step | Label | Esempio |
|---|---|---|
| 1 | `exact` | `"eat"` → `"eat"` |
| 2 | `lemma` | `"eating"` → `"eat"` via spaCy |
| 3 | `hyphen` | `"go out"` → `"go-out"` |
| 3b | `lemma_alt` | lemma della forma normalizzata |
| 4 | `token` | `"wash hands"` → `["wash", "hand"]` |
| — | `none` | nessun match → concetto saltato |

`resolve_concept(concept, kw_set, return_method=True)` ritorna `(queries, method)`.
La costante `RESOLVE_METHODS` elenca tutti i label nell'ordine canonico.

---

## 15. Eval — come funziona (`eval/eval.ipynb`)

Notebook self-contained per il cluster. Unico entry point per la valutazione.

Carica `eval_filtered.parquet`, esegue `agent.run(raw_input)` su ogni riga, misura:
- `gold_in_candidates` — il gold ID era nel pool prima del ranking?
- `hit` (`gold_in_window`) — il gold ID è nella finestra finale?
- `overlap_level` — livello semantico migliore (`synset` > `category` > `keyword` > `tag`)

Teacher forcing: dopo ogni turno inietta il gold come selezione per simulare multi-turn.

**Colonne CSV di output:**

| Colonna | Tipo | Descrizione |
|---|---|---|
| `model` | str | nome modello HF |
| `row_idx` | int | indice riga dataset |
| `split` | str | `clear` o `vague` |
| `turn_pos` | int | posizione turno (0-based) |
| `n_turns_total` | int | lunghezza sequenza |
| `caregiver_input` | str | input caregiver (vuoto per turni > 0) |
| `concept` | str | concetto gold del turno |
| `gold_id` | int | ID pittogramma gold |
| `all_gold_ids` | list | tutti i gold della sequenza |
| `predicted_ids` | list | IDs finestra prodotta dall'agente |
| `gold_in_candidates` | bool | gold nel pool pre-ranking |
| `hit` | bool | gold nella finestra finale |
| `n_candidates` | int | dimensione pool candidati |
| `window_len` | int | dimensione finestra |
| `called_get_time` | bool | planner ha chiamato get_time |
| `called_get_schedule` | bool | planner ha chiamato get_schedule |
| `overlap_level` | str\|None | miglior overlap semantico |
| `resolve_method` | str | step usato da resolve_concept |
| `resolve_queries` | list | keyword passate a search_pictograms |
| `plan_method` | str | `llm` / `fallback_spacy` / `fallback_empty` |
| `synset_added` | int | pittogrammi aggiunti da synset expansion |
| `fresh_count` | int | pittogrammi fresh nella finestra |
| `planner_had_gold_concept` | bool | planner ha generato esattamente il gold concept (aggiunto R20) |
| `input_triggered_tools` | bool | True solo a turn_pos==0 (aggiunto R20) |

**Uso corretto delle nuove colonne (R20):**
```python
# resolve_method significativo: solo dove il planner mirava al gold
df[df['planner_had_gold_concept']]['resolve_method'].value_counts()

# tool call su input reale vs autonoma dalla history
df[df['input_triggered_tools']]['called_get_time'].mean()   # caregiver → tool
df[~df['input_triggered_tools']]['called_get_time'].mean()  # autonomo
```

---

## 16. Infrastruttura eval: cluster vs Colab

### Cluster (SLURM)

```bash
git pull
NB_MODELS="Qwen/Qwen2.5-3B-Instruct" NB_N_ROWS=500 sbatch eval/cluster_work/run_eval_cluster.sh
```

Output: `eval/results/eval_hf_<JOB_ID>.csv`

### Colab

```python
os.environ["NB_IS_COLAB"] = "True"
os.environ["NB_MODELS"]   = "Qwen/Qwen2.5-3B-Instruct"
os.environ["NB_N_ROWS"]   = "200"
os.environ["NB_LANG"]     = "en_eval"
os.environ["NB_SPLIT"]    = "both"
```

| | Cluster | Colab |
|---|---|---|
| GPU | L40 (40 GB) | T4 (15 GB) o A100 Pro |
| Dataset completo | ✅ | ⚠️ subsample consigliato |
| Persistenza | Scratch HPC | nessuna |

---

## 17. Possibili prossimi passi

- **Sezione 5 `arasaac_vs_eval.ipynb`**: filtrare `eval_final.parquet` rimuovendo le sequenze con gold ID irrecuperabili (924 ID deprecati da ARASAAC). Produrre `eval_filtered.parquet`.
- **Free-run eval**: modalità senza teacher forcing.
- **Gold multipli**: estendere dataset con top-3 IDs per concetto.
- **Retrieval BM25/embedding**: alternative al planner LLM per la selezione delle keyword.
- **Produzione offline**: pre-scaricare PNG con `DatasetPanel` + `download_images=true`.

---

## 18. Teacher forcing e ricerca manuale dei pittogrammi

L'eval usa **teacher forcing**: se il `gold_id` non è nella finestra, si assume che il
caregiver lo trovi manualmente — rendendo la sequenza multi-turn deterministica.
**La funzionalità non esiste nell'app** (mancanza da segnalare nella tesi come sviluppo futuro).

Nell'app attuale il caregiver può solo riscrivere il testo nell'InputBar o usare
il `CategoryBrowser` per navigare per categoria. L'endpoint `GET /search?keyword=...`
esiste già nel backend ma non ha un'interfaccia frontend dedicata.

---

## 19. Ricerca manuale per categoria

`CategoryBrowser.jsx` implementa la navigazione a 3 livelli:
- Livello 0: ~15 macro-categorie (emoji + immagine + conteggio)
- Livello 1: sottocategorie ARASAAC della macro selezionata
- Livello 2: `PictogramGrid` della categoria specifica

Backend: `GET /categories` e `GET /by_category` in `api/server.py`.
`MACRO_CATEGORIES` è definita nel codice (non nei dati) e include tutte le
categorie biologiche animali (`mammal`, `viviparous`, ecc.) dalla R9.
Logica conteggio: set di ID unici per macro (nessun doppio conteggio);
"Other" contiene solo pittogrammi non coperti da nessuna macro nominata.

---

## 20. Note non banali sul dataset ARASAAC

- **13.780 pittogrammi** (non 15.757 — quello è il numero di keyword uniche).
- **567 categorie** ARASAAC sono tag piatti, non una gerarchia; nomi sempre in inglese.
- **~40% dei pittogrammi** non ha synsets → non espandibili via WordNet.
- **`aac: true`** indica pittogrammi validati per core vocabulary (~60% del totale).
- **`update_datasets.py`** usa l'endpoint bulk `/pictograms/all/{lang}` (non le MCP tools).
- **`en_eval/`** è uno snapshot frozen: 13.804 pittogrammi (merge local + 24 only-HF clean).
  Non viene toccato da `update_datasets.py`. I 24 only-HF hanno `synsets=[]`, `aac=False`.

---

## 21. Cosa è stato fatto in R6

- Bigrammi in `_terms_from_schedule` (compound ARASAAC labels → `exact`/`lemma` match)
- Fix path setup `test/tools_test.ipynb` (`SRC = PROJECT_ROOT / 'app' / 'src'`)
- Fix docstring colonne CSV in `run_eval_hf.py`
- Test `return_method=True` aggiunto a `tools_test.ipynb`
- Review completa codebase — nessun problema trovato

---

## 22. Cosa è stato fatto in R7

- Creato `app/frontend/src/components/CategoryBrowser.jsx` (3 livelli di navigazione)
- Modificato `App.jsx`: stato `showCategories`, bottone `🔍 Cerca` in header, rendering condizionale

---

## 23. Cosa è stato fatto in R8

- `CategoryBrowser.jsx`: testi UI in inglese
- `useAgent.js`: helper `_errorMessage` per errori FastAPI leggibili
- `api/server.py` `/categories`: riscritta logica con set ID unici; "Other" solo uncovered

---

## 24. Cosa è stato fatto in R9

- `MACRO_CATEGORIES["Animals"]` estesa con 8 categorie biologiche (`mammal`, `viviparous`, `herbivorous`, `omnivorous`, `carnivorous`, `oviparous`, `invertebrate`, `arachnid`)

---

## 25. Cosa è stato fatto in R10

- Diagnostica lentezza: `granite4:3b-h` non ha reasoning attivo di default (il `-h` è hybrid mamba-2)
- `settings.py`: aggiunto `num_predict=150` per tutti i modelli; modello default → `qwen2.5:3b`
- `agent.py._plan()`: options dinamiche per modello; timing con `perf_counter()`
- Roadmap con tutor: Step1 diagnostica → Step2 backend astratto → Step3 BM25/embedding

---

## 26. Cosa è stato fatto in R11

- Fix errata `has_thinking`: rimossa logica `think=False` (non necessaria)
- `settings.py`: corretti `size_gb`; `num_ctx` 2048 → 512
- Prompt split FULL/SHORT: `build_planner_prompt(*, full=False)` (SHORT di default)
- Nuovo file `app/src/agent/backends.py`: `OllamaBackend`, `LlamaCppBackend`, `HuggingFaceBackend`

---

## 27. Cosa è stato fatto in R12

- Rimosso cap `4-8` sui concepts in entrambi i prompt FULL e SHORT
- Aggiunta regola espansione semantica esplicita + few-shot inline nel SHORT
- Invariante: il LLM genera liberamente, `_rank_and_fill` taglia a `max_results`

---

## 28. Cosa è stato fatto in R14

- `InputBar.jsx`: prop `warmingUp`; textarea gialla durante warmup; bottone `⏳`
- GGUF scaricati in `app/models/` (Q4_K_M: Qwen2.5-3B, Llama-3.2-3B, Granite-4.1-3B, Mistral-7B)
- `settings.py`: sezione `gguf_models` (alias → path GGUF)
- `agent.py`: accetta `backend: Optional[LLMBackend] = None`; usa backend se presente, altrimenti Ollama

---

## 29. Cosa è stato fatto in R15 — REVISIONE CRITICA

⚠️ Le conclusioni di R15 erano errate: la copertura HF sembrava 100% perché
includeva 1.817 righe con `keywords = categories = tags = None`. Dopo cleaning
corretto, copertura HF = 86.3% — non meglio di local.
`eval/build_eval_dataset.ipynb` è obsoleto.

---

## 30. Cosa è stato fatto in R16

- Cleaning `df_hf`: rimossi 10 duplicati + 1.817 righe None → `df_hf_clean` = 10.657 righe
- Numeri definitivi copertura gold: local 86.1%, HF 86.3%, union 86.3%
- 924 gold ID irrecuperabili (deprecati da ARASAAC — non esistono più online)
- Piano merge: local vince nell'overlap (ha synsets, type, plural, più recente)

---

## 31. Analisi `arasaac_vs_eval.ipynb` §3

Stato finale sezioni:
- §3.1: ID partitioning, gold coverage, None rows anatomy ✅
- §3.2: deeper checks ✅
- §3.3: Jaccard keyword agreement + visualizzazione HTML ✅
- §4: merge + scrittura `en_eval/` ✅
- §5: filtro `eval_final.parquet` → `eval_filtered.parquet` ⬜ **da fare**

Conclusione §3.3: local vince sempre nell'overlap (Jaccard basso = granularità diversa, stesso pittogramma).
Switch `FIELD_SOURCE` per ablazioni future su `keywords`/`categories`/`tags`.

---

## 32. Cosa è stato fatto in R17

- Merge local + HF clean eseguito (§4 notebook): 13.804 pittogrammi
- Dataset `app/datasets/en_eval/` creato (5 file: `pictograms.json`, `keyword_index.json`, `keywords.json`, `synset_index.json`, `_meta.json`)
- `settings.py`: aggiunto `"en_eval"` a `dataset_langs`
- `run_eval_hf.py`: aggiunto `--lang` (default `"en_eval"`); `_eval_lang` modulo-level

---

## 33. Cosa è stato fatto in R19

**Fix 1 — `get_resolve_info` gonfiava `none_pct`:**
Per `turn_pos > 0` il planner non genera il gold concept → `get_resolve_info` restituiva
sempre `('none', [])`. Fix: 3° valore di ritorno `planner_had_gold` (bool).
- `planner_had_gold=True`: resolve_method è significativo
- `planner_had_gold=False`: `none` è atteso, non è un fallimento

**Fix 2 — `called_get_time`/`called_get_schedule` fuorvianti:**
Aggiunta colonna `input_triggered_tools` (True solo a `turn_pos==0`).

**Fix 3 — `hf_agent.py`:**
`build_planner_prompt(full=False)` esplicito (evita cambio silenzioso se il default cambia).

File toccati: `app/src/agent/hf_agent.py` ✅ (applicato); `eval/eval.ipynb` ⬜ (applicato in R20).

---

## 34. Cosa è stato fatto in R20

Completamento fix `eval/eval.ipynb` (i fix R19 erano stati descritti ma non applicati al notebook):

| Cella | Modifica |
|---|---|
| 10 (`CSV_COLUMNS`) | Aggiunte `planner_had_gold_concept` e `input_triggered_tools` |
| 18 (`get_resolve_info`) | Firma estesa a `tuple[str, list[str], bool]`; docstring dettagliata |
| 19 (`run_multi_turn`) | Unpacking 3 valori; 2 nuovi campi nel dict risultato |

File consegnato: `eval_patched.ipynb` → rinominare `eval.ipynb`.

**Stato finale di tutti i fix:**

| Fix | File | Stato |
|---|---|---|
| Bug #6 (get_resolve_info → 3 valori) | `eval/eval.ipynb` celle 10, 18, 19 | ✅ |
| Bug #5 (input_triggered_tools) | `eval/eval.ipynb` celle 10, 19 | ✅ |
| Log #2 (build_planner_prompt esplicito) | `app/src/agent/hf_agent.py` | ✅ |

---

*Ultimo aggiornamento: maggio 2026 — R20: fix eval pipeline completato (`eval.ipynb` aggiornato, colonne `planner_had_gold_concept` e `input_triggered_tools` aggiunte al CSV).*
