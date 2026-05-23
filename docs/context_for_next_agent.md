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
    agent_eval.ipynb
    cluster_work/
      run_eval_hf.py      # unico entry point eval (self-contained)
      run_eval_cluster.sh
      results/
  hf_dataset_annotation/  # DO NOT TOUCH — fase a monte completata
    eval_final.parquet
    eval_annotated.parquet
    eval_raw.parquet
    annotated_dataset_eval.ipynb
    cluster_work/
  datasets/               # dentro app/ — parte operativa dell'app
    en/
      pictograms.json
      keyword_index.json
      synset_index.json
      keywords.json
    pictograms/
      {id}.png
    update_datasets.py    # script per aggiornare i dataset locali
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

**Percorsi chiave in `config.py`:**

```python
_SRC  = Path(__file__).resolve().parent   # app/src/
_APP  = _SRC.parent                       # app/
_ROOT = _APP.parent                       # <project_root>/

load_dotenv(_APP / ".env")
_CREDENTIALS_DIR = _APP / "credentials"   # app/credentials/
DATASETS_DIR     = _APP / "datasets"     # app/datasets/
```

---

## 4. Come avviare il sistema

```bash
# Prerequisiti (una tantum)
pip install -r app/requirements.txt
python -m spacy download en_core_web_sm
ollama pull granite4:3b-h

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

**`POST /run` risposta:**

```json
{
  "pictograms": [
    {"id": 2248, "image_url": "https://static.arasaac.org/...",
     "label": "water", "categories": ["beverage"], "aac": true}
  ],
  "turn": 1,
  "tools_called": false
}
```

**`POST /select`** aggiorna l'ultimo turno in memoria: sostituisce `pictograms`
con il solo pittogramma scelto e ricalcola `topics`. `presented` rimane intatto
(usato da `recently_presented_ids()` per escludere l'intera finestra passata).

**Logica immagini (R4):**

- `_image_url(pictogram_id)` in `arasaac.py` — unica definizione, importata
  anche da `api/server.py`:
  - `USE_LOCAL_DATASETS=True` → `/api/images/{id}` (Vite proxy → backend)
  - `USE_LOCAL_DATASETS=False` → URL CDN ARASAAC diretta
- `get_pictogram_image(pictogram_id)` in `arasaac.py` — chiamato da `GET /images/{id}`:
  - `USE_LOCAL_DATASETS=True` → legge `app/datasets/pictograms/{id}.png`
    (download on-demand se mancante)
  - `USE_LOCAL_DATASETS=False` → GET CDN ARASAAC
  - usa `ARASAAC_IMG_PATTERN` direttamente (non `_image_url`) per evitare loop

**`POST /datasets/update` — SSE stream:**
Richiesta: `{"langs": ["en"], "force": false, "download_images": false}`
Eventi emessi:
```
data: {"type": "log",  "msg": "INFO Wrote pictograms[en] ..."}
data: {"type": "done", "ok": true}
```
- Gira in un thread background, un solo job alla volta (`_update_lock`).
- Al termine invalida `_DatasetCache` così la query successiva usa i dati freschi.
- Se un job è già in corso risponde 409.
- `download_images=true` scarica i PNG mancanti (lento, ~50k file).

`update_datasets.py` non usa le funzioni MCP di `arasaac.py` — usa
diretti la `GET /pictograms/all/{lang}` (endpoint bulk, nessun corrispettivo
in `arasaac.py`). Le costanti comuni (`ARASAAC_API_BASE`, `ARASAAC_IMG_PATTERN`,
`ARASAAC_TIMEOUT`) vengono da `config.py` — nessuna duplicazione.

---

## 6. Frontend React

```
┌───────────────────────────────────────────────────────────┐
│  AAC Pictogram Agent  [granite4:3b-h ▾]  [⚙ Settings] [↺] │
├───────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐  │
│  │ He wants something to eat before leaving            │  │
│  └─────────────────────────────────────────────────────┘  │
│                                              [→ Search]   │
├───────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
│  │  [img]  │  │  [img]  │  │  [img]  │  ← tap → select    │
│  │  apple  │  │ yogurt  │  │  snack  │                    │
│  └─────────┘  └─────────┘  └─────────┘                    │
├───────────────────────────────────────────────────────────┤
│  Session:  [🍎] [🧥] [👟]                                  │
└───────────────────────────────────────────────────────────┘
```

- Caregiver scrive → Enter o `→ Search` → spinner → griglia pittogrammi
- Soggetto tocca una card → POST `/select` → card appare nella Session bar
  → input svuotato per il concetto successivo
- `[↺]` → POST `/reset` → sessione azzerata
- Settings panel: modal con tutti i valori di `user_settings.json`
  (le credenziali sensibili si editano solo in `app/.env`)
- Dataset panel (`DatasetPanel.jsx`): modale raggiungibile dal pulsante
  `⬇ Datasets` in header:
  - mostra stato corrente per ogni lingua (record count, ultimo aggiornamento)
    e numero PNG cachati
  - opzioni: force re-fetch, download immagini
  - pulsante **Update now** → SSE stream con log in tempo reale
  - al termine lo stato si aggiorna automaticamente
- Il dropdown modello in header riflette `health.model`; cambiarlo fa
  PATCH `/settings` → al turno successivo `_get_agent()` ricostruisce l'agente

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

**Tipi di keyword:** 1=nomi propri, 2=nomi comuni, 3=verbi, 4=aggettivi,
5=sociale, 6=misc.

**Nota:** circa il 40% dei pittogrammi non ha synsets.

---

## 8. Pipeline agente (un LLM call per turno)

```
input caregiver
  └─► _plan() — LLM planner
        ├─ call_tools=true  → _collect_context() → get_time, get_schedule
        └─ call_tools=false → time_of_day=None

  └─► se planner fallisce o concepts=[] → fallback _extract_terms() [spaCy]
        └─► _enrich_terms() — aggiunge topic recenti dalla memoria

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

**Una sola chiamata LLM per turno** (il planner). Non c'è un secondo LLM call
di filtraggio — la selezione è interamente deterministica dopo la fase 1.

---

## 9. Parametri funzionali

Tutti configurabili tramite `user_settings.json` / Settings panel.
I valori default sono in `app/src/settings.py`.

```python
AGENT_MAX_RESULTS         = 24   # dimensione finestra pittogrammi
AGENT_CANDIDATES_PER_TERM = 10   # candidati per keyword ARASAAC
AGENT_MEMORY_TURNS        = 3    # turni di storia in memoria
AGENT_DEFAULT_MODEL       = "granite4:3b-h"
AGENT_SYNSET_EXPAND       = True
AGENT_SYNSET_EXPAND_MAX   = 8    # max synset da esplorare per turno
LANG                      = "en"
AGENT_FETCH_SCHEDULE      = True  # abilitato di default
```

> I valori di `config.py` sono snapshot al momento dell'import (modulo caricato
> una volta sola). Le modifiche live passano per `settings` e si riflettono
> all'import successivo (restart backend) oppure tramite accesso diretto a
> `settings.<campo>` nel codice che ne ha bisogno in tempo reale.

---

## 10. Uso di spaCy nel progetto

spaCy (`en_core_web_sm`) è usato in **due punti**:

1. **`app/src/agent/resolve.py`** — `resolve_concept()`: lemmatizza il concetto
   per trovare la keyword ARASAAC più vicina (exact match, poi fallback token).
2. **`app/src/agent/session.py`** — `_nlp()` (lazy, `lru_cache`):

   - `extract_topics()`: lemmatizza le keyword dei pittogrammi selezionati,
     filtra le stop-word di spaCy → salva in memoria semantica
   - importato anche da `agent.py` per `_extract_terms()` (fallback regex-free)
3. **`app/src/agent/agent.py`** — `_extract_terms()`: fallback quando il planner
   LLM fallisce; usa spaCy POS tagging per tenere solo `NOUN`, `VERB`, `PROPN`,
   lemmatizzati e filtrati dalle stop-word spaCy.

Non esiste più nessuna lista di stop-word hardcoded (`AGENT_STOPWORDS` rimossa
da `config.py` in R3).

**Prerequisito:** `python -m spacy download en_core_web_sm`
(già dichiarato in `requirements.txt` con commento).

---

## 11. Tabella implementazioni

| ID      | Descrizione                                                                             | File principali                                                |
| ------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Impl 1  | `prompt_summary` semantico                                                            | `session.py`                                                 |
| Impl 2  | `resolve_concept` con lemmatizzazione spaCy                                           | `resolve.py`, `agent.py`                                   |
| Impl 3  | `gold_in_candidates` + `last_candidates` per eval                                   | `agent.py`, `eval/`                                        |
| Impl 4  | `_plan` + `_parse_planner_response`                                                 | `agent.py`                                                   |
| Impl 5  | Metriche overlap semantico + mock temporale variabile                                   | `eval/`                                                      |
| Impl 6  | `_expand_pool_by_synset` (pool WordNet)                                               | `agent.py`                                                   |
| Impl 7  | Rimozione `search_pictograms_advanced`                                                | `mcp_server/tools/arasaac.py`                                |
| Impl 8  | Categorie nel prompt candidati                                                          | `prompts.py`                                                 |
| Impl 9  | Script CLI `run_eval.py`                                                              | `eval/run_eval.py`                                           |
| Impl 10 | `HFAACAgent` + infrastruttura eval cluster                                            | `hf_agent.py`, `eval/cluster_work/`                        |
| Impl 11 | Refactoring config/settings/env                                                         | `config.py`, `settings.py`, `app/.env`                   |
| Impl 12 | FastAPI backend + React frontend                                                        | `api/server.py`, `frontend/`                               |
| Impl 13 | README                                                                                  | `README.md`                                                  |
| R1      | Ristrutturazione `app/` (src/, logs/, frontend/, credentials/)                        | struttura intera                                               |
| R2      | Consolidamento eval:`run_eval.py` → `run_eval_hf.py`; rimossi config/settings root | `eval/cluster_work/`                                         |
| R4      | `datasets/` → `app/datasets/`, `_image_url` offline-aware, endpoint `/images` corretto | `config.py`, `arasaac.py`, `api/server.py` |
| R5      | Endpoint `/datasets/status` + `/datasets/update` SSE; UI `DatasetPanel`; tracing procedurale eval | `api/server.py`, `frontend/`, `resolve.py`, `agent.py`, `run_eval_hf.py` |

**Cosa è stato fatto in R3:**

- Rimosso tutto il blocco filter morto: `_llm_filter`, `_parse_id_list`,
  `_FILTER_SYSTEM_PROMPT`, `build_system_prompt`, `build_user_message`,
  `_format_pictogram` (erano irraggiungibili da qualsiasi call path)
- `_rank_and_fill` ora usa `recently_presented_ids()` (intera finestra passata)
  invece di `recent_pictogram_ids()` (solo selezionati) — semantica corretta
- `ContextBundle` rimosso da `_collect_context`: il return è ora solo
  `time_of_day`, senza costruire e scartare inutilmente il bundle
- `AGENT_STOPWORDS` rimossa da `config.py`; sostituita con lemmatizzazione
  e filtro stop-word nativo spaCy in `session.py` (`extract_topics`) e
  `agent.py` (`_extract_terms`)

**Cosa è stato fatto in R4:**

- `datasets/` spostato da `<root>/datasets/` a `app/datasets/`
- `DATASETS_DIR` in `config.py` aggiornato: `_APP / "datasets"`
- `update_datasets.py` path setup corretto
- `_image_url` in `arasaac.py` resa offline-aware: `USE_LOCAL_DATASETS=True`
  → `/api/images/{id}` (proxy Vite → backend), altrimenti CDN
- `get_pictogram_image` branch online usa `ARASAAC_IMG_PATTERN` diretto
- rimosso duplicato `_image_url` da `api/server.py`

**Cosa è stato fatto in R5 (sessione corrente):**

- `resolve_concept` esteso con `return_method=True`: ritorna
  `(queries, method)` con label `exact|lemma|hyphen|lemma_alt|token|none`;
  costante `RESOLVE_METHODS` esposta dal modulo
- `AACAgent` espone `last_resolve_info`, `last_plan_method`,
  `last_synset_added`, `last_fresh_count`; log `[RESOLVE]` e `[PLAN]`
  includono ora il metodo usato
- `run_eval_hf.py`: aggiunte colonne `resolve_method`, `resolve_queries`,
  `plan_method`, `synset_added`, `fresh_count` al CSV; sezioni
  corrispondenti in `_print_metrics_hf`; verbose `_print_turn` mostra
  il metodo di resolve
- Endpoint `GET /datasets/status` e `POST /datasets/update` (SSE) in
  `api/server.py`: job background con lock, log forwarding via
  `QueueHandler`, invalidazione cache al termine
- Frontend: `useDatasets.js` hook SSE, `DatasetPanel.jsx` modale con
  status card per lingua, log stream in tempo reale, opzioni
  force/download-images; pulsante `⬇ Datasets` in header `App.jsx`

---

## 12. Dubbi aperti — da chiarire con i tutor

**D1 — Tool-use: LLM decide o codice decide?**
Attualmente il planner LLM decide autonomamente se chiamare `get_time` /
`get_schedule`. È accettabile o la decisione dovrebbe essere hardcoded?

**D2 — Una sola chiamata LLM per turno: sufficiente?**
Il vecchio design prevedeva un secondo LLM call (filter). Ora è tutto
deterministico dopo il planner. L'output è abbastanza buono o serve un
secondo step LLM per il ranking?

**D3 — Funzionamento offline? → RISOLTO**
Quando `USE_LOCAL_DATASETS=True` (default in `settings.py`), `_image_url`
restituisce `/api/images/{id}`. Il proxy Vite riscrive `/api/*` → backend `:8000`,
quindi il browser chiama `GET /images/{id}` su FastAPI che serve il PNG da
`app/datasets/pictograms/{id}.png` (scaricandolo on-demand se mancante).
Il browser non raggiunge mai la CDN ARASAAC.

**D4 — Il comportamento tool-call entra nella valutazione formale?**
`tools_called` è esposto nell'API ma non è ancora una metrica di eval.

**D5 — Soglia di overlap per il successo semantico**
Qual è la soglia accettabile per `gold_in_candidates` e `gold_in_window`?

**D6 — Teacher forcing: metodologicamente accettabile?**
L'eval inietta `gold_id` come pittogramma selezionato per simulare turni
multi-step. È una scelta corretta per il contesto di tesi?

**D7 — Quante righe sono sufficienti per la valutazione?**
Il dataset ha ~54k righe. Quante servono per risultati statisticamente
significativi sul cluster?

**D8 — Gold multipli per concetto?**
Attualmente ogni riga ha un solo gold ID. Avrebbe senso estendere a top-3?

**D9 — Fine della sequenza nell'agente**
Non c'è un segnale esplicito di "frase completata". Il caregiver svuota
l'input manualmente. È il comportamento atteso?

---

## 13. Cose da NON fare

- **Non passare `concept` o `sentence` all'agente** durante l'eval
  (l'agente riceve solo `raw_input` come farebbe in produzione)
- **Non sovrascrivere parametri di `config.py` nel notebook**
- **Non usare MRR** come metrica primaria
- **Non valutare in "single-turn"** — ogni riga ha N pittogrammi gold
- **Non toccare `hf_dataset_annotation/`** — fase a monte completata
- **Non mettere logica applicativa fuori da `app/`**

---

## 14. Strategia `resolve_concept` (`app/src/agent/resolve.py`)

Mappa un concetto semantico libero su keyword reali presenti nel
`keyword_index.json` di ARASAAC. Ordine di tentativo (primo match vince):

| Step | Label | Esempio |
|---|---|---|
| 1 | `exact` | `"eat"` → `"eat"` |
| 2 | `lemma` | `"eating"` → `"eat"` via spaCy |
| 3 | `hyphen` | `"go out"` → `"go-out"` |
| 3b | `lemma_alt` | lemma della forma normalizzata |
| 4 | `token` | `"wash hands"` → `["wash", "hand"]` (token singoli) |
| — | `none` | nessun match → concetto saltato |

`resolve_concept(concept, kw_set, return_method=True)` ritorna
`(queries: list[str], method: str)` — usato dall'agente per tracciare
la risoluzione nel log e nell'eval. I caller che non passano
`return_method=True` ricevono solo `list[str]` — API invariata.

La costante `RESOLVE_METHODS` esposta dal modulo elenca tutti i label
possibili nell'ordine canonico.

---

## 15. Eval — come funziona (`eval/cluster_work/run_eval_hf.py`)

Script self-contained per il cluster. Non dipende dall'ambiente app.
Unico entry point per la valutazione.

Carica il dataset `eval_final.parquet`, esegue `agent.run(raw_input)` su
ogni riga, misura:

- `gold_in_candidates` — il gold ID era nel pool prima del ranking?
- `hit` (`gold_in_window`) — il gold ID è nella finestra finale?
- `overlap_level` — livello semantico migliore tra finestra e gold
  (`synset` > `category` > `keyword` > `tag` > `None`)

Teacher forcing: dopo ogni turno inietta il gold come selezione per
simulare multi-turn realistici.

**Colonne del CSV di output** (per riga di dataset × turno):

| Colonna | Tipo | Descrizione |
|---|---|---|
| `model` | str | nome modello HF |
| `row_idx` | int | indice riga dataset |
| `split` | str | `clear` o `vague` |
| `turn_pos` | int | posizione turno nella sequenza (0-based) |
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
| `synset_added` | int | pittogrammi aggiunti da espansione synset |
| `fresh_count` | int | pittogrammi fresh (non stale padding) nella finestra |

**Attributi `last_*` esposti da `AACAgent` per l'eval:**

```python
agent.last_candidates     # list[Pictogram] — pool completo prima del ranking
agent.last_call_tools     # bool — il planner ha chiamato i tool?
agent.last_resolve_info   # list[{concept, queries, method}] — tracing resolve
agent.last_plan_method    # str — "llm" | "fallback_spacy" | "fallback_empty"
agent.last_synset_added   # int — pittogrammi aggiunti da synset expansion
agent.last_fresh_count    # int — pittogrammi fresh nella finestra
```

**Sezioni del summary finale (`_print_metrics_hf`):**
1. HIT@WINDOW (metrica primaria)
2. FAILURE BREAKDOWN (retrieval fail / LLM fail / success)
3. SEMANTIC OVERLAP LEVELS
4. RESOLVE METHOD DISTRIBUTION (con hit_rate per metodo)
5. PLAN METHOD DISTRIBUTION (con hit_rate per metodo)
6. SYNSET EXPANSION & RANK SOURCE
7. TOOL-CALL BEHAVIOUR

---

## 16. Possibili prossimi passi

- **Free-run eval**: flag `--no-teacher-forcing` in `run_eval_hf.py`.
- **Gold multipli**: estendere dataset con top-3 IDs per concetto.
- **Model selector live**: il dropdown in header chiama già PATCH `/settings`;
  verificare che `_get_agent()` ricostruisca correttamente l'istanza.
- **Multilingua**: `LANG` è già parametrico; i dataset locali hanno solo `en/`
  ma le API ARASAAC supportano altre lingue.
- **Produzione offline completa**: i PNG scaricati on-demand da
  `_DatasetCache.get_pictogram_image` non sono pre-scaricati tutti.
  Usare `DatasetPanel` con `download_images=true` per pre-popolare la
  cache prima di portare il tablet offline.

---

## 17. Teacher forcing e ricerca manuale dei pittogrammi

### Assunzione nell'eval

L'eval usa **teacher forcing**: se il `gold_id` non è nella finestra restituita dall'agente, si assume che il caregiver lo trovi e lo selezioni manualmente — rendendo la sequenza multi-turn deterministica. Questa assunzione è metodologicamente necessaria per valutare in modo pulito il retrieval, ma **la funzionalità non esiste nell'app**.

### Come si cercano pittogrammi a mano oggi

Nell'app attuale non c'è nessun meccanismo di ricerca manuale. Il caregiver può solo:
1. Modificare il testo nell'`InputBar` e rilanciare l'agente con una descrizione più precisa.
2. Aspettare che un turno successivo proponga pittogrammi diversi.

Su ARASAAC online il caregiver può cercare a mano per keyword sul sito, ma questo è fuori dall'app.

### Come potrebbe essere implementata la ricerca manuale

L'endpoint `GET /search?keyword=...&lang=...` esiste già nel backend (`api/server.py`). Quello che manca è solo il frontend. Le opzioni più semplici:

- **Search bar nella griglia**: un campo di testo sopra la `PictogramGrid` che chiama `/search` direttamente, senza passare per l'agente. Il risultato si aggiunge o sostituisce la griglia corrente.
- **Modale di ricerca**: un bottone "Search manually" nell'header apre un modale con input + griglia secondaria; selezionando un pittogramma da lì si chiama comunque `POST /select` per registrarlo nella sessione.

Entrambe le opzioni non richiedono modifiche al backend.

### Implicazione per la tesi

Va esplicitato che il teacher forcing nell'eval **presuppone** questa funzionalità, che attualmente non è implementata. È una limitazione dell'app reale, non dell'eval. Va segnato come possible sviluppo futuro o domanda aperta per i tutor (vedi FAQ §3.1).

---

## 18. Ricerca manuale per categoria (da implementare)

### Motivazione

La ricerca manuale è necessaria per chiudere il gap tra eval (teacher forcing) e app reale
(vedi §17). Invece di mostrare 13.780 pittogrammi o richiedere di conoscere la keyword
esatta, si naviga per categoria gerarchica.

### Struttura dei dati esistente

Ogni pittogramma in `pictograms.json` ha già un campo `categories` (lista di stringhe).
Nel dataset `en/` ci sono **567 categorie uniche** che coprono 13.745 pittogrammi su 13.780
(solo 35 senza categoria). Le categorie più popolari per un uso AAC:

| Categoria | # pittogrammi |
|---|---|
| verb | 2668 |
| core vocabulary-communication | 357 |
| routine | 300 |
| clothes | 188 |
| feeling | 170 |
| terrestrial animal | 166 |
| food | 64 |
| beverage | 141 |
| furniture | 114 |
| toy | 113 |

### Gerarchia proposta (2 livelli)

567 categorie è ancora troppo per un browser visivo. Si raggruppano a mano in
~15 **macro-categorie** (livello 1), ognuna contenente le categorie ARASAAC originali
(livello 2):

| Macro-categoria | Categorie ARASAAC incluse (esempi) |
|---|---|
| Actions | verb, usual verbs, routine, body position |
| People & Body | family, human anatomy, child, adult, elderly, feeling |
| Animals | terrestrial animal, marine animal, bird, insect, domestic animal, pet |
| Food & Drink | food, beverage, fruit, vegetable, gastronomy, baking |
| Places & Buildings | residential building, commercial building, building room, educational space |
| Objects & Tools | work tool, utensil, electrical appliance, toy, educational material |
| Clothes | clothes, footwear, accessories |
| Health & Medicine | symptom, disease, medicament, medical procedure, hygiene product |
| School & Work | educational task, educational material, subject, professional |
| Transport | land transport, aerial transport, water transport, vehicle component |
| Nature | terrestrial animal, atmospheric phenomena, landform, plant |
| Time & Numbers | number, day hours, unit of time, month, day |
| Feelings & Behaviour | feeling, human response, disruptive behavior, expression |
| Communication | core vocabulary-communication, mass media, computing |
| Other | tutto il resto |

### Implementazione (backend)

Aggiungere un solo endpoint a `api/server.py`:

```python
@app.get("/categories")
def list_categories(lang: str = "en"):
    """Returns all categories with pictogram count."""
    ...

@app.get("/by_category")
def by_category(category: str, lang: str = "en", max_results: int = 50):
    """Returns pictograms that have `category` in their categories list."""
    ...
```

Entrambi leggono `pictograms.json` già in memoria nel `_DatasetCache` — nessuna
modifica al dataset, nessun nuovo indice necessario.

### Implementazione (frontend)

Un componente `CategoryBrowser` (modale o pannello laterale) con tre stati di navigazione, ognuno con breadcrumb/back per tornare indietro:

1. **Livello 0 — macro-categorie**: griglia di ~15 card, ognuna con un'**immagine rappresentativa** (un pittogramma scelto come icona della macro) + nome della macro-categoria.
2. **Livello 1 — categorie ARASAAC**: griglia delle categorie figlie, ognuna con un'**immagine rappresentativa** della categoria (primo pittogramma della lista o uno scelto a mano) + nome categoria + conteggio.
3. **Livello 2 — pittogrammi specifici**: `PictogramGrid` esistente riusata as-is, con immagine reale + nome/keyword del pittogramma.

Ogni livello deve caricarsi **velocemente** — livello 0 e 1 usano al massimo 1 immagine per card (non tutte), livello 2 usa `max_results=50` con scroll. Nessuna chiamata agente, solo fetch diretti a `/categories` e `/by_category`.

Navigazione: breadcrumb fisso in cima al browser (es. `Animals > terrestrial animal`) con freccia back per tornare al livello precedente senza ricaricare.

### Note
- Le categorie sono in inglese anche nel dataset `it/` e `es/` (nomi ARASAAC non tradotti).
- Un pittogramma può apparire in più macro-categorie (es. "verb" + "routine");
  va bene mostrarlo in entrambe.
- La macro-categoria "Other" fa da catch-all e può essere nascosta di default.

---

## 19. Cosa è stato fatto in R6 (sessione corrente)

**Bigrammi in `_terms_from_schedule` (fix da sessione precedente — già in codice):**
- Il metodo ora produce prima bigram (`"ice cream"`, `"cream breakfast"`, …) poi
  i singoli token (≥3 char), tutti deduplicati via `seen`.
- Compound ARASAAC labels (e.g. `"ice cream"`) vengono tentati prima dei token
  singoli — `resolve_concept` li centra al passo `exact` o `lemma`.
- Consiglio uso: inserire gli eventi in calendario in **inglese** per massimizzare
  il match con il dataset ARASAAC (es. `"ice cream breakfast"` invece di
  `"colazione con gelato"`).

**Fix 1 — `test/tools_test.ipynb` path setup (bug silenzioso):**
- `SRC = PROJECT_ROOT / 'src'` puntava a `<root>/src/` inesistente.
- Corretto in `SRC = PROJECT_ROOT / 'app' / 'src'`.
- La logica di rilevamento CWD è stata resa esplicita con un `if/else`
  invece del ternario (più leggibile, stessa semantica).

**Fix 2 — `run_eval_hf.py` docstring colonne CSV:**
- L'intestazione del modulo listava solo le colonne pre-R5.
- Aggiunto in docstring: `resolve_method, resolve_queries, plan_method,
  synset_added, fresh_count` — ora allineato con `CSV_COLUMNS` e `CSV_COLUMNS_HF`.

**Fix 3 — `test/tools_test.ipynb` test `return_method=True`:**
- Aggiunta cella finale nella sezione `resolve_concept` che verifica:
  - `resolve_concept('water', kw_set, return_method=True)` → `(['water'], 'exact')`
  - Concetto sconosciuto → `([], 'none')`
  - Tutti i label in `RESOLVE_METHODS` sono stringhe.
- Copre il path usato dall'eval (`_get_resolve_info` in `run_eval_hf.py`).

**Review complessiva — nessun altro problema trovato:**
- `agent.py`: pipeline, bigrammi, last_* attributes, fallback spaCy — tutto corretto.
- `resolve.py`: tutti i 5 step + `return_method` — corretti.
- `hf_agent.py`: override solo `_plan`, eredita tutto il resto — corretto.
- `api/server.py`: tutti gli endpoint, logica immagini offline-aware — corretti.
- `run_eval_hf.py`: CSV_COLUMNS, run_multi_turn, _get_resolve_info,
  _print_metrics_hf, sezioni R5 — tutti corretti e allineati.
- `session.py`, `prompts.py`, `config.py`, `settings.py` — nessun problema.
- Tutto il codebase è in **inglese** (commenti, log, docstring, nomi variabili).

---

## 32. Cosa è stato fatto in R17 (sessione corrente)

### Dataset `en_eval` — merge e creazione

**Analisi Jaccard (§3.3 notebook):**
- Aggiunta visualizzazione HTML con immagine local (CDN `static.arasaac.org`) e HF (bytes base64) per i k pittogrammi più divergenti per Jaccard.
- Bin Jaccard: esattamente 0.0, (0, 0.5), [0.5, 1.0), esattamente 1.0.
- Conferma visiva: anche i pittogrammi con Jaccard basso o 0 sono lo stesso pittogramma — le differenze sono solo di granularità keyword (HF atomico vs local composto). Local vince sempre nell'overlap.

**Merge (§4 notebook):**
- `FIELD_SOURCE` switch in cima alla cella per ablazioni future su `keywords`, `categories`, `tags`.
- Helper `_safe_list` e `_safe_bool` per gestire l'ambiguità numpy array / pandas Series.
- HF ha solo 5 colonne (`image`, `pictogram_id`, `tags`, `categories`, `keywords`) — tutti i booleani e le date dei 24 only-HF sono a default (`False` / `None`).
- Schema keyword normalizzato: `type=2`, `plural=None` per i 24 only-HF.
- Sanity checks: `len(merged_json)`, unicità ID, gold coverage.

**Creazione `app/datasets/en_eval/` (§4 notebook):**

I 5 file attesi da `_DatasetCache.load_*`:

| File | Contenuto | Count |
|---|---|---|
| `pictograms.json` | `{ id_str → record }` | 13.804 |
| `keyword_index.json` | `{ keyword → [id_str, ...] }` | ricostruito dal merged |
| `keywords.json` | `[keyword, ...]` sorted | ricostruito dal merged |
| `synset_index.json` | `{ synset_id → [id_str, ...] }` | ricostruito dal merged |
| `_meta.json` | conteggi + timestamp | aggiornato |

`en/` rimane **invariato** — dataset di produzione. `en_eval/` è snapshot frozen per l'eval.

### `app/src/settings.py` — `dataset_langs`

Aggiunto `"en_eval"` a `dataset_langs`:
```python
"dataset_langs": ["en", "en_eval", "it", "es"],
```
Con commento che spiega che `en_eval` è frozen e non viene toccato da `update_datasets.py`.

### `eval/cluster_work/run_eval_hf.py` — argomento `--lang`

Tre modifiche coordinate:

1. **`_eval_lang`** — variabile modulo-level (`str = "en_eval"`) che `get_gold_meta` usa per `get_pictogram_metadata(lang=_eval_lang)` invece del precedente hardcode `lang="en"`.

2. **`parse_args`** — aggiunto `--lang` (default `"en_eval"`). Documentato nel docstring con esempi `--lang en_eval` e `--lang en` (ablazione).

3. **`main()`** — `global _eval_lang; _eval_lang = args.lang` + `lang=args.lang` passato a `HFAACAgent(...)`.

**Uso:**
```bash
# eval con dataset merged (default)
python eval/cluster_work/run_eval_hf.py --models Qwen/Qwen2.5-3B-Instruct --lang en_eval

# ablazione con dataset local standard
python eval/cluster_work/run_eval_hf.py --models Qwen/Qwen2.5-3B-Instruct --lang en
```

### Prossimo passo

Sezione 5 del notebook: filtrare `eval_final.parquet` rimuovendo le sequenze il cui `best_id` è in `missing` (924 gold irrecuperabili deprecati da ARASAAC). Questo produce il parquet pulito da dare a `run_eval_hf.py`.

---

*Ultimo aggiornamento: maggio 2026 — R17: merge dataset en_eval, run_eval_hf.py --lang, arasaac_vs_eval.ipynb sezioni 3.2–3.3.*

---

## 29. Cosa è stato fatto in R15 — REVISIONE CRITICA

> ⚠️ Le conclusioni originali di R15 erano errate. Vedere §30 per l'analisi corretta.

### Problema originale (confermato)

- **SRC_LOCAL** (13.780 pittogrammi) copre l'**86.1%** dei gold ID dell'eval.
- **SRC_HF** grezzo (12.474 righe) sembrava coprire il **100%** dei gold ID.
- La conclusione era: merge LOCAL + HF → copertura 100%.

### Perché quella conclusione era sbagliata

L'analisi pre-cleaning contava `hf_ids` includendo **1.817 righe con `keywords = categories = tags = None`** — pittogrammi con solo l'immagine ma nessun metadato testuale. Quegli ID erano contati come "coperti" da HF, ma non sono usabili dall'agente (che ricerca per keyword). Dopo il cleaning corretto, la copertura reale di HF crolla all'86.3%.

**Il notebook `eval/build_eval_dataset.ipynb` si basava su premesse errate e va considerato obsoleto.**

---

## 30. Cosa è stato fatto in R16 (sessione corrente)

### Analisi dataset: `eval/arasaac_vs_eval.ipynb`

Notebook di analisi comparativa tra `df_local` (dataset ARASAAC locale), `df_hf` (dataset HuggingFace `disi-unibo-nlp-students/ARASAAC-Pictograms`) e `df_eval` (gold annotation in `eval_raw.parquet`).

### Cleaning eseguito (§2.3)

**`df_local`**: nessun problema strutturale. I `meaning: None` nelle keyword sono accettabili — l'agente non legge mai il campo `meaning`. Copiato as-is in `df_local_clean`.

**`df_hf`**: due problemi risolti:
- **10 ID duplicati** (20 righe): ogni coppia ha una riga buona e una con tutti i metadati a `None`. Tenuta la riga non-None.
- **1.817 righe con `keywords = categories = tags = None`**: dropping completo.
- Risultato: `df_hf_clean` = **10.657 righe** (da 12.474).

### Analisi overlap e copertura gold (§3.1) — numeri definitivi

| Dataset | Pittogrammi | Gold ID coperti | Gold ID mancanti |
|---|---|---|---|
| `df_local_clean` | 13.780 | 5.812 (86.1%) | 936 |
| `df_hf_clean` | 10.657 | 5.823 (86.3%) | 925 |
| union | 13.804 | 5.824 (86.3%) | **924** |

**Partizione degli ID:**
| Gruppo | Count |
|---|---|
| solo in local | 3.147 |
| solo in HF clean | 24 |
| overlap | 10.633 |

Dopo il cleaning, HF clean è quasi interamente un sottoinsieme di local. I 24 ID `only_hf` sono trascurabili. Il merge local + HF clean aggiunge **1 solo gold ID** rispetto a local da solo.

### Anatomia delle 1.817 righe None rimosse da HF

Queste righe sono il cuore del problema. Analisi completa:

| | Count |
|---|---|
| None rows totali | 1.817 |
| anche in local (metadati disponibili lì) | 11 |
| **NON in local** (deprecati da ARASAAC) | **1.806** |
| di cui gold eval ID | 930 |
| gold + in local (recuperabili via merge) | 6 |
| **gold + NON in local (irrecuperabili)** | **924** |

**Cross-check:** i 924 gold irrecuperabili coincidono esattamente con `missing_from_union` (924). ✓

### Interpretazione

- `local` è una copia recente dell'API ARASAAC online. **Se un pittogramma non è in local, non esiste più su ARASAAC.**
- Le 1.806 righe None non in local sono pittogrammi che **esistevano al momento dello snapshot HF ma che ARASAAC ha da allora deprecato/rimosso**. Non sono recuperabili.
- I **924 gold ID mancanti sono irrecuperabili**: il dataset di eval è stato annotato contro una versione di ARASAAC che includeva pittogrammi poi rimossi.
- Il merge local + HF clean è ancora utile (recupera gli 11 None che sono in local, e i 24 only_hf clean), ma **non chiude il gap di copertura gold**.

### Piano d'azione per l'eval dataset

**Cosa è stato fatto (R17):**

1. **Merge local + HF clean** — eseguito in `eval/arasaac_vs_eval.ipynb` §4. Risultato: 13.804 pittogrammi. I 24 only-HF hanno `synsets=[]`, booleani a `False` (campi non presenti in HF), schema keyword normalizzato al formato local (`type`, `keyword`, `plural`, `meaning`).

2. **Dataset `en_eval` creato** — `app/datasets/en_eval/` contiene i 5 file che l'app si aspetta: `pictograms.json`, `keyword_index.json`, `keywords.json`, `synset_index.json`, `_meta.json`. È uno snapshot frozen — `update_datasets.py` non lo tocca mai.

3. **Filtrare l'eval dataset** — da fare: rimuovere da `eval_final.parquet` le sequenze il cui `best_id` è in `missing` (924 gold irrecuperabili). Cella da aggiungere nella sezione 5 del notebook.

**Cosa NON fare:**
- ~~Cercare di recuperare i 924 ID dall'API ARASAAC~~ — non esistono più online.
- ~~Usare le righe None di HF come fonte di metadati~~ — sono vuote.
- ~~Considerare il `build_eval_dataset.ipynb` di R15 come valido~~ — basato su premesse errate.

### Dimensione effettiva dell'eval dopo il filtro

Le righe di `eval_final.parquet` con almeno un gold ID irrecuperabile vanno escluse. Il numero esatto dipende da quante *sequenze* (non singoli turni) hanno tutti i gold ID coperti — da calcolare in una cella dedicata del notebook prima di procedere all'eval.

### File coinvolti

| File | Stato | Note |
|---|---|---|
| `eval/arasaac_vs_eval.ipynb` | ✅ in corso | sezioni 1–3.3 complete; §4 merge completo; §5 filtro eval da fare |
| `eval/build_eval_dataset.ipynb` | ⚠️ obsoleto | basato su HF = 100% coverage (errato) |
| `app/datasets/en_eval/` | ✅ creato | snapshot frozen del merged dataset |
| `app/datasets/en/` | ✅ invariato | dataset produzione, non toccare |
| `hf_dataset_annotation/eval_raw.parquet` | da filtrare | rimuovere righe con gold ID irrecuperabili |
| `hf_dataset_annotation/eval_final.parquet` | da filtrare | idem |

---

## 31. Analisi sezione 3 di `arasaac_vs_eval.ipynb` — stato e decisioni

### 3.1 — completata

Vedi §30 per i numeri definitivi. Le variabili `local_ids`, `hf_ids`, `gold_ids`, `only_local`, `only_hf`, `in_overlap`, `merged_ids`, `none_pids`, `irrecoverable`, `missing` sono definite qui e usate nelle sezioni successive.

### 3.2 — keyword agreement nell'overlap

**Domanda:** per i 10.633 pittogrammi presenti in entrambi i dataset, le keyword sono le stesse?

**Perché serve:** nel merge dobbiamo scegliere quale versione preferire per i pittogrammi in overlap. Se le keyword coincidono quasi sempre, possiamo prendere local (che ha anche synsets). Se divergono significativamente, dobbiamo capire perché e decidere caso per caso.

**Schema keyword diverso tra i due dataset:**
- local: `{type: int, keyword: str, plural: str, meaning: str}`
- HF: `{hasLocution: bool, keyword: str, meaning: str}`

Il confronto va fatto solo sul campo `keyword` (lowercase, stripped), che è l'unico usato dall'agente.

**Metrica:** Jaccard tra i set di keyword di local e HF per ogni pittogramma in overlap. Jaccard = 1.0 → identici, 0 → nessun overlap.

**Conclusione attesa:** la maggior parte dei pittogrammi in overlap avrà Jaccard alto (keyword quasi identiche), perché entrambi derivano da ARASAAC. Le divergenze saranno dovute a versioni diverse dello snapshot (keyword aggiunte/rimosse nel tempo) o a differenze nel campo `plural` che HF non ha.

**Decisione di merge:** per i pittogrammi in overlap, usare **local** come fonte primaria perché:
1. ha i synsets (HF non li ha)
2. è più recente (mirror live di ARASAAC)
3. ha il campo `type` per ogni keyword

### 3.3 — analisi only_local e only_hf

**only_local (3.147 pittogrammi):** presenti in local ma non in HF. Vanno nel merged as-is — sono già in locale con metadati completi. L'analisi mostra che non sono filtrati per contenuto (sex/violence minoritari) né sono solo pittogrammi nuovi (date dal 2013 al 2026): l'HF snapshot era semplicemente incompleto.

**only_hf (24 pittogrammi):** presenti in HF clean ma non in local. Sono pochissimi e nessuno o quasi è gold. Vanno comunque aggiunti al merged (hanno metadati completi in HF clean). Non hanno synsets — da aggiungere con `synsets=[]`.

**Non serve un'analisi approfondita di questi due gruppi** oltre a confermare i numeri e la strategia di merge. Le celle devono essere concise.

### Struttura finale sezione 3 — COMPLETATA

| Sezione | Contenuto | Stato |
|---|---|---|
| 3.1 | ID partitioning, gold coverage, None rows anatomy | ✅ |
| 3.2 | Deeper checks: None-in-HF-but-local, only_hf, only_local, gold irrecuperabili | ✅ |
| 3.3 | Jaccard keyword agreement nell'overlap + visualizzazione HTML con immagini | ✅ |
| 4 | Merge: `merged_json`, scrittura `en_eval/` con tutti e 5 i file | ✅ |
| 5 | Filtro `eval_final.parquet` — da fare | ⬜ |

### Conclusioni sezione 3.3 — keyword agreement

La visualizzazione HTML (immagini local CDN + HF base64, keyword per Jaccard=0) ha confermato che anche i pittogrammi con Jaccard basso si riferiscono allo stesso pittogramma con minime variazioni grafiche. Le differenze di keyword sono esclusivamente di granularità (HF più atomico, local più composto). Decisione: **local vince sempre nell'overlap** — ha synsets, `type`, `plural`, ed è più recente.

### Switch `FIELD_SOURCE` per ablazione

```python
FIELD_SOURCE = {
    "keywords"   : "local",   # cambia in "hf" per ablazione
    "categories" : "local",
    "tags"       : "local",
}
```

HF ha solo 5 colonne (`image`, `pictogram_id`, `tags`, `categories`, `keywords`) — nessun booleano (`aac`, `sex`, ecc.), nessun `synsets`. Per i 24 only-HF tutti i booleani sono `False` per default, `synsets=[]`.

### Schema keyword normalizzato per i 24 only-HF

HF non ha `type` né `plural` — normalizzati a `type=2` (Common_Names) e `plural=None`. Funziona perché `_raw_to_pictogram` e `Keyword.model_validate` hanno default per entrambi.

---

## 28. Cosa è stato fatto in R14 (sessione corrente)

### InputBar bloccata durante il warmup del modello

**Problema:** quando il caregiver inviava input mentre llama.cpp stava ancora
caricando il GGUF in memoria, il backend andava in `socket hang up` perché
`agent.run()` veniva chiamato prima che il modello fosse pronto.

**Fix:** due modifiche coordinate.

**`app/frontend/src/components/InputBar.jsx`**
- Aggiunta prop `warmingUp = false`
- `isDisabled = loading || warmingUp` — blocca sia durante chiamate in corso
  sia durante il caricamento del modello
- `placeholder` dinamico: `'Loading model, please wait…'` quando `warmingUp=true`
- Stile textarea giallo (`#fefce8`, bordo `#fde68a`) per segnalare visivamente lo stato
- Bottone mostra `⏳` durante warmup, `…` durante loading, `→ Search` altrimenti
- Aggiunto stile `textareaWarmingUp` nell'oggetto `styles`

**`app/frontend/src/App.jsx`**
- `<InputBar>` riceve ora `warmingUp={warmingUp}` — unica modifica
- `warmingUp` era già presente in stato e gestito correttamente dal polling
  `/health` esistente — nessuna altra modifica necessaria

**Comportamento risultante:**
- Al primo avvio (o dopo switch modello): input grigio/giallo, placeholder
  `'Loading model, please wait…'`, bottone `⏳`, Enter ignorato
- Non appena `/health` risponde con `warming_up: false`: input si sblocca
  automaticamente, placeholder torna normale, bottone `→ Search`
- Il banner giallo in header (già presente in R13) continua a mostrare
  il messaggio globale — l'InputBar aggiunge il feedback contestuale
  direttamente dove l'utente interagisce


### GGUF scaricati in `app/models/`

Tutti e quattro i modelli in formato Q4_K_M da `bartowski` su HuggingFace:
- `Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- `ibm-granite_granite-4.1-3b-Q4_K_M.gguf` (granite 4.1, aggiornamento rispetto a granite4:3b-h in Ollama)
- `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`

`app/models/` e `*.gguf` aggiunti a `.gitignore`.

### Modifiche ai file

**`app/src/settings.py`**
- Aggiunta sezione `gguf_models` nei defaults: mappa alias Ollama → path relativo al GGUF
- Aggiunta property `gguf_models` al `_SettingsManager`

**`app/src/agent/agent.py`**
- Import di `LLMBackend`, `LlamaCppBackend` da `agent.backends`
- `AACAgent.__init__` ora accetta `backend: Optional[LLMBackend] = None`
- `AACAgent._plan()` usa il backend se presente, altrimenti Ollama come prima
- Zero breaking changes: comportamento identico se `backend=None` (default)

**`app/requirements.txt`**
- Aggiunto `llama-cpp-python>=0.3.0` con commento installazione OpenBLAS

### Come usare il backend llama.cpp

```python
from pathlib import Path
from agent.backends import LlamaCppBackend
from agent.agent import AACAgent

_APP = Path(__file__).resolve().parent.parent  # app/
gguf = _APP / "models" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"

backend = LlamaCppBackend(
    model_path = str(gguf),
    n_ctx      = 2048,   # safe per il prompt FULL; 512 per SHORT
    n_threads  = 4,      # Intel Core i5 2020 = 4 core fisici
    verbose    = False,
)
agent = AACAgent(backend=backend)
result = agent.run("he seems hungry")
```

### Installazione llama-cpp-python (Intel Mac con OpenBLAS)

```bash
brew install openblas
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

### Prossimo passo: R14

Esporre il backend llama.cpp via API/frontend: aggiungere un parametro
`use_llamacpp: bool` alle settings o un endpoint di switch, così il frontend
possa scegliere tra Ollama e llama.cpp senza modificare il codice.

---

## 26. Cosa è stato fatto in R12 (sessione corrente)

### Problema identificato — cap artificiale sui concepts

Entrambi i prompt (FULL e SHORT) limitavano i concepts a `4-8`. Questo era
sbagliato per due motivi:
1. Con pochi concepts il pool candidati è piccolo e la window non si riempie
   (o si riempie con pittogrammi dello stesso concetto, monotona).
2. Il limite massimo non ha senso architetturale: è `_rank_and_fill` che
   taglia alla window size `max_results=24`. Il LLM deve generare liberamente,
   poi il codice seleziona i migliori.

Secondo problema: il prompt SHORT non spiegava l'espansione semantica — con
qwen2.5:3b (modello default, il più leggero) il risultato era estrazione
letterale: `"he seems hungry"` → `["person", "appear", "hungry"]`.

### Fix applicato — `app/src/agent/prompts.py`

**`_PLANNER_SYSTEM_PROMPT_SHORT`** — tre modifiche:
- Rimosso il cap `4-8`; sostituito con `"as many ... as needed"`
- Aggiunta regola di espansione semantica esplicita
- Esempio inline che usa esattamente l'input problematico come few-shot:
  `"he seems hungry"` → `["hungry", "food", "eat", "meal", "snack", "plate", "drink", "stomach"]`

**`_PLANNER_SYSTEM_PROMPT_FULL`** — stesse modifiche per coerenza:
- Rimosso il cap `4-8`; sostituito con `"as many ... as needed, no upper limit"`
- Esempio `"he wants water"` ampliato da 2 a 4 concepts
- Tutti gli esempi mostrano liste più ricche per modellare il comportamento atteso

### Invarianti architetturali confermati

- Il LLM genera liberamente quanti concepts vuole (no cap)
- `search_pictograms` + synset expansion → pool grande
- `_rank_and_fill` taglia alla window `max_results` (default 24 in `settings.py`, configurabile — es. 50 in `user_settings.json`)
- Nessuna modifica ad `agent.py`, `settings.py` o altri file

---

## 25. Cosa è stato fatto in R11 (sessione corrente)

### Fix `has_thinking` — errata diagnosi corretta

In R10 era stato aggiunto `has_thinking: True` ai metadati di `granite4:3b-h` e
logica `think: False` in `_plan()`, basandosi sull'ipotesi che la lentezza fosse
causata dal reasoning attivo. Questa ipotesi era **errata**.

Da documentazione Ollama confermata:
- La suffisso `-h` in `granite4:3b-h` indica la **architettura hybrid mamba-2**,
  non una modalità reasoning.
- Il reasoning su modelli Granite (granite3.3, granite4.0-preview) si attiva
  **esplicitamente** con `think=True` o un messaggio `{"role": "control", "content": "thinking"}`.
  Non è mai attivo di default.
- Nessuno dei 4 modelli in `settings.py` ha reasoning attivo di default.

**Fix applicati:**
- `settings.py`: rimosso `has_thinking` da tutti i metadati modello; corretti
  `size_gb` (granite era 4.4 → 2.1 corretto); `num_ctx` abbassato da 2048 → 512
  per tutti (il planner prompt è sempre <512 token, riduce KV-cache su CPU).
- `agent.py` `_plan()`: rimosso `has_thinking`, rimossa logica `if has_thinking: options["think"] = False`,
  semplificato log `[PLAN CALL]`.

### Prompt split FULL / SHORT (`app/src/agent/prompts.py`)

Il prompt originale (~380 token, ~40s prefill su CPU) è stato rinominato
`_PLANNER_SYSTEM_PROMPT_FULL` e affiancato da `_PLANNER_SYSTEM_PROMPT_SHORT`
(~100 token, ~8s prefill su CPU, misurato con `qwen2.5:3b + num_ctx=2048`).

`build_planner_prompt(*, full=False)` restituisce il SHORT di default (produzione).
Passare `full=True` per il FULL (debug, eval comparativo).

Non è un parametro di settings — la scelta è nel codice. Il FULL è mantenuto
per documentare l'evoluzione del prompt e confrontare qualità/velocità in eval.

### Nuovo file `app/src/agent/backends.py`

Introduce l'astrazione `LLMBackend` (ABC) con tre implementazioni concrete:

| Classe | Backend | Uso |
|---|---|---|
| `OllamaBackend` | Ollama HTTP daemon | Produzione attuale |
| `LlamaCppBackend` | llama-cpp-python, GGUF in-process | Produzione futura su CPU |
| `HuggingFaceBackend` | HuggingFace Transformers | Eval cluster (GPU) |

Tutti e tre espongono la stessa interfaccia: `chat(system, user) → str`.
Tutta la logica di parsing, timing, logging rimane in `AACAgent`.

Il file è **pronto ma non ancora agganciato** ad `AACAgent`. Il wiring
(far accettare ad `AACAgent` un argomento `backend: LLMBackend`) è il passo
successivo (R12). Quando sarà completato:
- `AACAgent` non importerà più `ollama` direttamente
- `HFAACAgent` diventerà semplicemente `AACAgent(backend=HuggingFaceBackend(...))`
  e il file `hf_agent.py` potrà essere deprecato

**Mapping modelli per `LlamaCppBackend`:** i file GGUF si scaricano da HuggingFace
(namespace `bartowski` o `lmstudio-community`). Quantizzazione raccomandata su CPU:
`Q4_K_M`. Il mapping alias → path andrà in `settings.py` sotto una nuova chiave
`gguf_models` quando il backend sarà agganciato.

**`LlamaCppBackend` install note:**
```bash
pip install llama-cpp-python
# Build ottimizzato per CPU (OpenBLAS):
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

---

## 20. Cosa è stato fatto in R7 (sessione corrente)

**Ricerca manuale dei pittogrammi per categoria (frontend) — COMPLETATA:**

Il backend aveva già implementato `/categories` e `/by_category` nella sessione precedente (R6/fine R5).
Questa sessione ha completato il frontend.

**File creato: `app/frontend/src/components/CategoryBrowser.jsx`**

Componente a 3 livelli di navigazione:

| Livello | Contenuto | API usata |
|---|---|---|
| 0 | Griglia ~15 macro-categorie (emoji + immagine rappresentativa + conteggio) | `GET /categories` |
| 1 | Griglia sottocategorie ARASAAC della macro selezionata (immagine + conteggio) | (dati già in risposta /categories) |
| 2 | Griglia pittogrammi della categoria selezionata | `GET /by_category` |

Caratteristiche:
- Breadcrumb fisso nell'header del pannello con pulsante ← Indietro
- Ogni card mostra l'immagine rappresentativa (un pittogramma del gruppo) con fallback emoji se l'immagine non carica (lazy loading)
- Il pannello è un overlay laterale (drawer) che non copre l'intera UI
- Chiusura cliccando fuori dal pannello o sul ✕
- Al tap su un pittogramma: chiude il browser, chiama `handleSelect(pic)` in App.jsx → POST `/select` → aggiorna sessione (stesso contratto di PictogramGrid)
- Caricamento con spinner ⏳; gestione errori inline
- Scroll automatico in cima ad ogni cambio livello

**File modificato: `app/frontend/src/App.jsx`**
- Import `CategoryBrowser`
- Stato `showCategories` (boolean)
- Bottone `🔍 Cerca` nell'header affianco a ⬇ Datasets
- Rendering condizionale `{showCategories && <CategoryBrowser ... />}` nella sezione modali
- Al `onSelect`: chiude il browser e chiama `handleSelect(pic)` per registrare la selezione nella sessione e auto-continuare

**Nessuna modifica al backend** — `/categories` e `/by_category` già implementati e funzionanti.

**Nota per il prossimo agente:** i commenti del frontend in `CategoryBrowser.jsx` sono in italiano (coerente con la lingua del caregiver e con i commenti di App.jsx già esistenti); il codice Python rimane tutto in inglese.

---

## 21. Cosa è stato fatto in R8 (sessione corrente)

**Fix CategoryBrowser e `/categories` endpoint — 5 problemi risolti:**

### Problema 1 — UI non in inglese
- `CategoryBrowser.jsx`: tutti i testi visibili all'utente tradotti in inglese:
  - `"Scegli una categoria"` → `"Choose a category"`
  - `"scegli una sottocategoria"` → `"choose a subcategory"`
  - `"pittogrammi"` → `"pictograms"`
  - `"← Indietro"` → `"← Back"`
  - `"Chiudi"` → `"Close"`
  - `"Nessuna categoria trovata"` → `"No categories found."`
  - Messaggi di errore inline

### Problema 2 — Errori mostrati come dizionari JSON
- `hooks/useAgent.js`: aggiunto helper `_errorMessage(res)` che:
  - Tenta di parsare il body come JSON
  - Se trova `{"detail": "..."}` (formato errore FastAPI), estrae solo il testo del `detail`
  - Fallback: testo grezzo o `HTTP {status}`
- Usato in `runAgent`, `selectPictogram`, `resetSession`, `getSession`

### Problema 3 — Immagini duplicate
- Il count inflated era la causa principale dell'effetto visivo di duplicazione
- Root fix nel backend (vedi sotto): ogni pittogramma ora conta una sola volta

### Problema 4 — "Other" mostrava 15k pittogrammi
- **Causa reale**: `macro_count` sommava i count di ogni sotto-categoria, ma un
  pittogramma con N categorie veniva contato N volte. Inoltre moltissime categorie
  ARASAAC non erano mappate nelle macro → tutte in Other.
- **Fix in `/categories`** (`api/server.py`): riscritta la logica con set di ID:
  - `cat_ids[cat]` = set di ID unici per quella categoria ARASAAC
  - `macro_pids` = union degli ID di tutte le sotto-categorie della macro → count = `len(macro_pids)`
  - `covered_pids` = tutti i pittogrammi assegnati a una macro con nome
  - **"Other" ora contiene solo i pittogrammi NON coperti da nessuna macro con nome**
  - I pittogrammi di Other sono raggruppati per la loro *prima* categoria ARASAAC
    → niente bucket duplicati, ogni pittogramma appare in un solo gruppo

### Problema 5 — Mulo ripetuto 4 volte in Other
- **Causa**: le categorie `mammal`, `viviparous`, `herbivorous`, `omnivorous`
  non erano mappate → 4 bucket separati in Other con gli stessi pittogrammi
- **Fix**: il nuovo algoritmo assegna ogni pittogramma al bucket della sua *prima*
  categoria. Se `mammal` è prima per l'asino, appare una sola volta sotto `mammal`.
  Le altre categorie dell'asino (`viviparous`, ecc.) appaiono come bucket separati
  ma senza l'asino (già assegnato).
- **Fix strutturale applicato in R9**: aggiunte `mammal`, `viviparous`, `herbivorous`,
  `omnivorous`, `carnivorous`, `oviparous`, `invertebrate`, `arachnid` alla macro
  `Animals` in `MACRO_CATEGORIES` (vedi §22).

**File modificati:**
- `app/src/api/server.py` — `/categories` endpoint riscritto (conteggi unici + Other solo uncovered)
- `app/frontend/src/components/CategoryBrowser.jsx` — testi UI in inglese
- `app/frontend/src/hooks/useAgent.js` — `_errorMessage` helper, errori leggibili

---

## 22. Cosa è stato fatto in R9 (sessione corrente)

**Fix `MACRO_CATEGORIES` — categorie biologiche animali aggiunte ad Animals:**

Categorie ARASAAC che descrivono tratti biologici degli animali (`mammal`, `viviparous`,
`herbivorous`, `omnivorous`, `carnivorous`, `oviparous`, `invertebrate`, `arachnid`)
non erano mappate in nessuna macro → finivano tutte in "Other" come bucket separati,
gonfiando "Other" e frammentando animali semanticamente correlati.

Fix: aggiunte tutte alla lista `categories` della macro `Animals` in `MACRO_CATEGORIES`
(`app/src/api/server.py`). Non è stata modificata la logica dell'endpoint — il fix
è solo dichiarativo (aggiunta di stringhe alla lista).

**File modificato:**
- `app/src/api/server.py` — `MACRO_CATEGORIES["Animals"]["categories"]` estesa con
  8 nuove categorie biologiche

---

## 23. Note non banali sul dataset ARASAAC e la gerarchia delle categorie

Questa sezione documenta aspetti del dataset che non sono ovvi dalla struttura
dei file e che sono stati chiariti durante lo sviluppo.

### 23.1 Pittogrammi vs keyword: i numeri non sono la stessa cosa

Il dataset ha **due popolazioni distinte**:

| Cosa si conta | Valore | Fonte |
|---|---|---|
| Pittogrammi unici | **13.780** | `_meta.json ["pictograms"]["count"]`, `ls pictograms/ \| wc -l` |
| Keyword uniche nel keyword_index | **15.757** | `_meta.json ["keywords"]["count"]` |

Il divario (~2.000) nasce perché ogni pittogramma può avere **più keyword**
(es. il pittogramma "water" ha `["water", "waters"]`). Sommando le keyword
per gruppo si ottiene un numero gonfiato rispetto ai pittogrammi reali.
Il conteggio corretto da citare è sempre **13.780 pittogrammi**.

La stessa distinzione vale per `keyword_index.json` (mappa keyword → lista ID)
e `keywords.json` (lista piatta di tutte le keyword): entrambi parlano di keyword,
non di pittogrammi.

### 23.2 Le categorie ARASAAC sono tag piatti, non una gerarchia

Nel JSON di ogni pittogramma il campo `categories` è una **lista di stringhe piatte**
assegnate da ARASAAC — non c'è nessuna gerarchia parent/child nei dati grezzi.
Esempio:
```json
{"id": 2248, "categories": ["beverage", "mineral rich food"]}
```
Le 567 categorie uniche nel dataset `en/` sono etichette editoriali ARASAAC,
assegnate manualmente dai curatori del progetto. Non esiste un ontologia formale
é i nomi non sono tradotti nelle lingue diverse dall'inglese (anche in `it/` e
`es/` le categorie rimangono in inglese).

### 23.3 Come viene costruita la gerarchia dei macro-gruppi

I macro-gruppi visibili nel `CategoryBrowser` sono costruiti **a mano** nel codice
(`MACRO_CATEGORIES` in `app/src/api/server.py`) e non derivano dai dati ARASAAC.
La logica dell'endpoint `/categories` è:

1. Costruisce `cat_ids[cat]` = set di ID unici per ogni categoria ARASAAC
2. Per ogni macro in `MACRO_CATEGORIES`, aggrega i set delle sue sotto-categorie
   → `macro_pids` = union (deduplicata) di tutti gli ID; `count = len(macro_pids)`
3. Tiene traccia di `covered_pids` (tutti i pittogrammi già assegnati a una macro)
4. "Other" raccoglie solo i pittogrammi **non** in `covered_pids`, raggruppati
   per la loro prima categoria ARASAAC

**Regola per il rappresentante visivo** (immagine preview della categoria/macro):
primo pittogramma con `aac: true` trovato nella categoria; se nessuno ha `aac: true`,
il primo in assoluto. `aac: true` indica pittogrammi validati per uso AAC — circa
il 60% del totale. Questo fa sì che le anteprime siano icone più pulite e
riconoscibili.

**Un pittogramma con N categorie viene contato una sola volta per macro** anche se
appartiene a più sotto-categorie della stessa macro. Il conto può quindi differire
da quello che si otterrebbe sommando i count delle singole sotto-categorie.

### 23.4 Il campo `aac` e `aac_color`

```json
{"aac": false, "aac_color": false}
```

- `aac: true` → pittogramma raccomandato per vocabolario AAC di base (core vocabulary);
  usato nel ranking (`quality_score`) e per scegliere i rappresentativi nelle categorie
- `aac_color: true` → esiste una variante colorata raccomandata
- Circa il 40% dei pittogrammi non ha `aac: true`; questi vengono mostrati comunque
  ma ricevono un peso inferiore nel ranking dell'agente

### 23.5 Il campo `synsets` e la copertura WordNet

Circa il **40% dei pittogrammi non ha synsets** (`synsets: []`). L'espansione
WordNet (`_expand_pool_by_synset`) opera solo sui pittogrammi che hanno synsets;
pittogrammi senza synsets vengono trovati solo tramite keyword match diretto.
Questo è un limite strutturale del dataset ARASAAC, non un bug dell'agente.

### 23.6 Tipi di keyword e implicazioni per `resolve_concept`

Ogni keyword ha un campo `type` numerico:

| Tipo | Significato | Implicazioni per resolve |
|---|---|---|
| 1 | Nome proprio | Raramente utile per AAC generale |
| 2 | Nome comune | Match principale per sostantivi |
| 3 | Verbo | Fondamentale per categorie Actions |
| 4 | Aggettivo | Usato per Feelings e descrizioni |
| 5 | Sociale | Formule di cortesia, saluti |
| 6 | Misc | Catchall |

`resolve_concept` non filtra per tipo — usa tutte le keyword del `keyword_index`.
Se si volesse dare priorità ai tipi 2 e 3 (nomi + verbi) si potrebbe pesare
i risultati in `_rank_and_fill`, ma attualmente non è implementato.

### 23.7 Pittogrammi senza categoria (35 su 13.780)

35 pittogrammi non hanno categorie (`categories: []`). Nell'endpoint `/categories`
finiscono nel bucket `"uncategorised"` dentro "Other". Nell'agente non hanno
impatto perché il retrieval avviene per keyword, non per categoria.

### 23.8 `update_datasets.py` e l'endpoint bulk ARASAAC

`update_datasets.py` usa l'endpoint bulk `GET /pictograms/all/{lang}` di ARASAAC,
che non ha un corrispettivo in `arasaac.py` (che usa invece endpoint per singolo ID).
Questo endpoint restituisce tutti i pittogrammi in una sola risposta JSON — l'unica
ragione per cui il dataset locale può essere costruito in tempi ragionevoli.
Le costanti condivise (`ARASAAC_API_BASE`, `ARASAAC_IMG_PATTERN`, `ARASAAC_TIMEOUT`)
vengono tutte da `config.py` per evitare duplicazioni.



---

## 24. Cosa è stato fatto in R10 (sessione corrente)

### Diagnosi lentezza inferenziale e fix immediati

**Problema identificato:** il sistema impiegava ~1 minuto per turno a causa di tre concause:

1. **`granite4:3b-h` con reasoning attivo (causa principale):** la variante `h` (Hybrid)
   di Granite 4 esegue un blocco di extended thinking prima di rispondere. Per un task
   di generazione JSON da ~50 token, il costo del reasoning è completamente sprecato.
   Ollama non disabilita il thinking automaticamente anche con `temperature: 0.0`.

2. **Nessun cap sui token generabili:** senza `num_predict`, Ollama usa il default
   del modello (spesso 2048+ token), lasciando il modello libero di generare molto
   più del necessario in caso di output verboso o malformato.

3. **Modello default sbagliato:** `granite4:3b-h` era il default, ma è la scelta
   peggiore per latenza su CPU dato il reasoning attivo.

**Fix applicati:**

**`app/src/settings.py`** — due modifiche:
- Aggiunto `has_thinking` e `num_predict` ai metadati di ogni modello in `_DEFAULTS["models"]`:
  ```python
  "granite4:3b-h": {"size_gb": 4.4, "has_thinking": True,  "num_predict": 150},
  "qwen2.5:3b":    {"size_gb": 2.0, "has_thinking": False, "num_predict": 150},
  "llama3.2:3b":   {"size_gb": 1.9, "has_thinking": False, "num_predict": 150},
  "mistral:7b":    {"size_gb": 1.9, "has_thinking": False, "num_predict": 150},
  ```
- `agent_default_model` cambiato da `"granite4:3b-h"` a `"qwen2.5:3b"`.

**`app/src/agent/agent.py`** — metodo `_plan()` aggiornato:
- Importa `MODELS` da `config` per leggere i metadati del modello corrente.
- Importa `time` per il timing preciso.
- Costruisce le Ollama options dinamicamente per modello:
  ```python
  model_meta   = MODELS.get(self.model, {})
  has_thinking = model_meta.get("has_thinking", False)
  num_predict  = model_meta.get("num_predict", 150)
  options = {"temperature": 0.0, "num_predict": num_predict}
  if has_thinking:
      options["think"] = False   # disabilita reasoning su granite4:3b-h
  ```
- `think: False` viene passato **solo** ai modelli con `has_thinking: True`
  (passarlo a modelli standard causa warning su Ollama).
- Aggiunto timing con `time.perf_counter()` attorno alla chiamata `_ollama.chat()`.
- Log `[PLAN CALL]` prima della chiamata: mostra `model`, `has_thinking`, `num_predict`, `options`.
- Log `[PLAN OUT]` dopo la chiamata: aggiunto `elapsed=X.XXs` prima del raw output.

**Perché questa sequenza:** prima di refactoring architetturali (backend multipli,
llama.cpp, embedding retrieval), il tutor ha suggerito di verificare che i modelli
già in uso vengano usati correttamente. Il reasoning attivo su granite era un
problema banale ma ad alto impatto che andava misurato prima.

**Prossimo passo:** misurare il `elapsed` nei log con le fix applicate (su tutti
e tre i modelli disponibili: `qwen2.5:3b`, `llama3.2:3b`, `granite4:3b-h` con e
senza `think: False`) per avere dati reali prima di decidere se procedere con
il refactoring backend o strategie più profonde.

### Roadmap discussa con il tutor

**Step 1 (corrente):** diagnostica e fix uso corretto dei modelli esistenti.
**Step 2:** se la latenza rimane inaccettabile → refactoring a backend singolo astratto:
- `OllamaBackend`, `HuggingFaceBackend`, `LlamaCppBackend`
- `AACAgent` riceve il backend come dipendenza, non sottoclassa più
- `HFAACAgent` sparisce — l'eval usa `AACAgent(llm=HuggingFaceBackend(...))`
**Step 3:** se neanche quello basta → strategie di retrieval più efficienti
(BM25 o embedding-based) per ridurre il carico sul planner LLM.

### Dataset preprocessato dal tutor

Il tutor ha un dataset ARASAAC già strutturato come dataframe, ripulito e arricchito,
che risolve i problemi di matching degli ID riscontrati scaricando da API online.
Va ricevuto e convertito nel formato JSON atteso da `_DatasetCache`
(`pictograms.json`, `keyword_index.json`, ecc.) e piazzato in `app/datasets/en/`.
Nessuna modifica all'architettura necessaria.

### Retrieval keyword — punto aperto

Il tutor ha segnalato che la selezione delle keyword (attualmente affidata
interamente al planner LLM) è un punto critico per qualità e velocità.
Opzioni da valutare dopo la diagnostica:
- **BM25** (`rank_bm25`): indice leggero su `keyword_index`, zero GPU, velocissimo.
- **Embedding-based** (`sentence-transformers`, `all-MiniLM-L6-v2` ~80 MB):
  nearest neighbor tra il concetto e le keyword ARASAAC, migliore qualità su
  concetti ambigui o non presenti nel keyword index.

### Orario come segnale contestuale — nota

Il tutor ha precisato che nel contesto attuale l’orario è un segnale approssimativo
per filtrare gli schedule. Il suo valore reale emerge in futuro con una memoria
persistente tra sessioni (abitudini, pattern utente). Non vale la pena investirci
ora; `AGENT_FETCH_SCHEDULE` è già configurabile e può essere disabilitato per
ridurre latenza se necessario.