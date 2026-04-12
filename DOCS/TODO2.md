# TODO — Progetto 1: MCP Agentico per Selezione Pittogrammi CAA

> Stack: Python · FastMCP · LangChain · Ollama (LLM locale) · ARASAAC API

---

## Struttura del Progetto

```
caa-mcp-agent/
├── mcp_server/
│   ├── server.py            # FastMCP server — registra tutti i tool
│   ├── models.py            # Pydantic models (PictogramResult, TimeInfo, ecc.)
│   └── tools/
│       ├── arasaac.py       # Tool: search_pictograms, get_pictogram_metadata
│       ├── time_tool.py     # Tool: get_time (ora, momento giornata, stagione)
│       └── schedule.py      # Tool: get_schedule (agenda mock o Google Calendar)
├── agent/
│   ├── agent.py             # ReAct agent LangChain + connessione MCP
│   └── prompts.py           # System prompt + prompt di filtro LLM
├── pipeline/
│   └── pipeline.py          # Orchestrazione end-to-end (input → pittogrammi)
├── output/
│   └── renderer.py          # Genera HTML/JSON con i pittogrammi selezionati
├── tests/
│   ├── test_arasaac.py
│   ├── test_time_tool.py
│   └── test_pipeline.py
├── config.py                # Costanti, parametri, lingua ARASAAC
├── requirements.txt
├── README.md
└── notebook_demo.ipynb      # Notebook finale di presentazione
```

---

## FASE 0 — Setup

- [X] Creare il repo con la struttura sopra
- [X] Creare `requirements.txt` con:
  - `mcp==1.24.0`
  - `langchain`, `langchain-community`, `langchain-ollama`
  - `requests`, `pydantic`
  - `python-dotenv` (per config opzionale)
- [X] Configurare Ollama in locale (o Colab con colab-xterm come nel notebook del prof)
  - granite4:3b-h ===> https://ollama.com/library/granite4:3b-h
  - qwen2.5:3b    ===> https://ollama.com/library/qwen2.5:3b
  - llama3.2:3b   ===> https://ollama.com/library/llama3.2:3b
  - mistral:7b    ===> https://ollama.com/library/mistral:7b
- [X] Testare che `ollama serve` risponda su `localhost:11434`
- [X] Creare `config.py` con: lingua ARASAAC (`it`/`en`), nome modello Ollama, numero massimo pittogrammi da restituire

---

## FASE 1 — MCP Server (`mcp_server/`)

### 1.1 — `models.py`

- [X] Definire `PictogramResult` (Pydantic): `id`, `keyword`, `image_url`, `tags: List[str]`, `relevance_score: float`
- [X] Definire `TimeInfo`: `hour`, `time_of_day` (mattina/pomeriggio/sera/notte), `day_of_week`, `season`
- [X] Definire `ScheduleEvent`: `title`, `start_time`, `location`, `description`
- [X] Definire `ContextBundle`: aggrega `TimeInfo` + `List[ScheduleEvent]` + `raw_input` — è ciò che viene passato al filtro LLM


**Time e Schedule assumono di vedere l'informazione in quel momento, quinndi non è molto ben strutturata ==> potrebbe dover essere cambiata**

### 1.2 — `tools/time_tool.py`

- [ ] Implementare `get_time() -> TimeInfo`
  - Usa `datetime.now()` per ora e giorno
  - Calcola `time_of_day` con soglie (es. 6-12 mattina, 12-18 pomeriggio…)
  - Calcola stagione dal mese
- [ ] Decorare con `@mcp.tool()` e docstring chiara (il LLM la usa per decidere quando invocarla)

### 1.3 — `tools/schedule.py`

- [ ] Implementare `get_schedule(date: Optional[str] = None) -> List[ScheduleEvent]`
  - **Versione mock**: restituisce eventi hardcoded plausibili (terapia, scuola, pasto, ecc.)
  - **Bonus**: integrare con Google Calendar via `gcal.mcp.claude.com` o libreria `google-api-python-client`
- [ ] Decorare con `@mcp.tool()`

### 1.4 — `tools/arasaac.py`

- [ ] Leggere le API ARASAAC: `https://api.arasaac.org/v1/`
  - Endpoint utile: `GET /pictograms/{lang}/search/{keyword}`
  - Ritorna lista di pittogrammi con `_id`, `keywords`, URL immagine costruibile come `https://static.arasaac.org/pictograms/{id}/{id}_2500.png`
- [ ] Implementare `search_pictograms(keyword: str, lang: str = "it", max_results: int = 5) -> List[PictogramResult]`
  - Chiama l'API per ogni keyword
  - Parsing della risposta JSON
  - Gestione errori (timeout, keyword non trovata)
- [ ] Implementare `get_pictogram_metadata(pictogram_id: int) -> dict`
  - Recupera tags, categorie e descrizione di un singolo pittogramma
  - Utile per il filtro LLM
- [ ] Decorare entrambi con `@mcp.tool()`

### 1.5 — `server.py`

- [ ] Istanziare `mcp = FastMCP("CAA Pictogram Service")`
- [ ] Importare e registrare tutti i tool dai sottomoduli
- [ ] Testare il server in isolamento con `client_session` (come nel notebook del prof)

---

## FASE 2 — Agent (`agent/`)

### 2.1 — `prompts.py`

- [ ] Scrivere il **system prompt** dell'agente ReAct:
  - Ruolo: assistente per la CAA
  - Quando usare `get_time` vs `get_schedule` vs `search_pictograms`
  - Regola chiave: se il contesto è vago → arricchire prima; se è dettagliato → andare diretto all'API
- [ ] Scrivere il **prompt di filtro** (usato nella fase di selezione LLM):
  - Input: contesto arricchito + lista pittogrammi con tags
  - Output: JSON con i pittogrammi selezionati e motivazione sintetica

### 2.2 — `agent.py`

- [ ] Inizializzare il modello Ollama con `langchain-ollama`
- [ ] Connettersi al MCP server con `client_session` e raccogliere i tool
- [ ] Creare il ReAct agent con `create_react_agent(model, tools)`
- [ ] Esporre funzione `run_agent(user_input: str) -> dict` che ritorna i messaggi finali

---

## FASE 3 — Pipeline (`pipeline/pipeline.py`)

Questa è la parte centrale del progetto.

- [ ] Implementare `run_pipeline(context_description: str) -> PipelineResult`
  con i seguenti step espliciti:

  **Step 1 — Analisi del contesto**

  - [ ] Classificare l'input come "vago" o "dettagliato" (semplice euristica: lunghezza, presenza di orari/luoghi espliciti)

  **Step 2 — Arricchimento del contesto**

  - [ ] Se vago: invocare `get_time` e `get_schedule` tramite l'agente
  - [ ] Costruire un `ContextBundle` con le info recuperate

  **Step 3 — Estrazione keywords**

  - [ ] Usare l'LLM (chiamata separata, non agentica) per estrarre 3-6 keyword rilevanti dal contesto arricchito
  - [ ] Esempio: "Siamo in biblioteca la mattina" → `["biblioteca", "libro", "silenzio", "leggere"]`

  **Step 4 — Retrieval pittogrammi**

  - [ ] Per ogni keyword chiamare `search_pictograms`
  - [ ] Raccogliere un pool grezzo (es. 20-30 pittogrammi totali, con duplicati)
  - [ ] Deduplicare per `id`

  **Step 5 — Filtro LLM**

  - [ ] Chiamare l'LLM con il prompt di filtro + il pool grezzo
  - [ ] Ottenere i 5-8 pittogrammi più adatti al contesto
  - [ ] Il LLM restituisce anche una breve motivazione per ognuno

  **Step 6 — Output strutturato**

  - [ ] Restituire `PipelineResult`: pittogrammi selezionati + contesto arricchito + log degli step

---

## FASE 4 — Output & Visualizzazione (`output/renderer.py`)

*Aggiunta extra — non richiesta dalla consegna ma utile per la presentazione.*

- [ ] Implementare `render_html(pipeline_result: PipelineResult) -> str`
  - Genera una pagina HTML con:
    - Griglia dei pittogrammi (immagine + etichetta)
    - Badge con contesto arricchito (ora, eventi del giorno)
    - Motivazione del LLM sotto ogni pittogramma
- [ ] Implementare `render_json(pipeline_result: PipelineResult) -> str`
  - Export pulito del risultato per eventuali integrazioni future
- [ ] **Bonus**: aggiungere `render_pdf()` con `fpdf2` per un output stampabile (utile per caregiver)

---

## FASE 5 — Testing (`tests/`)

*Aggiunta extra — dimostra rigore nel progetto.*

- [ ] `test_time_tool.py`: verifica che `TimeInfo` sia corretta per diverse ore simulate
- [ ] `test_arasaac.py`: verifica che la ricerca ritorni risultati validi per keyword comuni (mock dell'API con `responses` o chiamate reali)
- [ ] `test_pipeline.py`: testa l'intera pipeline con 3 scenari:
  - Input vago: `"Siamo fuori casa"`
  - Input contestualizzato: `"È mattina e stiamo andando a scuola"`
  - Input specifico: `"Voglio fare colazione, ho fame"`

---

## FASE 6 — Notebook Demo (`notebook_demo.ipynb`)

*Da fare per ultimo, raccoglie tutto in modo presentabile.*

- [ ] Sezione 1: Setup e installazione dipendenze
- [ ] Sezione 2: Avvio Ollama (con colab-xterm se su Colab)
- [ ] Sezione 3: Avvio MCP Server e test dei singoli tool
- [ ] Sezione 4: Demo della pipeline completa con 2-3 scenari
  - Mostrare i log step-by-step (come fa il notebook del prof con il trace ReAct)
  - Visualizzare i pittogrammi inline con `IPython.display.Image`
- [ ] Sezione 5: Visualizzazione HTML del risultato finale (opzionale)

---

## Aggiunte Extra (opzionali ma consigliate)

| Idea                                                                       | Motivazione                                     |
| -------------------------------------------------------------------------- | ----------------------------------------------- |
| **Caching ARASAAC** con `functools.lru_cache` o file JSON          | Le keyword ripetute non rifanno richieste API   |
| **Multi-keyword parallela** con `asyncio.gather`                   | Velocizza il retrieval                          |
| **Score di rilevanza** calcolato dall'LLM (0-1) per ogni pittogramma | Rende il filtro più trasparente                |
| **Supporto multilingua** (parametro `lang` in config)              | ARASAAC supporta 20+ lingue                     |
| **CLI minimale** con `argparse`                                    | Permette di testare fuori dal notebook          |
| **Logging strutturato** con `logging` module                       | Traccia ogni step della pipeline in modo pulito |

---

## Ordine di Sviluppo Consigliato

```
1. config.py + models.py
2. tools/time_tool.py  →  test manuale
3. tools/arasaac.py    →  test manuale con 1-2 keyword
4. tools/schedule.py   →  versione mock
5. server.py           →  verifica con client_session
6. prompts.py          →  iterare finché il filtro LLM funziona bene
7. agent.py            →  test ReAct su scenario semplice
8. pipeline.py         →  integrare tutto
9. renderer.py         →  output HTML
10. tests/             →  copertura minima
11. notebook_demo.ipynb →  ultima cosa
```

---

## Note Tecniche

- **ARASAAC API**: nessuna autenticazione richiesta, ma rispetta i rate limit (non spammare)
- **Immagine pittogramma**: URL pattern → `https://static.arasaac.org/pictograms/{id}/{id}_2500.png`
- **Ollama**: modelli raccomandati `granite4:3b` o `llama3.2:3b` per restare leggeri su Colab
- **LangChain**: usare `create_react_agent` da `langgraph.prebuilt` come nel notebook del prof
- La chiamata di **filtro LLM** (Step 5 della pipeline) può essere una semplice `ChatOllama.invoke()` — non serve passare per l'agente ReAct, è una chiamata diretta con prompt strutturato
