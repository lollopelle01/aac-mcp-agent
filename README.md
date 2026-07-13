# AAC MCP Agent

This is the course project for **Big Data and Text Mining** at the University of Bologna (UniBo).

For someone who communicates through AAC (Augmentative and Alternative Communication), building a sentence usually means paging through hundreds of pictograms one tap at a time, hoping the right one shows up. This project explores a different approach: the caregiver describes the situation in plain language (e.g. *"he wants something to eat before going out"*) and a local LLM works out what's actually being said, then surfaces the [ARASAAC](https://arasaac.org/) pictograms that match, ready to tap. The agent also remembers what's already been picked in the session, so a sentence gets built one concept at a time across turns instead of restarting from zero on every message.

Under the hood, each caregiver message goes through a 5-phase pipeline:

1. deciding if time/schedule context is needed
2. planning 5-10 candidate concepts with the LLM
3. resolving each one to an exact ARASAAC keyword through a 5-step fallback chain (exact match → lemmatization → hyphenation → tokenization → embedding search)
4. expanding via WordNet synsets
5. ranking and trimming to a final grid.

The agent exposes this through **MCP tools** (pictogram search, time, calendar) and runs fully offline with a local 3B model (anyway online is a fallback or selectable option). Its quality is measured against a purpose-built, annotated evaluation set of caregiver sentences (split into *clear*/*vague* cases).

---

## 1. Project map

```
aac-mcp-agent/

  app/             	  # the complete application used in eval and the local app
  annotation/         # offline pipeline that built the evaluation dataset
  eval/               # notebooks evaluating the agent on that dataset
  test/               # quick/manual testing of the MCP tools in isolation
  launch.sh           # quick start for local app (macOS, two Terminal tabs)
  LICENSE
```

---

## 2. Setup and running

### 2.1 Prerequisites (one-time)

| Dependency                        | Minimum version | Notes                                          |
| --------------------------------- | --------------- | ---------------------------------------------- |
| Python                            | 3.11            |                                                |
| Node.js                           | 18              | for the frontend                               |
| Ollama                            | any             | used as a fallback/alternative to llama.cpp    |
| A GGUF model (e.g.`qwen2.5:3b`) | —              | see`app/models/` and `app/src/settings.py` |

### 2.2 Python environment

Use a virtual environment rather than installing dependencies system-wide:

```bash
cd app
python3 -m venv venv
source venv/bin/activate  

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Remember to run `source venv/bin/activate` again in any new terminal session before working on the project.

If you use Ollama instead of (or in addition to) llama.cpp:

```bash
ollama pull qwen2.5:3b
```

By default the agent uses **in-process llama.cpp** (`agent_use_llamacpp: true` in `user_settings.json`), so Ollama isn't strictly required unless you want to use it as an alternative backend.

### 2.3 Configuration (one-time, optional)

- Copy `app/.env.example` → `app/.env` and fill in only the credentials you actually need (Google Calendar, Apple iCloud, HuggingFace for cluster evaluation). **None of these are required** for basic operation (pictograms + local LLM).
- Operational settings (active model, number of results, etc.) are managed from the interface (Settings) or directly in `app/user_settings.json` (created automatically on first run).

### 2.4 Starting the app

```bash
# Terminal 1: backend (FastAPI on :8000)
cd app
source venv/bin/activate
uvicorn src.api.server:app --reload --port 8000

# Terminal 2: frontend (Vite on :5173)
cd app/frontend
npm install        # first time only
npm run dev
```

Open **http://localhost:5173**. The Vite proxy forwards `/api/*` → `:8000`.

On macOS, `./launch.sh` automatically opens the two Terminal tabs with both commands, i made it for quicker debug.

---

## 3. Where to go next

- [`app/README.md`](https://github.com/lollopelle01/aac-mcp-agent/blob/main/app/README.md) : how to set up and run the app, and how the code works (agent pipeline, MCP tools, API, frontend). Start here to run or understand the system.
- [`app/datasets/README.md`](https://github.com/lollopelle01/aac-mcp-agent/blob/main/app/datasets/README.md) : where the pictogram data comes from and how the local dataset is structured.
- [`annotation/README.md`](https://github.com/lollopelle01/aac-mcp-agent/blob/main/annotation/README.md) : how the evaluation dataset was built.
- [`eval/README.md`](https://github.com/lollopelle01/aac-mcp-agent/blob/main/eval/README.md) : how the agent's quality is measured against that dataset.
