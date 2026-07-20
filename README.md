# AAC MCP Agent

Course project for **Big Data and Text Mining** at the University of Bologna (UniBo).

For someone who communicates through AAC (Augmentative and Alternative Communication), building a sentence usually means paging through hundreds of pictograms one tap at a time, hoping the right one shows up. This project explores a different approach: the caregiver describes the situation in plain language (es. *"he wants something to eat before going out"*) and a local LLM works out what's actually being said, then surfaces the [ARASAAC](https://arasaac.org/) pictograms that match, ready to tap. The agent also remembers what's already been picked in the session, so a sentence gets built one concept at a time across turns instead of restarting from zero on every message.

Each caregiver message goes through a 5-phase pipeline: deciding if time/schedule context is needed, planning 5-10 candidate concepts with the LLM, resolving each one to an exact ARASAAC keyword through a 5-step fallback chain (exact match → lemmatization → hyphenation → tokenization → embedding search), expanding via WordNet synsets, then ranking and trimming to a final grid.

The agent exposes this through MCP tools (pictogram search, time, calendar) and runs fully offline with a local 3B model (online is a fallback/selectable option). Its quality is measured against a purpose-built, annotated evaluation set of caregiver sentences, split into *clear*/*vague* cases.

## Project map

```
aac-mcp-agent/
  app/              the complete application used in eval and the local app
  annotation/       offline pipeline that built the evaluation dataset
  eval/             notebooks evaluating the agent on that dataset
  test/             manual testing of the MCP tools, including a real MCP protocol demo
  launch.sh         quick start for local app (macOS, two Terminal tabs)
```

`test/mcp_protocol_demo.ipynb` is worth a special mention: everywhere else the agent calls its MCP tools as plain Python imports (see [`app/README.md` § MCP server and tools](app/README.md#mcp-server-and-tools-srcmcp_server)), this notebook is the one place in the project where the same tools run behind a real client/server connection, JSON-RPC over stdio, a separate subprocess.

## Setup and running

Prerequisites, one-time: Python 3.11, Node.js 18 for the frontend, Ollama as a fallback/alternative to llama.cpp, and a GGUF model such as `qwen2.5:3b` (see `app/models/` and `app/src/settings.py`).

Python environment:

```bash
cd app
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run `source venv/bin/activate` again in any new terminal session before working on the project.

If you use Ollama instead of (or in addition to) llama.cpp:

```bash
ollama pull qwen2.5:3b
```

By default the agent uses in-process llama.cpp (`agent_use_llamacpp: true` in `user_settings.json`), Ollama isn't required unless you want it as an alternative backend.

Configuration, one-time, optional: copy `app/.env.example` to `app/.env` and fill in only the credentials you actually need (Google Calendar, Apple iCloud, HuggingFace for cluster evaluation). None of these are required for basic operation, pictograms + local LLM work without them. Operational settings (active model, number of results, etc.) are managed from the interface (Settings) or directly in `app/user_settings.json`, created automatically on first run.

Starting the app:

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

Open http://localhost:5173. The Vite proxy forwards `/api/*` → `:8000`.

On macOS, `./launch.sh` opens both Terminal tabs automatically.

## Missing data (this is an essential archive)

To fit this project in a 100MB submission zip, two things were removed. Both are fully regenerable, and neither is required to read or grade the code, only to actually run the app fully offline or view pictograms as images.

1. `app/models/*.gguf` (~10 GB): needed to run the agent with the in-process llama.cpp backend. Download a GGUF build of the model (e.g. `qwen2.5:3b`) into `app/models/`, or use the Ollama backend instead (`ollama pull qwen2.5:3b`), see [`app/README.md`](app/README.md#interchangeable-llm-backends-srcagentbackendspy).
2. `app/datasets/pictograms/*.png` (~3GB, thousands of images): needed to view pictogram images in the UI, not for retrieval/eval logic. Regenerate with `python datasets/update_datasets.py --download-images`, a pre-downloaded archive via gdown, or let the app fetch them lazily from the ARASAAC CDN, see [`app/datasets/README.md § Images`](app/datasets/README.md#images).
3. `app/datasets/en/keyword_embeddings.npz`, `app/datasets/en_eval/keyword_embeddings.npz` (~40 MB total): the semantic-embedding fallback step in concept resolution, optional, the other 4 resolution steps work without it. Regenerate `en/` with `python datasets/build_keyword_embeddings.py --lang en` while `en_eval/` by re-running `annotation/arasaac_vs_hf_vs_eval.ipynb`, see [`app/datasets/README.md § Keyword embeddings`](app/datasets/README.md#keyword-embeddings) and [`annotation/README.md`](annotation/README.md).

All other dataset files (`pictograms.json`, `keywords.json`, `keyword_index.json`, `synset_index.json` for both `en` and `en_eval`) are included as-is in this archive.

## Where to go next

`app/README.md`: how to set up and run the app, and how the code works, agent pipeline, MCP tools, API, frontend.

`app/datasets/README.md`: where the pictogram data comes from and how the local dataset is structured.

`annotation/README.md`: how the evaluation dataset was built.

`eval/README.md`: how the agent's quality is measured against that dataset.
