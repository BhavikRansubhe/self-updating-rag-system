# Self-Updating RAG Demo (Incremental Indexing + Versioning)

This project is a **self-updating RAG** system:
- Ingests documents from `./docs`
- Detects document changes via hashing
- Re-chunks and **re-embeds only changed chunks**
- Stores metadata in SQLite and vectors in a FAISS index
- Provides a FastAPI backend + a Streamlit demo UI
- Includes a tiny evaluation harness with "golden" questions

## Quick Start (works without API keys)

### Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run backend
```bash
cd backend
uvicorn app:app --reload --port 8000
```

### Run demo UI
In another terminal:
```bash
cd frontend
streamlit run streamlit_app.py
```

Open Streamlit, click **Ingest / Update Index**, then start asking questions.


## Using OpenAI for best answers (recommended)

The demo works without an API key (fallback mode), but you'll get the cleanest, most "assistant-like" answers with OpenAI.

Set these environment variables before starting the backend:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export LLM_PROVIDER="openai"
export EMBED_PROVIDER="openai"
# Optional:
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_EMBED_MODEL="text-embedding-3-small"
```

Then restart the backend.

## Environment (.env)
Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The backend loads `.env` automatically.
