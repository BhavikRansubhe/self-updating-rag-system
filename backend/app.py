import os
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from config import (
    DATA_DIR, DOCS_DIR, EMBED_PROVIDER, LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBED_MODEL,
    CHUNK_CHARS, CHUNK_OVERLAP_CHARS, TOP_K
)
from rag.db import DB
from rag.embedder import Embedder
from rag.index import VectorIndex
from rag.ingest import ingest_docs
from rag.llm import LLM
from rag.eval import run_eval
from rag.diffing import diff_doc_versions

from dotenv import load_dotenv

load_dotenv()  # <-- THIS loads .env into os.environ
# Retrieval confidence gate (from .env)
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))
SCORE_WINDOW = float(os.getenv("SCORE_WINDOW", "0.08"))
MAX_CONTEXTS = int(os.getenv("MAX_CONTEXTS", "4"))

app = FastAPI(title="Self-Updating RAG Demo")

os.makedirs(DATA_DIR, exist_ok=True)
db = DB(os.path.join(DATA_DIR, "rag.sqlite"))

embedder = Embedder(EMBED_PROVIDER, OPENAI_API_KEY, OPENAI_EMBED_MODEL)
index = VectorIndex(DATA_DIR, dim=embedder.dim)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

llm = LLM(
    provider=LLM_PROVIDER,
    api_key=OPENROUTER_API_KEY,
    model=OPENROUTER_MODEL,
    base_url=OPENROUTER_BASE_URL,
)

class ChatRequest(BaseModel):
    query: str

class Citation(BaseModel):
    source_path: str
    chunk_id: int
    score: float
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    llm_meta: Dict[str, Any] = {}


class RollbackRequest(BaseModel):
    path: str
    version: int

@app.get("/health")
def health():
    return {"ok": True, "embed_provider": EMBED_PROVIDER, "llm_provider": LLM_PROVIDER}

@app.post("/documents/ingest")
def documents_ingest():
    return ingest_docs(db, index, embedder, DOCS_DIR, CHUNK_CHARS, CHUNK_OVERLAP_CHARS)

@app.get("/documents/status")
def documents_status():
    docs = db.list_documents()
    out = []
    for d in docs:
        chunks = db.list_chunks_for_doc_version(d.doc_id, d.version)
        out.append({
            "path": d.path,
            "doc_hash": d.doc_hash[:12],
            "active_version": d.version,
            "max_version": int(d.max_version or d.version),
            "active_chunks": len(chunks),
            "updated_at": d.updated_at
        })
    return {"documents": out, "total_chunks_all_versions": db.count_chunks()}



@app.post("/documents/upload")
async def documents_upload(files: List[UploadFile] = File(...)):
    """Upload one or more .md/.txt docs into the docs folder."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    saved = []
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name:
            continue
        if not (name.endswith(".md") or name.endswith(".txt")):
            raise HTTPException(status_code=400, detail="Only .md or .txt files are supported in this demo")
        data = await f.read()
        # Limit to ~2MB per file for safety in simple hosting.
        if len(data) > 2 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File too large: {name}")
        abs_path = os.path.abspath(os.path.join(DOCS_DIR, name))
        docs_root = os.path.abspath(DOCS_DIR)
        if not abs_path.startswith(docs_root + os.sep) and abs_path != docs_root:
            raise HTTPException(status_code=400, detail="Invalid filename")
        with open(abs_path, "wb") as out:
            out.write(data)
        saved.append(name)
    return {"saved": saved}

def _safe_doc_path(rel_path: str) -> str:
    name = rel_path.replace("\\", "/").lstrip("/")
    abs_path = os.path.abspath(os.path.join(DOCS_DIR, name))
    docs_root = os.path.abspath(DOCS_DIR)
    if not abs_path.startswith(docs_root + os.sep):
        raise HTTPException(status_code=400, detail="Invalid document path")
    return abs_path

@app.get("/documents/content", response_class=PlainTextResponse)
def documents_content(path: str):
    abs_path = _safe_doc_path(path)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="Document not found")
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

class ContentUpdate(BaseModel):
    path: str
    content: str

@app.post("/documents/content")
def documents_update(req: ContentUpdate):
    abs_path = _safe_doc_path(req.path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(req.content)
    return {"ok": True, "path": req.path}

@app.get("/documents/versions")
def documents_versions(path: str):
    doc = db.get_document_by_path(path)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    max_v = int(doc.max_version or doc.version)
    return {"path": doc.path, "active_version": doc.version, "max_version": max_v, "versions": list(range(1, max_v + 1))}


@app.post("/documents/rollback")
def documents_rollback(req: RollbackRequest):
    doc = db.get_document_by_path(req.path)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    max_v = int(doc.max_version or doc.version)
    if req.version < 1 or req.version > max_v:
        raise HTTPException(status_code=400, detail=f"Version must be between 1 and {max_v}")
    now = int(time.time())
    updated = db.set_active_version(req.path, req.version, now)
    return {"path": updated.path, "active_version": updated.version, "max_version": int(updated.max_version or updated.version), "updated_at": updated.updated_at}


@app.get("/documents/diff")
def documents_diff(path: str, from_version: int, to_version: int):
    doc = db.get_document_by_path(path)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    max_v = int(doc.max_version or doc.version)
    for v in (from_version, to_version):
        if v < 1 or v > max_v:
            raise HTTPException(status_code=400, detail=f"Version {v} out of range (1..{max_v})")
    return diff_doc_versions(db, path, from_version, to_version)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = req.query.strip()

    # Embed query
    qvec = embedder.embed_texts([q])[0]

    # Retrieve candidates (cosine similarity via inner product of normalized vectors)
    hits = index.search(qvec, top_k=TOP_K)

    docs = db.list_documents()
    docs_map = {d.doc_id: {"path": d.path, "active_version": d.version} for d in docs}

    citations = []
    contexts = []

    for cid, score in hits:
        crow = db.get_chunk_by_id(cid)
        if not crow:
            continue

        meta = docs_map.get(crow.doc_id)
        if not meta:
            continue

        # Only use chunks from the ACTIVE version of that document
        if int(crow.version) != int(meta["active_version"]):
            continue

        src = meta["path"]
        contexts.append({"source_path": src, "chunk_id": cid, "score": float(score), "text": crow.text})

    # Sort by score desc
    contexts.sort(key=lambda x: x["score"], reverse=True)

    # ---- Retrieval confidence gate (prevents out-of-context questions) ----
    if not contexts or contexts[0]["score"] < MIN_RELEVANCE_SCORE:
        contexts = []
        citations = []
    else:
        best = contexts[0]["score"]
        threshold = max(MIN_RELEVANCE_SCORE, best - SCORE_WINDOW)
        contexts = [c for c in contexts if c["score"] >= threshold]
        contexts = contexts[:MAX_CONTEXTS]

        # Build citations from the filtered contexts
        citations = []
        for c in contexts:
            snippet = (c["text"] or "").strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "…"
            citations.append({
                "source_path": c["source_path"],
                "chunk_id": c["chunk_id"],
                "score": c["score"],
                "snippet": snippet
            })

    answer, llm_meta = llm.answer(q, contexts)
    return {"answer": answer, "citations": citations, "llm_meta": llm_meta}

@app.post("/eval/run")
def eval_run():
    golden_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval_golden.json"))
    return run_eval(golden_path, db, index, embedder, llm, top_k=TOP_K)
