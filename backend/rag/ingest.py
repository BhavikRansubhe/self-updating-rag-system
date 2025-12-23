import os
import time
import hashlib
from typing import List, Dict, Any

from .db import DB
from .embedder import Embedder
from .index import VectorIndex

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap_chars)
    return out

def ingest_docs(db: DB, index: VectorIndex, embedder: Embedder, docs_dir: str,
                chunk_chars: int, overlap_chars: int) -> Dict[str, Any]:
    started = time.time()
    now = int(time.time())
    summary = {
        "docs_scanned": 0,
        "docs_changed": 0,
        "docs_unchanged": 0,
        "chunks_added": 0,
        "chunks_updated": 0,
        "chunks_deactivated": 0,
        "embed_calls": 0,
        "seconds": 0.0,
    }

    # Cache documents list for lookup
    existing_docs = {d.path: d for d in db.list_documents()}

    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if fn.startswith("."):
                continue
            if not (fn.endswith(".md") or fn.endswith(".txt")):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, docs_dir)

            summary["docs_scanned"] += 1
            raw = read_text(path)
            doc_hash = sha256_text(raw)

            doc_row = existing_docs.get(rel)
            if doc_row and doc_row.doc_hash == doc_hash:
                summary["docs_unchanged"] += 1
                continue

            # Create a new immutable version for this document.
            new_max = 1 if not doc_row else int(doc_row.max_version or doc_row.version) + 1
            active_version = new_max
            doc_row = db.upsert_document(rel, doc_hash, active_version, new_max, now)
            existing_chunks = {c.chunk_index: c for c in db.list_chunks_for_doc(doc_row.doc_id)}

            chunks = chunk_text(raw, chunk_chars, overlap_chars)
            new_hashes = [sha256_text(c) for c in chunks]

            to_embed_texts = []
            to_embed_chunk_ids = []
            deleted_chunk_ids = []

            new_indices = set(range(len(chunks)))

            # updates/additions
            for idx, (txt, h) in enumerate(zip(chunks, new_hashes)):
                if idx in existing_chunks and existing_chunks[idx].chunk_hash == h:
                    continue
                if idx in existing_chunks:
                    deleted_chunk_ids.append(existing_chunks[idx].chunk_id)
                    summary["chunks_updated"] += 1
                else:
                    summary["chunks_added"] += 1

                cid = db.insert_chunk(doc_row.doc_id, idx, h, txt, active_version, now)
                to_embed_texts.append(txt)
                to_embed_chunk_ids.append(cid)

            # deletions (doc got shorter)
            for idx, old in existing_chunks.items():
                if idx not in new_indices:
                    deleted_chunk_ids.append(old.chunk_id)

            # We keep old chunks/vectors to support rollback & diffing.
            # Deletions/updates are treated as "deactivated" for the active version.
            deleted_chunk_ids = sorted(set(deleted_chunk_ids))
            if deleted_chunk_ids:
                summary["chunks_deactivated"] += len(deleted_chunk_ids)

            if to_embed_texts:
                vecs = embedder.embed_texts(to_embed_texts)
                summary["embed_calls"] += 1
                rows = index.add_vectors(vecs, to_embed_chunk_ids)
                for cid, row in zip(to_embed_chunk_ids, rows):
                    db.set_vector_row(cid, row)

            summary["docs_changed"] += 1

    summary["seconds"] = round(time.time() - started, 3)
    return summary
