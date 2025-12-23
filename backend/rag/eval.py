import json
from typing import Dict, Any
from .db import DB
from .embedder import Embedder
from .index import VectorIndex
from .llm import LLM

def run_eval(golden_path: str, db: DB, index: VectorIndex, embedder: Embedder, llm: LLM, top_k: int) -> Dict[str, Any]:
    with open(golden_path, "r") as f:
        golden = json.load(f)

    docs = db.list_documents()
    docs_map = {d.doc_id: {"path": d.path, "active_version": d.version} for d in docs}

    results = []
    passed = 0
    for item in golden:
        q = item["question"]
        must = set(item.get("must_cite", []))

        qvec = embedder.embed_texts([q])[0]
        hits = index.search(qvec, top_k=top_k)

        sources = set()
        ctx = []
        for cid, score in hits:
            crow = db.get_chunk_by_id(cid)
            if not crow:
                continue
            meta = docs_map.get(crow.doc_id)
            if not meta:
                continue
            if crow.version != meta["active_version"]:
                continue
            src = meta["path"]
            sources.add(src)
            ctx.append({"source_path": src, "chunk_id": cid, "score": score, "text": crow.text})

        ans = llm.answer(q, ctx)
        ok = True if not must else any(m in sources for m in must)
        if ok:
            passed += 1
        results.append({
            "id": item.get("id"),
            "question": q,
            "must_cite": list(must),
            "retrieved_sources": sorted(list(sources)),
            "pass": ok,
            "answer_preview": ans[:240] + ("…" if len(ans) > 240 else "")
        })

    total = len(results)
    return {"total": total, "passed": passed, "pass_rate": round(passed / max(1,total), 3), "results": results}
