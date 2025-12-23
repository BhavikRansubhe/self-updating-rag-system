import difflib
from typing import Dict, Any, List

from .db import DB


def _chunks_by_index(db: DB, doc_id: int, version: int) -> Dict[int, str]:
    rows = db.list_chunks_for_doc_version(doc_id, version)
    return {r.chunk_index: r.text for r in rows}


def _unified(a: str, b: str, from_name: str, to_name: str) -> str:
    a_lines = a.splitlines(keepends=False)
    b_lines = b.splitlines(keepends=False)
    diff = difflib.unified_diff(
        a_lines,
        b_lines,
        fromfile=from_name,
        tofile=to_name,
        lineterm="",
    )
    return "\n".join(diff)


def diff_doc_versions(db: DB, path: str, from_version: int, to_version: int) -> Dict[str, Any]:
    """Compute per-chunk unified diffs between two stored versions.

    Returns a dict suitable for UI consumption:
      - summary counts
      - per_chunk diffs
      - combined diff string
    """
    doc = db.get_document_by_path(path)
    if not doc:
        return {"error": "Document not found"}

    a = _chunks_by_index(db, doc.doc_id, from_version)
    b = _chunks_by_index(db, doc.doc_id, to_version)

    all_idx = sorted(set(a.keys()) | set(b.keys()))
    per_chunk: List[Dict[str, Any]] = []

    added = changed = removed = unchanged = 0
    combined_parts: List[str] = []

    for idx in all_idx:
        a_txt = a.get(idx, "")
        b_txt = b.get(idx, "")
        if idx not in a:
            status = "added"
            added += 1
        elif idx not in b:
            status = "removed"
            removed += 1
        elif a_txt == b_txt:
            status = "unchanged"
            unchanged += 1
        else:
            status = "changed"
            changed += 1

        diff_str = ""
        if status != "unchanged":
            diff_str = _unified(
                a_txt,
                b_txt,
                from_name=f"{path}@v{from_version}:chunk_{idx}",
                to_name=f"{path}@v{to_version}:chunk_{idx}",
            )
            combined_parts.append(diff_str)

        per_chunk.append(
            {
                "chunk_index": idx,
                "status": status,
                "from_len": len(a_txt),
                "to_len": len(b_txt),
                "diff": diff_str,
            }
        )

    return {
        "path": path,
        "from_version": from_version,
        "to_version": to_version,
        "summary": {
            "added": added,
            "changed": changed,
            "removed": removed,
            "unchanged": unchanged,
            "total": len(all_idx),
        },
        "per_chunk": per_chunk,
        "combined_diff": "\n\n".join(combined_parts),
    }
