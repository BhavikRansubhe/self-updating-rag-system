import os
import sqlite3
from dataclasses import dataclass
from typing import Optional, List

SCHEMA = '''
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
  doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT UNIQUE NOT NULL,
  doc_hash TEXT NOT NULL,
  version INTEGER NOT NULL,
  max_version INTEGER,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL,
  chunk_hash TEXT NOT NULL,
  text TEXT NOT NULL,
  version INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS vectors (
  chunk_id INTEGER PRIMARY KEY,
  vector_row INTEGER NOT NULL
);
'''

@dataclass
class DocRow:
    doc_id: int
    path: str
    doc_hash: str
    version: int
    max_version: Optional[int]
    updated_at: int

@dataclass
class ChunkRow:
    chunk_id: int
    doc_id: int
    chunk_index: int
    chunk_hash: str
    text: str
    version: int
    updated_at: int

class DB:
    def __init__(self, sqlite_path: str):
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        for stmt in SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                cur.execute(s + ";")
        # Lightweight migrations for older DBs
        cur.execute("PRAGMA table_info(documents)")
        cols = {r[1] for r in cur.fetchall()}
        if "max_version" not in cols:
            cur.execute("ALTER TABLE documents ADD COLUMN max_version INTEGER")
            cur.execute("UPDATE documents SET max_version = version")
        self.conn.commit()

    def get_document_by_path(self, path: str) -> Optional[DocRow]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM documents WHERE path=?", (path,))
        row = cur.fetchone()
        return DocRow(**dict(row)) if row else None

    def upsert_document(self, path: str, doc_hash: str, version: int, max_version: int, updated_at: int) -> DocRow:
        cur = self.conn.cursor()
        existing = self.get_document_by_path(path)
        if existing:
            cur.execute("UPDATE documents SET doc_hash=?, version=?, max_version=?, updated_at=? WHERE path=?",
                        (doc_hash, version, max_version, updated_at, path))
        else:
            cur.execute("INSERT INTO documents(path, doc_hash, version, max_version, updated_at) VALUES (?,?,?,?,?)",
                        (path, doc_hash, version, max_version, updated_at))
        self.conn.commit()
        return self.get_document_by_path(path)

    def set_active_version(self, path: str, version: int, updated_at: int) -> Optional[DocRow]:
        cur = self.conn.cursor()
        cur.execute("UPDATE documents SET version=?, updated_at=? WHERE path=?", (version, updated_at, path))
        self.conn.commit()
        return self.get_document_by_path(path)

    def list_documents(self) -> List[DocRow]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM documents ORDER BY path")
        return [DocRow(**dict(r)) for r in cur.fetchall()]

    def list_chunks_for_doc(self, doc_id: int) -> List[ChunkRow]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,))
        return [ChunkRow(**dict(r)) for r in cur.fetchall()]

    def list_chunks_for_doc_version(self, doc_id: int, version: int) -> List[ChunkRow]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM chunks WHERE doc_id=? AND version=? ORDER BY chunk_index",
            (doc_id, version),
        )
        return [ChunkRow(**dict(r)) for r in cur.fetchall()]

    def insert_chunk(self, doc_id: int, chunk_index: int, chunk_hash: str, text: str, version: int, updated_at: int) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO chunks(doc_id, chunk_index, chunk_hash, text, version, updated_at) VALUES (?,?,?,?,?,?)",
                    (doc_id, chunk_index, chunk_hash, text, version, updated_at))
        cid = cur.lastrowid
        self.conn.commit()
        return int(cid)

    def set_vector_row(self, chunk_id: int, vector_row: int):
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO vectors(chunk_id, vector_row) VALUES (?,?)", (chunk_id, vector_row))
        self.conn.commit()

    def get_chunk_by_id(self, chunk_id: int) -> Optional[ChunkRow]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        return ChunkRow(**dict(row)) if row else None

    def count_chunks(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM chunks")
        return int(cur.fetchone()["n"])
