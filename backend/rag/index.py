import os
import json
from typing import List, Tuple
import numpy as np

class VectorIndex:
    def __init__(self, data_dir: str, dim: int):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.index_path = os.path.join(self.data_dir, "index.faiss")
        self.map_path = os.path.join(self.data_dir, "index_map.json")
        self.dim = dim

        import faiss
        self.faiss = faiss

        self.index = None
        self.row_to_chunk: List[int] = []
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            self.index = self.faiss.read_index(self.index_path)
            with open(self.map_path, "r") as f:
                self.row_to_chunk = json.load(f)
        else:
            self.index = self.faiss.IndexFlatIP(self.dim)
            self.row_to_chunk = []

    def save(self):
        self.faiss.write_index(self.index, self.index_path)
        with open(self.map_path, "w") as f:
            json.dump(self.row_to_chunk, f)

    def add_vectors(self, vectors: np.ndarray, chunk_ids: List[int]) -> List[int]:
        start = len(self.row_to_chunk)
        self.index.add(vectors.astype("float32"))
        self.row_to_chunk.extend([int(x) for x in chunk_ids])
        self.save()
        return list(range(start, start + len(chunk_ids)))

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        q = query_vec.reshape(1, -1).astype("float32")
        scores, idxs = self.index.search(q, top_k)
        out = []
        for row, score in zip(idxs[0], scores[0]):
            if row == -1:
                continue
            cid = int(self.row_to_chunk[row])
            if cid == -1:
                continue
            out.append((cid, float(score)))
        return out

    def logical_delete_chunk_ids(self, deleted_chunk_ids: List[int]):
        """FAISS flat index does not support deletions.
        For a demo-friendly incremental approach, mark deleted rows in mapping as -1."""
        if not deleted_chunk_ids:
            return
        s = set(int(x) for x in deleted_chunk_ids)
        changed = False
        for i, cid in enumerate(self.row_to_chunk):
            if cid in s:
                self.row_to_chunk[i] = -1
                changed = True
        if changed:
            self.save()
