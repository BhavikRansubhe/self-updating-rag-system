from typing import List
import numpy as np

class Embedder:
    def __init__(self, provider: str, openai_api_key: str = "", openai_embed_model: str = "text-embedding-3-small"):
        self.provider = provider.lower()
        self.openai_api_key = openai_api_key
        self.openai_embed_model = openai_embed_model
        self._model = None
        self._openai = None

        if self.provider == "sbert":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        elif self.provider == "openai":
            if not openai_api_key:
                raise RuntimeError("OPENAI_API_KEY required for EMBED_PROVIDER=openai")
            from openai import OpenAI
            self._openai = OpenAI(api_key=openai_api_key)
        else:
            raise ValueError(f"Unknown EMBED_PROVIDER: {provider}")

    @property
    def dim(self) -> int:
        if self.provider == "sbert":
            return int(self._model.get_sentence_embedding_dimension())
        vec = self.embed_texts(["test"])
        return int(vec.shape[1])

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.provider == "sbert":
            arr = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return arr.astype("float32")
        resp = self._openai.embeddings.create(model=self.openai_embed_model, input=texts)
        data = [d.embedding for d in resp.data]
        arr = np.array(data, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype("float32")
