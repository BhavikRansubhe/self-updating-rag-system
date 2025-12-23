from typing import List, Dict, Any, Tuple, Optional
import os
import requests
import re


class LLM:
    def __init__(
        self,
        provider: str,
        api_key: str = "",
        model: str = "openai/gpt-oss-120b:free",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: int = 60,
    ):
        self.provider = (provider or "").lower()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

        if self.provider == "openrouter" and not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY required for LLM_PROVIDER=openrouter")

    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        if not contexts:
            return (
                "I don't have enough information in the indexed documents to answer that.",
                {"llm_provider_used": "none", "reason": "no_contexts"},
            )

        # compact context
        top = contexts[:4]
        ctx = "\n\n".join(
            f"[{c.get('source','unknown')}:{c.get('chunk_index','?')}]\n{(c.get('text') or '').strip()}"
            for c in top
            if (c.get("text") or "").strip()
        ).strip()

        if not ctx:
            return (
                "I don't have enough information in the indexed documents to answer that.",
                {"llm_provider_used": "none", "reason": "empty_context"},
            )

        if self.provider == "openrouter":
            return self._openrouter_answer(question, ctx)

        # fallback local
        return self._local_fallback(question, contexts), {"llm_provider_used": "local"}

    def _openrouter_answer(self, question: str, ctx: str) -> Tuple[str, Dict[str, Any]]:
        sys = (
            "You are a helpful assistant answering questions using ONLY the provided context. "
            "Answer directly and concisely in 2-5 sentences. "
            "Do NOT list excerpts. Do NOT mention retrieval. "
            "If the context is insufficient, reply exactly: "
            "'I don't have enough information in the indexed documents to answer that.'"
        )
        user = f"Question: {question}\n\nContext:\n{ctx}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter:
            # "HTTP-Referer": "http://localhost",
            # "X-Title": "Self-Updating-RAG-Demo",
        }

        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
            if r.status_code != 200:
                return (
                    "I don't have enough information in the indexed documents to answer that.",
                    {"llm_provider_used": "openrouter", "error": r.text},
                )

            data = r.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()

            meta = {
                "llm_provider_used": "openrouter",
                "model_used": self.model,
                "openrouter_id": data.get("id"),
                "usage": data.get("usage", {}),
            }
            return text, meta

        except Exception as e:
            return (
                "I don't have enough information in the indexed documents to answer that.",
                {"llm_provider_used": "openrouter", "error": repr(e)},
            )

    def _local_fallback(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        best = contexts[0]
        text = (best.get("text") or "").strip()
        if not text:
            return "I don't have enough information in the indexed documents to answer that."

        q = question.lower()
        stop = {
            "what","when","where","which","that","this","with","from","your",
            "have","will","should","does","do","is","are","the","and","for","into","about"
        }
        q_terms = [t for t in re.findall(r"[a-zA-Z]{3,}", q) if t not in stop]
        sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))

        scored = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            sl = s.lower()
            score = sum(1 for t in q_terms if t in sl)
            scored.append((score, s))

        if not scored:
            return text[:350].strip() + ("…" if len(text) > 350 else "")

        scored.sort(key=lambda x: (-x[0], len(x[1])))
        return " ".join([s for _, s in scored[:3]]).strip() or text[:350].strip()