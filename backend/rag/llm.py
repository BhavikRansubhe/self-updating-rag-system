from typing import List, Dict, Any, Tuple
import re


class LLM:
    def __init__(
        self,
        provider: str,
        openai_api_key: str = "",
        openai_model: str = "gpt-4o-mini",
    ):
        self.provider = (provider or "").lower()
        if self.provider == "fallback":
            self.provider = "local"
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self._openai = None

        if self.provider == "openai":
            if not openai_api_key:
                raise RuntimeError("OPENAI_API_KEY required for LLM_PROVIDER=openai")
            from openai import OpenAI
            self._openai = OpenAI(api_key=openai_api_key)

    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (answer_text, meta)

        contexts: list of dicts like:
          {
            "text": "...",
            "source_path": "oncall_runbook.md",
            "chunk_id": 123,
            "score": 0.78
          }
        """
        if not contexts:
            return (
                "I don't have enough information in the indexed documents to answer that.",
                {"llm_provider_used": "local", "reason": "no_contexts"},
            )

        # Build a compact context string for the model (or fallback).
        top_contexts = contexts[:4]
        ctx = "\n\n".join(
            f"[{c.get('source_path','unknown')}:{c.get('chunk_id','?')} score={float(c.get('score', 0.0)):.3f}]\n{(c.get('text') or '').strip()}"
            for c in top_contexts
            if (c.get("text") or "").strip()
        ).strip()

        if not ctx:
            return (
                "I don't have enough information in the indexed documents to answer that.",
                {"llm_provider_used": "local", "reason": "empty_context_string"},
            )

        # OpenAI path
        if self.provider == "openai":
            if self._openai is None:
                return (
                    "I don't have enough information in the indexed documents to answer that.",
                    {"llm_provider_used": "local", "reason": "openai_client_missing"},
                )

            sys = (
                "You MUST answer using ONLY the provided context. "
                "Answer the user's question directly and concisely in 2-5 sentences. "
                "Do NOT list excerpts. Do NOT mention retrieval, embeddings, or vector databases. "
                "If the question asks for personal details (family, identity) or anything not stated in the context, "
                "reply exactly: 'I don't have enough information in the indexed documents to answer that.'"
            )
            user = f"Question: {question}\n\nContext:\n{ctx}"

            try:
                resp = self._openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                )
                text = (resp.choices[0].message.content or "").strip()
                meta: Dict[str, Any] = {
                    "llm_provider_used": "openai",
                    "model_used": self.openai_model,
                    "openai_request_id": getattr(resp, "id", None),
                }
                usage = getattr(resp, "usage", None)
                if usage:
                    meta["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                return text, meta
            except Exception as e:
                # fall back locally but record why
                fallback = self._local_fallback(question, contexts)
                return fallback, {
                    "llm_provider_used": "local",
                    "reason": f"openai_error: {type(e).__name__}",
                    "error": str(e),
                }

        # Local fallback
        return self._local_fallback(question, contexts), {"llm_provider_used": "local"}

    def _local_fallback(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        best = contexts[0]
        text = (best.get("text") or "").strip()
        if not text:
            return "I don't have enough information in the indexed documents to answer that."

        q = question.lower()
        stop = {
            "what", "when", "where", "which", "that", "this", "with", "from",
            "your", "have", "will", "should", "does", "do", "is", "are", "the",
            "and", "for", "into", "about"
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
        picked = [s for _, s in scored[:3]]
        answer = " ".join(picked).strip()
        if not answer:
            answer = text[:350].strip() + ("…" if len(text) > 350 else "")
        return answer
