"""
Core RAG engine.

Pipeline: embed chunks -> store vectors in memory -> embed the query ->
cosine-similarity search -> pass top matches to the LLM -> generate a
cited answer.

NOTE: This uses an in-memory NumPy vector store. For a demo this is ideal
(no infra, fully explainable). For production scale you'd swap this for a
dedicated vector DB (FAISS / Chroma / Pinecone) — the retrieval interface
would stay the same.

⚠️ The two functions calling the OpenAI SDK (embed_texts, generate_answer)
are the parts most likely to change with SDK versions. VERIFY them against
the current OpenAI Python SDK docs. Everything else is stable NumPy/Python.
"""

import os
import numpy as np
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

_API_KEY = os.getenv("OPENAI_API_KEY")
_client = None

EMBED_MODEL = "text-embedding-3-small"   # verify current model name
CHAT_MODEL = "gpt-4o-mini"               # verify current model name


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=_API_KEY)
    return _client


def has_api_key() -> bool:
    return bool(_API_KEY)


# ---------------------------------------------------------------------------
# VERIFY: embeddings call signature against current OpenAI SDK docs.
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    """Return an array of embedding vectors, one per input text."""
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype=np.float32)


class VectorStore:
    """Minimal in-memory vector store using cosine similarity."""

    def __init__(self):
        self.records: list[dict] = []      # chunk metadata + text
        self.vectors: np.ndarray | None = None

    def add(self, records: list[dict], vectors: np.ndarray):
        self.records = records
        # Normalize once so cosine similarity is just a dot product.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.vectors = vectors / norms

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        if self.vectors is None or len(self.records) == 0:
            return []
        q = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        scores = self.vectors @ q          # cosine similarity for all chunks
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            rec = dict(self.records[i])
            rec["score"] = float(scores[i])
            results.append(rec)
        return results


# ---------------------------------------------------------------------------
# VERIFY: chat completion call signature against current OpenAI SDK docs.
# ---------------------------------------------------------------------------
def generate_answer(question: str, retrieved: list[dict]) -> str:
    """Ask the LLM to answer using only the retrieved context."""
    client = _get_client()
    user_prompt = build_user_prompt(question, retrieved)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,   # factual, low creativity — we want grounded answers
    )
    return resp.choices[0].message.content.strip()


def build_index(records: list[dict]) -> VectorStore:
    """Embed all chunks and load them into a vector store."""
    store = VectorStore()
    if not records:
        return store
    texts = [r["text"] for r in records]
    vectors = embed_texts(texts)
    store.add(records, vectors)
    return store


def answer_question(store: VectorStore, question: str, top_k: int = 4,
                    confidence_threshold: float = 0.25) -> dict:
    """Full query pipeline: embed query -> retrieve -> generate."""
    q_vec = embed_texts([question])[0]
    retrieved = store.search(q_vec, top_k=top_k)
    if not retrieved:
        return {"answer": "No documents indexed yet.", "sources": [],
                "low_confidence": True}

    top_score = retrieved[0]["score"]
    low_confidence = top_score < confidence_threshold

    answer = generate_answer(question, retrieved)
    return {"answer": answer, "sources": retrieved,
            "low_confidence": low_confidence, "top_score": top_score}