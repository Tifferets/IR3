from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Data model for a chunk
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str          # file name / unique document id
    text: str
    meta: Dict[str, Any] # e.g., {"source_path": "...", "start_idx": ..., "end_idx": ...}

# -----------------------------
# Tokenization for BM25 (simple but solid)
# -----------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def bm25_tokenize(text: str) -> List[str]:
    # Lowercase + keep alnum/' tokens
    return [t.lower() for t in _TOKEN_RE.findall(text)]

# -----------------------------
# Hybrid Retriever
# -----------------------------
class HybridRetriever:
    """
    Hybrid = BM25 (lexical) + Dense embeddings (semantic)
    score = alpha * normalized_dense + (1-alpha) * normalized_bm25
    """
    def __init__(
        self,
        chunks: List[Chunk],
        dense_model_name: str = "intfloat/e5-base",
        alpha: float = 0.6,
        device: str | None = None
    ):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")

        self.chunks = chunks
        self.alpha = alpha

        # --- BM25 index ---
        self._bm25_tokens = [bm25_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens)

        # --- Dense embeddings index ---
        self._dense_model = SentenceTransformer(dense_model_name, device=device)
        self._chunk_texts_for_dense = [self._dense_prepare(c.text) for c in chunks]
        self._dense_matrix = self._dense_model.encode(
            self._chunk_texts_for_dense,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

    def _dense_prepare(self, text: str) -> str:
        # E5 models work best with prefixes: "passage:" for docs, "query:" for query
        return f"passage: {text}"

    def _dense_prepare_query(self, query: str) -> str:
        return f"query: {query}"

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be > 0")

        # --- BM25 scores ---
        q_tokens = bm25_tokenize(query)
        bm25_scores = np.array(self._bm25.get_scores(q_tokens), dtype=np.float32)

        # --- Dense scores (cosine since embeddings normalized => dot == cosine) ---
        q_vec = self._dense_model.encode(
            [self._dense_prepare_query(query)],
            normalize_embeddings=True
        ).astype(np.float32)
        dense_scores = cosine_similarity(q_vec, self._dense_matrix)[0].astype(np.float32)

        # --- Normalize both to [0,1] so alpha is meaningful ---
        bm25_norm = minmax_scale(bm25_scores) if bm25_scores.max() != bm25_scores.min() else np.zeros_like(bm25_scores)
        dense_norm = minmax_scale(dense_scores) if dense_scores.max() != dense_scores.min() else np.zeros_like(dense_scores)

        hybrid = self.alpha * dense_norm + (1.0 - self.alpha) * bm25_norm

        top_idx = np.argsort(-hybrid)[:k]

        results = []
        for rank, i in enumerate(top_idx, start=1):
            c = self.chunks[i]
            results.append({
                "rank": rank,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "score_hybrid": float(hybrid[i]),
                "score_dense": float(dense_scores[i]),
                "score_bm25": float(bm25_scores[i]),
                "text": c.text,
                "meta": c.meta,
            })
        return results

# -----------------------------
# Example usage (replace load_chunks)
# -----------------------------
def load_chunks_dummy() -> List[Chunk]:
    # Replace this with your real chunking output.
    return [
        Chunk("c1", "docA", "The Prime Minister spoke about the defense budget on 12 March 2023.", {"source_path": "docA.txt"}),
        Chunk("c2", "docA", "Immigration bill arguments focused on humanitarian obligations and border control.", {"source_path": "docA.txt"}),
        Chunk("c3", "docB", "Investment in education was linked to reduced crime rates over time.", {"source_path": "docB.txt"}),
    ]

if __name__ == "__main__":
    chunks = load_chunks_dummy()
    retriever = HybridRetriever(chunks, dense_model_name="intfloat/e5-base", alpha=0.6)

    q = "On what date was the defense budget discussed?"
    hits = retriever.retrieve(q, k=2)
    for h in hits:
        print(f"[{h['rank']}] {h['doc_id']} {h['chunk_id']} hybrid={h['score_hybrid']:.3f}")
        print("source:", h["meta"].get("source_path"))
        print(h["text"])
        print("---")
