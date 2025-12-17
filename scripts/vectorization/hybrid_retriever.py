from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# Global debug flag
# =============================
VERBOSE = True

# -----------------------------
# Data model for a chunk
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: Dict[str, Any]

# -----------------------------
# Tokenization for BM25
# -----------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def bm25_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

# -----------------------------
# Hybrid Retriever
# -----------------------------
class HybridRetriever:
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

        if VERBOSE:
            print("[INIT] HybridRetriever")
            print(f"       #chunks = {len(chunks)}")
            print(f"       dense model = {dense_model_name}")
            print(f"       alpha = {alpha}")

        # --- BM25 ---
        self._bm25_tokens = [bm25_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens)

        if VERBOSE:
            print("[INIT] BM25 index built")

        # --- Dense ---
        self._dense_model = SentenceTransformer(dense_model_name, device=device)
        self._chunk_texts_for_dense = [self._dense_prepare(c.text) for c in chunks]
        self._dense_matrix = self._dense_model.encode(
            self._chunk_texts_for_dense,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        if VERBOSE:
            print(f"[INIT] Dense embeddings shape = {self._dense_matrix.shape}\n")

    def _dense_prepare(self, text: str) -> str:
        return f"passage: {text}"

    def _dense_prepare_query(self, query: str) -> str:
        return f"query: {query}"

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if VERBOSE:
            print(f"\n[QUERY] {query}")
            print(f"[RETRIEVE] top-k = {k}")

        q_tokens = bm25_tokenize(query)
        bm25_scores = np.array(self._bm25.get_scores(q_tokens), dtype=np.float32)

        if VERBOSE:
            print("[SCORES] BM25 computed")

        q_vec = self._dense_model.encode(
            [self._dense_prepare_query(query)],
            normalize_embeddings=True
        ).astype(np.float32)

        dense_scores = cosine_similarity(q_vec, self._dense_matrix)[0].astype(np.float32)

        if VERBOSE:
            print("[SCORES] Dense similarity computed")

        bm25_norm = minmax_scale(bm25_scores) if bm25_scores.max() != bm25_scores.min() else np.zeros_like(bm25_scores)
        dense_norm = minmax_scale(dense_scores) if dense_scores.max() != dense_scores.min() else np.zeros_like(dense_scores)

        hybrid = self.alpha * dense_norm + (1.0 - self.alpha) * bm25_norm

        if VERBOSE:
            print("[SCORES] Hybrid score computed")

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

        if VERBOSE:
            print(f"[RESULTS] Returning {len(results)} chunks\n")

        return results

# -----------------------------
# Chunk loaders
# -----------------------------
def load_chunks_from_hierarchical(root_dir: str) -> List[Chunk]:
    if VERBOSE:
        print(f"[LOAD] Hierarchical chunks from {root_dir}")

    chunks: List[Chunk] = []
    root = Path(root_dir)

    for doc_folder in sorted(root.iterdir()):
        if not doc_folder.is_dir():
            continue

        doc_id = doc_folder.name.replace("_chunks", "")
        if VERBOSE:
            print(f"[DOC] {doc_id}")

        for chunk_file in sorted(doc_folder.glob("*.txt")):
            text = chunk_file.read_text(encoding="utf-8").strip()
            if not text:
                continue

            if VERBOSE:
                print(f"  [CHUNK] {chunk_file.name} | {len(text.split())} words")

            chunks.append(
                Chunk(
                    chunk_id=chunk_file.stem,
                    doc_id=doc_id,
                    text=text,
                    meta={"source_path": str(chunk_file), "chunking": "hierarchical"}
                )
            )

    if VERBOSE:
        print(f"[DONE] Loaded {len(chunks)} hierarchical chunks\n")

    return chunks

def load_chunks_660(root_dir: str) -> List[Chunk]:
    if VERBOSE:
        print(f"[LOAD] Fixed 660 chunks from {root_dir}")

    chunks: List[Chunk] = []
    root = Path(root_dir)

    for doc_folder in sorted(root.iterdir()):
        if not doc_folder.is_dir():
            continue

        doc_id = doc_folder.name.replace("_chunks", "")
        if VERBOSE:
            print(f"[DOC] {doc_id}")

        for chunk_file in sorted(doc_folder.glob("*.txt")):
            text = chunk_file.read_text(encoding="utf-8").strip()
            if not text:
                continue

            if VERBOSE:
                print(f"  [CHUNK] {chunk_file.name} | {len(text.split())} words")

            chunks.append(
                Chunk(
                    chunk_id=chunk_file.stem,
                    doc_id=doc_id,
                    text=text,
                    meta={"source_path": str(chunk_file), "chunking": "fixed_660"}
                )
            )

    if VERBOSE:
        print(f"[DONE] Loaded {len(chunks)} fixed-660 chunks\n")

    return chunks

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    USE_HIERARCHICAL = False  # Set to True to use hierarchical chunks
    print("[MODE]", "Hierarchical" if USE_HIERARCHICAL else "Fixed 660")
 
    if USE_HIERARCHICAL:
        chunks = load_chunks_from_hierarchical(r"C:\Users\USER\Desktop\school work\Year 5\IR3\hierarchical_chunks")
    else:
        chunks = load_chunks_660("../chunks_output_660")

    retriever = HybridRetriever(chunks, alpha=0.6)

    print("[RUN] Retriever ready")

    q = "On what date was the defense budget discussed?"
    hits = retriever.retrieve(q, k=2)

    for h in hits:
        print(f"[{h['rank']}] {h['doc_id']} | {h['chunk_id']}")
        print("source:", h["meta"]["source_path"])
        print(h["text"][:300], "...")
        print("---")
