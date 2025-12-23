# retriever.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scripts.vectorization.vector_index import VectorIndex
from scripts.common.models import Chunk


INDEX_PATHS = {
    "fixed_660": "vector_indexes/fixed_660/vector_index.pkl",
    "hierarchical": "vector_indexes/hierarchical/vector_index.pkl",
}





def retrieve(query: str, method: str, chunking_type: str, k: int):
    # 1. load index
    index = VectorIndex.load(INDEX_PATHS[chunking_type])

    # 2. score
    if method == "bm25":
        scores = index.bm25_scores(query)

    elif method == "dense":
        q_vec = index.encode_query_dense(query)
        scores = cosine_similarity(q_vec, index.dense_matrix)[0]

    else:
        raise ValueError("Unknown method")

    # 3. top-k
    top_idx = np.argsort(-scores)[:k]

    # 4. build output
    results = []
    for i in top_idx:
        c = index.chunks[i]
        results.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "source_path": c.meta["source_path"],
            "score": float(scores[i]),
        })

    return results


# if __name__ == "__main__":
#     query = "When was the defense budget discussed?"

#     results = retrieve(
#         query=query,
#         method="bm25",          # או "dense"
#         chunking_type="fixed_660",  # או "hierarchical"
#         k=5
#     )

#     for r in results:
#         print("DOC:", r["doc_id"])
#         print("CHUNK:", r["chunk_id"])
#         print("SCORE:", r["score"])
#         print("SOURCE:", r["source_path"])
#         print(r["text"][:300])
#         print("-" * 50)
