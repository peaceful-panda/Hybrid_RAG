import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np


# -------------------------
# LOAD CHUNKS
# -------------------------

from app.paths import EMBEDDED_CHUNKS_FILE

with open(EMBEDDED_CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# -------------------------
# BM25 SETUP
# -------------------------

tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# -------------------------
# DENSE MODEL
# -------------------------

dense_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -------------------------
# CROSS-ENCODER MODEL
# -------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# -------------------------
# DENSE RETRIEVAL
# -------------------------

def dense_retrieve(question, top_k=3):

    question_embedding = dense_model.encode(question)

    results = []

    for chunk in chunks:

        chunk_embedding = np.array(
            chunk["embedding"]
        )

        similarity = np.dot(
            question_embedding,
            chunk_embedding
        ) / (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(chunk_embedding)
        )

        results.append({
            "chunk_id": chunk["chunk_id"],
            "score": float(similarity),
            "text": chunk["text"]
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# -------------------------
# BM25 RETRIEVAL
# -------------------------

def bm25_retrieve(question, top_k=3):

    tokenized_question = question.lower().split()

    scores = bm25.get_scores(
        tokenized_question
    )

    results = []

    for chunk, score in zip(chunks, scores):

        results.append({
            "chunk_id": chunk["chunk_id"],
            "score": float(score),
            "text": chunk["text"]
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# -------------------------
# HYBRID RRF
# -------------------------

def hybrid_retrieve(question, top_k=3):

    dense_results = dense_retrieve(
        question,
        top_k=3
    )

    bm25_results = bm25_retrieve(
        question,
        top_k=3
    )

    rrf_scores = {}

    k = 60

    # BM25 rankings
    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = result["chunk_id"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    # Dense rankings
    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        chunk_id = result["chunk_id"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    results = []

    for chunk in chunks:

        chunk_id = chunk["chunk_id"]

        if chunk_id in rrf_scores:

            results.append({
                "chunk_id": chunk_id,
                "rrf_score": rrf_scores[chunk_id],
                "text": chunk["text"]
            })

    results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return results[:top_k]


# -------------------------
# CROSS-ENCODER RERANKING
# -------------------------

def rerank(question, results, top_k=None):

    pairs = [
        [question, result["text"]]
        for result in results
    ]

    scores = reranker.predict(pairs)

    reranked_results = []

    for result, score in zip(
        results,
        scores
    ):

        reranked_results.append({
            "chunk_id": result["chunk_id"],
            "rrf_score": result["rrf_score"],
            "reranker_score": float(score),
            "text": result["text"]
        })

    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    if top_k is None:
        return reranked_results

    return reranked_results[:top_k]


# -------------------------
# TEST / DEMO
# -------------------------

if __name__ == "__main__":

    question = "What is document ingestion?"

    # Hybrid retrieval
    hybrid_results = hybrid_retrieve(
        question,
        top_k=3
    )

    print("\nHybrid results:")

    for result in hybrid_results:

        print(
            "\nChunk:",
            result["chunk_id"],
            "| RRF Score:",
            round(result["rrf_score"], 6)
        )

        print("Text:", result["text"])

    # Reranking
    final_results = rerank(
        question,
        hybrid_results,
        top_k=3
    )

    print("\n\nFinal reranked results:")

    for result in final_results:

        print(
            "\nChunk:",
            result["chunk_id"],
            "| RRF:",
            round(result["rrf_score"], 6),
            "| Reranker:",
            round(result["reranker_score"], 4)
        )

        print("Text:", result["text"])