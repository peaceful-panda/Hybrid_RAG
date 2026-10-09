import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np


# Load chunks
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
# DENSE SEARCH SETUP
# -------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


def dense_retrieve(question, top_k=3):

    question_embedding = model.encode(question)

    results = []

    for chunk in chunks:

        chunk_embedding = np.array(chunk["embedding"])

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
# BM25 SEARCH
# -------------------------

def bm25_retrieve(question, top_k=3):

    tokenized_question = question.lower().split()

    scores = bm25.get_scores(tokenized_question)

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
# RRF FUSION
# -------------------------

def hybrid_retrieve(question, top_k=3):

    dense_results = dense_retrieve(question, top_k=5)
    bm25_results = bm25_retrieve(question, top_k=5)

    rrf_scores = {}

    # RRF constant
    k = 60

    # Add BM25 rankings
    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["chunk_id"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    # Add Dense rankings
    for rank, result in enumerate(dense_results, start=1):

        chunk_id = result["chunk_id"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    # Create final hybrid results
    results = []

    for chunk in chunks:

        chunk_id = chunk["chunk_id"]

        if chunk_id in rrf_scores:

            results.append({
                "chunk_id": chunk_id,
                "rrf_score": rrf_scores[chunk_id],
                "text": chunk["text"]
            })

    # Sort by RRF score
    results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return results[:top_k]


# -------------------------
# TEST
# -------------------------

if __name__ == "__main__":

    question = "What does RAG stand for?"

    results = hybrid_retrieve(question)

    print("\nHybrid results:")

    for result in results:
        print(
            f"\nChunk: {result['chunk_id']} | "
            f"RRF Score: {result['rrf_score']:.6f}"
        )

        print(f"Text: {result['text']}")