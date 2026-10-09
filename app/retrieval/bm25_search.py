import json
from rank_bm25 import BM25Okapi


# Load chunks
from app.paths import EMBEDDED_CHUNKS_FILE

with open(EMBEDDED_CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# Tokenize each chunk
tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]


# Create BM25 index
bm25 = BM25Okapi(tokenized_chunks)


def bm25_retrieve(question, top_k=5):

    # Tokenize the question
    tokenized_question = question.lower().split()

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_question)

    # Pair each score with its chunk
    results = []

    for chunk, score in zip(chunks, scores):

        results.append({
            "chunk_id": chunk["chunk_id"],
            "score": float(score),
            "text": chunk["text"]
        })

    # Sort highest score first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Return Top-K results
    return results[:top_k]


# Test question
if __name__ == "__main__":

    question = "What does RAG stand for?"

    # Retrieve using BM25
    results = bm25_retrieve(question)

    # Display results
    print("\nBM25 results:")

    for result in results:

        print(
            "\nChunk:",
            result["chunk_id"],
            "| BM25 Score:",
            round(result["score"], 4)
        )

        print("Text:", result["text"])