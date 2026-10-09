import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from dotenv import load_dotenv

from app.paths import EMBEDDED_CHUNKS_FILE


# ============================================================
# SETUP
# ============================================================

load_dotenv()
client = genai.Client()


# ============================================================
# LOAD CHUNKS
# ============================================================

with open(EMBEDDED_CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError(
        "No embedded chunks found. Run ingestion and embedding first."
    )


# ============================================================
# LOAD MODELS
# ============================================================

dense_model = SentenceTransformer("all-MiniLM-L6-v2")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================================================
# BM25 INDEX
# ============================================================

tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# ============================================================
# DENSE RETRIEVAL
# ============================================================

def dense_retrieve(question, top_k=3):
    question_embedding = dense_model.encode(question)

    results = []

    for chunk in chunks:
        chunk_embedding = np.array(chunk["embedding"])

        denominator = (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(chunk_embedding)
        )

        similarity = (
            np.dot(question_embedding, chunk_embedding) / denominator
            if denominator != 0
            else 0.0
        )

        results.append({
            "chunk_id": chunk["chunk_id"],
            "score": float(similarity),
            "text": chunk["text"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


# ============================================================
# BM25 RETRIEVAL
# ============================================================

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

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


# ============================================================
# RRF HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(question, top_k=3):
    dense_results = dense_retrieve(question, top_k=3)
    bm25_results = bm25_retrieve(question, top_k=3)

    rrf_scores = {}
    rrf_k = 60

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank)
        )

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank)
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

    results.sort(key=lambda x: x["rrf_score"], reverse=True)

    return results[:top_k]


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank(question, results, top_k=2):
    if not results:
        return []

    pairs = [
        [question, result["text"]]
        for result in results
    ]

    scores = reranker.predict(pairs)
    reranked_results = []

    for result, score in zip(results, scores):
        reranked_results.append({
            **result,
            "reranker_score": float(score)
        })

    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):
    context_parts = []

    for result in results:
        context_parts.append(
            f"[{result['chunk_id']}]\n{result['text']}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(question, context):
    return f"""
You are a helpful assistant.

Answer the user's question using ONLY the provided context.

Cite supporting chunk IDs in square brackets, using the exact IDs
shown in the context.

If the answer cannot be found in the context, say that you don't
have enough information.

Do not use outside knowledge.
Do not make up information.

Context:
{context}

Question:
{question}

Answer:
""".strip()


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text or "GENERATION_FAILED"

    except Exception as error:
        print("\n" + "!" * 60)
        print("GEMINI ERROR")
        print("!" * 60)
        print(type(error).__name__)
        print(error)

        return "GENERATION_FAILED"


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def answer_question(question):
    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    # Hybrid retrieval
    hybrid_results = hybrid_retrieve(question, top_k=3)

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL")
    print("=" * 60)

    for result in hybrid_results:
        print(
            f"\n{result['chunk_id']}"
            f" | RRF: {result['rrf_score']:.6f}"
        )

    # Reranking
    final_results = rerank(
        question,
        hybrid_results,
        top_k=2
    )

    print("\n" + "=" * 60)
    print("RERANKED RESULTS")
    print("=" * 60)

    for result in final_results:
        print(
            f"\n{result['chunk_id']}"
            f" | Reranker: {result['reranker_score']:.4f}"
        )

    # Build context
    context = build_context(final_results)

    print("\n" + "=" * 60)
    print("FINAL CONTEXT")
    print("=" * 60)
    print(context)

    # Generate answer
    prompt = build_prompt(question, context)
    answer = generate_answer(prompt)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)

    return {
        "question": question,
        "answer": answer,
        "results": final_results,
        "context": context
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    question = "What is document ingestion?"
    answer_question(question)
