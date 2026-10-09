import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Create Gemini client
client = genai.Client()


# Load chunks and their embeddings
from app.paths import EMBEDDED_CHUNKS_FILE

with open(EMBEDDED_CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def dense_retrieve(question, top_k=5):

    # Convert the question into an embedding
    question_embedding = model.encode(question)

    results = []

    # Compare the question with every chunk
    for chunk in chunks:

        chunk_embedding = np.array(chunk["embedding"])

        similarity = np.dot(question_embedding, chunk_embedding) / (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(chunk_embedding)
        )

        results.append({
            "chunk_id": chunk["chunk_id"],
            "similarity": float(similarity),
            "text": chunk["text"]
        })

    # Sort highest similarity first
    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # Keep only the top K results
    top_results = results[:top_k]

    return top_results


def build_context(results):

    # Combine the retrieved chunks into one context string
    context = "\n\n".join(
        result["text"] for result in results
    )

    return context


def build_prompt(question, context):

    # Create the prompt for the LLM
    prompt = f"""
You are a helpful assistant.

Answer the user's question using the provided context.
If the answer cannot be found in the context, say that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def generate_answer(prompt):

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# Standalone testing
# This code runs only when semantic_search.py is executed directly.
# It will NOT run when another file imports dense_retrieve().
if __name__ == "__main__":

    # User's question
    question = "What does RAG stand for?"

    # RETRIEVAL
    results = dense_retrieve(question)

    # Display retrieval results
    print("\nTop results:")

    for result in results:

        print(
            "\nChunk:",
            result["chunk_id"],
            "| Similarity:",
            round(result["similarity"], 4)
        )

        print("Text:", result["text"])

    # Build context
    context = build_context(results)

    # Build prompt
    prompt = build_prompt(question, context)

    # GENERATION
    answer = generate_answer(prompt)

    # Display context
    print("\nContext:")
    print(context)

    # Display final answer
    print("\nAnswer:")
    print(answer)