import json
from sentence_transformers import SentenceTransformer

from app.paths import CHUNKS_FILE, EMBEDDED_CHUNKS_FILE

# Load the chunks saved during ingestion.
with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError("No chunks found. Run ingestion before generating embeddings.")

# Load the embedding model.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert every chunk's text into an embedding.
for chunk in chunks:
    embedding = model.encode(chunk["text"])
    chunk["embedding"] = embedding.tolist()

# Save the chunks along with their embeddings.
with open(EMBEDDED_CHUNKS_FILE, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=4)

print("Number of chunks:", len(chunks))
print("Embedding dimensions:", len(chunks[0]["embedding"]))
print("Saved to:", EMBEDDED_CHUNKS_FILE)