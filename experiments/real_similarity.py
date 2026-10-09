
from sentence_transformers import SentenceTransformer
import numpy as np

# Load our embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Two sentences to compare
text_a = "RAG retrieves relevant documents to answer questions."
text_b = "Retrieval helps a system find useful information."

# Convert sentences into numerical vectors
embedding_a = model.encode(text_a)
embedding_b = model.encode(text_b)

# Calculate cosine similarity
similarity = np.dot(embedding_a, embedding_b) / (
    np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
)

print("Embedding A dimensions:", len(embedding_a))
print("Embedding B dimensions:", len(embedding_b))
print("Cosine similarity:", round(float(similarity), 4))