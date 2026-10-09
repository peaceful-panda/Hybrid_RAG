from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Retrieval-Augmented Generation combines retrieval with generation."

embedding = model.encode(text)

print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])