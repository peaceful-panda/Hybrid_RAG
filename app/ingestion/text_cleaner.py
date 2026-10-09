import json
import re

from app.paths import RAW_DATA_DIR, CHUNKS_FILE

# Read the source document.
document_path = RAW_DATA_DIR / "sample_document.txt"
text = document_path.read_text(encoding="utf-8")

# Clean the text.
text = text.strip()
text = re.sub(r"\s+", " ", text)

# Split the cleaned text into chunks.
chunk_size = 40
overlap = 8

words = text.split()
chunks = []

step = chunk_size - overlap

for start in range(0, len(words), step):
    chunk_words = words[start:start + chunk_size]
    chunk = " ".join(chunk_words)
    chunks.append(chunk)

# Display the chunks.
print("TOTAL WORDS:", len(words))
print("TOTAL CHUNKS:", len(chunks))

chunk_data = []

for i, chunk in enumerate(chunks, start=1):
    chunk_id = f"sample_document_chunk_{i}"

    chunk_info = {
        "chunk_id": chunk_id,
        "source": "sample_document.txt",
        "word_count": len(chunk.split()),
        "text": chunk
    }

    chunk_data.append(chunk_info)

    print(f"\nCHUNK ID: {chunk_id}")
    print("WORD COUNT:", len(chunk.split()))
    print("TEXT:", chunk)

# Save chunks in data/processed/chunks.json.
with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
    json.dump(chunk_data, file, indent=4)

print("\nChunks saved to:", CHUNKS_FILE)