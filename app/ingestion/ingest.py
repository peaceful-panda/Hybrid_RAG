import json
import re
from pathlib import Path


# Configuration
DOCUMENTS_DIR = Path("data/raw/documents")
from app.paths import CHUNKS_FILE

OUTPUT_FILE = CHUNKS_FILE
CHUNK_SIZE = 40
OVERLAP = 8


def clean_text(text):
    """
    Clean unnecessary whitespace from document text.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Split text into overlapping word-based chunks.
    """
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def create_chunks():
    """
    Load every TXT document and create chunks with metadata.
    """
    all_chunks = []

    documents = sorted(DOCUMENTS_DIR.glob("*.txt"))

    print("Documents found:", len(documents))

    for document_path in documents:

        print("Processing:", document_path.name)

        with open(document_path, "r", encoding="utf-8") as file:
            text = file.read()

        text = clean_text(text)

        chunks = chunk_text(text)

        source_name = document_path.stem

        for index, chunk in enumerate(chunks, start=1):

            chunk_data = {
                "chunk_id": f"{source_name}_chunk_{index}",
                "source": document_path.name,
                "text": chunk
            }

            all_chunks.append(chunk_data)

        print("  Chunks created:", len(chunks))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, indent=4)

    print("\nTotal chunks:", len(all_chunks))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    create_chunks()