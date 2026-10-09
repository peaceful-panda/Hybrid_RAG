
from pathlib import Path

# Find the project's root folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Build the path to our sample text document.
document_path = PROJECT_ROOT / "data" / "raw" / "sample_document.txt"

# Read the document using UTF-8 text encoding.
document_text = document_path.read_text(encoding="utf-8")

# Display the document and its statistics.
print("DOCUMENT LOADED SUCCESSFULLY")
print("-" * 40)
print(document_text)

print("\nDOCUMENT STATISTICS")
print("File name:", document_path.name)
print("Number of characters:", len(document_text))
print("Number of words:", len(document_text.split()))