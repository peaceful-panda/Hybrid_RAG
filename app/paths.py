from pathlib import Path

# Project root: the folder containing README.md
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.json"
EMBEDDED_CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks_with_embeddings.json"

BENCHMARK_FILE = PROJECT_ROOT / "benchmarks" / "benchmark.json"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RETRIEVAL_RESULTS_FILE = ARTIFACTS_DIR / "retrieval_results.json"
GENERATION_RESULTS_FILE = ARTIFACTS_DIR / "generation_results.json"