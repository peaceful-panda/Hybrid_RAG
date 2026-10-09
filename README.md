# Hybrid RAG: Retrieval-Augmented Generation with Hybrid Search and Reranking

A Retrieval-Augmented Generation (RAG) system that combines semantic search, keyword-based retrieval, Reciprocal Rank Fusion (RRF), and cross-encoder reranking to retrieve relevant document passages and generate grounded answers using Google's Gemini model.

## Overview

Large Language Models (LLMs) can generate incorrect information when they lack relevant context. Retrieval-Augmented Generation addresses this by retrieving relevant information from a document collection and providing it to the language model before generating an answer.

This project implements a **Hybrid RAG pipeline** that combines dense vector retrieval and BM25 keyword retrieval. Their results are merged using Reciprocal Rank Fusion and reordered using a cross-encoder before being passed to Gemini for answer generation.

The system is designed to evaluate how each retrieval approach affects retrieval quality.

## Architecture

```text
Raw Documents
      |
      v
Text Cleaning and Chunking
      |
      v
Document Chunks
      |
      v
Embedding Generation
      |
      v
User Query
      |
      +---------------------+
      |                     |
      v                     v
 Dense Retrieval       BM25 Retrieval
      |                     |
      +----------+----------+
                 |
                 v
       Reciprocal Rank Fusion
                 |
                 v
       Cross-Encoder Reranking
                 |
                 v
        Relevant Context
                 |
                 v
       Gemini Answer Generation
                 |
                 v
      Grounded Answer + Citations
```

## Key Features

- **Document ingestion:** Loads text documents from a local document collection.
- **Text preprocessing:** Cleans unnecessary whitespace and splits documents into overlapping chunks.
- **Dense retrieval:** Uses sentence embeddings and cosine similarity to retrieve semantically related passages.
- **BM25 retrieval:** Retrieves passages using keyword-based relevance scoring.
- **Hybrid retrieval:** Combines dense and BM25 rankings using Reciprocal Rank Fusion.
- **Cross-encoder reranking:** Reorders retrieved passages according to their relevance to the query.
- **Grounded generation:** Uses Google's Gemini model to answer questions based on retrieved context.
- **Source citations:** Includes chunk identifiers in generated answers to help trace supporting passages.
- **Retrieval evaluation:** Compares BM25, dense retrieval, hybrid retrieval, and hybrid retrieval with reranking.
- **Generation evaluation:** Records generated answers, context relevance, and response latency.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| Keyword retrieval | BM25 using `rank-bm25` |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder — `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation | Google Gemini via `google-genai` |
| Numerical computation | NumPy |
| Configuration | `python-dotenv` |
| Evaluation | Custom Python evaluation scripts |

## Project Structure

```text
hybrid-rag/
├── app/
│   ├── ingestion/
│   │   ├── ingest.py
│   │   ├── text_cleaner.py
│   │   └── embedded_chunks.py
│   ├── retrieval/
│   │   ├── semantic_search.py
│   │   ├── bm25_search.py
│   │   └── hybrid_search.py
│   ├── reranking/
│   │   └── reranker.py
│   ├── generation/
│   │   └── rag_pipeline.py
│   ├── evaluation/
│   │   ├── evaluate_retrieval.py
│   │   ├── evaluate_generation.py
│   │   └── evaluate_reranker.py
│   └── paths.py
├── data/
│   ├── raw/
│   │   └── documents/
│   └── processed/
│       ├── chunks.json
│       └── chunks_with_embeddings.json
├── benchmarks/
│   └── benchmark.json
├── experiments/
├── tests/
├── artifacts/
│   ├── retrieval_results.json
│   └── generation_results.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd hybrid-rag
```

Replace the placeholder with your repository URL after publishing the project.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
```

Keep this file private. Never commit API keys or other secrets to GitHub.

## Running the Project

Run commands from the project root.

### Step 1: Ingest documents

Place your `.txt` documents in `data/raw/documents/`, then run:

```bash
python -m app.ingestion.ingest
```

This creates text chunks in `data/processed/chunks.json`.

### Step 2: Generate embeddings

```bash
python -m app.ingestion.embedded_chunks
```

This saves the chunks and their embedding vectors in `data/processed/chunks_with_embeddings.json`.

### Step 3: Run the RAG pipeline

```bash
python -m app.generation.rag_pipeline
```

The pipeline retrieves relevant passages, reranks them, constructs the context, and requests a grounded answer from Gemini.

### Step 4: Evaluate retrieval

```bash
python -m app.evaluation.evaluate_retrieval
```

Results are saved to `artifacts/retrieval_results.json`.

### Step 5: Evaluate generation

```bash
python -m app.evaluation.evaluate_generation
```

Results are saved to `artifacts/generation_results.json`.

The Gemini API must be available for generation evaluation. API availability, rate limits, and response latency may affect individual runs.

## Retrieval Evaluation Results

The following results were obtained from the project's 60-question retrieval benchmark.

| Metric | BM25 | Dense | Hybrid | Hybrid + Reranking |
|---|---:|---:|---:|---:|
| Recall@1 | 0.633 | 0.675 | 0.658 | 0.792 |
| Recall@3 | 0.767 | 0.867 | 0.892 | 0.908 |
| Recall@5 | 0.792 | 0.917 | 0.933 | 0.933 |
| Precision@1 | 0.717 | 0.733 | 0.717 | 0.867 |
| Precision@3 | 0.300 | 0.333 | 0.344 | 0.350 |
| Precision@5 | 0.187 | 0.217 | 0.264 | 0.264 |
| MRR | 0.774 | 0.814 | 0.827 | 0.911 |

*These are the results from the current benchmark run; rerun the evaluation if the corpus, benchmark, or retrieval implementation changes.*

### Observations

- Dense retrieval outperformed BM25 on several retrieval metrics, indicating the value of semantic matching for this dataset.
- Hybrid retrieval improved Recall@3, Recall@5, and MRR compared with dense retrieval alone.
- Cross-encoder reranking increased Recall@1 from 0.658 to 0.792 and Precision@1 from 0.717 to 0.867 compared with hybrid retrieval.
- Hybrid retrieval with reranking achieved the highest MRR of 0.911 among the evaluated approaches.
- Recall@5 remained unchanged after reranking, suggesting that its main benefit was improving the ordering of relevant passages rather than adding new candidates to the top-five set.

These results illustrate why combining complementary retrieval strategies and reranking candidates can improve the quality of retrieved context.

## Evaluation Approach

### Retrieval evaluation

The retrieval evaluator measures:

- **Recall@K:** The fraction of known relevant chunks retrieved in the top K results.
- **Precision@K:** The fraction of retrieved chunks that are relevant.
- **Mean Reciprocal Rank (MRR):** How highly the first relevant result appears in the ranking.

The benchmark associates questions with their expected relevant chunk identifiers.

### Generation evaluation

The generation evaluator records generated answers, retrieval-context relevance based on benchmark chunk identifiers, and response latency.

Generation evaluation is still an initial implementation. Context relevance is a benchmark-based proxy, not a complete semantic evaluation of faithfulness, factual correctness, or citation correctness.

## Limitations

- The current implementation focuses on local text documents.
- Retrieval and generation quality depend on the source documents and benchmark annotations.
- The generation evaluator does not yet provide a comprehensive, independently validated faithfulness or correctness score.
- Gemini responses depend on external API availability and may experience rate limits or high latency.
- The current chunking strategy uses fixed-size word chunks with overlap.

## Future Improvements

- Expand and diversify the evaluation benchmark.
- Add stronger automated checks for faithfulness, answer correctness, and citation correctness.
- Improve document parsing and metadata-based filtering.
- Add a user interface for interactive document question answering.
- Explore persistent vector storage and scalable document ingestion.
- Add further experiments and reproducible performance comparisons.

## Learning Outcomes

Through this project, I explored:

- Document preprocessing and chunking.
- Semantic embeddings and vector similarity.
- Keyword-based information retrieval with BM25.
- Hybrid retrieval and Reciprocal Rank Fusion.
- Cross-encoder reranking.
- Retrieval-Augmented Generation and grounded prompting.
- Benchmark-based evaluation and metric interpretation.
- Python project organization and environment configuration.

## Author

**[Your Name]**

Computer Science Engineering

[GitHub Profile URL] | [LinkedIn Profile URL]