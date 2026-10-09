import json

from app.paths import BENCHMARK_FILE, RETRIEVAL_RESULTS_FILE
from app.retrieval.semantic_search import dense_retrieve
from app.retrieval.bm25_search import bm25_retrieve
from app.retrieval.hybrid_search import hybrid_retrieve
from app.reranking.reranker import rerank

# ============================================================
# CONFIGURATION
# ============================================================

K_VALUES = [1, 3, 5]


# ============================================================
# LOAD BENCHMARK
# ============================================================

with open(BENCHMARK_FILE, "r", encoding="utf-8") as file:
    benchmark = json.load(file)

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_recall(retrieved_ids, relevant_ids):
    """
    Recall@K:
    How many of the relevant chunks were retrieved?
    """

    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0.0

    retrieved_relevant = (
        set(retrieved_ids) & relevant_ids
    )

    return len(retrieved_relevant) / len(relevant_ids)


def calculate_precision(retrieved_ids, relevant_ids):
    """
    Precision@K:
    How many of the retrieved chunks were relevant?
    """

    relevant_ids = set(relevant_ids)

    if not retrieved_ids:
        return 0.0

    retrieved_relevant = (
        set(retrieved_ids) & relevant_ids
    )

    return len(retrieved_relevant) / len(retrieved_ids)


def calculate_mrr(retrieved_ids, relevant_ids):
    """
    Mean Reciprocal Rank:
    Measures how high the first relevant chunk appears.
    """

    relevant_ids = set(relevant_ids)

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


# ============================================================
# EVALUATE ONE RETRIEVAL SYSTEM
# ============================================================

def evaluate_system(
    system_name,
    retrieve_function,
    use_reranker=False
):
    """
    Evaluate one retrieval system over the full benchmark.

    Systems:
        BM25
        Dense
        Hybrid
        Hybrid + Reranking
    """

    metric_values = {
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "precision@1": [],
        "precision@3": [],
        "precision@5": [],
        "mrr": []
    }

    print("\n" + "=" * 60)
    print(system_name)
    print("=" * 60)

    for index, item in enumerate(
        benchmark,
        start=1
    ):

        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        # ----------------------------------------------------
        # RETRIEVE CANDIDATES
        # ----------------------------------------------------

        retrieval_results = retrieve_function(
            question,
            top_k=5
        )

        # ----------------------------------------------------
        # OPTIONAL RERANKING
        # ----------------------------------------------------

        if use_reranker:

            final_results = rerank(
                question,
                retrieval_results
            )

        else:

            final_results = retrieval_results

        # ----------------------------------------------------
        # EXTRACT CHUNK IDS
        # ----------------------------------------------------

        retrieved_ids = [
            result["chunk_id"]
            for result in final_results
        ]

        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        for k in K_VALUES:

            top_k_ids = retrieved_ids[:k]

            recall = calculate_recall(
                top_k_ids,
                relevant_ids
            )

            precision = calculate_precision(
                top_k_ids,
                relevant_ids
            )

            metric_values[
                f"recall@{k}"
            ].append(recall)

            metric_values[
                f"precision@{k}"
            ].append(precision)

        # MRR is calculated using the returned ranking.
        mrr = calculate_mrr(
            retrieved_ids,
            relevant_ids
        )

        metric_values["mrr"].append(mrr)

        print(
            f"Question {index}/{len(benchmark)}",
            end="\r"
        )

    # --------------------------------------------------------
    # AVERAGE METRICS
    # --------------------------------------------------------

    averages = {}

    for metric, values in metric_values.items():

        if values:

            averages[metric] = (
                sum(values) / len(values)
            )

        else:

            averages[metric] = 0.0

    return averages


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    system_name,
    results
):

    print("\n" + "=" * 60)
    print(system_name)
    print("=" * 60)

    print(
        f"Recall@1:     {results['recall@1']:.3f}"
    )

    print(
        f"Recall@3:     {results['recall@3']:.3f}"
    )

    print(
        f"Recall@5:     {results['recall@5']:.3f}"
    )

    print(
        f"Precision@1:  {results['precision@1']:.3f}"
    )

    print(
        f"Precision@3:  {results['precision@3']:.3f}"
    )

    print(
        f"Precision@5:  {results['precision@5']:.3f}"
    )

    print(
        f"MRR:          {results['mrr']:.3f}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    print(
        f"\nQuestions in benchmark: {len(benchmark)}"
    )

    # --------------------------------------------------------
    # 1. BM25
    # --------------------------------------------------------

    bm25_results = evaluate_system(
        "BM25 Retrieval",
        bm25_retrieve
    )

    # --------------------------------------------------------
    # 2. DENSE
    # --------------------------------------------------------

    dense_results = evaluate_system(
        "Dense Retrieval",
        dense_retrieve
    )

    # --------------------------------------------------------
    # 3. HYBRID
    # --------------------------------------------------------

    hybrid_results = evaluate_system(
        "Hybrid Retrieval",
        hybrid_retrieve
    )

    # --------------------------------------------------------
    # 4. HYBRID + RERANKING
    # --------------------------------------------------------

    reranked_results = evaluate_system(
        "Hybrid + Reranking",
        hybrid_retrieve,
        use_reranker=True
    )

    # ========================================================
    # PRINT INDIVIDUAL RESULTS
    # ========================================================

    print_results(
        "BM25 Retrieval",
        bm25_results
    )

    print_results(
        "Dense Retrieval",
        dense_results
    )

    print_results(
        "Hybrid Retrieval",
        hybrid_results
    )

    print_results(
        "Hybrid + Reranking",
        reranked_results
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 60)

    systems = {
        "BM25": bm25_results,
        "Dense": dense_results,
        "Hybrid": hybrid_results,
        "Hybrid + Reranking": reranked_results
    }

    metrics = [
        "recall@1",
        "recall@3",
        "recall@5",
        "precision@1",
        "precision@3",
        "precision@5",
        "mrr"
    ]

    for metric in metrics:

        print(
            f"\n{metric.upper()}"
        )

        for system_name, results in systems.items():

            print(
                f"{system_name:<22}"
                f"{results[metric]:.3f}"
            )

    # ========================================================
    # RERANKING IMPROVEMENT
    # ========================================================

    print("\n" + "=" * 60)
    print("RERANKING IMPROVEMENT")
    print("=" * 60)

    for metric in metrics:

        hybrid_value = hybrid_results[metric]
        reranked_value = reranked_results[metric]

        change = (
            reranked_value - hybrid_value
        )

        print(
            f"{metric.upper():<15}"
            f"Hybrid: {hybrid_value:.3f} | "
            f"Reranked: {reranked_value:.3f} | "
            f"Change: {change:+.3f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

        evaluation_results = {
        "benchmark_questions": len(benchmark),
        "k_values": K_VALUES,
        "systems": {
            "BM25": bm25_results,
            "Dense": dense_results,
            "Hybrid": hybrid_results,
            "Hybrid + Reranking": reranked_results
        },
        "reranking_improvement": {
            metric: (
                reranked_results[metric] - hybrid_results[metric]
            )
            for metric in metrics
        }
    }

    with open(RETRIEVAL_RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(evaluation_results, file, indent=4)

    print(f"\nResults saved to: {RETRIEVAL_RESULTS_FILE}")