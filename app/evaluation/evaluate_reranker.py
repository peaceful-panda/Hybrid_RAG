import json

from app.paths import BENCHMARK_FILE
from app.retrieval.hybrid_search import hybrid_retrieve
from app.reranking.reranker import rerank


# Load benchmark

with open(BENCHMARK_FILE, "r", encoding="utf-8") as file:
    benchmark = json.load(file)


def calculate_recall(retrieved_ids, relevant_ids):
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0.0

    retrieved_relevant = len(set(retrieved_ids) & relevant_ids)

    return retrieved_relevant / len(relevant_ids)


def calculate_precision(retrieved_ids, relevant_ids):
    relevant_ids = set(relevant_ids)

    if not retrieved_ids:
        return 0.0

    retrieved_relevant = len(set(retrieved_ids) & relevant_ids)

    return retrieved_relevant / len(retrieved_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_ids = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def evaluate_system(use_reranker=False):
    results = {
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "precision@1": [],
        "precision@3": [],
        "precision@5": [],
        "mrr": []
    }

    for item in benchmark:

        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        # Get hybrid candidates
        hybrid_results = hybrid_retrieve(question, top_k=5)

        if use_reranker:
            final_results = rerank(question, hybrid_results)
        else:
            final_results = hybrid_results

        retrieved_ids = [
            result["chunk_id"]
            for result in final_results
        ]

        for k in [1, 3, 5]:

            top_k_ids = retrieved_ids[:k]

            results[f"recall@{k}"].append(
                calculate_recall(top_k_ids, relevant_ids)
            )

            results[f"precision@{k}"].append(
                calculate_precision(top_k_ids, relevant_ids)
            )

        results["mrr"].append(
            reciprocal_rank(retrieved_ids, relevant_ids)
        )

    # Average all results
    averages = {}

    for metric, values in results.items():
        averages[metric] = sum(values) / len(values)

    return averages


def print_results(title, results):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Recall@1:     {results['recall@1']:.3f}")
    print(f"Recall@3:     {results['recall@3']:.3f}")
    print(f"Recall@5:     {results['recall@5']:.3f}")

    print(f"Precision@1:  {results['precision@1']:.3f}")
    print(f"Precision@3:  {results['precision@3']:.3f}")
    print(f"Precision@5:  {results['precision@5']:.3f}")

    print(f"MRR:          {results['mrr']:.3f}")


if __name__ == "__main__":

    hybrid_results = evaluate_system(use_reranker=False)

    reranked_results = evaluate_system(use_reranker=True)

    print_results(
        "Hybrid Retrieval",
        hybrid_results
    )

    print_results(
        "Hybrid + Reranking",
        reranked_results
    )

    print("\n" + "=" * 60)
    print("RERANKING COMPARISON")
    print("=" * 60)

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

        hybrid = hybrid_results[metric]
        reranked = reranked_results[metric]

        change = reranked - hybrid

        print(
            f"{metric.upper():<12} "
            f"Hybrid: {hybrid:.3f} | "
            f"Reranked: {reranked:.3f} | "
            f"Change: {change:+.3f}"
        )