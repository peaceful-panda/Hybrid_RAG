import json
import time

from app.paths import BENCHMARK_FILE, GENERATION_RESULTS_FILE
from app.generation.rag_pipeline import answer_question


# ============================================================
# CONFIGURATION
# ============================================================

# Start with 1 question while verifying Gemini generation.
# Increase to 10 after generation works reliably.
GENERATION_QUESTION_LIMIT = 3


# ============================================================
# LOAD BENCHMARK
# ============================================================

with open(BENCHMARK_FILE, "r", encoding="utf-8") as file:
    benchmark = json.load(file)

# Use only a representative subset for LLM generation evaluation.
generation_benchmark = benchmark[:GENERATION_QUESTION_LIMIT]


# ============================================================
# GENERATION EVALUATION
# ============================================================

def evaluate_generation():

    results = []

    total_questions = len(generation_benchmark)

    for index, item in enumerate(generation_benchmark, start=1):

        question = item["question"]
        expected_answer = item["expected_answer"]
        relevant_ids = set(item["relevant_chunk_ids"])

        print("\n" + "=" * 60)
        print(f"QUESTION {index}/{total_questions}")
        print("=" * 60)

        print(question)

        # -----------------------------------------
        # RUN RAG PIPELINE
        # -----------------------------------------

        start_time = time.time()

        try:
            output = answer_question(question)

        except Exception as error:

            error_message = str(error)

            # -----------------------------------------
            # GEMINI API UNAVAILABLE / QUOTA ERROR
            # -----------------------------------------

            if (
                "429" in error_message
                or "503" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "UNAVAILABLE" in error_message
                or "quota" in error_message.lower()
            ):

                print("\n" + "!" * 60)
                print("GEMINI API UNAVAILABLE")
                print("!" * 60)

                print(
                    "\nThe Gemini API is currently unavailable "
                    "or the quota is exhausted."
                )

                print(
                    "Stopping generation evaluation immediately."
                )

                print(
                    f"\nQuestions successfully evaluated: "
                    f"{len(results)}"
                )

                return results

            # -----------------------------------------
            # OTHER ERROR
            # -----------------------------------------

            print("\nERROR while generating answer:")
            print(error)

            print(
                "\nSkipping this question and continuing..."
            )

            continue

        end_time = time.time()

        latency = end_time - start_time

        # -----------------------------------------
        # RETRIEVED CHUNKS
        # -----------------------------------------

        retrieved_ids = [
            result["chunk_id"]
            for result in output["results"]
        ]

        # -----------------------------------------
        # CONTEXT RELEVANCE
        # -----------------------------------------

        retrieved_relevant = (
            set(retrieved_ids) & relevant_ids
        )

        context_relevance = (
            len(retrieved_relevant)
            / len(retrieved_ids)
            if retrieved_ids
            else 0
        )

        # -----------------------------------------
        # SAVE RESULT
        # -----------------------------------------

        result = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": output["answer"],
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_chunk_ids": list(relevant_ids),
            "context_relevance": context_relevance,
            "latency_seconds": latency
        }

        results.append(result)

        # -----------------------------------------
        # DISPLAY ANSWER
        # -----------------------------------------

        print("\nGenerated Answer:")
        print(output["answer"])

        print("\nRetrieved Chunks:")
        print(retrieved_ids)

        print("\nContext Relevance:")
        print(f"{context_relevance:.3f}")

        print("\nLatency:")
        print(f"{latency:.2f} seconds")

    return results


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Questions evaluated: {len(results)}"
    )

    # -----------------------------------------
    # HANDLE NO SUCCESSFUL RESULTS
    # -----------------------------------------

    if not results:

        print(
            "\nNo questions were successfully evaluated."
        )

        print(
            "The Gemini API may still be unavailable "
            "or the quota may be exhausted."
        )

        return

    # -----------------------------------------
    # AVERAGE LATENCY
    # -----------------------------------------

    average_latency = (
        sum(
            result["latency_seconds"]
            for result in results
        )
        / len(results)
    )

    # -----------------------------------------
    # AVERAGE CONTEXT RELEVANCE
    # -----------------------------------------

    average_context_relevance = (
        sum(
            result["context_relevance"]
            for result in results
        )
        / len(results)
    )

    print(
        f"Average Context Relevance: "
        f"{average_context_relevance:.3f}"
    )

    print(
        f"Average Latency: "
        f"{average_latency:.2f} seconds"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION")
    print("=" * 60)

    print(
        f"\nGeneration benchmark size: "
        f"{len(generation_benchmark)} questions"
    )

    print(
        "Retrieval evaluation remains independent "
        "and can use the full benchmark."
    )

    evaluation_results = evaluate_generation()

    print_summary(evaluation_results)

    # -----------------------------------------
    # SAVE RESULTS
    # -----------------------------------------

    with open(
        GENERATION_RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nResults saved to: {GENERATION_RESULTS_FILE}")