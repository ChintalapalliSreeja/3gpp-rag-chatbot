import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag import ThreeGPPRAG
from evaluation.evaluation_questions import TEST_QUESTIONS


def run_evaluation():

    print("=" * 80)
    print("3GPP RAG EVALUATION")
    print("=" * 80)

    rag = ThreeGPPRAG()

    total = len(TEST_QUESTIONS)
    passed = 0

    results = []

    for index, test in enumerate(TEST_QUESTIONS, start=1):

        question = test["question"]
        expected = test["expected"]

        print("\n" + "-" * 80)
        print(f"Test {index}/{total}")
        print(f"Question: {question}")
        print(f"Expected: {expected}")

        result = rag.ask(question)

        answer = result["answer"]
        sources = result["sources"]

        refusal = (
            "I couldn't find sufficient information"
            in answer
        )

        if expected == "answerable":

            success = (
                not refusal
                and len(sources) > 0
            )

        else:

            success = (
                refusal
                and len(sources) == 0
            )

        if success:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"Result: {status}")
        print(f"Sources: {len(sources)}")

        results.append({
            "question": question,
            "expected": expected,
            "status": status
        })


    # -----------------------------------------
    # Summary
    # -----------------------------------------

    accuracy = (passed / total) * 100

    print("\n")
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {total - passed}")
    print(f"Pass rate   : {accuracy:.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()