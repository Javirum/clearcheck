"""Evaluate NOPE against the test dataset and report metrics."""

import json
import time
from pathlib import Path

from src.agent import check_claim
from src.audit_log import log_check

TEST_DATA_PATH = Path(__file__).resolve().parent / "data" / "test_dataset.json"


def load_test_data() -> list[dict]:
    with open(TEST_DATA_PATH) as f:
        return json.load(f)


def run_evaluation():
    tests = load_test_data()
    print(f"Running evaluation on {len(tests)} test items...\n")

    results = []
    total_time = 0
    correct = 0
    multi_source_count = 0
    false_confidence_count = 0

    for i, test in enumerate(tests):
        claim = test["claim"]
        expected = test["expected_verdict"]
        print(f"[{i + 1}/{len(tests)}] Checking: {claim[:70]}...")

        start = time.time()
        try:
            verdict, validation, evidence = check_claim(claim)
            elapsed = time.time() - start
            total_time += elapsed

            log_check(verdict, evidence, validation, response_time=elapsed)

            actual = verdict.verdict.value
            is_correct = actual == expected

            # Count sources consulted
            num_sources = sum([
                1 if evidence.pinecone_results else 0,
                1 if evidence.tavily_results else 0,
                1 if evidence.factcheck_results else 0,
            ])
            is_multi_source = num_sources >= 2

            # Check for false confidence on uncertain items
            is_false_confident = (
                expected == "uncertain"
                and actual != "uncertain"
                and verdict.confidence > 0.7
            )

            if is_correct:
                correct += 1
            if is_multi_source:
                multi_source_count += 1
            if is_false_confident:
                false_confidence_count += 1

            result = {
                "id": test["id"],
                "claim": claim[:60],
                "expected": expected,
                "actual": actual,
                "correct": is_correct,
                "confidence": verdict.confidence,
                "sources_consulted": num_sources,
                "multi_source": is_multi_source,
                "validation_passed": validation.is_valid,
                "time_seconds": round(elapsed, 1),
            }
            results.append(result)

            status = "✓" if is_correct else "✗"
            print(f"  {status} Expected: {expected}, Got: {actual} "
                  f"(conf: {verdict.confidence:.0%}, {elapsed:.1f}s, "
                  f"{num_sources} source types)")

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  ERROR: {e}")
            results.append({
                "id": test["id"],
                "claim": claim[:60],
                "expected": expected,
                "actual": "error",
                "correct": False,
                "confidence": 0,
                "sources_consulted": 0,
                "multi_source": False,
                "validation_passed": False,
                "time_seconds": round(elapsed, 1),
            })

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    n = len(tests)
    accuracy = correct / n * 100
    avg_time = total_time / n
    multi_source_pct = multi_source_count / n * 100
    source_citation_rate = sum(1 for r in results if r.get("sources_consulted", 0) > 0) / n * 100
    validation_pass_rate = sum(1 for r in results if r.get("validation_passed")) / n * 100

    print(f"\n{'Metric':<35} {'Target':<15} {'Actual':<15} {'Pass'}")
    print("-" * 70)
    print(f"{'Verdict accuracy':<35} {'≥85%':<15} {accuracy:.0f}%{'':<10} {'✓' if accuracy >= 85 else '✗'}")
    print(f"{'Multi-source checks':<35} {'≥80%':<15} {multi_source_pct:.0f}%{'':<10} {'✓' if multi_source_pct >= 80 else '✗'}")
    print(f"{'Source citation rate':<35} {'100%':<15} {source_citation_rate:.0f}%{'':<10} {'✓' if source_citation_rate == 100 else '✗'}")
    print(f"{'False confidence (uncertain)':<35} {'0':<15} {false_confidence_count}{'':<12} {'✓' if false_confidence_count == 0 else '✗'}")
    print(f"{'Avg response time':<35} {'<20s':<15} {avg_time:.1f}s{'':<10} {'✓' if avg_time < 20 else '✗'}")
    print(f"{'Output validity':<35} {'100%':<15} {validation_pass_rate:.0f}%{'':<10} {'✓' if validation_pass_rate == 100 else '✗'}")

    print(f"\nTotal time: {total_time:.0f}s for {n} checks")

    # Save results
    output_path = Path(__file__).resolve().parent / "data" / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()
