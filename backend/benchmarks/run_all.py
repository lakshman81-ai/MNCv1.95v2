"""Entry point to run all available benchmarks in this package.

This script invokes each benchmark runner and aggregates their results.
Currently only the mono benchmarks are implemented.  Extend this script
to call additional benchmarks (e.g., poly_dominant, full_poly, real_songs)
as they are added.
"""

from __future__ import annotations

import json
import os

import os
import sys

# Ensure root path is importable for run_bench_mono
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from .run_bench_mono import run_benchmarks as run_mono


def main() -> None:
    results_all = {}
    # Run mono benchmarks
    mono_results = run_mono()
    results_all["mono"] = mono_results
    # Write aggregated results
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)
    print("Aggregated benchmark results saved to results/leaderboard.json")


if __name__ == "__main__":
    main()