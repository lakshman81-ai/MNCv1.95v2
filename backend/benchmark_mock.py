"""
Simple benchmarking harness for the transcription pipeline.

By default this runs in mock mode so it does not require heavy audio
libraries to be installed. Supply ``--use-mock 0`` and an input audio
path to exercise the full pipeline once the dependencies are available.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.transcription import transcribe_audio_pipeline


def run_benchmark(iterations: int, use_mock: bool, input_path: str | None) -> None:
    fixture_path = Path(__file__).parent / "mock_data" / "happy_birthday.xml"
    target_path = Path(input_path) if input_path else fixture_path

    durations: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = transcribe_audio_pipeline(str(target_path), use_mock=use_mock)
        durations.append(time.perf_counter() - start)

    print("Benchmark results")
    print(f"  Iterations: {iterations}")
    print(f"  Mode: {'mock' if use_mock else 'full'}")
    print(f"  Input: {target_path}")
    print(f"  Output length: {len(result.musicxml)} characters")
    print(f"  Min / Avg / Max: {min(durations):.4f}s / {statistics.mean(durations):.4f}s / {max(durations):.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the transcription pipeline")
    parser.add_argument("--iterations", type=int, default=5, help="Number of runs to average")
    parser.add_argument(
        "--use-mock",
        type=int,
        default=1,
        help="Set to 0 to run the full pipeline (requires audio dependencies)",
    )
    parser.add_argument("--input", type=str, default=None, help="Optional audio file path")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, use_mock=bool(args.use_mock), input_path=args.input)
