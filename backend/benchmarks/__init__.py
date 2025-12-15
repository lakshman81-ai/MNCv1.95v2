"""Benchmark suite for the music‑note pipeline.

This package exposes utilities and entrypoints for running synthetic
transcription benchmarks on the pipeline.  Benchmarks are organized
by instrument and polyphony level (mono, poly‑dominant, full‑poly).
Each benchmark script produces metrics and intermediate JSON files
according to the specification in the work instructions.

The goal of these benchmarks is to measure and optimize the pipeline
without changing public APIs.  Scripts in this package should run
standalone via ``python -m backend.benchmarks.run_bench_mono`` or
``python -m backend.benchmarks.run_all`` and write results under
``results/``.
"""

# Nothing is executed on import; see individual modules for details.