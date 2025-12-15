# Benchmark Plan

This directory holds reusable fixtures and notes for evaluating the transcription pipeline. Use the structure and checklists below to keep inputs, ground truths, and timing results consistent across scenarios.

The unified benchmark runner and logic now reside in `backend/benchmarks/benchmark_runner.py`.

## Directory layout

```
benchmarks/
  01_scales/
    README.md          # scenario-specific notes and fixture checklist
    results.md         # recorded runs using the template below
    audio/             # raw audio fixtures (wav, mp3, etc.)
    references/        # MIDI or MusicXML ground truth when available
  02_simple_melodies/
  03_melody_plus_chords/
  04_pop_loops/
```
Alternatively use MIDIs from https://github.com/bytedance/GiantMIDI-Piano/tree/master/midis_for_evaluation
Add new scenarios by creating additional numbered folders that follow the same pattern (README + results.md + fixture subfolders).

## Naming conventions
- Use short, descriptive filenames such as `c_major_scale_100bpm.wav` or `folk_tune_in_g.mid`.
- Keep corresponding reference files aligned by basename (e.g., `folk_tune_in_g.mid` and `folk_tune_in_g_reference.musicxml`).
- Store mock-friendly XML/MIDI fixtures under `references/` so the mock pipeline can be exercised without audio dependencies.

## Running benchmarks

Benchmarks are now run using the unified runner in `backend.benchmarks`.

1. **Synthesize & Run L0-L2 benchmarks (synthetic)**:
   ```bash
   python -m backend.benchmarks.benchmark_runner --level L0 --output results
   python -m backend.benchmarks.benchmark_runner --level L1 --output results
   ```

2. **Run L4 benchmarks (Real Songs)**:
   This runs against the known "Real Songs" dataset (Happy Birthday, Old Macdonald).
   ```bash
   python -m backend.benchmarks.benchmark_runner --level L4 --output results
   ```
   (Ensure you have the ground truth JSONs in `backend/benchmarks/` or `benchmarks/`).

## Results template
Copy this template into each scenario's `results.md` (already seeded in the starter files):

```
# Results

| Date       | Environment (commit, OS, deps) | Fixture                          | Mode  | Iterations | Min (s) | Avg (s) | Max (s) | Notes |
|------------|--------------------------------|----------------------------------|-------|------------|---------|---------|---------|-------|
| 2025-02-15 | abc1234, Ubuntu, full pipeline | c_major_scale_100bpm.wav         | full  | 5          | 0.480   | 0.525   | 0.610   | Initial full run |
```

Include extra context below the table when comparing different model versions, noise levels, or quantization settings.
