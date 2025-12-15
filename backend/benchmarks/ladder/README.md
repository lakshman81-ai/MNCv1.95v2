# Ladder benchmarks

This folder defines the ladder benchmark levels and utilities used by the automated
checks in `test_ladder_benchmarks.py`.

## Generation structure expectations

The test suite iterates over `BENCHMARK_LEVELS` and calls `generate_benchmark_example`
for every `example_id`. For monophonic levels the generated `Score` must contain a
single part with only `music21.note.Note` events. Polyphonic levels must add an
accompaniment part that overlaps the melody so the resulting score is truly
polyphonic. The L3 `polyphonic_dominant` examples are expected to place the melody
above softer chordal accompaniment, while L4 `polyphonic_full` examples need an
active accompaniment line.

## Exported artifact schema

When benchmark runs emit JSON artifacts, the tests assert the following schema:

- `*_metrics.json` files must include the string fields `level` and `name`, numeric
  `note_f1`, `onset_mae_ms` (float or null), and integer `predicted_count` and
  `gt_count` values.
- `*_run_info.json` files must provide a `detectors_ran` list of strings and a
  `config` object containing dictionaries for `stage_a`, `stage_b`, `stage_c`, and
  `stage_d`.

Any newly added level should ship example artifact files in
`backend/benchmarks/ladder/test_artifacts` that follow this schema so the
regression tests continue to cover the full ladder.
