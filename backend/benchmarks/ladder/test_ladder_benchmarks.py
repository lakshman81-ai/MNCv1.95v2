import json
from pathlib import Path

import pytest
from music21 import chord, note, stream

from backend.benchmarks.ladder.generators import generate_benchmark_example
from backend.benchmarks.ladder.levels import BENCHMARK_LEVELS


def _collect_timed_events(part: stream.Part):
    events = []
    for element in part.recurse().notesAndRests:
        if getattr(element, "quarterLength", None) is None:
            continue
        start = float(element.offset)
        end = start + float(element.quarterLength)
        events.append((start, end, element))
    return events


def _has_overlaps(melody_events, accompaniment_events) -> bool:
    for m_start, m_end, _ in melody_events:
        for a_start, a_end, _ in accompaniment_events:
            if m_start < a_end and a_start < m_end:
                return True
    return False


def _validate_metrics_schema(data: dict):
    required = {
        "level": str,
        "name": str,
        "note_f1": (int, float),
        "onset_mae_ms": (int, float, type(None)),
        "predicted_count": int,
        "gt_count": int,
    }
    for key, expected_type in required.items():
        assert key in data, f"Missing '{key}' in metrics payload"
        assert isinstance(data[key], expected_type), f"{key} should be {expected_type}"


def _validate_run_info_schema(data: dict):
    assert "detectors_ran" in data, "detectors_ran missing from run info"
    assert isinstance(data["detectors_ran"], list)
    assert all(isinstance(det, str) for det in data["detectors_ran"])

    assert "config" in data, "config missing from run info"
    assert isinstance(data["config"], dict)
    for stage_key in ["stage_a", "stage_b", "stage_c", "stage_d"]:
        assert stage_key in data["config"], f"{stage_key} missing from run info config"
        assert isinstance(data["config"][stage_key], dict)


@pytest.mark.parametrize("level", BENCHMARK_LEVELS)
def test_generate_examples_structure(level):
    for example_id in level["examples"]:
        score = generate_benchmark_example(example_id)
        assert isinstance(score, stream.Score)

        parts = list(score.parts)
        if level["polyphony"].startswith("mono"):
            assert len(parts) == 1
            mono_events = _collect_timed_events(parts[0])
            assert mono_events
            assert all(isinstance(ev, note.Note) for _, _, ev in mono_events)
            continue

        assert len(parts) >= 2, "Polyphonic levels should include accompaniment"
        melody, accompaniment = parts[0], parts[1]
        melody_events = _collect_timed_events(melody)
        accompaniment_events = _collect_timed_events(accompaniment)

        assert melody_events and accompaniment_events
        assert _has_overlaps(melody_events, accompaniment_events), "Parts should overlap to create polyphony"

        if level["polyphony"] == "polyphonic_dominant":
            assert any(isinstance(ev, chord.Chord) for _, _, ev in accompaniment_events)
            melody_high = max(ev.pitch.midi for _, _, ev in melody_events if hasattr(ev, "pitch"))
            accomp_high = max(
                max(p.midi for p in ev.pitches) if isinstance(ev, chord.Chord) else ev.pitch.midi
                for _, _, ev in accompaniment_events
                if hasattr(ev, "pitches") or hasattr(ev, "pitch")
            )
            assert melody_high > accomp_high, "Melody should sit above softer accompaniment"

        if level["polyphony"] == "polyphonic_full":
            assert any(isinstance(ev, note.Note) for _, _, ev in accompaniment_events)


def _artifact_directories():
    repo_root = Path(__file__).resolve().parents[3]
    return [
        repo_root / "results" / "benchmark_1765724693",
        Path(__file__).resolve().parent / "test_artifacts",
    ]


@pytest.mark.parametrize("artifact_dir", _artifact_directories())
def test_benchmark_artifact_schema(artifact_dir):
    if not artifact_dir.exists():
        pytest.skip(f"Artifact directory {artifact_dir} not available")

    metric_files = sorted(artifact_dir.glob("*_metrics.json"))
    run_info_files = sorted(artifact_dir.glob("*_run_info.json"))

    assert metric_files, f"No metrics files found in {artifact_dir}"
    assert run_info_files, f"No run_info files found in {artifact_dir}"

    for metrics_path in metric_files:
        with open(metrics_path) as f:
            data = json.load(f)
        _validate_metrics_schema(data)

    for run_info_path in run_info_files:
        with open(run_info_path) as f:
            data = json.load(f)
        _validate_run_info_schema(data)
