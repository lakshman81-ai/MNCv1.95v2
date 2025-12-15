# tests/test_pipeline_flow.py

import numpy as np
import pytest
import copy
from pathlib import Path

from backend.pipeline.config import PIANO_61KEY_CONFIG
from backend.pipeline.stage_a import load_and_preprocess
from backend.pipeline.stage_b import extract_features
from backend.pipeline.stage_c import apply_theory
from backend.pipeline.stage_d import quantize_and_render
from backend.pipeline.models import StageAOutput, AnalysisData, TranscriptionResult
# Updated: import transcribe from its specific module to avoid confusing it with the package
from backend.pipeline.transcribe import transcribe


def create_sine_wave(freq: float = 440.0, duration: float = 1.0, sr: int = 44100):
    """Simple pure tone generator for regression tests."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    return y, sr


def test_full_pipeline_flow_low_level(tmp_path):
    """
    Stage-by-stage test:
        Stage A → Stage B → Stage C → Stage D

    Uses a pure sine wave at 440 Hz and expects a note near A4 (MIDI 69).
    """

    # 1. Create dummy audio file
    sr_target = PIANO_61KEY_CONFIG.stage_a.target_sample_rate
    y, sr = create_sine_wave(freq=440.0, duration=1.0, sr=sr_target)

    audio_path = tmp_path / "test_sine.wav"
    import soundfile as sf

    sf.write(audio_path, y, sr)

    # 2. Stage A
    print("Running Stage A...")
    # Updated: Pass FULL config so Stage A can resolve hop/window from Stage B defaults
    stage_a_out = load_and_preprocess(
        str(audio_path),
        config=PIANO_61KEY_CONFIG,
    )
    assert isinstance(stage_a_out, StageAOutput)
    assert stage_a_out.meta.sample_rate == sr_target
    assert stage_a_out.meta.target_sr == sr_target

    # 3. Stage B
    print("Running Stage B...")
    # Use a deep copy so we don't mutate the global config
    test_config = copy.deepcopy(PIANO_61KEY_CONFIG)

    # For this test we want a minimal, deterministic detector setup:
    test_config.stage_b.separation["enabled"] = False

    # Disable neural / heavy detectors for pure sine
    test_config.stage_b.detectors["swiftf0"]["enabled"] = False
    test_config.stage_b.detectors["sacf"]["enabled"] = False
    test_config.stage_b.detectors["cqt"]["enabled"] = False
    test_config.stage_b.detectors["rmvpe"]["enabled"] = False
    test_config.stage_b.detectors["crepe"]["enabled"] = False

    # Enable YIN (robust for sinusoidal)
    test_config.stage_b.detectors["yin"]["enabled"] = True

    stage_b_out = extract_features(stage_a_out, config=test_config)

    # Debug: show detector averages
    print("Per Detector Results:")
    for stem, dets in stage_b_out.per_detector.items():
        for dname, (f0, conf) in dets.items():
            avg_f0 = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0.0
            print(f"Stem: {stem}, Det: {dname}, Avg F0: {avg_f0:.2f} Hz")

    # 4. Stage C
    print("Running Stage C...")
    analysis_data = AnalysisData(
        meta=stage_a_out.meta,
        timeline=[],
        stem_timelines=stage_b_out.stem_timelines,
    )

    notes = apply_theory(analysis_data, config=test_config)

    if len(notes) == 0:
        pytest.fail("No notes detected by YIN on sine wave.")

    print(f"Detected {len(notes)} notes.")
    print(f"First note: {notes[0]}")

    # YIN should be accurate for 440 Hz → A4 (~69)
    # Allow ±1 semitone to be safe (68–70)
    assert 68 <= notes[0].midi_note <= 70

    # 5. Stage D
    print("Running Stage D...")
    result_d = quantize_and_render(notes, analysis_data, config=test_config)

    # Updated: quantize_and_render returns TranscriptionResult
    assert isinstance(result_d, TranscriptionResult)
    assert "<?xml" in result_d.musicxml
    assert "<score-partwise" in result_d.musicxml

    print("Low-level pipeline test passed!")


def test_transcribe_orchestrator(tmp_path):
    """
    High-level test of backend.pipeline.transcribe().
    Ensures the orchestration Stage A→B→C→D works end-to-end.
    """

    # 1. Create simple sine file again
    sr_target = PIANO_61KEY_CONFIG.stage_a.target_sample_rate
    y, sr = create_sine_wave(freq=440.0, duration=1.0, sr=sr_target)

    audio_path = tmp_path / "test_sine_orchestrator.wav"
    import soundfile as sf

    sf.write(audio_path, y, sr)

    # 2. Call high-level API
    result = transcribe(str(audio_path), config=PIANO_61KEY_CONFIG)

    # 3. Basic checks
    # Updated: result is TranscriptionResult
    assert isinstance(result, TranscriptionResult)
    assert isinstance(result.musicxml, str)
    assert "<?xml" in result.musicxml
    assert "<score-partwise" in result.musicxml
    assert isinstance(result.midi_bytes, bytes)

    analysis = result.analysis_data
    assert analysis.meta.sample_rate == sr_target
    assert analysis.meta.target_sr == sr_target

    # There should be at least one note
    assert len(analysis.notes) > 0
    # Sanity check on first note pitch
    first_note = analysis.notes[0]
    print("Orchestrator first note:", first_note)
    assert 20 <= first_note.midi_note <= 110  # just a broad sanity band

    print("Orchestrator pipeline test passed!")
