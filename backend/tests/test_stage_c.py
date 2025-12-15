
import pytest
import numpy as np
from backend.pipeline.stage_c import apply_theory, quantize_notes
from backend.pipeline.models import NoteEvent, AnalysisData, MetaData, FramePitch

def test_stage_c_segmentation():
    # Setup timeline with stable pitch
    timeline = []
    # 5 frames of silence
    for i in range(5):
        timeline.append(FramePitch(time=i*0.01, pitch_hz=0, midi=None, confidence=0.0, rms=0.0))

    # 10 frames of A4 (440Hz, midi 69)
    for i in range(10):
        timeline.append(FramePitch(time=(i+5)*0.01, pitch_hz=440.0, midi=69, confidence=0.9, rms=0.5))

    # 5 frames of silence
    for i in range(5):
        timeline.append(FramePitch(time=(i+15)*0.01, pitch_hz=0, midi=None, confidence=0.0, rms=0.0))

    analysis = AnalysisData(timeline=timeline)

    notes = apply_theory([], analysis)

    # Expect 1 note
    assert len(notes) == 1
    n = notes[0]
    assert n.midi_note == 69
    # Start at 0.05s. BPM 120 -> 1/16th = 0.125s.
    # 0.05 is closer to 0.0 than 0.125. Quantization snaps to 0.0.
    assert abs(n.start_sec - 0.0) < 0.001

    # End at ~0.15s.
    # 0.15 is closer to 0.125 than 0.25.
    # Snaps to 0.125.
    # Note duration min is 1/16th (0.125).
    # If start 0.0, end 0.125.
    assert abs(n.end_sec - 0.125) < 0.001

def test_stage_c_quantization():
    # Note at 1.02s
    # BPM 120 -> Quarter=0.5s, 16th=0.125s
    # 1.0s is exactly on beat (2nd second, start of measure 3? No, start of measure 1 beat 3)
    # 1.02s should snap to 1.0s

    n = NoteEvent(start_sec=1.02, end_sec=1.14, midi_note=60, pitch_hz=261.6)

    analysis = AnalysisData(meta=MetaData(tempo_bpm=120.0))

    quantized = quantize_notes([n], analysis_data=analysis)
    q = quantized[0]

    # 1.0s is 8th 16th note (0.125 * 8 = 1.0)
    # 1.02s / 0.125 = 8.16 -> rounds to 8

    assert abs(q.start_sec - 1.0) < 0.001

    # End 1.14 / 0.125 = 9.12 -> rounds to 9 -> 1.125
    assert abs(q.end_sec - 1.125) < 0.001

    # Check measure/beat
    # 1.0s = 2 beats (0-based index?) -> Beat 3.0 (1-based)
    # Measure 1, Beat 3.0
    assert q.measure == 1
    assert q.beat == 3.0
