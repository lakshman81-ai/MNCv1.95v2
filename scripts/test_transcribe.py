import math

from scripts.transcribe import quantize_duration


def test_quantize_duration_snaps_to_denominator():
    quantized, raw_beats = quantize_duration(0.5, bpm=120, denominators=[1.0, 0.5, 0.25])

    assert math.isclose(raw_beats, 1.0)
    assert quantized == 1.0
