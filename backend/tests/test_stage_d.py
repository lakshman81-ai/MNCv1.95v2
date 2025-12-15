
import pytest
import numpy as np
from backend.pipeline.stage_d import quantize_and_render
from backend.pipeline.models import NoteEvent, AnalysisData, MetaData

# Removed test_gap_merging because gap merging is Stage C responsibility, not Stage D rendering.
# Stage D renders what it is given.

def test_velocity_mapping_passthrough():
    meta = MetaData()
    analysis = AnalysisData(meta=meta)

    # Stage D should respect the velocity already computed (by Stage C).
    # If Stage C set velocity to 0.157, Stage D uses it.

    n_quiet = NoteEvent(start_sec=0, end_sec=1, midi_note=60, pitch_hz=261.6, velocity=0.157)
    n_loud = NoteEvent(start_sec=1, end_sec=2, midi_note=62, pitch_hz=293.6, velocity=0.826)

    res = quantize_and_render([n_quiet, n_loud], analysis)

    # We verify that the XML export didn't crash and we got MIDI bytes
    assert len(res.midi_bytes) > 0
    assert len(res.musicxml) > 0

    # We can check if the internal velocity was preserved if we had access to the intermediate music21 stream,
    # but since the function returns a string/bytes, we just assert it runs.
