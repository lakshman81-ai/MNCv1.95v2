# backend/pipeline/midi_export.py

from __future__ import annotations

from typing import List

from music21 import (
    stream,
    tempo as m21tempo,
    meter as m21meter,
    instrument as m21instrument,
    midi as m21midi,
    note as m21note,
)

from .models import NoteEvent, MetaData


def _get_bpm(meta: MetaData) -> float:
    """
    Safe BPM getter with a musical default.
    """
    bpm = getattr(meta, "tempo_bpm", None)
    if bpm is None or bpm <= 0:
        return 120.0
    return float(bpm)


def _get_time_signature(meta: MetaData) -> str:
    """
    Safe time-signature getter with a default of 4/4.
    """
    ts = getattr(meta, "time_signature", None)
    if not ts:
        return "4/4"
    return str(ts)


def notes_to_midi_stream(notes: List[NoteEvent], meta: MetaData) -> stream.Stream:
    """
    Build a music21 Stream from NoteEvent objects and MetaData.

    Mapping rules (aligned with Stage C / Stage D):
    - Timing:
        * Beats are quarter-note units (1 beat = quarter note).
        * If NoteEvent has .duration_beats / .start_beats, we use them.
        * Otherwise we derive from start_sec/end_sec using meta.tempo_bpm.
    - Pitch:
        * NoteEvent.midi_note → MIDI pitch number.
    - Velocity:
        * NoteEvent.velocity is expected to be 0–1 (as set in Stage C).
        * Mapped to MIDI 1–127 and stored in note.volume.velocity.
    """
    s = stream.Stream()

    bpm = _get_bpm(meta)
    ts_str = _get_time_signature(meta)

    # Global tempo and time signature
    s.append(m21tempo.MetronomeMark(number=bpm))
    s.append(m21meter.TimeSignature(ts_str))

    # Single instrument part (Piano by default; frontend can change later)
    s.append(m21instrument.Piano())

    quarter_dur = 60.0 / bpm

    for ev in notes:
        # Duration in beats
        dur_beats = getattr(ev, "duration_beats", None)
        if dur_beats is None:
            dur_beats = (ev.end_sec - ev.start_sec) / quarter_dur

        # Start in beats
        start_beats = getattr(ev, "start_beats", None)
        if start_beats is None:
            start_beats = ev.start_sec / quarter_dur

        # Guard against non-positive duration
        if dur_beats <= 0.0:
            dur_beats = 1.0 / 16.0  # minimum 1/16 note

        # Create music21 Note
        try:
            pitch_midi = int(ev.midi_note)
        except Exception:
            # Skip events with invalid pitch
            continue

        n = m21note.Note(pitch_midi)
        n.quarterLength = float(dur_beats)

        # Map normalized velocity (0–1) to MIDI 1–127 if present
        vel_norm = getattr(ev, "velocity", None)
        if vel_norm is not None:
            vel = int(round(max(0.0, min(1.0, float(vel_norm))) * 127.0))
            vel = max(1, min(127, vel))
            n.volume.velocity = vel

        # Insert into stream by offset in beats
        s.insert(float(start_beats), n)

    return s


def notes_to_midi_bytes(notes: List[NoteEvent], meta: MetaData) -> bytes:
    """
    Convert NoteEvent list + MetaData into a Standard MIDI File as bytes.

    This is used by backend.pipeline.__init__.transcribe(), which wraps:
        midi_bytes = notes_to_midi_bytes(notes, analysis.meta)

    If anything goes wrong in this function, the caller is already
    wrapped in a try/except and will fall back to b"".
    """
    # Build a music21 stream first
    s = notes_to_midi_stream(notes, meta)

    # Convert to MIDI file and then to bytes
    mf = m21midi.translate.streamToMidiFile(s)

    try:
        # music21's MidiFile has writestr() returning bytes
        midi_bytes: bytes = mf.writestr()
    except Exception:
        # Fallback: write to an in-memory buffer if needed
        import io

        buf = io.BytesIO()
        mf.open(buf)
        mf.write()
        mf.close()
        midi_bytes = buf.getvalue()

    return midi_bytes
