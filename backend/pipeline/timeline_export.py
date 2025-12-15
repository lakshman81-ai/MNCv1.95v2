# backend/pipeline/timeline_export.py

from __future__ import annotations

from typing import Dict, Any, List
from dataclasses import asdict
import base64

from .models import AnalysisData, FramePitch, NoteEvent, TranscriptionResult


def frame_pitch_to_dict(fp: FramePitch) -> Dict[str, Any]:
    """
    Convert a FramePitch into a JSON-friendly dictionary.

    This is intended for per-frame timeline visualisation on the frontend:
    - time (seconds)
    - pitch_hz (0 if unvoiced)
    - midi (None if unvoiced)
    - confidence (0–1)
    - rms (linear)
    - active_pitches: list of {pitch_hz, confidence} for polyphonic layers
    """
    return {
        "time": float(fp.time),
        "pitch_hz": float(fp.pitch_hz),
        "midi": int(fp.midi) if fp.midi is not None else None,
        "confidence": float(fp.confidence),
        "rms": float(fp.rms),
        "active_pitches": [
            {
                "pitch_hz": float(p),
                "confidence": float(c),
            }
            for (p, c) in fp.active_pitches
        ],
    }


def note_event_to_dict(note: NoteEvent) -> Dict[str, Any]:
    """
    Convert a NoteEvent into a JSON-friendly dictionary for UI timelines.

    This mostly mirrors AnalysisData.to_dict() but is explicit here so the
    timeline export is self-contained and stable for API/Frontend use.
    """
    # Some fields (staff, measure, beat, duration_beats, alternatives, spec_thumb)
    # may or may not be present depending on which stage populated them;
    # getattr with defaults keeps this robust.
    return {
        "start_sec": float(note.start_sec),
        "end_sec": float(note.end_sec),
        "midi_note": int(note.midi_note),
        "pitch_hz": float(note.pitch_hz),
        "confidence": float(getattr(note, "confidence", 0.0)),
        "velocity": float(getattr(note, "velocity", 0.8)),
        "rms_value": float(getattr(note, "rms_value", 0.0)),
        "is_grace": bool(getattr(note, "is_grace", False)),
        "dynamic": getattr(note, "dynamic", "mf"),
        "voice": int(getattr(note, "voice", 1)),
        "staff": getattr(note, "staff", None),
        "measure": getattr(note, "measure", None),
        "beat": getattr(note, "beat", None),
        "duration_beats": float(getattr(note, "duration_beats", 0.0)),
        "alternatives": [
            asdict(a) for a in getattr(note, "alternatives", []) or []
        ],
        "spec_thumb": getattr(note, "spec_thumb", None),
    }


def analysis_to_timeline_dict(analysis: AnalysisData) -> Dict[str, Any]:
    """
    Build a consolidated, JSON-friendly payload for timeline visualisation.

    Structure:

    {
      "meta": { ... },                # MetaData (sample_rate, tempo, etc.)
      "beats": [ ... ],               # Beat times in seconds
      "global_timeline": [ ... ],     # Optional: aggregate FramePitch list
      "stems": {
        "piano": [ {FramePitch dict}, ... ],
        "vocals": [ ... ],
        ...
      },
      "notes": [ {NoteEvent dict}, ... ]
    }

    - Uses analysis.notes if available, else falls back to analysis.events.
    - Uses analysis.stem_timelines (multi-stem) for per-stem curves.
    - Uses analysis.timeline (if populated) for a single global curve.
    """
    # Choose which note list to expose
    notes_source: List[NoteEvent] = (
        analysis.notes if analysis.notes else analysis.events
    )

    # Per-stem timelines
    stems: Dict[str, List[Dict[str, Any]]] = {}
    for stem_name, fp_list in analysis.stem_timelines.items():
        stems[stem_name] = [frame_pitch_to_dict(fp) for fp in fp_list]

    # Global timeline (may be empty if you only use stem_timelines)
    global_timeline: List[Dict[str, Any]] = [
        frame_pitch_to_dict(fp) for fp in analysis.timeline
    ]

    return {
        "meta": asdict(analysis.meta),
        "beats": [float(b) for b in analysis.beats],
        "global_timeline": global_timeline,
        "stems": stems,
        "notes": [note_event_to_dict(n) for n in notes_source],
    }


def transcription_result_to_payload(result: TranscriptionResult) -> Dict[str, Any]:
    """
    High-level helper: convert a TranscriptionResult into a single
    JSON-friendly payload that UI / API layers can return directly.

    Structure:

    {
      "musicxml": "<score-partwise ...>",
      "midi_bytes_b64": "....",          # base64-encoded MIDI (optional)
      "analysis": { ... }                # output of analysis_to_timeline_dict(...)
    }

    If midi_bytes is empty, midi_bytes_b64 will be None.
    """
    # Timeline + notes + meta
    analysis_payload = analysis_to_timeline_dict(result.analysis_data)

    # Encode MIDI bytes for transport if available
    midi_bytes_b64: str | None
    if result.midi_bytes:
        midi_bytes_b64 = base64.b64encode(result.midi_bytes).decode("ascii")
    else:
        midi_bytes_b64 = None

    return {
        "musicxml": result.musicxml,
        "midi_bytes_b64": midi_bytes_b64,
        "analysis": analysis_payload,
    }
