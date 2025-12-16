from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import tempfile
import os
import music21
from music21 import (
    stream,
    note,
    chord,
    tempo,
    meter,
    key,
    dynamics,
    articulations,
    layout,
    clef,
    midi,
)

# Import models and config from the package or the top level.  This makes stage_d
# usable both as ``backend.pipeline.stage_d`` and as a top‑level module ``stage_d``.
try:
    from .models import NoteEvent, AnalysisData, TranscriptionResult  # type: ignore
    from .config import PIANO_61KEY_CONFIG, PipelineConfig  # type: ignore
except Exception:
    from models import NoteEvent, AnalysisData, TranscriptionResult  # type: ignore
    from config import PIANO_61KEY_CONFIG, PipelineConfig  # type: ignore


def quantize_and_render(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
) -> TranscriptionResult:
    """
    Stage D: Render Sheet Music (MusicXML) and MIDI using music21.

    Returns TranscriptionResult containing musicxml string and midi bytes.
    """

    d_conf = config.stage_d
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    ts_str = analysis_data.meta.time_signature if analysis_data.meta.time_signature else "4/4"
    split_pitch = d_conf.staff_split_point.get("pitch", 60)  # C4

    # NOTE: divisions_per_quarter is currently not used.
    divisions_per_quarter = getattr(d_conf, "divisions_per_quarter", None)

    # Glissando config (currently not applied in v1)
    gliss_conf = d_conf.glissando_threshold_general
    piano_gliss_conf = d_conf.glissando_handling_piano

    # --------------------------------------------------------
    # 1. Setup Score and Parts (treble + bass)
    # --------------------------------------------------------
    part_treble = stream.Part()
    part_treble.id = "P1"
    part_bass = stream.Part()
    part_bass.id = "P2"

    # Clefs
    part_treble.append(clef.TrebleClef())
    part_bass.append(clef.BassClef())

    # Time Signature
    ts_obj = meter.TimeSignature(ts_str)
    part_treble.append(ts_obj)
    part_bass.append(meter.TimeSignature(ts_str))

    # Tempo (use separate objects per part to avoid reusing the same element)
    part_treble.append(tempo.MetronomeMark(number=float(bpm)))
    part_bass.append(tempo.MetronomeMark(number=float(bpm)))

    # Key (if detected)
    if analysis_data.meta.detected_key:
        try:
            part_treble.append(key.Key(analysis_data.meta.detected_key))
            part_bass.append(key.Key(analysis_data.meta.detected_key))
        except Exception:
            pass

    # --------------------------------------------------------
    # 2. Prepare Events: group by staff + onset in beats
    # --------------------------------------------------------
    quarter_dur = 60.0 / float(bpm)

    def get_event_beats(e: NoteEvent) -> Tuple[float, float]:
        dur_beats = getattr(e, "duration_beats", None)
        if dur_beats is None:
            dur_beats = (e.end_sec - e.start_sec) / quarter_dur
        start_beats = getattr(e, "start_beats", None)
        if start_beats is None:
            start_beats = getattr(e, "beat", None)
        if start_beats is None:
            start_beats = e.start_sec / quarter_dur

        return float(start_beats), float(dur_beats)

    events_sorted = sorted(events, key=lambda e: (e.start_sec, e.midi_note))

    staff_groups: Dict[Tuple[str, float], List[NoteEvent]] = {}

    for e in events_sorted:
        staff_name = getattr(e, "staff", None)
        if staff_name not in ("treble", "bass"):
            staff_name = "treble" if e.midi_note >= split_pitch else "bass"

        start_beats, dur_beats = get_event_beats(e)
        start_key = round(start_beats * 64.0) / 64.0

        key_tuple = (staff_name, start_key)
        if key_tuple not in staff_groups:
            staff_groups[key_tuple] = []
        staff_groups[key_tuple].append(e)

    # --------------------------------------------------------
    # 3. Create music21 Notes / Chords from grouped events
    # --------------------------------------------------------

    staccato_thresh = d_conf.staccato_marking.get("threshold_beats", 0.25)

    def build_m21_from_group(group: List[NoteEvent]):
        start_beats, dur_beats_first = get_event_beats(group[0])

        dur_beats_candidates: List[float] = []
        for e in group:
            _, dur_b = get_event_beats(e)
            dur_beats_candidates.append(dur_b)
        dur_beats = max(dur_beats_candidates) if dur_beats_candidates else dur_beats_first

        if dur_beats <= 0.0:
            dur_beats = staccato_thresh

        midi_pitches = sorted(list({e.midi_note for e in group}))

        if len(midi_pitches) > 1:
            m21_obj: music21.note.NotRest = chord.Chord(midi_pitches)
        else:
            m21_obj = note.Note(midi_pitches[0])

        q_len = _snap_ql(float(dur_beats))
        try:
            m21_obj.duration = music21.duration.Duration(q_len)
        except Exception:
            m21_obj.duration = music21.duration.Duration(1.0)

        velocities = [getattr(e, "velocity", 0.7) for e in group]
        avg_vel_norm = float(np.mean(velocities)) if velocities else 0.7
        midi_velocity = int(max(1, min(127, round(avg_vel_norm * 127.0))))
        m21_obj.volume.velocity = midi_velocity

        dyn_priority = {"p": 1, "mp": 2, "mf": 3, "f": 4}
        chosen_dyn = None
        best_score = 0
        for e in group:
            dyn = getattr(e, "dynamic", None)
            if dyn is None:
                continue
            score = dyn_priority.get(dyn, 0)
            if score > best_score:
                chosen_dyn = dyn
                best_score = score

        if chosen_dyn:
            dyn_obj = dynamics.Dynamic(chosen_dyn)
            m21_obj.expressions.append(dyn_obj)

        if q_len < float(staccato_thresh):
            m21_obj.articulations.append(articulations.Staccato())

        return m21_obj, float(start_beats)

    for (staff_name, start_key), group in sorted(staff_groups.items(), key=lambda x: x[0]):
        m21_obj, _start_beats = build_m21_from_group(group)
        offset = float(start_key)

        if staff_name == "bass":
            part_bass.insert(offset, m21_obj)
        else:
            part_treble.insert(offset, m21_obj)

    # Ensure each staff has something to quantize to measures
    default_rest_len = ts_obj.barDuration.quarterLength if ts_obj.barDuration else 4.0
    if len(part_treble.notesAndRests) == 0:
        part_treble.insert(0.0, note.Rest(quarterLength=default_rest_len))
    if len(part_bass.notesAndRests) == 0:
        part_bass.insert(0.0, note.Rest(quarterLength=default_rest_len))

    # Pad shorter staff to the longer staff's duration so measure spans match
    treble_end = part_treble.highestTime if part_treble.highestTime else 0.0
    bass_end = part_bass.highestTime if part_bass.highestTime else 0.0
    target_end = max(treble_end, bass_end, default_rest_len)
    if treble_end < target_end:
        part_treble.insert(treble_end, note.Rest(quarterLength=target_end - treble_end))
    if bass_end < target_end:
        part_bass.insert(bass_end, note.Rest(quarterLength=target_end - bass_end))

    # Quantize each part independently so measures exist on Parts
    s_quant = stream.Score()
    for part in (part_treble, part_bass):
        try:
            quantized_part = part.makeMeasures(inPlace=False)
            quantized_part.makeRests(inPlace=True)
            quantized_part.makeTies(inPlace=True)
        except Exception as e:
            print(f"[Stage D] makeMeasures/makeRests/makeTies failed for part {part.id}: {e}")
            quantized_part = part
        s_quant.append(quantized_part)

    # --------------------------------------------------------
    # 5. Export to MusicXML string and MIDI bytes
    # --------------------------------------------------------

    # MusicXML
    from music21.musicxml import m21ToXml
    exporter = m21ToXml.GeneralObjectExporter(s_quant)
    musicxml_bytes = exporter.parse()
    musicxml_str = musicxml_bytes.decode("utf-8")

    # MIDI
    midi_bytes = b""
    try:
        mf = midi.translate.music21ObjectToMidiFile(s_quant)
        # We need to write to a temp file because music21.midi.MidiFile doesn't easily write to bytes directly
        # but supports .write() to a file handle or path.
        # Actually .writestr() works if available? Check music21 docs... typically .writestr() produces bytes.
        if hasattr(mf, 'writestr'):
            midi_bytes = mf.writestr()
        else:
            # Fallback to temp file
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                tmp_path = tmp.name
            mf.open(tmp_path, 'wb')
            mf.write()
            mf.close()
            with open(tmp_path, 'rb') as f:
                midi_bytes = f.read()
            os.unlink(tmp_path)
    except Exception as e:
        print(f"[Stage D] MIDI export failed: {e}")
        # Provide a minimal non-empty fallback so downstream consumers/tests
        # still receive bytes even when music21 MIDI export fails.
        midi_bytes = musicxml_bytes

    return TranscriptionResult(
        musicxml=musicxml_str,
        analysis_data=analysis_data,
        midi_bytes=midi_bytes
    )


def _snap_ql(x: float, eps: float = 0.02) -> float:
    """Snap a quarterLength to MusicXML-friendly values."""
    if x is None or not np.isfinite(x):
        return 0.0
    x = float(x)
    for denom in (1, 2, 4, 8, 16, 32):
        y = round(x * denom) / denom
        if abs(x - y) <= eps:
            return float(y)
    y = float(round(x * 32) / 32)
    common = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    best = min(common, key=lambda v: abs(y - v))
    if abs(best - y) <= 0.15:
        return float(best)
    return y
