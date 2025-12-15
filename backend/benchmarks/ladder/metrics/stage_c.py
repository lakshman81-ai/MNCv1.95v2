import numpy as np
import mir_eval
from typing import Dict, Any, List

def calculate_stage_c_metrics(
    notes_predicted: List[Any], # List of NoteEvent/ChordEvent
    ground_truth_midi_path: str
) -> Dict[str, float]:
    """
    Calculates Stage C (Transcription) metrics.
    Onset F1, Onset-Offset F1.
    """
    import music21

    # 1. Load Ground Truth intervals and pitches
    mf = music21.midi.MidiFile()
    mf.open(ground_truth_midi_path)
    mf.read()
    mf.close()
    s = music21.midi.translate.midiFileToStream(mf)

    ref_intervals = []
    ref_pitches = []

    # Need accurate timing. Assuming 100bpm or reading tempo.
    qpm = 100.0
    sec_per_q = 60.0 / qpm
    tempos = s.flat.getElementsByClass(music21.tempo.MetronomeMark)
    if tempos:
        qpm = tempos[0].number
        sec_per_q = 60.0 / qpm

    for n in s.flat.notes:
        start = n.offset * sec_per_q
        end = (n.offset + n.quarterLength) * sec_per_q

        if isinstance(n, music21.note.Note):
            ref_intervals.append([start, end])
            ref_pitches.append(n.pitch.frequency)
        elif isinstance(n, music21.chord.Chord):
            # For transcription eval, chords are usually broken into notes
            for p in n.pitches:
                ref_intervals.append([start, end])
                ref_pitches.append(p.frequency)

    ref_intervals = np.array(ref_intervals)
    ref_pitches = np.array(ref_pitches)

    # 2. Parse Predictions
    # notes_predicted comes from Stage C output.
    # Assuming it's a list of objects with start, end, pitch_hz or similar.
    # Looking at Stage C implementation (not provided in detail in memories,
    # assuming standard structure or adapting).

    est_intervals = []
    est_pitches = []

    for n in notes_predicted:
        # Check type
        # Assuming NoteEvent objects from models.py
        # They have start_time, end_time, pitch_hz (or similar)
        if hasattr(n, 'start_time') and hasattr(n, 'end_time'):
            est_intervals.append([n.start_time, n.end_time])
            if hasattr(n, 'pitch_hz'):
                est_pitches.append(n.pitch_hz)
            elif hasattr(n, 'pitches'): # Chord
                # Expand chord? Or just take root?
                # Benchmark usually compares individual notes.
                # If Stage C returns ChordEvents, we should expand.
                for p in n.pitches:
                    est_intervals.append([n.start_time, n.end_time])
                    est_pitches.append(p) # Hz
            else:
                 est_pitches.append(440.0) # Fallback

    est_intervals = np.array(est_intervals)
    est_pitches = np.array(est_pitches)

    if len(est_intervals) == 0:
        return {
            "Onset_F1": 0.0,
            "Onset_Offset_F1": 0.0,
            "Note_F1": 0.0 # With pitch
        }

    if len(ref_intervals) == 0:
        return {
            "Onset_F1": 0.0,
            "Onset_Offset_F1": 0.0,
            "Note_F1": 0.0
        }

    # 3. Calculate Metrics
    # Onset only
    onset_scores = mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches, est_intervals, est_pitches, onset_only=True
    )

    # Onset + Offset
    offset_scores = mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=0.2
    )

    metrics = {
        "Onset_F1": onset_scores['F-measure_no_offset'],
        "Onset_Offset_F1": offset_scores['F-measure_onset_offset'],
        "Note_With_Pitch_F1": onset_scores['F-measure_no_offset'] # mir_eval checks pitch by default even if offset ignored?
        # Actually: 'F-measure_no_offset' checks Onset + Pitch.
        # 'F-measure_onset_offset' checks Onset + Offset + Pitch.
    }

    return metrics
