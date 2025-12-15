import numpy as np
import mir_eval
from typing import Dict, Any, List

def calculate_stage_b_metrics(
    stage_b_output: Any,
    ground_truth_midi_path: str,
    sr: int = 44100
) -> Dict[str, float]:
    """
    Calculates Stage B (F0) metrics comparing pipeline output to MIDI ground truth.
    Uses mir_eval.melody for monophonic/skyline metrics.
    """
    import music21

    # 1. Load Ground Truth F0 from MIDI
    # We need to sample the MIDI at the same time grid as stage_b_output
    time_grid = stage_b_output.time_grid

    # Parse MIDI
    mf = music21.midi.MidiFile()
    mf.open(ground_truth_midi_path)
    mf.read()
    mf.close()
    s = music21.midi.translate.midiFileToStream(mf)
    flat = s.flat.notes

    # Construct Ground Truth F0 array
    ref_f0 = np.zeros_like(time_grid)

    # Simplified MIDI -> F0 (Skyline/Max Pitch)
    # For each frame, find the active notes, pick the highest pitch (Skyline)
    # This matches "Polyphonic Dominant" or "Mono" evaluation.

    # Convert notes to (start_time, end_time, freq)
    # Assumption: 100bpm default or read from midi?
    # If midi file has tempo events, music21 handles it in secondsMap usually?
    # music21.midi.translate handles tempo.

    # Let's trust secondsMap or simple calculation.
    # Actually, we generated the WAV with our Synth which assumed simple timing.
    # The pipeline should align if tempo matches.

    # For robustness, we iterate notes and map to time grid.
    # We need to know the tempo used for the MIDI.
    # The generated MIDI has 100bpm.
    qpm = 100.0
    sec_per_q = 60.0 / qpm

    # Check if midi has tempo
    tempos = s.flat.getElementsByClass(music21.tempo.MetronomeMark)
    if tempos:
        qpm = tempos[0].number
        sec_per_q = 60.0 / qpm

    # Build GT
    for n in flat:
        if isinstance(n, music21.note.Note):
            start = n.offset * sec_per_q
            end = (n.offset + n.quarterLength) * sec_per_q
            freq = n.pitch.frequency

            # Fill grid
            # Find indices
            start_idx = np.searchsorted(time_grid, start)
            end_idx = np.searchsorted(time_grid, end)

            # Simple Skyline: overwrite if higher or zero
            # Note: This simple overwrite assumes sorted notes or simple melody.
            # For polyphony, we want the highest pitch for skyline.
            for i in range(start_idx, min(end_idx, len(ref_f0))):
                if freq > ref_f0[i]:
                    ref_f0[i] = freq
        elif isinstance(n, music21.chord.Chord):
            start = n.offset * sec_per_q
            end = (n.offset + n.quarterLength) * sec_per_q
            # Skyline = max pitch in chord
            freq = max(p.frequency for p in n.pitches)

            start_idx = np.searchsorted(time_grid, start)
            end_idx = np.searchsorted(time_grid, end)

            for i in range(start_idx, min(end_idx, len(ref_f0))):
                if freq > ref_f0[i]:
                    ref_f0[i] = freq

    # 2. Compare using mir_eval
    est_f0 = stage_b_output.f0_main # The skyline/main estimation

    # mir_eval expects (ref_time, ref_freq, est_time, est_freq)
    # But for frame-level, we can pass arrays directly to mir_eval.melody.evaluate

    # Create voicing arrays
    ref_voicing = ref_f0 > 0
    est_voicing = est_f0 > 0

    # mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)
    # It resamples if times differ. Ours match.

    scores = mir_eval.melody.evaluate(time_grid, ref_f0, time_grid, est_f0)

    # scores keys: Raw Pitch Accuracy, Recall, Voicing False Alarm, etc.
    metrics = {
        "RPA": scores['Raw Pitch Accuracy'],
        "RCA": scores['Raw Chroma Accuracy'],
        "VR": scores['Voicing Recall'],
        "VFA": scores['Voicing False Alarm'],
        "OverallAccuracy": scores['Overall Accuracy']
    }

    return metrics
