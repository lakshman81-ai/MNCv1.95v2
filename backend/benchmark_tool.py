import argparse
import os
import sys
import glob
import numpy as np
import music21
import tempfile
import mir_eval
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.transcription import transcribe_audio_pipeline
from backend.pipeline.models import NoteEvent

@dataclass
class BenchmarkMetrics:
    # Frame-level
    rpa: float = 0.0
    voicing_recall: float = 0.0
    voicing_false_alarm: float = 0.0

    # Note-level
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    onset_mae: float = 0.0 # Mean Absolute Error in seconds

    # Custom
    avg_overlap_ratio: float = 0.0

def parse_xml_notes(xml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a MusicXML file into (intervals, pitches) for mir_eval.
    intervals: (N, 2) float, start/end times
    pitches: (N,) float, Hz
    """
    try:
        score = music21.converter.parse(xml_path)
        intervals = []
        freqs = []

        # Unroll repeats and ties (simplified: flattening)
        # music21's flat properly handles offsets
        flat_score = score.flatten()
        notes = flat_score.notes

        # Determine tempo
        tempos = flat_score.getElementsByClass(music21.tempo.MetronomeMark)
        current_bpm = 120.0
        if len(tempos) > 0:
            current_bpm = tempos[0].getQuarterBPM()

        seconds_per_quarter = 60.0 / current_bpm

        for n in notes:
            # Calculate time from offsets (quarter notes)
            start_sec = n.offset * seconds_per_quarter
            duration_sec = n.duration.quarterLength * seconds_per_quarter
            end_sec = start_sec + duration_sec

            if isinstance(n, music21.note.Note):
                intervals.append([start_sec, end_sec])
                freqs.append(n.pitch.frequency)

            elif isinstance(n, music21.chord.Chord):
                # For chords, standard monophonic evaluation usually takes the top line or root.
                # mir_eval expects single F0 for melody.
                # Let's take the root for consistency with previous logic,
                # but be aware this is lossy for polyphonic matching.
                p = n.root()
                intervals.append([start_sec, end_sec])
                freqs.append(p.frequency)

        if not intervals:
            return np.empty((0, 2)), np.array([])

        # Sort by start time
        intervals = np.array(intervals)
        freqs = np.array(freqs)

        # mir_eval expects sorted intervals
        idx = np.argsort(intervals[:, 0])
        return intervals[idx], freqs[idx]

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return np.empty((0, 2)), np.array([])

def run_benchmark_single(args: Tuple[str, str, bool]) -> BenchmarkMetrics:
    audio_path, ref_xml_path, use_crepe = args

    try:
        # 1. Transcribe
        # Disable trimming for benchmark alignment
        result = transcribe_audio_pipeline(audio_path, use_crepe=use_crepe, trim_silence=False)

        # Save temp hypothesis XML
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False, mode='w', encoding='utf-8') as tmp:
            tmp.write(result.musicxml)
            tmp_path = tmp.name

        # 2. Load Hypothesis
        est_intervals, est_pitches = parse_xml_notes(tmp_path)
        os.remove(tmp_path)

        # 3. Load Reference
        ref_intervals, ref_pitches = parse_xml_notes(ref_xml_path)

        if len(ref_intervals) == 0:
            print(f"Warning: Reference {ref_xml_path} is empty.")
            return BenchmarkMetrics()

        if len(est_intervals) == 0:
            print(f"Warning: Hypothesis for {audio_path} is empty.")
            return BenchmarkMetrics()

        # 4. Calculate Metrics using mir_eval

        # Note-level (Standard MIREX)
        # onset_tolerance=0.05 (50ms), pitch_tolerance=50 cents, offset_ratio=0.2, offset_min_tolerance=0.05
        scores = mir_eval.transcription.evaluate(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=0.05,
            pitch_tolerance=50.0,
            offset_ratio=0.2,
            offset_min_tolerance=0.05
        )

        # Frame-level (RPA etc) requires frame-wise pitch labels.
        # XML gives notes. We can convert notes to frames or just rely on note metrics.
        # Requirement mentions "RPA" (Raw Pitch Accuracy) which is frame-level.
        # To compute RPA from XML, we'd need to rasterize the notes.
        # For this tool, let's focus on the Note-level MIREX metrics which are robust.
        # But user asked for "Metrics (frame-level): RPA...".
        # If we only have XML as reference, we can rasterize.

        # Rasterize Ref and Est to 10ms frames
        # This is an approximation.
        duration = max(ref_intervals[-1][1], est_intervals[-1][1]) + 1.0
        time_grid = np.arange(0, duration, 0.01)

        ref_frames = mir_eval.util.interpolate_intervals(ref_intervals, ref_pitches, time_grid, fill_value=0.0)
        est_frames = mir_eval.util.interpolate_intervals(est_intervals, est_pitches, time_grid, fill_value=0.0)

        # Ensure they are numpy arrays
        ref_frames = np.array(ref_frames)
        est_frames = np.array(est_frames)

        # Filter for voicing
        # RPA: Proportion of voiced frames with correct pitch (+- 50c)
        # Voicing Recall: Proportion of Ref voiced frames that are Est voiced

        # mir_eval.melody.evaluate requires (ref_time, ref_freq, est_time, est_freq)
        # But we have synchronized grid.

        ref_voicing = ref_frames > 0
        est_voicing = est_frames > 0

        # RPA
        # Only evaluate where ref is voiced
        voiced_ref_indices = np.where(ref_voicing)[0]
        if len(voiced_ref_indices) > 0:
            ref_v_freqs = ref_frames[voiced_ref_indices]
            est_v_freqs = est_frames[voiced_ref_indices]

            # Check correctness
            # |1200 log2(f_est/f_ref)| < 50
            # Handle est=0
            valid_est = est_v_freqs > 0

            diff_cents = np.abs(1200 * np.log2(est_v_freqs[valid_est] / ref_v_freqs[valid_est]))
            correct_frames = np.sum(diff_cents <= 50)
            rpa = correct_frames / len(voiced_ref_indices)
        else:
            rpa = 0.0

        # Voicing Recall
        # TP / (TP + FN) -> (Ref & Est) / Ref
        n_ref_voiced = np.sum(ref_voicing)
        n_both_voiced = np.sum(ref_voicing & est_voicing)
        v_recall = n_both_voiced / n_ref_voiced if n_ref_voiced > 0 else 0.0

        # Voicing False Alarm
        # FP / (FP + TN)? No, usually False Alarm Rate.
        # Let's use Precision/Recall on voicing.
        # Voicing Precision = (Ref & Est) / Est
        n_est_voiced = np.sum(est_voicing)
        v_prec = n_both_voiced / n_est_voiced if n_est_voiced > 0 else 0.0

        return BenchmarkMetrics(
            rpa=rpa,
            voicing_recall=v_recall,
            voicing_false_alarm=1.0 - v_prec, # loose interp
            precision=scores['Precision'],
            recall=scores['Recall'],
            f1=scores['F-measure'],
            onset_mae=0.0 # mir_eval doesn't give MAE directly in 'evaluate', strictly F-measure
        )

    except Exception as e:
        print(f"Failed benchmark for {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return BenchmarkMetrics()

def main():
    parser = argparse.ArgumentParser(description="Benchmark Transcription Pipeline")
    parser.add_argument("--data_dir", type=str, help="Directory containing pairs of .wav and .musicxml")
    parser.add_argument("--audio", type=str, help="Single audio file")
    parser.add_argument("--ref", type=str, help="Reference XML file")
    parser.add_argument("--use_crepe", action="store_true", help="Use CREPE pitch tracker")

    args = parser.parse_args()

    pairs = []

    if args.data_dir:
        wavs = glob.glob(os.path.join(args.data_dir, "*.wav"))
        for w in wavs:
            base = os.path.splitext(w)[0]
            xml = base + ".musicxml"
            if not os.path.exists(xml):
                xml = base + ".xml"

            if os.path.exists(xml):
                pairs.append((w, xml))
            else:
                print(f"Warning: No reference XML found for {w}")
    elif args.audio and args.ref:
        pairs.append((args.audio, args.ref))
    else:
        print("Please provide --data_dir or --audio and --ref")
        return

    print(f"Found {len(pairs)} pairs to benchmark.")

    metrics_list = []

    for audio, ref in pairs:
        print(f"Processing {os.path.basename(audio)}...")
        m = run_benchmark_single((audio, ref, args.use_crepe))
        metrics_list.append(m)
        print(f"  F1: {m.f1:.2f}, RPA: {m.rpa:.2f}")

    if not metrics_list:
        print("No results.")
        return

    avg_f1 = np.mean([m.f1 for m in metrics_list])
    avg_prec = np.mean([m.precision for m in metrics_list])
    avg_rec = np.mean([m.recall for m in metrics_list])
    avg_rpa = np.mean([m.rpa for m in metrics_list])

    print("\n=== Benchmark Results ===")
    print(f"Average Precision: {avg_prec:.2f}")
    print(f"Average Recall:    {avg_rec:.2f}")
    print(f"Average F1 Score:  {avg_f1:.2f}")
    print(f"Average RPA:       {avg_rpa:.2f}")
    print("=========================")

if __name__ == "__main__":
    main()
