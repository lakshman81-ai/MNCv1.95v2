import os
import sys
import numpy as np
import music21
import librosa
import subprocess
from typing import List, Dict, Any, Tuple, Union

# Ensure we can import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from backend.tools.synthesize import synthesize_xml

def generate_test_data(xml_path: str, wav_path: str, force: bool = False):
    """Generates WAV from XML if not exists or forced."""
    if not os.path.exists(wav_path) or force:
        print(f"[INFO] Synthesizing {xml_path} -> {wav_path}")
        synthesize_xml(xml_path, wav_path)
    else:
        print(f"[INFO] Using existing {wav_path}")

def load_ground_truth(xml_path: str, hop_length: int = 512, sr: int = 22050, polyphony: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[List[float]], np.ndarray]]:
    """
    Parses XML and returns (f0_curve, voiced_mask) for frames.
    If polyphony=True, returns (List[List[float]], voiced_mask).
    """
    try:
        score = music21.converter.parse(xml_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse XML: {e}")
        return ([], np.array([])) if polyphony else (np.array([]), np.array([]))

    # Estimate duration
    bpm = 120.0
    mm = score.flat.getElementsByClass('MetronomeMark')
    if mm:
        bpm = mm[0].number

    highest_time = score.highestTime
    duration_sec = highest_time * (60.0 / bpm)

    n_frames = int(duration_sec * sr / hop_length) + 1

    if not polyphony:
        f0 = np.zeros(n_frames)
    else:
        f0 = [[] for _ in range(n_frames)]

    voiced = np.zeros(n_frames, dtype=bool)

    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    notes = score.flat.notes

    for n in notes:
        if n.isRest: continue

        start_sec = n.offset * (60.0 / bpm)
        dur_sec = n.quarterLength * (60.0 / bpm)
        gap = min(0.05, dur_sec * 0.2)
        end_sec = start_sec + dur_sec - gap

        # Mask
        mask_idx = np.where((times >= start_sec) & (times < end_sec))[0]

        pitches = []
        if isinstance(n, music21.chord.Chord):
            pitches = [p.frequency for p in n.pitches]
        else:
            pitches = [n.pitch.frequency]

        if not polyphony:
            # Take root/lowest for mono GT
            p = min(pitches)
            if len(mask_idx) > 0:
                f0[mask_idx] = p
                voiced[mask_idx] = True
        else:
            for idx in mask_idx:
                f0[idx].extend(pitches)
                voiced[idx] = True

    return f0, voiced

def calculate_metrics(pred_f0: np.ndarray, gt_f0: np.ndarray, gt_voiced: np.ndarray) -> Dict[str, float]:
    """
    Calculates Precision, Recall, F1, RPA for Monophonic.
    """
    min_len = min(len(pred_f0), len(gt_f0))
    pred = pred_f0[:min_len]
    gt = gt_f0[:min_len]
    gt_v = gt_voiced[:min_len]

    pred_v = (pred > 10.0)

    valid_idx = (pred > 0) & (gt > 0)

    diff_semitones = np.abs(12 * np.log2(pred[valid_idx] / gt[valid_idx]))
    match_mask = diff_semitones < 0.5

    pitch_match = np.zeros(min_len, dtype=bool)
    pitch_match[valid_idx] = match_mask

    tp = np.sum(pred_v & gt_v & pitch_match)
    fp = np.sum(pred_v) - tp
    fn = np.sum(gt_v) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "RPA": recall
    }

def calculate_poly_metrics(pred_timelines: List[Any], gt_frames: List[List[float]], gt_voiced: np.ndarray) -> Dict[str, float]:
    """
    Polyphonic Metrics.
    pred_timelines: List of FramePitch objects (global timeline) or just use stems?
    The user wants "F1 > 0.80 on Polyphonic Mix test".
    The pipeline merges everything into a global timeline (dominant pitch?)
    OR returns `stem_timelines`.
    For polyphony, we should check if *all* notes in GT are present in the output.
    But `Stage D` (Midi generation) produces notes.
    For this benchmark, let's compare the extracted notes (events) or the frame-wise active pitches.
    Our `FramePitch` has `active_pitches` list!
    """
    # pred_timelines is list of FramePitch
    n_frames = min(len(pred_timelines), len(gt_frames))

    tp = 0
    fp = 0
    fn = 0

    for i in range(n_frames):
        gt_pitches = gt_frames[i]
        pred_obj = pred_timelines[i]

        # Get all active pitches in prediction
        # active_pitches is List[(hz, conf)]
        pred_pitches = [p[0] for p in pred_obj.active_pitches if p[1] > 0.3] # Threshold?

        # Match sets
        # For each GT pitch, is there a match in Pred?
        matched_gt_indices = set()
        matched_pred_indices = set()

        for g_idx, g_hz in enumerate(gt_pitches):
            for p_idx, p_hz in enumerate(pred_pitches):
                if p_idx in matched_pred_indices: continue

                diff = abs(12 * np.log2(p_hz / g_hz))
                if diff < 0.5:
                    tp += 1
                    matched_gt_indices.add(g_idx)
                    matched_pred_indices.add(p_idx)
                    break

        # False Negatives: GT pitches not matched
        fn += len(gt_pitches) - len(matched_gt_indices)

        # False Positives: Pred pitches not matched
        fp += len(pred_pitches) - len(matched_pred_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }
