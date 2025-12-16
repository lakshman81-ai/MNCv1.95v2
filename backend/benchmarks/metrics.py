"""Utility functions for computing transcription metrics.

These functions implement a minimal set of metrics suitable for
synthetic audio benchmarks.  They are intentionally lightweight and
require only NumPy; additional metrics from mir_eval or sound
libraries can be integrated later if desired.

Metric definitions
------------------

Pitch metrics:
    - ``cents_error``: mean absolute error in cents on voiced frames.
    - ``voicing_precision`` and ``voicing_recall``: proportion of
      correctly voiced/unvoiced detections.

Note metrics:
    - ``note_f1``: harmonic mean of precision and recall for
      correctly detected notes (by pitch and onset within tolerance).
    - ``onset_mae`` and ``offset_mae``: mean absolute error on start
      and end times of notes.

These metrics operate on simple Python lists or NumPy arrays; they
do not depend on the music21 library and thus can be used in any
environment.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


def cents_error(pred_hz: np.ndarray, gt_hz: np.ndarray) -> float:
    """Compute the mean absolute cents error on voiced frames.

    Parameters
    ----------
    pred_hz : np.ndarray
        Predicted fundamental frequency per frame (Hz).
    gt_hz : np.ndarray
        Ground‑truth fundamental frequency per frame (Hz).

    Returns
    -------
    float
        Mean absolute error in cents.  Returns ``nan`` if no voiced
        frames are present in either prediction or ground truth.
    """
    pred_hz = np.asarray(pred_hz, dtype=np.float64).reshape(-1)
    gt_hz = np.asarray(gt_hz, dtype=np.float64).reshape(-1)
    n = min(len(pred_hz), len(gt_hz))
    if n == 0:
        return float('nan')
    pred_hz = pred_hz[:n]
    gt_hz = gt_hz[:n]
    voiced = (gt_hz > 0.0)
    if not np.any(voiced):
        return float('nan')
    pred_voiced = np.where(pred_hz > 0.0, pred_hz, np.nan)
    err_cents = 1200.0 * np.log2(np.maximum(pred_voiced, 1e-9) / np.maximum(gt_hz, 1e-9))
    return float(np.nanmean(np.abs(err_cents[voiced])))


def voicing_precision_recall(pred_hz: np.ndarray, gt_hz: np.ndarray) -> Tuple[float, float]:
    """Compute voicing precision and recall.

    Parameters
    ----------
    pred_hz : np.ndarray
        Predicted fundamental frequency per frame (Hz).
    gt_hz : np.ndarray
        Ground‑truth fundamental frequency per frame (Hz).

    Returns
    -------
    (float, float)
        (precision, recall) of voiced frame detection.  Precision is
        the fraction of predicted voiced frames that are actually
        voiced; recall is the fraction of ground‑truth voiced frames
        that are correctly predicted as voiced.  If there are no
        voiced frames in ground truth, both precision and recall are
        returned as ``nan``.
    """
    pred_hz = np.asarray(pred_hz, dtype=np.float64).reshape(-1)
    gt_hz = np.asarray(gt_hz, dtype=np.float64).reshape(-1)
    n = min(len(pred_hz), len(gt_hz))
    if n == 0:
        return float('nan'), float('nan')
    pred_voiced = pred_hz[:n] > 0.0
    gt_voiced = gt_hz[:n] > 0.0
    tp = np.sum(pred_voiced & gt_voiced)
    fp = np.sum(pred_voiced & ~gt_voiced)
    fn = np.sum(~pred_voiced & gt_voiced)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / float(tp + fn) if (tp + fn) > 0 else float('nan')
    return precision, recall


def note_f1(pred_notes: List[Tuple[int, float, float]], gt_notes: List[Tuple[int, float, float]],
            onset_tol: float = 0.05) -> float:
    """Compute note F1 score between predicted and ground‑truth notes.

    Parameters
    ----------
    pred_notes : List[Tuple[int, float, float]]
        Predicted notes as (midi, start_sec, end_sec).
    gt_notes : List[Tuple[int, float, float]]
        Ground‑truth notes as (midi, start_sec, end_sec).
    onset_tol : float, optional
        Onset tolerance in seconds.  If the absolute difference
        between predicted and ground‑truth onsets is within this
        tolerance and the MIDI pitches match, the note is considered
        correct.  Default is 0.05 (50 ms).

    Returns
    -------
    float
        Note F1 score: 2 * (precision * recall) / (precision + recall).
    """
    if not gt_notes and not pred_notes:
        return float('nan')
    used = [False] * len(gt_notes)
    tp = 0
    for p_midi, p_start, p_end in pred_notes:
        for i, (g_midi, g_start, g_end) in enumerate(gt_notes):
            if used[i]:
                continue
            if p_midi == g_midi and abs(p_start - g_start) <= onset_tol:
                tp += 1
                used[i] = True
                break
    fp = len(pred_notes) - tp
    fn = len(gt_notes) - tp
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def onset_offset_mae(pred_notes: List[Tuple[int, float, float]], gt_notes: List[Tuple[int, float, float]]) -> Tuple[float, float]:
    """Compute mean absolute error of note onsets and offsets.

    Returns (onset_mae, offset_mae).  If there are no matching notes
    (by MIDI pitch), returns (nan, nan).
    """
    if not gt_notes or not pred_notes:
        return float('nan'), float('nan')
    errors_start = []
    errors_end = []
    for p_midi, p_start, p_end in pred_notes:
        # find closest matching MIDI note in gt
        candidates = [(i, g_start, g_end) for i, (g_midi, g_start, g_end) in enumerate(gt_notes) if g_midi == p_midi]
        if not candidates:
            continue
        # choose candidate with minimal onset error
        i_best, g_start_best, g_end_best = min(candidates, key=lambda x: abs(x[1] - p_start))
        errors_start.append(abs(p_start - g_start_best))
        errors_end.append(abs(p_end - g_end_best))
    if not errors_start:
        return float('nan'), float('nan')
    return float(np.mean(errors_start)), float(np.mean(errors_end))


def si_sdr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in linear domain."""

    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    n = min(len(pred), len(target))
    if n == 0:
        return float('nan')
    pred = pred[:n]
    target = target[:n]

    target_energy = np.sum(target ** 2) + eps
    if target_energy <= eps:
        return float('nan')

    scale = np.sum(pred * target) / target_energy
    projection = scale * target
    noise = pred - projection

    return float(10 * np.log10((np.sum(projection ** 2) + eps) / (np.sum(noise ** 2) + eps)))
