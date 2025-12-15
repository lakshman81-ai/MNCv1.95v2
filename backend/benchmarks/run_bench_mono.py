"""Run synthetic mono benchmarks for the music‑note pipeline.

This script generates simple test signals (pure sine waves) to verify
that the pipeline detects pitch and segments notes correctly.  It
computes pitch and note metrics for each test case and writes a
summary table to the console.  The script can be executed directly
via ``python -m backend.benchmarks.run_bench_mono``.

Currently the benchmark suite covers the L0 (mono sanity) and L1
(mono musical) levels described in the work instructions.  For each
case, the metrics reported are:

    - Cents error on voiced frames (mean absolute)
    - Voicing precision & recall
    - Note F1 score
    - Onset MAE (ms) and Offset MAE (ms)

If you wish to extend this benchmark to include polyphonic cases or
parameter tuning, you can add more test signals and adjust the
configuration accordingly.  The results could also be saved to CSV
and JSON as described in the instructions.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import os
import sys

# Ensure the project root is on sys.path so that stage modules can be imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import PipelineConfig
from stage_b import extract_features
from stage_c import apply_theory
from models import (
    MetaData,
    StageAOutput,
    Stem,
    AnalysisData,
    AudioType,
    AudioQuality,
    NoteEvent,
)
from .metrics import (
    cents_error,
    voicing_precision_recall,
    note_f1,
    onset_offset_mae,
)


def synth_sine(frequency: float, sr: int = 44100, duration: float = 1.0) -> np.ndarray:
    """Generate a sine wave at a given frequency.

    Parameters
    ----------
    frequency : float
        Frequency of the sine wave in Hz.
    sr : int, optional
        Sample rate.  Default is 44100.
    duration : float, optional
        Duration of the signal in seconds.  Default is 1.0.

    Returns
    -------
    np.ndarray
        A 1‑D NumPy array containing the sine wave.
    """
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2.0 * np.pi * frequency * t).astype(np.float32)


def run_mono_case(name: str, freq: float, config: PipelineConfig) -> Dict[str, Any]:
    """Run a single mono benchmark case and return metrics.

    The case uses a pure sine wave of the given frequency.  It runs
    through Stage B and Stage C of the pipeline and computes basic
    transcription metrics.

    Parameters
    ----------
    name : str
        A label for the test case.
    freq : float
        Frequency of the sine wave in Hz.
    config : PipelineConfig
        Pipeline configuration to use.

    Returns
    -------
    Dict[str, Any]
        A dictionary of metrics for this test case.
    """
    sr = config.stage_a.target_sample_rate
    duration = 1.0
    audio = synth_sine(freq, sr=sr, duration=duration)

    # Build StageAOutput manually to bypass Stage A preprocessor.
    meta = MetaData(
        tuning_offset=0.0,
        detected_key="C",
        lufs=-23.0,
        processing_mode="monophonic",
        audio_type=AudioType.MONOPHONIC,
        audio_quality=AudioQuality.LOSSLESS,
        snr=0.0,
        window_size=config.stage_b.detectors.get('yin', {}).get('n_fft', 2048),
        hop_length=config.stage_b.detectors.get('yin', {}).get('hop_length', 512),
        sample_rate=sr,
        tempo_bpm=120.0,
        time_signature="4/4",
        original_sr=sr,
        target_sr=sr,
        duration_sec=duration,
        beats=[],
        audio_path=None,
        n_channels=1,
        normalization_gain_db=0.0,
        rms_db=-20.0,
        noise_floor_rms=0.0,
        noise_floor_db=-80.0,
        pipeline_version="bench",
    )
    stage_a_out = StageAOutput(
        stems={"mix": Stem(audio=audio, sr=sr, type="mix")},
        meta=meta,
        audio_type=AudioType.MONOPHONIC,
        noise_floor_rms=meta.noise_floor_rms,
        noise_floor_db=meta.noise_floor_db,
        beats=[],
    )

    # Prepare analysis data
    analysis = AnalysisData(meta=meta)
    analysis.beats = []

    # Stage B
    stage_b_out = extract_features(stage_a_output=stage_a_out, config=config)
    analysis.stem_timelines = stage_b_out.stem_timelines

    # Stage C
    notes: List[NoteEvent] = apply_theory(analysis, config=config)

    # Metrics
    # Get main f0 stream from Stage B's f0_main
    f0_pred = stage_b_out.f0_main
    f0_gt = np.full_like(f0_pred, float(freq), dtype=np.float32)
    cents_err = cents_error(f0_pred, f0_gt)
    prec, rec = voicing_precision_recall(f0_pred, f0_gt)

    # Note metrics: predicted vs ground truth note (one note across duration)
    pred_notes: List[Tuple[int, float, float]] = []
    for n in notes:
        pred_notes.append((n.midi_note, n.start_sec, n.end_sec))
    # Ground truth note: one note with full duration, MIDI from freq
    midi_gt = int(round(69 + 12 * np.log2(float(freq) / 440.0)))
    gt_notes = [(midi_gt, 0.0, duration)]
    f1 = note_f1(pred_notes, gt_notes, onset_tol=0.05)
    onset_mae, offset_mae = onset_offset_mae(pred_notes, gt_notes)

    return {
        "case": name,
        "freq": freq,
        "cents_error": cents_err,
        "voicing_precision": prec,
        "voicing_recall": rec,
        "note_f1": f1,
        "onset_mae_ms": onset_mae * 1000.0 if onset_mae == onset_mae else float('nan'),
        "offset_mae_ms": offset_mae * 1000.0 if offset_mae == offset_mae else float('nan'),
        "n_pred_notes": len(pred_notes),
    }


def run_benchmarks() -> List[Dict[str, Any]]:
    """Run all mono benchmarks and return summary list."""
    config = PipelineConfig()
    # For mono benchmarks, disable separation to speed up.
    config.stage_b.separation["enabled"] = False
    test_cases = [
        ("sine_440Hz", 440.0),
        ("sine_220Hz", 220.0),
        ("sine_523Hz", 523.25),
    ]
    results = []
    for name, freq in test_cases:
        results.append(run_mono_case(name, freq, config))
    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a tabulated summary of benchmark results to the console."""
    header = [
        "Case",
        "Freq (Hz)",
        "CentsErr",
        "VoicePrec",
        "VoiceRec",
        "NoteF1",
        "OnsetMAE (ms)",
        "OffsetMAE (ms)",
        "#Notes",
    ]
    print("\t".join(header))
    for r in results:
        row = [
            r["case"],
            f"{r['freq']:.2f}",
            f"{r['cents_error']:.2f}" if r["cents_error"] == r["cents_error"] else "nan",
            f"{r['voicing_precision']:.2f}" if r["voicing_precision"] == r["voicing_precision"] else "nan",
            f"{r['voicing_recall']:.2f}" if r["voicing_recall"] == r["voicing_recall"] else "nan",
            f"{r['note_f1']:.2f}" if r["note_f1"] == r["note_f1"] else "nan",
            f"{r['onset_mae_ms']:.1f}" if r["onset_mae_ms"] == r["onset_mae_ms"] else "nan",
            f"{r['offset_mae_ms']:.1f}" if r["offset_mae_ms"] == r["offset_mae_ms"] else "nan",
            str(r["n_pred_notes"]),
        ]
        print("\t".join(row))


def main() -> None:
    results = run_benchmarks()
    print_summary(results)
    # Optionally write results to a JSON file for later inspection
    out_dir = os.path.join("results", "mono")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_dir}/summary.json")


if __name__ == "__main__":
    main()