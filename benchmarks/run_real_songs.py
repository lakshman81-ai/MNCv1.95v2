"""Run real song benchmarks for the music‑note pipeline.

This script synthesizes simple audio from ground‑truth note sequences for
"Happy Birthday" and "Old MacDonald", runs the Stage B/C pipeline on the
generated audio, and computes note‑level metrics.  It saves the predicted
notes and metrics to ``results/run_<timestamp>``.

Usage:

    python -m backend.benchmarks.run_real_songs --song happy_birthday

    python -m backend.benchmarks.run_real_songs --song old_macdonald

    python -m backend.benchmarks.run_real_songs --song all

If no ``--song`` argument is provided, both songs are processed.

Note: This script assumes that ground truth note lists live in
``ground_truth/<song>_gt.json`` and synthesizes the audio itself.  It
does not rely on external MIDI or audio files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from backend.pipeline.config import PipelineConfig
from backend.pipeline.models import (
    StageAOutput,
    MetaData,
    Stem,
    AnalysisData,
    AudioType,
    AudioQuality,
)
from backend.pipeline.stage_b import extract_features
from backend.pipeline.stage_c import apply_theory
from backend.benchmarks.metrics import note_f1, onset_offset_mae


def midi_to_freq(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12.0)


def synthesize_audio(notes: List[Tuple[int, float]], sr: int = 44100) -> np.ndarray:
    """Generate a waveform from a sequence of (midi_note, duration) tuples."""
    signal: np.ndarray = np.array([], dtype=np.float32)
    for midi_note, dur in notes:
        freq = midi_to_freq(midi_note)
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
        wave = 0.3 * np.sin(2.0 * np.pi * freq * t)
        fade_len = int(0.01 * sr)
        fade = np.linspace(0, 1, fade_len)
        if fade_len > 0 and len(wave) >= fade_len:
            wave[:fade_len] *= fade
            wave[-fade_len:] *= fade[::-1]
        signal = np.concatenate((signal, wave))
    return signal


def load_ground_truth(song: str) -> List[Dict[str, Any]]:
    # Search in benchmarks folder first
    path = os.path.join('benchmarks', f'{song}_gt.json')
    if not os.path.exists(path):
         # fallback to current dir or ground_truth
         path = os.path.join('ground_truth', f'{song}_gt.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)["notes"]


def run_song(song: str) -> Dict[str, Any]:
    gt = load_ground_truth(song)
    # Build (midi_note, duration) list
    notes_dur = [(n['midi_note'], n['end_sec'] - n['start_sec']) for n in gt]
    sr = 44100
    audio = synthesize_audio(notes_dur, sr=sr)

    # Construct StageAOutput manually
    config = PipelineConfig()
    # disable separation for synthetic audio
    config.stage_b.separation['enabled'] = False
    # Tune Stage C for synthetic audio (tight segmentation)
    config.stage_c.velocity_map['min_db'] = -15.0
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
        duration_sec=float(len(audio) / sr),
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
    )
    analysis = AnalysisData(meta=meta)
    stage_b_out = extract_features(stage_a_out, config=config)
    analysis.stem_timelines = stage_b_out.stem_timelines
    notes_pred = apply_theory(analysis, config=config)

    # Convert predicted and ground truth to uniform tuples
    pred_list = [
        (n.midi_note, float(n.start_sec), float(n.end_sec)) for n in notes_pred
    ]
    gt_list = [
        (n['midi_note'], float(n['start_sec']), float(n['end_sec'])) for n in gt
    ]
    f1 = note_f1(pred_list, gt_list, onset_tol=0.05)
    onset_mae, offset_mae = onset_offset_mae(pred_list, gt_list)

    return {
        'song': song,
        'note_f1': f1,
        'onset_mae_ms': onset_mae * 1000.0 if onset_mae == onset_mae else None,
        'offset_mae_ms': offset_mae * 1000.0 if offset_mae == offset_mae else None,
        'predicted_notes': len(pred_list),
        'gt_notes': len(gt_list),
        'predicted': pred_list,
        'ground_truth': gt_list,
        'resolved_config': config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real songs benchmark")
    parser.add_argument('--song', type=str, default='all', help='Song name: happy_birthday, old_macdonald, or all')
    args = parser.parse_args()
    songs = []
    if args.song in ('happy_birthday', 'old_macdonald'):
        songs = [args.song]
    else:
        songs = ['happy_birthday', 'old_macdonald']

    timestamp = int(time.time())
    run_dir = os.path.join('results', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for song in songs:
        res = run_song(song)
        summary.append({k: res[k] for k in ['song','note_f1','onset_mae_ms','offset_mae_ms','predicted_notes','gt_notes']})
        # Save per-song artifacts
        with open(os.path.join(run_dir, f'{song}_predicted_notes.json'), 'w', encoding='utf-8') as f:
            json.dump([{'midi_note': m, 'start_sec': s, 'end_sec': e} for m,s,e in res['predicted']], f, indent=2)
        with open(os.path.join(run_dir, f'{song}_ground_truth_notes.json'), 'w', encoding='utf-8') as f:
            json.dump([{'midi_note': m, 'start_sec': s, 'end_sec': e} for m,s,e in res['ground_truth']], f, indent=2)
        # Save resolved config (serialized via dict)
        with open(os.path.join(run_dir, f'{song}_resolved_config.json'), 'w', encoding='utf-8') as f:
            json.dump(res['resolved_config'].__dict__, f, indent=2, default=lambda o: o.__dict__)
        # Save metrics
        with open(os.path.join(run_dir, f'{song}_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({k: res[k] for k in ['note_f1','onset_mae_ms','offset_mae_ms','predicted_notes','gt_notes']}, f, indent=2)

    # Write summary and leaderboard
    summary_path = os.path.join('results', 'summary.csv')
    leaderboard_path = os.path.join('results', 'leaderboard.json')
    # Append summary to CSV
    header = ['song','note_f1','onset_mae_ms','offset_mae_ms','predicted_notes','gt_notes']
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(','.join(header) + '\n')
    with open(summary_path, 'a', encoding='utf-8') as f:
        for row in summary:
            f.write(','.join(str(row[h]) for h in header) + '\n')
    # Leaderboard json
    leaderboard = {s['song']: {'note_f1': s['note_f1']} for s in summary}
    with open(leaderboard_path, 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=2)
    print('Completed real-song benchmark; results saved to', run_dir)


if __name__ == '__main__':
    main()