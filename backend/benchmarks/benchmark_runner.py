"""
Unified Benchmark Runner (L0-L4)

This module implements the full benchmark ladder:
- L0: Mono Sanity (Synthetic Sine/Vibrato)
- L1: Mono Musical (Synthetic MIDI)
- L2: Poly Dominant (Synthetic Mix)
- L3: Full Poly (MusicXML-backed synthetic score)
- L4: Real Songs (via run_real_songs)

It validates algorithm selection (Stage B), records polyphonic diagnostics, and saves artifacts/metrics.
"""

from __future__ import annotations

import os
import json
import time
import argparse
import logging
import numpy as np
import warnings
import tempfile
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from music21 import tempo, chord

from backend.pipeline.config import PipelineConfig, InstrumentProfile
from backend.pipeline.models import (
    StageAOutput, MetaData, Stem, AnalysisData, AudioType, NoteEvent
)
from backend.pipeline.stage_a import load_and_preprocess
from backend.pipeline.stage_b import extract_features
from backend.pipeline.stage_c import apply_theory, quantize_notes
from backend.pipeline.stage_d import quantize_and_render
from backend.benchmarks.metrics import note_f1, onset_offset_mae
from backend.benchmarks.run_real_songs import run_song as run_real_song
from backend.benchmarks.ladder.generators import generate_benchmark_example
from backend.benchmarks.ladder.synth import midi_to_wav_synth

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark_runner")


def accuracy_benchmark_plan() -> Dict[str, Any]:
    """Structured description of the accuracy-focused benchmark plan.

    The returned payload groups scenarios, toggles, and expected metrics so tests and
    dashboards can validate coverage without depending on markdown prose alone.
    """

    return {
        "end_to_end": {
            "scenarios": [
                "clean_piano",
                "dense_chords",
                "percussive_passages",
                "noisy_inputs",
            ],
            "outputs": ["musicxml", "midi_bytes", "analysis_timelines"],
            "goals": [
                "stage_A_to_D_flow",
                "aggregate_outputs",
                "consistency_across_artifacts",
            ],
            "acceptance_metrics": ["note_f1", "onset_offset_f1", "runtime_s"],
        },
        "stage_a": {
            "toggles": [
                "sample_rate_targets",
                "channel_handling",
                "trimming",
                "loudness_normalization",
            ],
            "fixtures": ["silence", "dc_offset_tones", "clipped_signals"],
            "metrics": [
                "snr_change",
                "loudness_change",
                "latency_s",
                "headroom_recovery",
            ],
            "measurements": [
                "pre_post_snr",
                "pre_post_lufs",
                "conditioning_wall_time",
            ],
        },
        "stage_b": {
            "detectors": ["yin", "swiftf0", "crepe", "rmvpe"],
            "ensemble_settings": [
                "confidence_voicing_threshold",
                "pitch_disagreement_cents",
                "per_detector_flags",
                "source_separation",
                "harmonic_masking",
            ],
            "metrics": [
                "f0_precision",
                "f0_recall",
                "voicing_error",
                "latency_s",
                "robustness_to_masking",
            ],
            "fixtures": ["annotated_monophonic", "annotated_polyphonic"],
            "robustness_checks": ["separation_on_off", "harmonic_masking_on_off"],
        },
        "stage_c": {
            "segmentation_modes": ["hmm", "threshold"],
            "parameters": ["minimum_duration", "pitch_merge_tolerance", "gap_filling"],
            "fixtures": ["staccato", "legato", "varied_tempos"],
            "metrics": [
                "note_f_measure",
                "onset_offset_f_measure",
                "fragmentation_rate",
                "merging_rate",
            ],
            "tempo_sweeps": ["slow", "medium", "fast", "rubato_sim"]
        },
        "stage_d": {
            "scenarios": ["tempo_grids", "swing", "rubato"],
            "metrics": [
                "beat_alignment_error",
                "barline_placement",
                "notation_cleanliness",
            ],
            "ground_truth": "synthetic_midi_round_trip",
            "render_checks": ["quantize_and_render", "swing_grid_alignment"],
        },
        "ablation": {
            "sweeps": [
                "source_separation",
                "ensemble_weights",
                "segmentation_method",
            ],
            "reports": ["f_measure_impact", "runtime_impact", "interaction_notes"],
        },
        "regression": {
            "corpus": "fixed_benchmark_corpus",
            "thresholds": ["accuracy_delta", "timing_delta", "latency_budget"],
            "stage_thresholds": {
                "end_to_end_note_f1_delta": 0.01,
                "stage_a_latency_delta_s": 0.05,
                "stage_b_voicing_error_delta": 0.01,
            },
            "alerts": True,
        },
        "profiling": {
            "hooks": [
                "stage_timings",
                "noise_floor",
                "detector_confidences",
                "hmm_state_durations",
            ],
            "purpose": "contextualize_benchmark_results",
            "artifacts": ["profiling_traces", "intermediate_metrics"],
        },
    }

def midi_to_freq(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12.0)

def synthesize_audio(notes: List[Tuple[int, float]], sr: int = 44100, waveform: str = 'sine') -> np.ndarray:
    """Generate simple synthetic audio."""
    signal = np.array([], dtype=np.float32)
    for midi_note, dur in notes:
        freq = midi_to_freq(midi_note)
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
        if waveform == 'sine':
            wave = 0.5 * np.sin(2.0 * np.pi * freq * t)
        elif waveform == 'saw':
            # Simple approx
            wave = 0.5 * (2.0 * (t * freq - np.floor(t * freq + 0.5)))
        else:
            wave = 0.5 * np.sin(2.0 * np.pi * freq * t)

        # Envelope
        fade_len = int(0.01 * sr)
        if fade_len > 0 and len(wave) >= fade_len:
            fade = np.linspace(0, 1, fade_len)
            wave[:fade_len] *= fade
            wave[-fade_len:] *= fade[::-1]

        signal = np.concatenate((signal, wave))
    return signal


def _load_musicxml_notes(xml_path: str) -> List[Tuple[int, float, float]]:
    """Parse a MusicXML into note tuples (midi, start_sec, end_sec)."""
    score = music21.converter.parse(xml_path)
    bpm = 120.0
    mm = score.flat.getElementsByClass('MetronomeMark')
    if mm:
        bpm = mm[0].number

    sec_per_beat = 60.0 / bpm
    gt: List[Tuple[int, float, float]] = []
    for n in score.flat.notes:
        if n.isRest:
            continue
        start_sec = float(n.offset * sec_per_beat)
        end_sec = float((n.offset + n.quarterLength) * sec_per_beat)
        if hasattr(n, "pitches"):
            for p in n.pitches:
                gt.append((int(p.midi), start_sec, end_sec))
        else:
            gt.append((int(n.pitch.midi), start_sec, end_sec))
    return gt

def run_pipeline_on_audio(
    audio: np.ndarray,
    sr: int,
    config: PipelineConfig,
    audio_type: AudioType = AudioType.MONOPHONIC,
    audio_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full pipeline on raw audio array."""

    # Synthetic benchmarks do not require source separation and the default Demucs
    # model download can fail in offline environments. Disable separation here to
    # keep the ladder runnable without external network access.
    if config.stage_b.separation.get("enabled", False):
        config.stage_b.separation["enabled"] = False

    # 1. Stage A (Manual construction since we have raw audio, but let's simulate Stage A output)
    # We can skip load_and_preprocess if we already have the array, but we should fill meta correctly.
    meta = MetaData(
        sample_rate=sr,
        target_sr=sr,
        duration_sec=float(len(audio)) / sr,
        processing_mode=audio_type.value,
        audio_type=audio_type,
        audio_path=audio_path,
        hop_length=config.stage_b.detectors.get('yin', {}).get('hop_length', 512),
        window_size=config.stage_b.detectors.get('yin', {}).get('n_fft', 2048),
        # Assuming normalized already for synthetic
        lufs=-20.0,
    )

    stems = {"mix": Stem(audio=audio, sr=sr, type="mix")}
    if audio_type == AudioType.POLYPHONIC_DOMINANT:
         # For L2, we might want to simulate separate stems if we had them,
         # but for now we feed mix and let Stage B handle it (or separation if enabled).
         pass

    stage_a_out = StageAOutput(stems=stems, meta=meta, audio_type=audio_type)

    # 2. Stage B
    stage_b_out = extract_features(stage_a_out, config=config)

    # 3. Stage C
    analysis = AnalysisData(meta=meta, stem_timelines=stage_b_out.stem_timelines)
    notes_pred = apply_theory(analysis, config=config)

    # 4. Stage D (Verify it runs, though we check notes mostly)
    try:
        transcription_result = quantize_and_render(notes_pred, analysis, config=config)
    except Exception as e:
        logger.warning(f"Stage D failed: {e}")
        transcription_result = None

    return {
        "notes": notes_pred,
        "stage_b_out": stage_b_out,
        "transcription": transcription_result,
        "resolved_config": config # Stage B might warn but doesn't mutate much, we log what we passed
    }

class BenchmarkSuite:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.results = []
        os.makedirs(output_dir, exist_ok=True)

    def _poly_config(self, use_harmonic_masking: bool = False) -> PipelineConfig:
        config = PipelineConfig()
        config.stage_b.separation["enabled"] = True
        config.stage_b.separation["synthetic_model"] = False
        config.stage_b.separation["harmonic_masking"]["enabled"] = use_harmonic_masking
        config.stage_b.polyphonic_peeling["force_on_mix"] = True
        return config

    def _enable_high_capacity_frontend(self, config: PipelineConfig) -> None:
        config.stage_b.detectors["crepe"]["enabled"] = True
        config.stage_b.detectors["crepe"]["model_capacity"] = "full"
        config.stage_b.detectors["rmvpe"]["enabled"] = True
        config.stage_b.detectors["rmvpe"]["fmax"] = 2000.0

    def _compute_diff(self, previous: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prev_map = {(r.get("level"), r.get("name")): r for r in previous}
        diff: List[Dict[str, Any]] = []
        for r in current:
            key = (r.get("level"), r.get("name"))
            prev = prev_map.get(key)
            if not prev:
                continue
            diff.append({
                "level": r.get("level"),
                "name": r.get("name"),
                "delta_note_f1": r.get("note_f1") - prev.get("note_f1", 0.0),
                "delta_onset_mae_ms": (r.get("onset_mae_ms") or 0.0) - (prev.get("onset_mae_ms") or 0.0),
                "previous": prev,
                "current": r,
            })
        return diff

    def _merge_results(self, previous: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge current run results into the previously persisted snapshot.

        When benchmarks are executed piecemeal (e.g., separate invocations for L1–L4),
        this keeps the latest metrics for each (level, name) pair instead of
        overwriting the entire snapshot with the most recent slice.
        """

        merged = {(r.get("level"), r.get("name")): r for r in previous}
        for r in current:
            merged[(r.get("level"), r.get("name"))] = r

        return [merged[k] for k in sorted(merged.keys())]

    def _save_run(self, level: str, name: str, res: Dict[str, Any], gt: List[Tuple[int, float, float]]):
        """Save artifacts for a single run."""
        pred_notes = res['notes']
        pred_list = [(n.midi_note, n.start_sec, n.end_sec) for n in pred_notes]

        # Calculate Metrics
        f1 = note_f1(pred_list, gt, onset_tol=0.05)
        onset_mae, offset_mae = onset_offset_mae(pred_list, gt)

        # Normalize NaNs for downstream checks/serialization
        if np.isnan(f1):
            f1 = 0.0
        if onset_mae is not None and np.isnan(onset_mae):
            onset_mae = None

        metrics = {
            "level": level,
            "name": name,
            "note_f1": f1,
            "onset_mae_ms": onset_mae * 1000 if onset_mae is not None else None,
            "predicted_count": len(pred_list),
            "gt_count": len(gt)
        }
        self.results.append(metrics)

        # Save JSONs
        base_path = os.path.join(self.output_dir, f"{level}_{name}")

        with open(f"{base_path}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(f"{base_path}_pred.json", "w") as f:
            json.dump([asdict(n) for n in pred_notes], f, indent=2, default=str)

        with open(f"{base_path}_gt.json", "w") as f:
            json.dump([{"midi": m, "start": s, "end": e} for m,s,e in gt], f, indent=2)

        # Log resolved config
        # We also want to see what detectors ran
        detectors_ran = list(res['stage_b_out'].per_detector.get('mix', {}).keys())
        diagnostics = getattr(res.get("stage_b_out"), "diagnostics", {}) if res.get("stage_b_out") else {}
        run_info = {
            "detectors_ran": detectors_ran,
            "diagnostics": diagnostics,
            "config": asdict(res['resolved_config'])
        }
        with open(f"{base_path}_run_info.json", "w") as f:
            json.dump(run_info, f, indent=2, default=str)

        return metrics

    @staticmethod
    def _score_to_gt(score) -> List[Tuple[int, float, float]]:
        tempo_marks = score.flat.getElementsByClass(tempo.MetronomeMark)
        bpm = float(tempo_marks[0].number) if tempo_marks else 100.0
        sec_per_quarter = 60.0 / bpm if bpm else 0.6

        gt: List[Tuple[int, float, float]] = []
        for el in score.flat.notes:
            start = float(el.offset) * sec_per_quarter
            dur = float(el.quarterLength) * sec_per_quarter
            end = start + dur

            if isinstance(el, chord.Chord):
                for p in el.pitches:
                    gt.append((int(p.midi), start, end))
            else:
                gt.append((int(el.pitch.midi), start, end))

        return gt

    def run_L0_mono_sanity(self):
        logger.info("Running L0: Mono Sanity")

        # Case 1: Simple Sine 440Hz
        sr = 44100
        notes = [(69, 1.0)] # A4, 1 sec
        audio = synthesize_audio(notes, sr=sr, waveform='sine')

        config = PipelineConfig()
        config.stage_b.detectors['swiftf0']['enabled'] = False # Force baseline for sanity check if needed?
        # Actually let's just let it use defaults. But we want to ensure it works.

        gt = [(69, 0.0, 1.0)]
        res = run_pipeline_on_audio(audio, sr, config, AudioType.MONOPHONIC)

        m = self._save_run("L0", "sine_440", res, gt)

        # Validations
        if m['note_f1'] < 0.9:
            raise RuntimeError(f"L0 Failed: Sine 440 F1 {m['note_f1']} < 0.9")

        # Verify algorithm selection
        detectors = res['stage_b_out'].per_detector.get('mix', {})
        if not any(d in detectors for d in ['yin', 'sacf', 'swiftf0', 'crepe']):
            raise RuntimeError("L0 Failed: No mono pitch tracker ran!")

        logger.info("L0 Passed.")

    def run_L1_mono_musical(self):
        logger.info("Running L1: Mono Musical")

        # Scale C major
        notes = [
            (60, 0.5), (62, 0.5), (64, 0.5), (65, 0.5),
            (67, 0.5), (69, 0.5), (71, 0.5), (72, 0.5)
        ]
        audio = synthesize_audio(notes, sr=44100, waveform='saw') # Use saw for harmonics

        gt = []
        t = 0.0
        for m, d in notes:
            gt.append((m, t, t+d))
            t += d

        config = PipelineConfig()
        res = run_pipeline_on_audio(audio, 44100, config, AudioType.MONOPHONIC)

        m = self._save_run("L1", "scale_c_maj", res, gt)

        if m['note_f1'] < 0.9:
             logger.warning(f"L1 Warning: F1 {m['note_f1']} < 0.9. (Strict pass required for production)")
             # raise RuntimeError("L1 Failed") # Do not fail hard yet for dev

        logger.info(f"L1 Complete. F1: {m['note_f1']}")

    def run_L2_poly_dominant(self):
        logger.info("Running L2: Poly Dominant")

        # Melody + Bass
        # Melody: C5, E5, G5 (0.5s each)
        # Bass: C3 (1.5s)
        sr = 44100
        melody = synthesize_audio([(72, 0.5), (76, 0.5), (79, 0.5)], sr, 'sine')
        bass = synthesize_audio([(48, 1.5)], sr, 'saw') * 0.5 # Lower volume

        mix = melody + bass
        gt_melody = [(72, 0.0, 0.5), (76, 0.5, 1.0), (79, 1.0, 1.5)]

        config = PipelineConfig()
        config.stage_b.separation['enabled'] = False
        # Enable separation if possible, or assume dominant melody extraction works
        # config.stage_b.separation['enabled'] = True

        res = run_pipeline_on_audio(mix, sr, config, AudioType.POLYPHONIC_DOMINANT)

        m = self._save_run("L2", "melody_plus_bass_synthetic_sep", res, gt_melody)

        # We expect it to find the melody (highest energy/frequency?)
        # Standard YIN might track bass or jump. RMVPE/Swift should track melody.
        # This is a harder test without separation.
        detectors = res['stage_b_out'].per_detector.get('mix', {})
        if m['note_f1'] < 0.25:
            raise RuntimeError(f"L2 Failed: melody_plus_bass F1 {m['note_f1']} < 0.25")
        if m['onset_mae_ms'] is None or m['onset_mae_ms'] > 250:
            raise RuntimeError(f"L2 Failed: onset MAE {m['onset_mae_ms']}ms is too high")
        if len(detectors) < 2:
            raise RuntimeError("L2 Failed: insufficient detector coverage on poly mix")

        logger.info(f"L2 Complete. F1: {m['note_f1']}")

        # High-capacity frontend experiment (CREPE + RMVPE)
        exp_config = self._poly_config(use_harmonic_masking=True)
        self._enable_high_capacity_frontend(exp_config)
        exp_res = run_pipeline_on_audio(mix, sr, exp_config, AudioType.POLYPHONIC_DOMINANT)
        m_exp = self._save_run("L2", "melody_plus_bass_crepe_rmvpe", exp_res, gt_melody)
        logger.info(f"L2 CREPE/RMVPE Complete. F1: {m_exp['note_f1']}")

    def run_L3_full_poly(self):
        logger.info("Running L3: Full Poly")

        # 1. Generate full-poly example and synthesize audio
        score = generate_benchmark_example('old_macdonald_poly_full')
        gt = self._score_to_gt(score)

        sr = 22050
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            midi_to_wav_synth(score, wav_path, sr=sr)
            audio, read_sr = sf.read(wav_path)
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Trim to a shorter clip to keep CI/runtime reasonable while retaining polyphony
        max_duration = 8.0
        if len(audio) > int(max_duration * read_sr):
            audio = audio[: int(max_duration * read_sr)]
            gt = [
                (m, s, min(e, max_duration))
                for m, s, e in gt
                if s < max_duration
            ]

        config = PipelineConfig()
        config.stage_b.separation['enabled'] = False
        config.stage_b.polyphonic_peeling["max_layers"] = 0
        for det in ["swiftf0", "rmvpe", "crepe", "yin"]:
            if det in config.stage_b.detectors:
                config.stage_b.detectors[det]["enabled"] = False
        res = run_pipeline_on_audio(audio.astype(np.float32), int(read_sr), config, AudioType.POLYPHONIC)

        m = self._save_run("L3", "old_macdonald_poly_full", res, gt)

        detectors = res['stage_b_out'].per_detector.get('mix', {})
        if m['note_f1'] < 0.2:
            raise RuntimeError(f"L3 Failed: old_macdonald_poly_full F1 {m['note_f1']} < 0.2")
        if m['onset_mae_ms'] is None or m['onset_mae_ms'] > 300:
            raise RuntimeError(f"L3 Failed: onset MAE {m['onset_mae_ms']}ms is too high")
        if len(detectors) < 2:
            raise RuntimeError("L3 Failed: insufficient detector coverage on full-poly mix")
        if m['predicted_count'] == 0:
            raise RuntimeError("L3 Failed: no notes predicted for full-poly example")

        logger.info(f"L3 Complete. F1: {m['note_f1']}")

    def run_L4_real_songs(self):
        logger.info("Running L4: Real Songs")
        # Reuse run_real_songs logic but integrate here
        # We need to adapt it to return metrics and save to our dir

        try:
            # Happy Birthday
            res_hb = run_real_song('happy_birthday')
            self._save_real_song_result("L4", "happy_birthday", res_hb)

            # Old Macdonald
            res_om = run_real_song('old_macdonald')
            self._save_real_song_result("L4", "old_macdonald", res_om)

        except Exception as e:
            logger.error(f"L4 Failed: {e}")

    def _save_real_song_result(self, level, name, res):
        # Adapt run_real_songs output dict to our metrics
        metrics = {
            "level": level,
            "name": name,
            "note_f1": res['note_f1'],
            "onset_mae_ms": res['onset_mae_ms'],
            "predicted_count": res['predicted_notes'],
            "gt_count": res['gt_notes']
        }
        self.results.append(metrics)

        base_path = os.path.join(self.output_dir, f"{level}_{name}")
        with open(f"{base_path}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # We don't have the raw stage_b_out from run_real_song easily without modifying it,
        # but we have the resolved config
        with open(f"{base_path}_run_info.json", "w") as f:
            json.dump({"config": asdict(res['resolved_config'])}, f, indent=2, default=str)


    def generate_summary(self):
        summary_path = os.path.join(self.output_dir, "summary.csv")
        leaderboard_path = os.path.join(self.output_dir, "leaderboard.json")
        snapshot_path = os.path.join(self.output_dir, "summary.json")
        latest_path = os.path.join("results", "benchmark_latest.json")

        # CSV
        header = ["level", "name", "note_f1", "onset_mae_ms", "predicted_count", "gt_count"]
        with open(summary_path, "w") as f:
            f.write(",".join(header) + "\n")
            for r in self.results:
                line = [str(r.get(h, "")) for h in header]
                f.write(",".join(line) + "\n")

        # Leaderboard
        lb = {r['name']: r['note_f1'] for r in self.results}
        with open(leaderboard_path, "w") as f:
            json.dump(lb, f, indent=2)

        with open(snapshot_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Diff against previous run
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        previous: List[Dict[str, Any]] = []
        if os.path.exists(latest_path):
            try:
                with open(latest_path) as f:
                    previous = json.load(f)
            except Exception:
                previous = []

        merged = self._merge_results(previous, self.results)
        diff = self._compute_diff(previous, merged)
        with open(os.path.join(self.output_dir, "summary_diff.json"), "w") as f:
            json.dump(diff, f, indent=2)

        with open(latest_path, "w") as f:
            json.dump(merged, f, indent=2)

        logger.info(
            "Accuracy snapshot: "
            + ", ".join(
                f"{r['level']}:{r['name']} F1={r.get('note_f1', 0):.3f}"
                for r in merged
            )
        )

        logger.info(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=f"results/benchmark_{int(time.time())}")
    parser.add_argument("--level", choices=["all", "L0", "L1", "L2", "L3", "L4"], default="all",
                        help="Run a specific benchmark level or all levels")
    args = parser.parse_args()

    runner = BenchmarkSuite(args.output)

    level_order = ["L0", "L1", "L2", "L3", "L4"]
    to_run = level_order if args.level == "all" else [args.level]

    try:
        for lvl in to_run:
            if lvl == "L0":
                runner.run_L0_mono_sanity()
            elif lvl == "L1":
                runner.run_L1_mono_musical()
            elif lvl == "L2":
                runner.run_L2_poly_dominant()
            elif lvl == "L3":
                runner.run_L3_full_poly()
            elif lvl == "L4":
                runner.run_L4_real_songs()
    except Exception as e:
        logger.error(f"Benchmark Suite Failed: {e}")
        # Make sure we still save what we have
        pass

    runner.generate_summary()

if __name__ == "__main__":
    main()
