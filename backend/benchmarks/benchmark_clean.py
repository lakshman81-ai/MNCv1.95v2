import argparse
import os
import sys
import numpy as np

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.transcription import transcribe_audio_pipeline
from backend.benchmarks.utils import generate_test_data, load_ground_truth, calculate_metrics

def run_benchmark(mode: str = "calibration"):
    # Setup Paths
    base_dir = os.path.dirname(__file__)
    xml_path = os.path.join(base_dir, "../mock_data/happy_birthday.xml")
    wav_path = os.path.join(base_dir, "../mock_data/happy_birthday.wav")

    # 1. Generate Data
    generate_test_data(xml_path, wav_path)

    # 2. Load Ground Truth
    print(f"[INFO] Loading Dataset: {wav_path}")
    # Pipeline uses hop_length=256 (Stage A)
    gt_f0, gt_voiced = load_ground_truth(xml_path, hop_length=256)
    print(f"[INFO] Ground Truth loaded. Voiced Frames: {np.sum(gt_voiced)} | Silence Frames: {len(gt_voiced) - np.sum(gt_voiced)}")

    # 3. Initial Benchmark
    print("\n--- STEP 1: INITIAL BENCHMARK (Default Params) ---")
    defaults = {"confidence_threshold": 0.4, "min_duration_ms": 10.0}
    print(f"[CONFIG] Conf_Thresh: {defaults['confidence_threshold']} | Min_Dur: {defaults['min_duration_ms']}ms")

    res = transcribe_audio_pipeline(
        wav_path,
        mode="quality",
        confidence_threshold=defaults["confidence_threshold"],
        min_duration_ms=defaults["min_duration_ms"]
    )

    # Extract predicted f0
    timeline = res.analysis_data.timeline
    pred_f0 = np.array([f.pitch_hz for f in timeline])

    metrics = calculate_metrics(pred_f0, gt_f0, gt_voiced)

    print("[METRICS]")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    if mode != "calibration":
        return

    # 4. Calibration Loop
    target_precision = 0.99
    target_recall = 0.98

    if metrics["Precision"] >= target_precision and metrics["Recall"] >= target_recall:
        print("[SUCCESS] Baseline meets criteria.")
        return

    print("\n--- STEP 2: AUTOMATED TUNING LOOP ---")

    best_candidate = None
    best_score = 0.0

    # Tune Confidence
    print("[TRIAL 1] Tuning Confidence Threshold...")
    thresholds = [0.45, 0.50, 0.55, 0.60]

    candidate_thresh = defaults["confidence_threshold"]

    for th in thresholds:
        res = transcribe_audio_pipeline(
            wav_path,
            mode="quality",
            confidence_threshold=th,
            min_duration_ms=defaults["min_duration_ms"]
        )
        timeline = res.analysis_data.timeline
        pred_f0 = np.array([f.pitch_hz for f in timeline])
        m = calculate_metrics(pred_f0, gt_f0, gt_voiced)

        print(f"Thresh {th:.2f} -> Precision: {m['Precision']:.3f} | Recall: {m['Recall']:.3f}")

        # Simple selection: max precision while recall > 0.98
        if m['Precision'] > 0.99 and m['Recall'] > 0.98:
             print(" <-- CANDIDATE FOUND")
             candidate_thresh = th
             break
        elif m['Precision'] > best_score: # Keep best precision if none pass
             best_score = m['Precision']
             candidate_thresh = th

    # Tune Duration
    print(f"\n[TRIAL 2] Applying Duration Filter to Candidate ({candidate_thresh})...")
    min_durs = [30.0, 50.0]

    final_dur = defaults["min_duration_ms"]

    for dur in min_durs:
        res = transcribe_audio_pipeline(
            wav_path,
            mode="quality",
            confidence_threshold=candidate_thresh,
            min_duration_ms=dur
        )
        timeline = res.analysis_data.timeline
        pred_f0 = np.array([f.pitch_hz for f in timeline])
        m = calculate_metrics(pred_f0, gt_f0, gt_voiced)

        print(f"Dur {dur}ms -> Precision: {m['Precision']:.3f} | Recall: {m['Recall']:.3f}")

        if m['Precision'] >= target_precision and m['Recall'] >= target_recall:
             final_dur = dur
             break

    print("\n--- FINAL RESULTS: OPTIMIZED CONFIGURATION ---")
    print(f"[CONFIG] Conf_Thresh: {candidate_thresh} | Min_Dur: {final_dur}ms")

    # Final Run
    res = transcribe_audio_pipeline(
        wav_path,
        mode="quality",
        confidence_threshold=candidate_thresh,
        min_duration_ms=final_dur
    )
    timeline = res.analysis_data.timeline
    pred_f0 = np.array([f.pitch_hz for f in timeline])
    m = calculate_metrics(pred_f0, gt_f0, gt_voiced)

    print("[METRICS]")
    for k, v in m.items():
        passed = "PASSED" if (k == "Precision" and v >= target_precision) or (k == "Recall" and v >= target_recall) else ""
        print(f"{k}: {v:.3f} {passed}")

    if m['Precision'] >= target_precision and m['Recall'] >= target_recall:
        print("\n[SUCCESS] Baseline Calibration Complete.")
    else:
        print("\n[WARNING] Optimization criteria not fully met.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="calibration", choices=["calibration", "run"])
    args = parser.parse_args()

    run_benchmark(args.mode)
