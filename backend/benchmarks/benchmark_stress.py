import argparse
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
import librosa

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.transcription import transcribe_audio_pipeline
from backend.benchmarks.utils import generate_test_data, load_ground_truth, calculate_metrics

def add_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    p_signal = np.mean(y ** 2)
    if p_signal == 0: return y
    p_noise = p_signal / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(p_noise), len(y))
    return y + noise

def run_stress_test():
    base_dir = os.path.dirname(__file__)
    xml_path = os.path.join(base_dir, "../mock_data/happy_birthday.xml")
    clean_wav_path = os.path.join(base_dir, "../mock_data/happy_birthday.wav")

    generate_test_data(xml_path, clean_wav_path)

    # Load GT
    gt_f0, gt_voiced = load_ground_truth(xml_path, hop_length=256)

    # Load Audio
    sr, y = wavfile.read(clean_wav_path)
    y = y.astype(np.float32) / 32768.0

    snr_levels = [20, 10]

    print("\n--- STEP 2: STRESS TESTING (Robustness) ---")

    for snr in snr_levels:
        print(f"\n[TEST] SNR: {snr}dB")
        y_noisy = add_noise(y, snr)

        # Save temp
        noisy_path = os.path.join(base_dir, f"temp_noise_{snr}db.wav")
        wavfile.write(noisy_path, sr, (y_noisy * 32767).astype(np.int16))

        try:
            # Run Pipeline
            # Use robust params found in calibration (hardcoded for now or use defaults)
            res = transcribe_audio_pipeline(
                noisy_path,
                mode="quality",
                confidence_threshold=0.5, # Assume reasonable default
                min_duration_ms=30.0
            )

            timeline = res.analysis_data.timeline
            pred_f0 = np.array([f.pitch_hz for f in timeline])
            m = calculate_metrics(pred_f0, gt_f0, gt_voiced)

            print(f"[METRICS] RPA: {m['RPA']:.3f} | F1: {m['F1 Score']:.3f}")

            if m['RPA'] > 0.90:
                print("[PASS] RPA > 0.90")
            else:
                print("[FAIL] RPA < 0.90")
                # Suggest WLP or HMM
                print("Recommendation: Enable Spectral Whitening (WLP) or adjust HMM params.")

        finally:
            if os.path.exists(noisy_path):
                os.remove(noisy_path)

if __name__ == "__main__":
    run_stress_test()
