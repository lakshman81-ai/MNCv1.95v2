import numpy as np
import librosa
import pyloudnorm as pyln
from typing import Dict, Any

def calculate_stage_a_metrics(
    processed_audio: np.ndarray,
    sr: int,
    meta: Any,
    target_lufs: float = -23.0
) -> Dict[str, float]:
    """
    Calculates Stage A (Signal Conditioning) metrics.
    """
    metrics = {}

    # 1. HPF Leakage Check (Heuristic)
    # We can't strictly check 'leakage' without a specific test signal (30Hz + 65Hz).
    # But we can measure energy below cutoff (55Hz) vs above.
    # For a general benchmark on music, this is less useful unless we use the specific test tone.
    # The requirement said "Generate test tone... Check amplitude".
    # But here we are running on the benchmark examples (Happy Birthday).
    # So we should probably skip specific HPF frequency checks on music data
    # OR implement a separate 'calibration' pass.
    # For this function, let's measure general LF energy ratio.

    # Energy < 50Hz vs Total Energy
    S = np.abs(librosa.stft(processed_audio))
    freqs = librosa.fft_frequencies(sr=sr)
    lf_idx = np.where(freqs < 50)[0]
    lf_energy = np.sum(S[lf_idx, :])
    total_energy = np.sum(S)
    metrics["lf_energy_ratio"] = float(lf_energy / (total_energy + 1e-9))

    # 2. LUFS Error
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(processed_audio)
        metrics["lufs_measured"] = loudness
        metrics["lufs_error"] = abs(loudness - target_lufs)
    except:
        metrics["lufs_measured"] = -99.0
        metrics["lufs_error"] = 99.0

    # 3. Noise Floor
    # Estimated from silent parts? Or just take the 10th percentile of RMS.
    rms = librosa.feature.rms(y=processed_audio)[0]
    noise_floor = np.percentile(rms, 10)
    metrics["noise_floor_db"] = 20 * np.log10(noise_floor + 1e-9)

    return metrics
