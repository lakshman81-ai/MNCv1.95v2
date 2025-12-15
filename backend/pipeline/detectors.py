# backend/pipeline/detectors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import warnings
import numpy as np
import scipy.signal


# --------------------------------------------------------------------------------------
# Optional dependencies (never fail import of this module)
# --------------------------------------------------------------------------------------
try:
    import librosa  # type: ignore
except Exception as e:  # pragma: no cover
    librosa = None  # type: ignore
    _LIBROSA_IMPORT_ERR = e  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import crepe  # type: ignore
except Exception:  # pragma: no cover
    crepe = None  # type: ignore


# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------
def hz_to_midi(hz: float) -> float:
    if hz <= 0.0:
        return 0.0
    return 69.0 + 12.0 * float(np.log2(hz / 440.0))


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return float(440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0)))


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _frame_audio(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) <= 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    if len(y) < frame_length:
        pad = frame_length - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    n_frames = 1 + (len(y) - frame_length) // hop_length
    if n_frames <= 0:
        n_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, frame_length),
        strides=(y.strides[0] * hop_length, y.strides[0]),
        writeable=False,
    )
    return np.asarray(frames, dtype=np.float32)


def _autocorr_pitch_per_frame(
    frames: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight ACF pitch estimator: returns (f0, conf) per frame.
    conf ~ normalized ACF peak (0..1).
    """
    n_frames, frame_length = frames.shape
    f0 = np.zeros((n_frames,), dtype=np.float32)
    conf = np.zeros((n_frames,), dtype=np.float32)

    if n_frames == 0:
        return f0, conf

    # Lags corresponding to frequency bounds
    lag_min = max(1, int(sr / max(fmax, 1e-6)))
    lag_max = max(lag_min + 1, int(sr / max(fmin, 1e-6)))
    lag_max = min(lag_max, frame_length - 2)

    win = np.hanning(frame_length).astype(np.float32)

    for i in range(n_frames):
        x = frames[i] * win
        x = x - np.mean(x)
        denom = float(np.dot(x, x)) + 1e-12
        if denom <= 1e-10:
            continue

        ac = scipy.signal.correlate(x, x, mode="full", method="auto")
        ac = ac[ac.size // 2 :]  # keep non-negative lags
        ac0 = float(ac[0]) + 1e-12

        seg = ac[lag_min:lag_max]
        if seg.size <= 0:
            continue

        k = int(np.argmax(seg)) + lag_min
        peak = float(ac[k]) / ac0
        peak = float(np.clip(peak, 0.0, 1.0))

        if peak <= 0.0:
            continue

        f0[i] = float(sr / k)
        conf[i] = float(peak)

    return f0, conf


# --------------------------------------------------------------------------------------
# Public polyphonic helpers (imported by tests / used by Stage B)
# --------------------------------------------------------------------------------------
def create_harmonic_mask(
    f0_hz: np.ndarray,
    sr: int,
    n_fft: int,
    mask_width: float = 0.03,
    n_harmonics: int = 8,
    min_band_hz: float = 6.0,
) -> np.ndarray:
    """
    Create a time-frequency mask that zeros bins around harmonics of f0.
    Returns mask shape: (n_fft//2 + 1, n_frames). 1.0 = keep, 0.0 = remove.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float32).reshape(-1)
    n_frames = int(f0_hz.shape[0])
    n_bins = n_fft // 2 + 1

    freqs = np.linspace(0.0, float(sr) / 2.0, n_bins, dtype=np.float32)
    mask = np.ones((n_bins, n_frames), dtype=np.float32)

    for t in range(n_frames):
        f0 = float(f0_hz[t])
        if f0 <= 0.0 or not np.isfinite(f0):
            continue

        for h in range(1, n_harmonics + 1):
            fh = f0 * h
            if fh >= float(sr) / 2.0:
                break

            bw = max(min_band_hz, abs(mask_width) * fh)
            lo = fh - bw
            hi = fh + bw

            idx = np.where((freqs >= lo) & (freqs <= hi))[0]
            if idx.size:
                mask[idx, t] = 0.0

    return mask


def iterative_spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    primary_detector: "BasePitchDetector",
    validator_detector: Optional["BasePitchDetector"] = None,
    max_polyphony: int = 8,
    mask_width: float = 0.03,
    min_mask_width: float = 0.02,
    max_mask_width: float = 0.08,
    mask_growth: float = 1.1,
    mask_shrink: float = 0.9,
    harmonic_snr_stop_db: float = 3.0,
    residual_rms_stop_ratio: float = 0.08,
    residual_flatness_stop: float = 0.45,
    validator_cents_tolerance: float = 50.0,
    validator_agree_window: int = 5,
    validator_disagree_decay: float = 0.6,
    validator_min_agree_frames: int = 2,
    validator_min_disagree_frames: int = 2,
    max_harmonics: int = 12,
    audio_path: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Iterative spectral subtraction ("peeling"):
      1) Detect dominant f0 on residual
      2) Build harmonic mask and suppress those bins
      3) ISTFT back to residual and repeat

    Returns list of (f0, conf) per extracted layer.
    """
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return []

    hop = _safe_int(getattr(primary_detector, "hop_length", 512), 512)
    n_fft = _safe_int(getattr(primary_detector, "n_fft", 2048), 2048)
    bin_hz = float(sr) / float(n_fft)

    layers: List[Tuple[np.ndarray, np.ndarray]] = []
    residual = y.copy()
    base_rms = float(np.sqrt(np.mean(residual**2)) + 1e-9)
    current_mask_width = float(mask_width)

    def _spectral_flatness(magnitude: np.ndarray) -> float:
        mag = np.asarray(magnitude, dtype=np.float32)
        if mag.size == 0:
            return 0.0
        mag = np.maximum(mag, 1e-9)
        log_mean = float(np.mean(np.log(mag)))
        geo = float(np.exp(log_mean))
        arith = float(np.mean(mag)) + 1e-12
        return float(geo / arith)

    def _rolling_mean(arr: np.ndarray, win: int) -> np.ndarray:
        if win <= 1:
            return arr.astype(np.float32)
        kernel = np.ones(win, dtype=np.float32) / float(win)
        return np.convolve(arr.astype(np.float32), kernel, mode="same")

    def _consecutive_lengths(flags: np.ndarray) -> np.ndarray:
        lengths = np.zeros_like(flags, dtype=np.int32)
        run = 0
        for i, v in enumerate(flags):
            run = run + 1 if v else 0
            lengths[i] = run
        return lengths

    for _layer in range(int(max_polyphony)):
        f0, conf = primary_detector.predict(residual, audio_path=audio_path)

        if f0 is None or conf is None:
            break
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)

        # Optional validator gate with temporal smoothing
        if validator_detector is not None:
            try:
                vf0, vconf = validator_detector.predict(residual, audio_path=audio_path)
                vf0 = np.asarray(vf0, dtype=np.float32).reshape(-1)
                vconf = np.asarray(vconf, dtype=np.float32).reshape(-1)

                # If validator mostly unvoiced, stop early
                if np.mean((vf0 > 0.0).astype(np.float32)) < 0.05:
                    break

                with np.errstate(divide="ignore", invalid="ignore"):
                    cents = 1200.0 * np.log2((f0 + 1e-9) / (vf0 + 1e-9))
                agree_raw = (np.abs(cents) <= float(validator_cents_tolerance)) & (f0 > 0.0) & (vf0 > 0.0)

                agree_smooth = _rolling_mean(agree_raw.astype(np.float32), int(max(1, validator_agree_window)))
                agree_runs = _consecutive_lengths(agree_raw)
                disagree_runs = _consecutive_lengths(~agree_raw & (f0 > 0.0) & (vf0 > 0.0))

                stable_agree = agree_runs >= int(max(1, validator_min_agree_frames))
                stable_disagree = disagree_runs >= int(max(1, validator_min_disagree_frames))

                gate = agree_smooth + (1.0 - agree_smooth) * float(validator_disagree_decay)
                gate = np.clip(gate, 0.0, 1.0)

                # Require consensus before fully accepting/rejecting
                gate = np.where(stable_disagree, gate * float(validator_disagree_decay), gate)
                gate = np.where(~stable_agree & ~stable_disagree, gate * 0.8, gate)

                conf = conf * gate.astype(np.float32)
            except Exception:
                # Validator should never crash peeling
                pass

        voiced_ratio = float(np.mean((conf > 0.1).astype(np.float32)))
        if voiced_ratio < 0.05:
            break

        # STFT -> apply harmonic mask -> iSTFT
        try:
            f, t, Z = scipy.signal.stft(
                residual,
                fs=sr,
                nperseg=n_fft,
                noverlap=max(0, n_fft - hop),
                boundary=None,
                padded=False,
            )
            # Ensure frame alignment (f0 length must match STFT time frames)
            n_frames = Z.shape[1]
            if f0.shape[0] != n_frames:
                # pad/trim f0/conf to n_frames
                if f0.shape[0] < n_frames:
                    pad = n_frames - f0.shape[0]
                    f0 = np.pad(f0, (0, pad))
                    conf = np.pad(conf, (0, pad))
                else:
                    f0 = f0[:n_frames]
                    conf = conf[:n_frames]

            # Adapt harmonics based on current F0 distribution and FFT
            f0_valid = f0[f0 > 0.0]
            median_f0 = float(np.median(f0_valid)) if f0_valid.size else 0.0
            max_possible_h = int(float(sr) / 2.0 / max(median_f0, 1e-6)) if median_f0 > 0 else 1
            adaptive_harmonics = int(max(1, min(int(max_harmonics), max_possible_h)))

            # adapt mask width to FFT bin spacing and current mask schedule
            effective_mask_width = float(np.clip(current_mask_width, min_mask_width, max_mask_width))
            min_band = max(bin_hz, float(effective_mask_width) * max(median_f0, 1.0) * 0.5)

            mask = create_harmonic_mask(
                f0_hz=f0,
                sr=sr,
                n_fft=n_fft,
                mask_width=effective_mask_width,
                n_harmonics=adaptive_harmonics,
                min_band_hz=min_band,
            )

            magnitude = np.abs(Z).astype(np.float32)
            harmonic_energy = float(np.mean(magnitude * (1.0 - mask)))
            residual_energy = float(np.mean(magnitude * mask))
            harmonic_snr = 10.0 * np.log10((harmonic_energy + 1e-9) / (residual_energy + 1e-9))

            if harmonic_snr < float(harmonic_snr_stop_db):
                break

            layers.append((f0, conf))

            # Apply mask softly based on conf (avoid over-subtraction)
            strength = np.clip(conf, 0.0, 1.0).reshape(1, -1)
            soft_mask = 1.0 - (1.0 - mask) * strength
            Z2 = Z * soft_mask

            _, residual2 = scipy.signal.istft(
                Z2,
                fs=sr,
                nperseg=n_fft,
                noverlap=max(0, n_fft - hop),
                input_onesided=True,
                boundary=None,
            )
            residual = np.asarray(residual2, dtype=np.float32).reshape(-1)
            if residual.size < y.size:
                residual = np.pad(residual, (0, y.size - residual.size))
            residual = residual[: y.size]

            residual_rms = float(np.sqrt(np.mean(residual**2)) + 1e-9)
            if residual_rms / base_rms < float(residual_rms_stop_ratio):
                break

            flatness = _spectral_flatness(magnitude)
            if flatness > float(residual_flatness_stop):
                break

            # Adapt mask width for the next iteration (widen when SNR drops)
            if harmonic_snr < float(harmonic_snr_stop_db) + 3.0:
                current_mask_width = min(float(max_mask_width), float(current_mask_width) * float(mask_growth))
            else:
                current_mask_width = max(float(min_mask_width), float(current_mask_width) * float(mask_shrink))
        except Exception:
            # If STFT fails, stop peeling (but keep what we extracted)
            break

    return layers


# --------------------------------------------------------------------------------------
# Detector base + implementations
# --------------------------------------------------------------------------------------
@dataclass
class DetectorOutput:
    f0_hz: np.ndarray
    confidence: np.ndarray


class BasePitchDetector:
    """
    Base class used by Stage B.
    Must implement: predict(audio, audio_path=None) -> (f0, conf)
    """

    def __init__(
        self,
        sr: int,
        hop_length: int,
        n_fft: int = 2048,
        fmin: float = 50.0,
        fmax: float = 1200.0,
        threshold: float = 0.10,
        **kwargs: Any,  # absorb unknown config keys safely
    ):
        self.sr = int(sr)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.threshold = float(threshold)
        self._warned: Dict[str, bool] = {}

    def _warn_once(self, key: str, msg: str) -> None:
        if not self._warned.get(key, False):
            warnings.warn(msg)
            self._warned[key] = True

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class YinDetector(BasePitchDetector):
    """
    Prefers librosa.pyin when available; otherwise falls back to lightweight ACF tracker.
    """

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        # librosa path
        if librosa is not None:
            try:
                f0, voiced_flag, voiced_prob = librosa.pyin(
                    y=y,
                    fmin=float(self.fmin),
                    fmax=float(self.fmax),
                    sr=int(self.sr),
                    frame_length=int(self.n_fft),
                    hop_length=int(self.hop_length),
                    fill_na=0.0,
                )
                f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
                voiced_prob = np.asarray(voiced_prob, dtype=np.float32).reshape(-1)
                f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
                conf = np.where(f0 > 0.0, voiced_prob, 0.0).astype(np.float32)
                conf = np.clip(conf, 0.0, 1.0)
                conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
                f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
                return f0, conf
            except Exception:
                # fall back below
                pass

        # fallback: ACF per frame
        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
        f0, conf = _autocorr_pitch_per_frame(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax)
        conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
        f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
        return f0, conf


class SACFDetector(BasePitchDetector):
    """
    Summary autocorrelation style: currently returns a dominant f0 + confidence using ACF.
    (Designed to be stable without extra dependencies.)
    """

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
        f0, conf = _autocorr_pitch_per_frame(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax)

        # SACF tends to be noisier; apply threshold a bit more strictly
        thr = max(self.threshold, 0.12)
        conf = np.where(conf >= thr, conf, 0.0).astype(np.float32)
        f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
        return f0, conf

    def validate_curve(self, curve_hz: np.ndarray, audio: np.ndarray) -> float:
        """Lightweight curve validator used in tests.

        Computes the agreement between a candidate F0 curve and the SACF
        estimate for the provided audio. The score is the proportion of frames
        whose deviation is within 50 cents, weighted by the SACF confidence.
        """

        cand = np.asarray(curve_hz, dtype=np.float32).reshape(-1)
        est_f0, est_conf = self.predict(audio)

        n = min(len(cand), len(est_f0))
        if n == 0:
            return 0.0

        cand = cand[:n]
        est_f0 = est_f0[:n]
        est_conf = est_conf[:n]

        cents_diff = 1200.0 * np.log2(np.divide(cand, est_f0 + 1e-9))
        agree = np.abs(cents_diff) <= 50.0
        if np.sum(est_conf) <= 0.0:
            return float(np.mean(agree))

        score = float(np.sum(agree * est_conf) / np.sum(est_conf))
        return score


class CQTDetector(BasePitchDetector):
    """
    CQT-based dominant pitch (requires librosa). If librosa missing, returns zeros with warning.
    """

    def __init__(self, sr: int, hop_length: int, n_fft: int = 2048, **kwargs: Any):
        super().__init__(sr=sr, hop_length=hop_length, n_fft=n_fft, **kwargs)
        self.bins_per_octave = _safe_int(kwargs.get("bins_per_octave", 36), 36)
        self.n_bins = _safe_int(kwargs.get("n_bins", 7 * self.bins_per_octave), 7 * self.bins_per_octave)

    def predict(
        self,
        audio: np.ndarray,
        audio_path: Optional[str] = None,
        polyphony: bool = False,
        top_k: int = 3,
    ) -> Tuple[Any, Any]:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return ([[]] if polyphony else empty), ([[]] if polyphony else empty)

        if librosa is None:
            self._warn_once("no_librosa", "CQTDetector disabled: librosa not available.")
            # match expected frame count approximately
            frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
            n = frames.shape[0]
            return np.zeros((n,), dtype=np.float32), np.zeros((n,), dtype=np.float32)

        try:
            C = librosa.cqt(
                y=y,
                sr=int(self.sr),
                hop_length=int(self.hop_length),
                fmin=float(self.fmin),
                n_bins=int(self.n_bins),
                bins_per_octave=int(self.bins_per_octave),
            )
            M = np.abs(C).astype(np.float32)  # (bins, frames)
            if M.size == 0:
                empty = np.zeros((0,), dtype=np.float32)
                return ([[]] if polyphony else empty), ([[]] if polyphony else empty)

            freqs = librosa.cqt_frequencies(
                n_bins=int(self.n_bins),
                fmin=float(self.fmin),
                bins_per_octave=int(self.bins_per_octave),
            ).astype(np.float32)

            # dominant bin per frame
            idx = np.argmax(M, axis=0)
            f0 = freqs[idx].astype(np.float32)

            # confidence from peak-to-mean ratio
            peak = M[idx, np.arange(M.shape[1])]
            mean = np.mean(M, axis=0) + 1e-9
            conf = (peak / mean).astype(np.float32)
            conf = np.clip((conf - 1.0) / 4.0, 0.0, 1.0)  # squash
            conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
            f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)

            if not polyphony:
                return f0, conf

            # Polyphonic mode: pick the top-k peaks per frame.
            top_k = max(1, int(top_k))
            pitches_list: List[List[float]] = []
            confs_list: List[List[float]] = []
            for t in range(M.shape[1]):
                mag = M[:, t]
                if mag.size == 0:
                    pitches_list.append([])
                    confs_list.append([])
                    continue

                top_idx = np.argsort(mag)[-top_k:][::-1]
                top_freqs = freqs[top_idx]
                top_conf = mag[top_idx] / (np.mean(mag) + 1e-9)
                top_conf = np.clip((top_conf - 1.0) / 4.0, 0.0, 1.0)

                pitches_list.append(top_freqs.tolist())
                confs_list.append(top_conf.tolist())

            return pitches_list, confs_list
        except Exception:
            # fallback to ACF
            frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
            f0, conf = _autocorr_pitch_per_frame(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax)
            conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
            f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
            return ([f0.tolist()] if polyphony else f0), ([conf.tolist()] if polyphony else conf)


class SwiftF0Detector(BasePitchDetector):
    """
    Placeholder wrapper: requires torch model in your project.
    If torch not available, returns zeros (and Stage B will warn).
    """

    def __init__(self, sr: int, hop_length: int, n_fft: int = 2048, **kwargs: Any):
        super().__init__(sr=sr, hop_length=hop_length, n_fft=n_fft, **kwargs)
        self.enabled = torch is not None

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
        n = frames.shape[0]

        if torch is None:
            self._warn_once("no_torch", "SwiftF0 disabled: torch not available. Falling back to ACF.")
            # Fall through to ACF fallback

        # If you later add real SwiftF0 inference, replace this block.
        # For now: stable fallback to ACF so pipeline still works deterministically.
        f0, conf = _autocorr_pitch_per_frame(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax)
        conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
        f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
        return f0, conf


class RMVPEDetector(BasePitchDetector):
    """
    Placeholder wrapper for RMVPE. If torch not available, returns zeros.
    """

    def __init__(self, sr: int, hop_length: int, n_fft: int = 2048, **kwargs: Any):
        super().__init__(sr=sr, hop_length=hop_length, n_fft=n_fft, **kwargs)
        self.enabled = torch is not None

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
        n = frames.shape[0]

        if torch is None:
            self._warn_once("no_torch", "RMVPE disabled: torch not available.")
            return np.zeros((n,), dtype=np.float32), np.zeros((n,), dtype=np.float32)

        # Replace with actual RMVPE inference later.
        f0, conf = _autocorr_pitch_per_frame(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax)
        conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
        f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
        return f0, conf


class CREPEDetector(BasePitchDetector):
    """
    CREPE wrapper. If crepe missing, returns zeros.
    Note: CREPE expects sr=16000 typically; you can resample in Stage B if needed.
    """

    def __init__(self, sr: int, hop_length: int, n_fft: int = 2048, **kwargs: Any):
        super().__init__(sr=sr, hop_length=hop_length, n_fft=n_fft, **kwargs)
        self.model_capacity = str(kwargs.get("model_capacity", "full"))

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
        n = frames.shape[0]

        if crepe is None:
            self._warn_once("no_crepe", "CREPE disabled: crepe not available.")
            return np.zeros((n,), dtype=np.float32), np.zeros((n,), dtype=np.float32)

        # Minimal safe stub: fall back to ACF unless you explicitly wire CREPE
        f0, conf = _autocorr_pitch_per_frame(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax)
        conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
        f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
        return f0, conf

# Alias for backwards compatibility with older code/tests.  Some
# unit tests import CQTPeaksDetector from detectors.py, expecting it
# to behave identically to CQTDetector.  Provide an alias so that
# ``from detectors import CQTPeaksDetector`` works without modifying
# those tests.
CQTPeaksDetector = CQTDetector  # type: ignore
