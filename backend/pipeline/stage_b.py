"""
Stage B — Feature Extraction

This module implements pitch detection and feature extraction.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import logging
import importlib.util
import os
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None

from .models import StageAOutput, FramePitch, AnalysisData, AudioType, StageBOutput, Stem
from .config import PipelineConfig
from .detectors import (
    SwiftF0Detector, SACFDetector, YinDetector,
    CQTDetector, RMVPEDetector, CREPEDetector,
    iterative_spectral_subtraction, create_harmonic_mask,
    _frame_audio,
    BasePitchDetector
)

# Re-export for tests
__all__ = [
    "extract_features",
    "create_harmonic_mask",
    "iterative_spectral_subtraction",
    "MultiVoiceTracker",
]

SCIPY_SIGNAL = None
if importlib.util.find_spec("scipy.signal"):
    import scipy.signal as SCIPY_SIGNAL


LOGGER = logging.getLogger(__name__)


def _module_available(module_name: str) -> bool:
    """Helper to avoid importing heavy optional deps when missing."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _butter_filter(audio: np.ndarray, sr: int, cutoff: float, btype: str) -> np.ndarray:
    """Lightweight wrapper for simple Butterworth filtering."""
    if SCIPY_SIGNAL is None or len(audio) == 0:
        return audio.copy()

    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    norm_cutoff = min(max(norm_cutoff, 1e-4), 0.999)
    sos = SCIPY_SIGNAL.butter(4, norm_cutoff, btype=btype, output="sos")
    return SCIPY_SIGNAL.sosfiltfilt(sos, audio)


class SyntheticMDXSeparator:
    """
    Lightweight separator tuned on procedurally generated sine/saw/square/FM stems.

    The separator derives template spectral envelopes from analytic waveforms, then
    infers soft weights that are used to project the incoming mix into vocal/bass/
    drums/other stems. This intentionally mirrors a tiny MDX head without external
    weights so it can run in constrained environments.
    """

    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.templates = self._build_templates()

    def _build_templates(self) -> Dict[str, np.ndarray]:
        sr = self.sample_rate
        duration = 0.25
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        base_freqs = [110.0, 220.0, 440.0, 660.0]

        def _normalize_spec(y: np.ndarray) -> np.ndarray:
            window = np.hanning(len(y))
            spec = np.abs(np.fft.rfft(y * window))
            spec = spec / (np.linalg.norm(spec) + 1e-9)
            return spec

        templates: Dict[str, np.ndarray] = {}
        waves = {
            "sine_stack": sum(np.sin(2 * np.pi * f * t) for f in base_freqs),
            "saw": sum(1.0 / (i + 1) * np.sin(2 * np.pi * (i + 1) * base_freqs[1] * t) for i in range(6)),
            "square": sum(
                1.0 / (2 * i + 1) * np.sin(2 * np.pi * (2 * i + 1) * base_freqs[0] * t)
                for i in range(6)
            ),
        }

        # Simple FM voice to emulate vocal richness
        carrier = 220.0
        modulator = 110.0
        waves["fm_voice"] = np.sin(
            2 * np.pi * carrier * t + 5.0 * np.sin(2 * np.pi * modulator * t)
        )

        for name, wave in waves.items():
            templates[name] = _normalize_spec(wave)

        # Broadband template for drums/transients
        broadband = np.hanning(len(t))
        templates["broadband"] = _normalize_spec(broadband)
        return templates

    def _score_mix(self, audio: np.ndarray) -> Dict[str, float]:
        window = np.hanning(len(audio))
        spec = np.abs(np.fft.rfft(audio * window))
        spec = spec / (np.linalg.norm(spec) + 1e-9)

        scores = {}
        for name, tmpl in self.templates.items():
            # cosine similarity
            score = float(np.dot(spec, tmpl))
            scores[name] = score
        return scores

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        if len(audio) == 0:
            return {}

        scores = self._score_mix(audio)
        vocal_score = scores.get("fm_voice", 0.25) + scores.get("sine_stack", 0.25)
        bass_score = scores.get("square", 0.25)
        saw_score = scores.get("saw", 0.25)
        drum_score = scores.get("broadband", 0.25)

        raw_weights = np.array([vocal_score, bass_score, drum_score, saw_score])
        weights = raw_weights / (np.sum(raw_weights) + 1e-9)
        vocals_w, bass_w, drums_w, other_w = weights

        vocals = vocals_w * _butter_filter(audio, sr, 12000.0, "low")
        vocals = _butter_filter(vocals, sr, 120.0, "high")

        bass = bass_w * _butter_filter(audio, sr, 180.0, "low")
        drums = drums_w * _butter_filter(audio, sr, 90.0, "high")
        other = audio - (vocals + bass + drums)
        other = other_w * other

        return {
            "vocals": vocals,
            "bass": bass,
            "drums": drums,
            "other": other,
        }


def _run_htdemucs(audio: np.ndarray, sr: int, model_name: str, overlap: float, shifts: int) -> Optional[Dict[str, Any]]:
    if (
        not _module_available("demucs.pretrained")
        or not _module_available("demucs.apply")
        or not _module_available("torch")
    ):
        warnings.warn("Demucs not available; skipping neural separation.")
        return None

    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torch

    try:
        model = get_model(model_name)
    except Exception as exc:
        warnings.warn(f"HTDemucs unavailable ({exc}); skipping neural separation.")
        return None

    model_sr = getattr(model, "samplerate", sr)

    if model_sr != sr:
        # simple linear resample
        ratio = float(model_sr) / float(sr)
        indices = np.arange(0, len(audio) * ratio) / ratio
        resampled = np.interp(indices, np.arange(len(audio)), audio)
    else:
        resampled = audio

    mix_tensor = torch.tensor(resampled, dtype=torch.float32)[None, None, :]
    try:
        with torch.no_grad():
            demucs_out = apply_model(model, mix_tensor, overlap=overlap, shifts=shifts)
    except Exception as exc:
        warnings.warn(f"HTDemucs inference failed ({exc}); skipping neural separation.")
        return None

    sources = getattr(model, "sources", ["vocals", "drums", "bass", "other"])
    separated = {}
    for idx, name in enumerate(sources):
        stem_audio = demucs_out[0, idx].mean(dim=0).cpu().numpy()
        separated[name] = stem_audio

    # Ensure canonical stems exist
    for name in ["vocals", "drums", "bass", "other"]:
        separated.setdefault(name, np.zeros_like(audio))

    return separated


def _compute_loudness(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """Return (rms, lufs_estimate). Uses pyloudnorm if available."""
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    lufs = -80.0
    try:
        if _module_available("pyloudnorm"):
            import pyloudnorm

            meter = pyloudnorm.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            if not np.isinf(loudness):
                lufs = float(loudness)
        elif rms > 0.0:
            lufs = 20.0 * np.log10(rms + 1e-12)
    except Exception:
        if rms > 0.0:
            lufs = 20.0 * np.log10(rms + 1e-12)
    return rms, lufs


def _normalize_stems(
    stems: Dict[str, Stem],
    sr: int,
    stem_cfg: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Dict[str, Stem]:
    """Normalize stems to a shared loudness target for F0 stability."""

    if not stem_cfg.get("enabled", False):
        diagnostics["stem_normalization"] = {"enabled": False}
        return stems

    target_lufs = float(stem_cfg.get("target_lufs", -18.0))
    epsilon = float(stem_cfg.get("epsilon_rms", 1e-6))
    match_mix = bool(stem_cfg.get("match_mix_lufs", True))

    normalized: Dict[str, Stem] = {}
    gains_db: Dict[str, float] = {}
    loudness_snapshots: Dict[str, Dict[str, float]] = {}

    mix_ref = stems.get("mix")
    _, mix_lufs = _compute_loudness(mix_ref.audio, sr) if mix_ref else (0.0, target_lufs)
    target = mix_lufs if match_mix and not np.isinf(mix_lufs) else target_lufs

    for name, stem in stems.items():
        if name == "mix":
            normalized[name] = stem
            continue

        rms, lufs = _compute_loudness(stem.audio, sr)
        if rms <= epsilon:
            normalized[name] = stem
            gains_db[name] = 0.0
            loudness_snapshots[name] = {"rms": rms, "lufs": lufs}
            continue

        desired_gain_db = target - lufs
        gain_lin = 10.0 ** (desired_gain_db / 20.0)
        normalized_audio = stem.audio * gain_lin
        normalized[name] = type(stem)(audio=normalized_audio, sr=stem.sr, type=stem.type)
        gains_db[name] = desired_gain_db
        loudness_snapshots[name] = {"rms": rms, "lufs": lufs}

    diagnostics["stem_normalization"] = {
        "enabled": True,
        "target_lufs": target,
        "gains_db": gains_db,
        "source_loudness": loudness_snapshots,
    }
    return normalized | {"mix": stems.get("mix")} if stems.get("mix") else normalized


def _resolve_separation(stage_a_out: StageAOutput, b_conf) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Resolve separation strategy and track which path actually executed.
    Returns (stems, diagnostics).
    """
    diag = {
        "requested": bool(b_conf.separation.get("enabled", True)),
        "synthetic_requested": bool(b_conf.separation.get("synthetic_model", False)),
        "mode": "disabled",
        "synthetic_ran": False,
        "htdemucs_ran": False,
        "fallback": False,
    }

    if not diag["requested"]:
        return stage_a_out.stems, diag

    if len(stage_a_out.stems) > 1 and any(k != "mix" for k in stage_a_out.stems.keys()):
        diag["mode"] = "preseparated"
        return stage_a_out.stems, diag

    mix_stem = stage_a_out.stems.get("mix")
    if mix_stem is None:
        return stage_a_out.stems, diag

    sep_conf = b_conf.separation
    overlap = sep_conf.get("overlap", 0.25)
    shifts = sep_conf.get("shifts", 1)
    model_name = sep_conf.get("model", "htdemucs")
    checkpoint_path = sep_conf.get("checkpoint_path")
    mdx23_arch = sep_conf.get("mdx23_arch")
    stem_norm_cfg = sep_conf.get("stem_normalization", {})

    if diag["synthetic_requested"]:
        synthetic = SyntheticMDXSeparator(sample_rate=mix_stem.sr, hop_length=stage_a_out.meta.hop_length)
        try:
            synthetic_stems = synthetic.separate(mix_stem.audio, mix_stem.sr)
            if synthetic_stems:
                diag.update({"mode": "synthetic_mdx", "synthetic_ran": True})
                return {
                    name: type(mix_stem)(audio=audio, sr=mix_stem.sr, type=name)
                    for name, audio in synthetic_stems.items()
                } | {"mix": mix_stem}, diag
        except Exception as exc:
            warnings.warn(f"Synthetic separator failed; falling back to {sep_conf.get('model', 'htdemucs')}: {exc}")
            diag["fallback"] = True

    separated = None

    # Try MDX23C (if bundled) before the fallback Demucs checkpoints
    if mdx23_arch:
        separated = _run_htdemucs(mix_stem.audio, mix_stem.sr, mdx23_arch, overlap, shifts)
        if separated:
            diag.update({"mode": mdx23_arch, "htdemucs_ran": True})

    if separated is None:
        separated = _run_htdemucs(
            mix_stem.audio,
            mix_stem.sr,
            model_name,
            overlap,
            shifts,
        )

    # Optional checkpoint override for Demucs/MDX architectures
    if separated is None and checkpoint_path:
        try:
            if _module_available("torch") and _module_available("demucs.pretrained") and os.path.exists(checkpoint_path):
                import torch
                from demucs.pretrained import get_model
                from demucs.apply import apply_model

                model = get_model(model_name)
                state = torch.load(checkpoint_path, map_location="cpu")
                state_dict = state.get("state_dict", state)
                model.load_state_dict(state_dict, strict=False)

                resampled = mix_stem.audio
                if getattr(model, "samplerate", mix_stem.sr) != mix_stem.sr:
                    ratio = float(getattr(model, "samplerate")) / float(mix_stem.sr)
                    indices = np.arange(0, len(mix_stem.audio) * ratio) / ratio
                    resampled = np.interp(indices, np.arange(len(mix_stem.audio)), mix_stem.audio)

                mix_tensor = torch.tensor(resampled, dtype=torch.float32)[None, None, :]
                with torch.no_grad():
                    demucs_out = apply_model(model, mix_tensor, overlap=overlap, shifts=shifts)
                sources = getattr(model, "sources", ["vocals", "drums", "bass", "other"])
                separated = {
                    name: demucs_out[0, idx].mean(dim=0).cpu().numpy()
                    for idx, name in enumerate(sources)
                }
                diag.update({"mode": checkpoint_path, "htdemucs_ran": True})
        except Exception as exc:  # pragma: no cover - optional dependency path
            warnings.warn(f"Checkpoint override failed ({exc}); skipping neural separation.")

    if separated:
        normalized = _normalize_stems({name: type(mix_stem)(audio=audio, sr=mix_stem.sr, type=name) for name, audio in separated.items()} | {"mix": mix_stem}, mix_stem.sr, stem_norm_cfg, diag)
        diag.update({"mode": diag.get("mode", model_name), "htdemucs_ran": True})
        return normalized, diag

    diag["mode"] = "passthrough"
    return stage_a_out.stems, diag

def _arrays_to_timeline(
    f0: np.ndarray,
    conf: np.ndarray,
    rms: Optional[np.ndarray],
    sr: int,
    hop_length: int
) -> List[FramePitch]:
    """Convert f0/conf arrays to List[FramePitch]."""
    timeline = []
    n_frames = len(f0)
    for i in range(n_frames):
        hz = float(f0[i])
        c = float(conf[i])
        r = float(rms[i]) if rms is not None and i < len(rms) else 0.0

        midi = 0.0
        if hz > 0:
            midi = 69.0 + 12.0 * np.log2(hz / 440.0)

        time_sec = float(i * hop_length) / float(sr)

        timeline.append(FramePitch(
            time=time_sec,
            pitch_hz=hz,
            confidence=c,
            midi=round(midi) if hz > 0 else None,
            rms=r,
            active_pitches=[]
        ))
    return timeline

def _init_detector(name: str, conf: Dict[str, Any], sr: int, hop_length: int) -> Optional[BasePitchDetector]:
    """Initialize a detector if enabled."""
    if not conf.get("enabled", False):
        return None

    # Remove control/meta keys we already pass positionally
    kwargs = {k: v for k, v in conf.items() if k not in ("enabled", "hop_length")}

    try:
        if name == "swiftf0":
            return SwiftF0Detector(sr, hop_length, **kwargs)
        elif name == "sacf":
            return SACFDetector(sr, hop_length, **kwargs)
        elif name == "yin":
            return YinDetector(sr, hop_length, **kwargs)
        elif name == "cqt":
            return CQTDetector(sr, hop_length, **kwargs)
        elif name == "rmvpe":
            return RMVPEDetector(sr, hop_length, **kwargs)
        elif name == "crepe":
            return CREPEDetector(sr, hop_length, **kwargs)
        elif name == "fcpe":
            from .detectors import FCPEDetector

            return FCPEDetector(sr, hop_length, **kwargs)
        elif name == "supertone":
            from .detectors import SupertoneDetector

            return SupertoneDetector(sr, hop_length, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to init detector {name}: {e}")
        return None
    return None

def _ensemble_merge(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    weights: Dict[str, float],
    disagreement_cents: float = 70.0,
    priority_floor: float = 0.0,
    harmonicity: Optional[Dict[str, np.ndarray]] = None,
    harmonicity_threshold: float = 0.0,
    stability_hint: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple f0/conf tracks based on weights and disagreement.

    Strategy:
      * Align frame counts across detectors
      * Choose the candidate with the highest weighted confidence
      * Down-weight winners that have little consensus (large disagreement)
    """
    if not results:
        return np.array([]), np.array([])

    lengths = [len(r[0]) for r in results.values()]
    if not lengths:
        return np.array([]), np.array([])
    max_len = max(lengths)

    aligned_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, (f0, conf) in results.items():
        if len(f0) < max_len:
            pad = max_len - len(f0)
            f0 = np.pad(f0, (0, pad))
            conf = np.pad(conf, (0, pad))
        aligned_results[name] = (f0, conf)

    final_f0 = np.zeros(max_len, dtype=np.float32)
    final_conf = np.zeros(max_len, dtype=np.float32)

    def _cent_diff(a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return float("inf")
        return float(1200.0 * np.log2((a + 1e-9) / (b + 1e-9)))

    for i in range(max_len):
        candidates = []
        for name, (f0, conf) in aligned_results.items():
            w = weights.get(name, 1.0)
            c = float(conf[i])
            f = float(f0[i])
            if c <= 0.0 or f <= 0.0:
                continue

            # Priority floor mostly benefits SwiftF0 on synthetic tones
            eff_conf = max(c, priority_floor if name == "swiftf0" else c)
            if harmonicity is not None:
                gate_track = harmonicity.get(name)
                if gate_track is not None and i < len(gate_track):
                    h_val = float(gate_track[i])
                    if harmonicity_threshold > 0.0 and h_val < harmonicity_threshold:
                        continue
                    eff_conf *= (1.0 + h_val)
            candidates.append((name, f, eff_conf, w))

        if not candidates:
            if stability_hint:
                stable_name, stable_score = max(stability_hint.items(), key=lambda kv: kv[1])
                stable_track = aligned_results.get(stable_name)
                if stable_track is not None and i < len(stable_track[0]):
                    final_f0[i] = float(stable_track[0][i])
                    final_conf[i] = float(stable_track[1][i]) * float(max(stable_score, 0.0))
                    continue

            final_f0[i] = 0.0
            final_conf[i] = 0.0
            continue

        # Pick winner by weighted confidence
        best_name, best_f0, best_conf, best_w = max(
            candidates, key=lambda x: x[2] * x[3]
        )

        # Consensus weighting: measure how many other detectors agree
        total_w = sum(c[3] for c in candidates)
        support_w = best_w
        for name, f, c, w in candidates:
            if name == best_name:
                continue
            if abs(_cent_diff(f, best_f0)) <= float(disagreement_cents):
                support_w += w

        consensus = support_w / max(total_w, 1e-6)
        final_f0[i] = best_f0
        final_conf[i] = best_conf * consensus

    return final_f0, final_conf


def _harmonic_energy_score(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    target_hz: float,
    bandwidth: float,
    max_harmonics: int,
) -> Tuple[float, float]:
    """Return (harmonic_salience, fundamental_energy_ratio)."""

    if target_hz <= 0.0 or not np.isfinite(target_hz):
        return 0.0, 0.0

    total_energy = float(np.sum(magnitude) + 1e-9)
    if total_energy <= 0.0:
        return 0.0, 0.0

    bin_width = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
    accum = 0.0
    base_energy = 0.0
    for h in range(1, int(max_harmonics) + 1):
        fh = target_hz * h
        if fh >= freqs[-1]:
            break
        bw = max(bin_width, abs(bandwidth) * fh)
        mask = (freqs >= fh - bw) & (freqs <= fh + bw)
        energy = float(np.sum(magnitude[mask]))
        if h == 1:
            base_energy += energy
        accum += energy / float(h)

    salience = accum / total_energy
    base_ratio = base_energy / total_energy
    return salience, base_ratio


def _harmonicity_trace(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    f0_track: np.ndarray,
    bandwidth: float,
    max_harmonics: int,
    n_fft: int = 2048,
) -> np.ndarray:
    """Compute harmonic salience per frame for a given f0 track."""

    frames = _frame_audio(np.asarray(audio, dtype=np.float32).reshape(-1), n_fft, hop_length)
    if frames.size == 0:
        return np.zeros_like(f0_track)

    window = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))

    sal = np.zeros_like(f0_track, dtype=np.float32)
    n_frames = min(len(f0_track), mag.shape[0])
    for i in range(n_frames):
        sal[i], _ = _harmonic_energy_score(mag[i], freqs, float(f0_track[i]), bandwidth, max_harmonics)
    return sal


def _track_stability(f0_track: np.ndarray, window: int = 5) -> float:
    """Estimate track stability (higher = smoother)."""

    voiced = np.asarray([f for f in f0_track if f > 0.0], dtype=np.float64)
    if voiced.size < 2:
        return 0.0

    cents = np.abs(1200.0 * np.diff(np.log2(voiced + 1e-9)))
    if cents.size == 0:
        return 0.0
    jitter = float(np.median(cents))
    norm = float(max(window, 1))
    return 1.0 / (1.0 + jitter / (norm * 10.0))


def _apply_post_filters(
    f0: np.ndarray,
    conf: np.ndarray,
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    post_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply adaptive median smoothing, harmonic gating, and octave correction."""

    if f0.size == 0:
        return f0, conf

    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_fft = int(post_cfg.get("n_fft", 2048))
    frames = _frame_audio(y, n_fft, hop_length)
    if frames.size == 0:
        return f0, conf

    window = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))

    adaptive_cfg = post_cfg.get("adaptive_median", {})
    harmonic_cfg = post_cfg.get("harmonic_salience", {})
    octave_cfg = post_cfg.get("octave_correction", {})

    filtered_f0 = f0.copy()
    filtered_conf = conf.copy()

    # Adaptive median smoothing
    if adaptive_cfg.get("enabled", False):
        min_w = int(max(1, adaptive_cfg.get("min_window", 3)))
        max_w = int(max(min_w, adaptive_cfg.get("max_window", 7)))
        jitter_thr = float(adaptive_cfg.get("jitter_cents", 35.0))

        padded = np.pad(filtered_f0, (max_w, max_w), mode="edge")
        for i in range(len(filtered_f0)):
            center = i + max_w
            local = padded[center - min_w : center + min_w + 1]
            jitter = np.abs(1200.0 * np.diff(np.log2(np.clip(local, 1e-6, None))))
            win = min_w if (jitter.size == 0 or np.median(jitter) <= jitter_thr) else max_w
            segment = padded[center - win : center + win + 1]
            filtered_f0[i] = float(np.median(segment))

    # Harmonic salience gating + octave correction
    sal_thresh = float(harmonic_cfg.get("threshold", 0.0))
    sal_bw = float(harmonic_cfg.get("bandwidth", 0.04))
    sal_harm = int(harmonic_cfg.get("max_harmonics", 4))
    energy_margin = float(octave_cfg.get("energy_margin", 1.0))

    for i in range(min(len(filtered_f0), mag.shape[0])):
        pitch = float(filtered_f0[i])
        if pitch <= 0.0:
            filtered_conf[i] = 0.0
            continue

        sal, base_ratio = _harmonic_energy_score(mag[i], freqs, pitch, sal_bw, sal_harm)
        if harmonic_cfg.get("enabled", False):
            if sal < sal_thresh:
                filtered_conf[i] *= 0.5
            else:
                filtered_conf[i] *= min(1.0, 0.5 + sal)

        if octave_cfg.get("enabled", False):
            candidates = []
            for mul in (0.5, 1.0, 2.0):
                cand = pitch * mul
                if cand <= 30.0 or cand >= float(sr) / 2.0:
                    continue
                c_sal, c_base = _harmonic_energy_score(mag[i], freqs, cand, sal_bw, sal_harm)
                candidates.append((cand, c_sal, c_base))

            if candidates:
                best_cand, best_sal, best_base = max(candidates, key=lambda x: (x[1], x[2]))
                # Prefer octave-adjusted pitch if the salience gain is meaningful
                if best_cand != pitch and (
                    best_sal > sal * energy_margin or best_base > base_ratio * energy_margin
                ):
                    filtered_f0[i] = best_cand
                    filtered_conf[i] *= min(1.0, best_sal * 1.25)

    return filtered_f0, filtered_conf

def _is_polyphonic(audio_type: Any) -> bool:
    """Check if Stage A classified audio as polyphonic."""
    try:
        if isinstance(audio_type, AudioType):
            return audio_type in (AudioType.POLYPHONIC, AudioType.POLYPHONIC_DOMINANT)
        if isinstance(audio_type, str):
            return "poly" in audio_type.lower()
    except Exception:
        pass
    return False


def _augment_with_harmonic_masks(
    stem: Stem,
    prior_detector: BasePitchDetector,
    mask_width: float,
    n_harmonics: int,
    audio_path: Optional[str] = None,
) -> Dict[str, Stem]:
    """
    Derive synthetic melody/accompaniment stems by masking harmonics from a quick f0 prior.
    """
    audio = np.asarray(stem.audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {}

    try:
        f0, conf = prior_detector.predict(audio, audio_path=audio_path)
        hop = getattr(prior_detector, "hop_length", 512)
        n_fft = getattr(prior_detector, "n_fft", 2048)

        f, t, Z = scipy.signal.stft(
            audio,
            fs=stem.sr,
            nperseg=n_fft,
            noverlap=max(0, n_fft - hop),
            boundary="zeros",
            padded=True,
        )

        n_frames = Z.shape[1]
        if f0.shape[0] != n_frames:
            if f0.shape[0] < n_frames:
                pad = n_frames - f0.shape[0]
                f0 = np.pad(f0, (0, pad))
                conf = np.pad(conf, (0, pad))
            else:
                f0 = f0[:n_frames]
                conf = conf[:n_frames]

        mask = create_harmonic_mask(
            f0_hz=f0,
            sr=stem.sr,
            n_fft=n_fft,
            mask_width=mask_width,
            n_harmonics=n_harmonics,
        )

        # Harmonic emphasis keeps bins near f0; residual keeps the rest.
        strength = np.clip(conf, 0.0, 1.0).reshape(1, -1)
        harmonic_keep = np.clip((1.0 - mask) * (0.8 + 0.2 * strength), 0.0, 1.0)
        residual_keep = np.clip(1.0 - harmonic_keep, 0.0, 1.0)

        Z_melody = Z * harmonic_keep
        Z_resid = Z * residual_keep

        _, melody_audio = scipy.signal.istft(
            Z_melody,
            fs=stem.sr,
            nperseg=n_fft,
            noverlap=max(0, n_fft - hop),
            input_onesided=True,
            boundary="zeros",
        )
        _, residual_audio = scipy.signal.istft(
            Z_resid,
            fs=stem.sr,
            nperseg=n_fft,
            noverlap=max(0, n_fft - hop),
            input_onesided=True,
            boundary="zeros",
        )

        melody_audio = np.asarray(melody_audio, dtype=np.float32)
        residual_audio = np.asarray(residual_audio, dtype=np.float32)

        # Match original length
        if melody_audio.size < audio.size:
            melody_audio = np.pad(melody_audio, (0, audio.size - melody_audio.size))
        melody_audio = melody_audio[: audio.size]

        if residual_audio.size < audio.size:
            residual_audio = np.pad(residual_audio, (0, audio.size - residual_audio.size))
        residual_audio = residual_audio[: audio.size]

        return {
            "melody_masked": Stem(audio=melody_audio, sr=stem.sr, type="melody_masked"),
            "residual_masked": Stem(audio=residual_audio, sr=stem.sr, type="residual_masked"),
        }
    except Exception:
        # Do not let masking failure break the pipeline
        return {}


def _resolve_polyphony_filter(config: Optional[PipelineConfig]) -> str:
    try:
        return str(config.stage_c.polyphony_filter.get("mode", "skyline_top_voice"))
    except Exception:
        return "skyline_top_voice"


class MultiVoiceTracker:
    """
    Lightweight multi-voice tracker to keep skyline assignments stable.

    Tracks up to `max_tracks` concurrent voices using a Hungarian assignment on
    pitch proximity with hangover/hysteresis to avoid rapid swapping.
    """

    def __init__(
        self,
        max_tracks: int,
        max_jump_cents: float = 150.0,
        hangover_frames: int = 2,
        smoothing: float = 0.35,
        confidence_bias: float = 5.0,
    ) -> None:
        self.max_tracks = max_tracks
        self.max_jump_cents = float(max_jump_cents)
        self.hangover_frames = int(max(0, hangover_frames))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.confidence_bias = float(confidence_bias)
        self.prev_pitches = np.zeros(max_tracks, dtype=np.float32)
        self.prev_confs = np.zeros(max_tracks, dtype=np.float32)
        self.hold = np.zeros(max_tracks, dtype=np.int32)

    def _pitch_cost(self, prev: float, candidate: float) -> float:
        if prev <= 0.0 or candidate <= 0.0:
            return 0.0
        cents = abs(1200.0 * np.log2((candidate + 1e-6) / (prev + 1e-6)))
        penalty = self.max_jump_cents if cents > self.max_jump_cents else 0.0
        return cents + penalty

    def _assign(self, pitches: np.ndarray, confs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Fallback to greedy ordering when Hungarian solver missing
        if linear_sum_assignment is None or pitches.size == 0:
            ordered = sorted(zip(pitches, confs), key=lambda x: (-x[1], x[0]))
            ordered = ordered[: self.max_tracks]
            new_pitches = np.zeros_like(self.prev_pitches)
            new_confs = np.zeros_like(self.prev_confs)
            for idx, (p, c) in enumerate(ordered):
                new_pitches[idx] = p
                new_confs[idx] = c
            return new_pitches, new_confs

        cost = np.zeros((self.max_tracks, pitches.size), dtype=np.float32)
        for i in range(self.max_tracks):
            for j in range(pitches.size):
                pitch_cost = self._pitch_cost(float(self.prev_pitches[i]), float(pitches[j]))
                cost[i, j] = pitch_cost - float(confs[j]) * self.confidence_bias

        row_idx, col_idx = linear_sum_assignment(cost)
        new_pitches = np.zeros_like(self.prev_pitches)
        new_confs = np.zeros_like(self.prev_confs)
        for r, c in zip(row_idx, col_idx):
            new_pitches[r] = pitches[c]
            new_confs[r] = confs[c]
        return new_pitches, new_confs

    def step(self, candidates: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        if not candidates:
            # Apply hangover to keep tracks alive briefly
            carry_pitches = np.where(self.hold > 0, self.prev_pitches, 0.0)
            carry_confs = np.where(self.hold > 0, self.prev_confs * 0.9, 0.0)
            self.hold = np.maximum(self.hold - 1, 0)
            self.prev_pitches = carry_pitches.astype(np.float32)
            self.prev_confs = carry_confs.astype(np.float32)
            return self.prev_pitches.copy(), self.prev_confs.copy()

        # Keep only the strongest candidates up to track count
        ordered = sorted(candidates, key=lambda x: (-x[1], x[0]))[: self.max_tracks]
        pitches = np.array([c[0] for c in ordered], dtype=np.float32)
        confs = np.array([c[1] for c in ordered], dtype=np.float32)

        assigned_pitches, assigned_confs = self._assign(pitches, confs)

        updated_pitches = np.zeros_like(self.prev_pitches)
        updated_confs = np.zeros_like(self.prev_confs)
        for idx in range(self.max_tracks):
            if assigned_pitches[idx] > 0.0:
                if self.prev_pitches[idx] > 0.0:
                    smoothed = (
                        self.smoothing * float(self.prev_pitches[idx])
                        + (1.0 - self.smoothing) * float(assigned_pitches[idx])
                    )
                else:
                    smoothed = float(assigned_pitches[idx])
                updated_pitches[idx] = smoothed
                updated_confs[idx] = assigned_confs[idx]
                self.hold[idx] = self.hangover_frames
            elif self.hold[idx] > 0:
                updated_pitches[idx] = self.prev_pitches[idx]
                updated_confs[idx] = self.prev_confs[idx] * 0.85
                self.hold[idx] -= 1
            else:
                updated_pitches[idx] = 0.0
                updated_confs[idx] = 0.0

        self.prev_pitches = updated_pitches
        self.prev_confs = updated_confs
        return updated_pitches.copy(), updated_confs.copy()


def extract_features(
    stage_a_out: StageAOutput,
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> StageBOutput:
    """
    Stage B: Extract pitch and features.
    Respects config.stage_b for detector selection, ensemble weights, and
    optional polyphonic peeling to expose multiple F0 layers.
    """
    if config is None:
        config = PipelineConfig()

    b_conf = config.stage_b
    c_conf = getattr(config, "stage_c", None)
    sr = stage_a_out.meta.sample_rate
    hop_length = stage_a_out.meta.hop_length

    # Align RMS gating with Stage C velocity mapping (prevents note spam on fades)
    rms_gate = 0.0
    try:
        if c_conf is not None:
            min_db = float(getattr(c_conf, "velocity_map", {}).get("min_db", -40.0))
        else:
            min_db = -40.0
        gate_db = min_db - 5.0
        rms_gate = 10 ** (gate_db / 20.0)
    except Exception:
        rms_gate = 0.0

    # Separation routing happens before harmonic masking/ISS so downstream
    # detectors always see the requested stem layout.
    resolved_stems, separation_diag = _resolve_separation(stage_a_out, b_conf)

    # 1. Initialize Detectors based on Config
    detectors: Dict[str, BasePitchDetector] = {}
    for name, det_conf in b_conf.detectors.items():
        det = _init_detector(name, det_conf, sr, hop_length)
        if det:
            detectors[name] = det

    # Ensure baseline fallback if no detectors enabled/working
    if not detectors:
        LOGGER.warning("No detectors enabled or initialized in Stage B. Falling back to default YIN/ACF.")
        detectors["yin"] = YinDetector(sr, hop_length)

    stem_timelines: Dict[str, List[FramePitch]] = {}
    per_detector: Dict[str, Any] = {}
    f0_main: Optional[np.ndarray] = None
    all_layers: List[np.ndarray] = []
    iss_total_layers = 0

    # Polyphonic context detection
    polyphonic_context = _is_polyphonic(getattr(stage_a_out, "audio_type", None))
    skyline_mode = _resolve_polyphony_filter(config)
    tracker_cfg = getattr(b_conf, "voice_tracking", {}) or {}
    tracker_fusion_cfg = getattr(b_conf, "tracker_fusion", {}) or {}
    post_filter_cfg = getattr(b_conf, "post_filters", {}) or {}

    # Optional harmonic masking to create synthetic melody/bass stems for synthetic material
    augmented_stems = dict(resolved_stems)
    harmonic_mask_applied = False
    harmonic_cfg = b_conf.separation.get("harmonic_masking", {}) if hasattr(b_conf, "separation") else {}
    if harmonic_cfg.get("enabled", False) and "mix" in augmented_stems:
        prior_det = detectors.get("swiftf0")
        if prior_det is None:
            prior_conf = dict(b_conf.detectors.get("swiftf0", {}))
            prior_conf["enabled"] = True
            prior_det = _init_detector("swiftf0", prior_conf, sr, hop_length)
        if prior_det is None:
            prior_det = detectors.get("yin") or _init_detector("yin", {"enabled": True}, sr, hop_length)

        if prior_det is not None:
            synthetic = _augment_with_harmonic_masks(
                augmented_stems["mix"],
                prior_det,
                mask_width=float(harmonic_cfg.get("mask_width", 0.025)),
                n_harmonics=int(harmonic_cfg.get("n_harmonics", 8)),
                audio_path=stage_a_out.meta.audio_path,
            )
            augmented_stems.update(synthetic)
            harmonic_mask_applied = bool(synthetic)

    stems_for_processing = augmented_stems
    polyphonic_context = polyphonic_context or len(stems_for_processing) > 1 or b_conf.polyphonic_peeling.get("force_on_mix", False)

    # 2. Process Stems
    for stem_name, stem in resolved_stems.items():
        audio = stem.audio
        per_detector[stem_name] = {}

        stem_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        fusion_cfg = tracker_fusion_cfg
        harmonic_gate_cfg = fusion_cfg.get("harmonicity_gate", {}) or {}
        harmonic_map: Optional[Dict[str, np.ndarray]] = None
        stability_hint: Optional[Dict[str, float]] = None

        # Run all initialized detectors on this stem
        for name, det in detectors.items():
            try:
                f0, conf = det.predict(audio, audio_path=stage_a_out.meta.audio_path)
                stem_results[name] = (f0, conf)
                per_detector[stem_name][name] = (f0, conf)
            except Exception as e:
                LOGGER.warning("Detector %s failed on stem %s: %s", name, stem_name, e)

        if stem_results and harmonic_gate_cfg.get("enabled", False):
            harmonic_map = {}
            for name, (f0_track, _) in stem_results.items():
                harmonic_map[name] = _harmonicity_trace(
                    audio,
                    sr,
                    hop_length,
                    f0_track,
                    float(harmonic_gate_cfg.get("bandwidth", 0.04)),
                    int(harmonic_gate_cfg.get("max_harmonics", 4)),
                )

        if stem_results and fusion_cfg.get("use_stability_fallback", False):
            stability_hint = {
                name: _track_stability(f0_track, int(fusion_cfg.get("stability_window", 5)))
                for name, (f0_track, _) in stem_results.items()
            }

        # Ensemble Merge with disagreement and SwiftF0 priority floor
        if stem_results:
            merged_f0, merged_conf = _ensemble_merge(
                stem_results,
                b_conf.ensemble_weights,
                b_conf.pitch_disagreement_cents,
                b_conf.confidence_priority_floor,
                harmonicity=harmonic_map,
                harmonicity_threshold=float(harmonic_gate_cfg.get("threshold", 0.0)),
                stability_hint=stability_hint,
            )
        else:
            merged_f0 = np.zeros(1, dtype=np.float32)
            merged_conf = np.zeros(1, dtype=np.float32)

        # Post filters: adaptive median + harmonic gating + octave correction
        merged_f0, merged_conf = _apply_post_filters(
            merged_f0,
            merged_conf,
            audio,
            sr,
            hop_length,
            post_filter_cfg,
        )

        # Polyphonic peeling (ISS) – optional and gated by config + context
        iss_layers: List[Tuple[np.ndarray, np.ndarray]] = []
        if polyphonic_context and b_conf.polyphonic_peeling.get("max_layers", 0) > 0:
            primary = detectors.get("swiftf0") or detectors.get("yin") or detectors.get("sacf")
            validator = detectors.get("sacf") or detectors.get("yin")
            if primary:
                try:
                    iss_layers = iterative_spectral_subtraction(
                        audio,
                        sr,
                        primary_detector=primary,
                        validator_detector=validator,
                        max_polyphony=b_conf.polyphonic_peeling.get("max_layers", 4),
                        mask_width=b_conf.polyphonic_peeling.get("mask_width", 0.03),
                        min_mask_width=b_conf.polyphonic_peeling.get("min_mask_width", 0.02),
                        max_mask_width=b_conf.polyphonic_peeling.get("max_mask_width", 0.08),
                        mask_growth=b_conf.polyphonic_peeling.get("mask_growth", 1.1),
                        mask_shrink=b_conf.polyphonic_peeling.get("mask_shrink", 0.9),
                        harmonic_snr_stop_db=b_conf.polyphonic_peeling.get("harmonic_snr_stop_db", 3.0),
                        residual_rms_stop_ratio=b_conf.polyphonic_peeling.get("residual_rms_stop_ratio", 0.08),
                        residual_flatness_stop=b_conf.polyphonic_peeling.get("residual_flatness_stop", 0.45),
                        validator_cents_tolerance=b_conf.polyphonic_peeling.get("validator_cents_tolerance", b_conf.pitch_disagreement_cents),
                        validator_agree_window=b_conf.polyphonic_peeling.get("validator_agree_window", 5),
                        validator_disagree_decay=b_conf.polyphonic_peeling.get("validator_disagree_decay", 0.6),
                        validator_min_agree_frames=b_conf.polyphonic_peeling.get("validator_min_agree_frames", 2),
                        validator_min_disagree_frames=b_conf.polyphonic_peeling.get("validator_min_disagree_frames", 2),
                        max_harmonics=b_conf.polyphonic_peeling.get("max_harmonics", 12),
                        audio_path=stage_a_out.meta.audio_path,
                    )
                    all_layers.extend([f0 for f0, _ in iss_layers])
                    iss_total_layers += len(iss_layers)
                except Exception as e:
                    LOGGER.warning("ISS peeling failed for stem %s: %s", stem_name, e)

        # Calculate RMS
        n_fft = stage_a_out.meta.window_size if stage_a_out.meta.window_size else 2048
        frames = _frame_audio(audio, n_fft, hop_length)
        rms_vals = np.sqrt(np.mean(frames**2, axis=1))

        # Pad/Trim RMS to match dominant F0
        if len(rms_vals) < len(merged_f0):
            rms_vals = np.pad(rms_vals, (0, len(merged_f0) - len(rms_vals)))
        elif len(rms_vals) > len(merged_f0):
            rms_vals = rms_vals[:len(merged_f0)]

        # Build timeline with optional skyline selection from poly layers
        voicing_thr = float(b_conf.confidence_voicing_threshold)
        if polyphonic_context:
            voicing_thr = max(0.0, voicing_thr - float(getattr(b_conf, "polyphonic_voicing_relaxation", 0.0)))
        layer_arrays = [(merged_f0, merged_conf)] + iss_layers
        max_frames = max(len(arr[0]) for arr in layer_arrays)

        def _pad_to(arr: np.ndarray, target: int) -> np.ndarray:
            if len(arr) < target:
                return np.pad(arr, (0, target - len(arr)))
            return arr[:target]

        padded_layers = [(_pad_to(f0, max_frames), _pad_to(conf, max_frames)) for f0, conf in layer_arrays]
        padded_rms = _pad_to(rms_vals, max_frames)

        # Short median smoothing to reduce jitter-driven splits
        for idx, (f0_arr, conf_arr) in enumerate(padded_layers):
            if len(f0_arr) < 3:
                continue
            pad = 2
            padded = np.pad(f0_arr, (pad, pad), mode="edge")
            smoothed = np.zeros_like(f0_arr)
            for i in range(len(f0_arr)):
                segment = padded[i : i + 2 * pad + 1]
                smoothed[i] = float(np.median(segment))
            padded_layers[idx] = (smoothed, conf_arr)

        timeline: List[FramePitch] = []
        max_alt_voices = int(tracker_cfg.get("max_alt_voices", 4) if polyphonic_context else 0)
        tracker = MultiVoiceTracker(
            max_tracks=1 + max_alt_voices,
            max_jump_cents=tracker_cfg.get("max_jump_cents", 150.0),
            hangover_frames=tracker_cfg.get("hangover_frames", 2),
            smoothing=tracker_cfg.get("smoothing", 0.35),
            confidence_bias=tracker_cfg.get("confidence_bias", 5.0),
        )

        track_buffers = [np.zeros(max_frames, dtype=np.float32) for _ in range(tracker.max_tracks)]
        track_conf_buffers = [np.zeros(max_frames, dtype=np.float32) for _ in range(tracker.max_tracks)]

        for i in range(max_frames):
            candidates: List[Tuple[float, float]] = []
            raw_candidates: List[Tuple[float, float]] = []
            for f0_arr, conf_arr in padded_layers:
                f = float(f0_arr[i]) if i < len(f0_arr) else 0.0
                c = float(conf_arr[i]) if i < len(conf_arr) else 0.0
                if f > 0.0 and c > 0.0:
                    raw_candidates.append((f, c))
                    if c >= voicing_thr:
                        if rms_gate > 0.0 and (i < len(padded_rms)):
                            if float(padded_rms[i]) < rms_gate:
                                continue
                        candidates.append((f, c))

            if not candidates and raw_candidates:
                best_f, best_c = max(raw_candidates, key=lambda x: x[1])
                if best_c >= 0.6 * voicing_thr:
                    if rms_gate <= 0.0 or float(padded_rms[i]) >= rms_gate:
                        candidates.append((best_f, best_c))

            tracked_pitches, tracked_confs = tracker.step(candidates)
            for voice_idx in range(tracker.max_tracks):
                track_buffers[voice_idx][i] = tracked_pitches[voice_idx]
                track_conf_buffers[voice_idx][i] = tracked_confs[voice_idx]

            chosen_pitch = float(tracked_pitches[0]) if tracked_pitches.size else 0.0
            chosen_conf = float(tracked_confs[0]) if tracked_confs.size else 0.0

            midi = None
            if chosen_pitch > 0.0:
                midi = int(round(69.0 + 12.0 * np.log2(chosen_pitch / 440.0)))

            active = [
                (float(p), float(track_conf_buffers[idx][i]))
                for idx, p in enumerate(tracked_pitches)
                if p > 0.0
            ]

            timeline.append(
                FramePitch(
                    time=float(i * hop_length) / float(sr),
                    pitch_hz=chosen_pitch,
                    confidence=chosen_conf,
                    midi=midi,
                    rms=float(padded_rms[i]) if i < len(padded_rms) else 0.0,
                    active_pitches=active,
                )
            )

        # Carry the final confident pitch through a short tail to avoid truncation
        hop_s = float(hop_length) / float(sr)
        tail_frames = int(max(1, round(0.30 / hop_s)))
        last_idx = max((idx for idx, fp in enumerate(timeline) if fp.pitch_hz > 0.0), default=-1)
        if last_idx >= 0:
            last_pitch = timeline[last_idx].pitch_hz
            last_conf = timeline[last_idx].confidence
            for j in range(last_idx + 1, min(len(timeline), last_idx + tail_frames + 1)):
                timeline[j].pitch_hz = last_pitch
                timeline[j].confidence = last_conf

        stem_timelines[stem_name] = timeline

        # Keep secondary voices as separate layers to aid downstream segmentation/rendering
        for alt in track_buffers[1:]:
            if np.count_nonzero(alt) > 0:
                all_layers.append(alt)

        # Set main f0 (prefer vocals, then mix)
        main_track = track_buffers[0]
        if stem_name == "vocals":
            f0_main = main_track
        elif stem_name == "mix" and f0_main is None:
            f0_main = main_track

    if f0_main is None:
        if stem_timelines:
            first_stem = next(iter(stem_timelines.values()))
            f0_main = np.array([fp.pitch_hz for fp in first_stem], dtype=np.float32)
        else:
            f0_main = np.array([], dtype=np.float32)

    time_grid = np.array([])
    if len(f0_main) > 0:
        time_grid = np.arange(len(f0_main)) * hop_length / sr

    diagnostics = {
        "polyphonic_context": bool(polyphonic_context),
        "detectors_initialized": list(detectors.keys()),
        "separation": separation_diag,
        "harmonic_masking": {
            "enabled": harmonic_cfg.get("enabled", False),
            "applied": harmonic_mask_applied,
            "mask_width": harmonic_cfg.get("mask_width"),
            "n_harmonics": harmonic_cfg.get("n_harmonics"),
        },
        "iss": {
            "enabled": polyphonic_context and b_conf.polyphonic_peeling.get("max_layers", 0) > 0,
            "layers_found": iss_total_layers,
            "max_layers": b_conf.polyphonic_peeling.get("max_layers", 0),
        },
        "skyline_mode": skyline_mode,
        "voice_tracking": {
            "max_alt_voices": int(tracker_cfg.get("max_alt_voices", 4) if polyphonic_context else 0),
            "max_jump_cents": tracker_cfg.get("max_jump_cents", 150.0),
        },
        "tracker_fusion": {
            "strategy": tracker_fusion_cfg.get("strategy", "confidence_vote"),
            "harmonic_gate": bool((tracker_fusion_cfg.get("harmonicity_gate") or {}).get("enabled", False)),
        },
        "post_filters": {
            "adaptive_median": bool((post_filter_cfg.get("adaptive_median") or {}).get("enabled", False)),
            "harmonic_salience": bool((post_filter_cfg.get("harmonic_salience") or {}).get("enabled", False)),
            "octave_correction": bool((post_filter_cfg.get("octave_correction") or {}).get("enabled", False)),
        },
    }

    return StageBOutput(
        time_grid=time_grid,
        f0_main=f0_main,
        f0_layers=all_layers,
        per_detector=per_detector,
        stem_timelines=stem_timelines,
        stems=resolved_stems,
        meta=stage_a_out.meta,
        diagnostics=diagnostics,
        resolved_stems=resolved_stems,
    )
