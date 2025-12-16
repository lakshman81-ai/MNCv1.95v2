"""Lightweight dual-head separator + F0 system with refinement utilities.

This module does not attempt to train a full neural network. Instead it
implements the core accounting that Stage B can use to reason about a
separator backbone, a pitch head that couples SI-SDR with pitch losses, and
optional distillation / iterative refinement helpers. The goal is to expose
training-ready primitives without requiring heavyweight ML dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .stage_b import SyntheticMDXSeparator


@dataclass
class PitchTargets:
    """Container for target F0 information."""

    f0_hz: np.ndarray
    voiced_mask: np.ndarray


@dataclass
class PredictionBundle:
    """Aggregated outputs and losses for the dual-head system."""

    separated_stems: Dict[str, np.ndarray]
    pitch_logits: np.ndarray
    pitch_regression: np.ndarray
    embeddings: np.ndarray
    si_sdr_loss: float
    pitch_loss: float
    total_loss: float


def _spectrogram(audio: np.ndarray, win: int = 1024, hop: int = 256) -> np.ndarray:
    """Compute a minimal magnitude spectrogram for use by the heads."""

    if audio.size == 0:
        return np.zeros((win // 2 + 1, 1), dtype=np.float32)

    window = np.hanning(win)
    frames = []
    for start in range(0, len(audio) - win + 1, hop):
        frame = audio[start : start + win] * window
        spec = np.abs(np.fft.rfft(frame))
        frames.append(spec)

    if not frames:
        frames.append(np.abs(np.fft.rfft(audio[:win] * window)))

    spec = np.stack(frames, axis=1)
    return spec.astype(np.float32)


def _safe_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    if est.size == 0 or ref.size == 0:
        return 0.0

    ref_energy = np.dot(ref, ref) + 1e-8
    proj = np.dot(est, ref) * ref / ref_energy
    noise = est - proj
    ratio = (np.dot(proj, proj) + 1e-8) / (np.dot(noise, noise) + 1e-8)
    return -10.0 * np.log10(ratio + 1e-8)


class F0Head:
    """Pitch head that mixes CE and regression losses."""

    def __init__(self, n_bins: int = 360, min_hz: float = 32.7, max_hz: float = 2093.0):
        self.n_bins = n_bins
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.bin_edges = np.linspace(np.log2(min_hz), np.log2(max_hz), n_bins)

    def _hz_to_bin(self, hz: np.ndarray) -> np.ndarray:
        hz = np.clip(hz, self.min_hz, self.max_hz)
        log_hz = np.log2(hz)
        idx = (log_hz - self.bin_edges[0]) / (self.bin_edges[-1] - self.bin_edges[0])
        idx = np.clip(idx * (self.n_bins - 1), 0, self.n_bins - 1)
        return idx

    def predict(self, magnitude_spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if magnitude_spec.ndim == 2:
            spec = magnitude_spec
        else:
            spec = magnitude_spec.reshape(magnitude_spec.shape[0], -1)

        energy = np.maximum(spec, 1e-6)
        norm_energy = energy / np.sum(energy, axis=0, keepdims=True)
        pitch_logits = np.log(norm_energy + 1e-9)

        freq_axis = np.linspace(self.min_hz, self.max_hz, spec.shape[0])[:, None]
        centroid = np.sum(freq_axis * norm_energy, axis=0)
        regression = centroid - self.min_hz

        spectral_flatness = np.exp(np.mean(np.log(energy), axis=0) - np.log(np.mean(energy, axis=0)))
        embeddings = np.stack([centroid, regression, spectral_flatness], axis=-1)
        return pitch_logits, regression, embeddings

    def loss(
        self,
        pitch_logits: np.ndarray,
        regression: np.ndarray,
        targets: PitchTargets,
        teacher_logits: Optional[np.ndarray] = None,
        ce_weight: float = 1.0,
        reg_weight: float = 0.5,
        distill_weight: float = 0.2,
    ) -> float:
        bin_targets = self._hz_to_bin(targets.f0_hz).astype(int)
        probs = _safe_log_softmax(pitch_logits, axis=0)
        ce_terms = -probs[bin_targets, np.arange(probs.shape[1])]
        ce_terms *= targets.voiced_mask
        ce_loss = float(np.mean(ce_terms)) if ce_terms.size else 0.0

        target_reg = targets.f0_hz - self.min_hz
        reg_terms = np.abs(regression - target_reg)
        reg_terms *= targets.voiced_mask
        reg_loss = float(np.mean(reg_terms)) if reg_terms.size else 0.0

        distill_loss = 0.0
        if teacher_logits is not None and teacher_logits.shape == pitch_logits.shape:
            student_p = _safe_log_softmax(pitch_logits, axis=0)
            teacher_p = _safe_log_softmax(teacher_logits, axis=0)
            kl = teacher_p * (np.log(teacher_p + 1e-9) - np.log(student_p + 1e-9))
            distill_loss = float(np.mean(np.sum(kl, axis=0)))

        return ce_weight * ce_loss + reg_weight * reg_loss + distill_weight * distill_loss


class TeacherStudentDistiller:
    """Helper to inject teacher distributions during early training."""

    def __init__(self, temperature: float = 2.0):
        self.temperature = temperature

    def soften(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / max(self.temperature, 1e-6)
        return _safe_log_softmax(scaled, axis=0)

    def distill(self, student_logits: np.ndarray, teacher_logits: np.ndarray) -> np.ndarray:
        teacher_soft = self.soften(teacher_logits)
        student_soft = _safe_log_softmax(student_logits, axis=0)
        return teacher_soft * (np.log(teacher_soft + 1e-9) - np.log(student_soft + 1e-9))


class CrossAttentionRefiner:
    """Iterative refinement to suppress octave slips."""

    def __init__(self, n_iters: int = 2):
        self.n_iters = n_iters

    def refine(
        self, spectrograms: Dict[str, np.ndarray], f0_embeddings: np.ndarray
    ) -> np.ndarray:
        refined = f0_embeddings.copy()
        if refined.size == 0:
            return refined

        energy_tracks: List[np.ndarray] = []
        for spec in spectrograms.values():
            if spec.ndim == 2:
                energy_tracks.append(np.sum(spec, axis=0, keepdims=True))
        if not energy_tracks:
            return refined

        energy_stack = np.concatenate(energy_tracks, axis=0)
        norm_energy = energy_stack / (np.sum(energy_stack, axis=0, keepdims=True) + 1e-9)

        for _ in range(self.n_iters):
            attn = norm_energy / (np.linalg.norm(norm_energy, axis=0, keepdims=True) + 1e-9)
            attended = attn.T @ refined[:, : attn.shape[0]]
            residual = refined[:, : attended.shape[1]] - attended
            refined[:, : attended.shape[1]] = attended + 0.5 * residual
        return refined


class TwoHeadF0Network:
    """End-to-end container that ties together separator, pitch head, and refinement."""

    def __init__(
        self,
        separator: Optional[SyntheticMDXSeparator] = None,
        f0_head: Optional[F0Head] = None,
        distiller: Optional[TeacherStudentDistiller] = None,
        refiner: Optional[CrossAttentionRefiner] = None,
        si_sdr_weight: float = 1.0,
    ):
        self.separator = separator or SyntheticMDXSeparator()
        self.f0_head = f0_head or F0Head()
        self.distiller = distiller
        self.refiner = refiner
        self.si_sdr_weight = si_sdr_weight

    def forward(
        self,
        audio: np.ndarray,
        sr: int,
        targets: Optional[Dict[str, np.ndarray]] = None,
        pitch_targets: Optional[PitchTargets] = None,
        teacher_logits: Optional[np.ndarray] = None,
    ) -> PredictionBundle:
        stems = self.separator.separate(audio, sr)
        spectrograms = {name: _spectrogram(stem) for name, stem in stems.items()}
        mix_spec = sum(spectrograms.values()) if spectrograms else _spectrogram(audio)

        pitch_logits, pitch_reg, embeddings = self.f0_head.predict(mix_spec)
        if self.refiner is not None:
            embeddings = self.refiner.refine(spectrograms, embeddings)

        si_sdr_losses: List[float] = []
        if targets:
            for key, ref in targets.items():
                if key in stems:
                    si_sdr_losses.append(_si_sdr(stems[key], ref))
        si_sdr_loss = float(np.mean(si_sdr_losses)) if si_sdr_losses else 0.0

        pitch_loss = 0.0
        if pitch_targets is not None:
            pitch_loss = self.f0_head.loss(pitch_logits, pitch_reg, pitch_targets, teacher_logits)

        total_loss = self.si_sdr_weight * si_sdr_loss + pitch_loss
        return PredictionBundle(
            separated_stems=stems,
            pitch_logits=pitch_logits,
            pitch_regression=pitch_reg,
            embeddings=embeddings,
            si_sdr_loss=si_sdr_loss,
            pitch_loss=pitch_loss,
            total_loss=total_loss,
        )


def select_best_variant(
    baseline_metrics: Dict[str, float], candidate_metrics: Dict[str, float], key: str = "L2"
) -> Tuple[str, Dict[str, float]]:
    """Pick the better variant with respect to a benchmark key (default: L2)."""

    base_score = baseline_metrics.get(key, -np.inf)
    cand_score = candidate_metrics.get(key, -np.inf)
    if cand_score >= base_score:
        return "candidate", candidate_metrics
    return "baseline", baseline_metrics
