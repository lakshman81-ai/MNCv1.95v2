"""Structured plan for L2 poly-dominant retraining and augmentation.

The plan mirrors the research goals outlined for improving melody
retention under bass masking and hardening the pipeline against
aggressive augmentations.  It is kept as data (rather than prose
in documentation) so that training and evaluation tooling can ingest
it for auditing, logging, or checklist-style gating.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass(frozen=True)
class AugmentationStep:
    """A single augmentation step with an ordered description."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurriculumPhase:
    """Represents a curriculum stage for L2 retraining."""

    name: str
    epochs: str
    focus: str
    notes: str


@dataclass(frozen=True)
class RetrainingPlan:
    """Container for the L2 poly-dominant retraining recipe."""

    objectives: List[str]
    dataset_curation: Dict[str, List[str]]
    synthetic_counterpoint: Dict[str, Any]
    quality_filters: Dict[str, Any]
    augmentation_chain: List[AugmentationStep]
    curriculum: List[CurriculumPhase]
    retraining_tasks: Dict[str, Any]
    benchmark: Dict[str, str]

    def as_dict(self) -> Dict[str, Any]:
        """Render the plan to plain Python structures for logging/serialization."""

        return {
            "objectives": self.objectives,
            "dataset_curation": self.dataset_curation,
            "synthetic_counterpoint": self.synthetic_counterpoint,
            "quality_filters": self.quality_filters,
            "augmentation_chain": [
                {
                    "name": step.name,
                    "description": step.description,
                    "parameters": step.parameters,
                }
                for step in self.augmentation_chain
            ],
            "curriculum": [
                {
                    "name": phase.name,
                    "epochs": phase.epochs,
                    "focus": phase.focus,
                    "notes": phase.notes,
                }
                for phase in self.curriculum
            ],
            "retraining_tasks": self.retraining_tasks,
            "benchmark": self.benchmark,
        }


L2_POLY_DOMINANT_PLAN = RetrainingPlan(
    objectives=[
        "Improve melody retention under bass masking via bass-forward training data and counterpoint coverage.",
        "Increase robustness of separation and pitch models to extreme pitch/formant shifts, distractor stems, and diverse rooms.",
        "Validate gains on the L2 benchmark before cascading to upper ladder levels.",
    ],
    dataset_curation={
        "bass_forward_corpora": [
            "Prioritize Jamendo solos/duets with verified bass stems and metadata (key/tempo, LUFS, instrument tags).",
            "Use MedleyDB tracks with isolated bass DI/mics and keep stem loudness for sampling.",
            "Pull RWC-Pop songs flagged for pronounced bass lines.",
            "Use class-balanced Slakh mixtures filtered for bass amplitude and crest factor.",
        ],
        "duet_focus": [
            "Build a melody+bass subset by pairing exposed vocal/lead stems with bass from the above corpora.",
            "Include real duets and semi-stem versions where accompaniment is suppressed via separation.",
        ],
    },
    synthetic_counterpoint={
        "intervals": ["3rd", "6th", "10th"],
        "entrances": "staggered counter-melody entrances with rhythm offsets",
        "rhythm_offsets": "quantized 1/8 or 1/16 offsets",
        "bass_timbres": [
            "electric_picked",
            "electric_fingered",
            "synth_saw",
            "synth_square",
            "synth_fm",
            "acoustic",
        ],
        "lead_timbres": ["sine", "woodwind", "strings", "voice_like_formants"],
        "render_constraints": {
            "tempo_bpm": "70-150",
            "quantization": "enforce musical subdivisions",
        },
    },
    quality_filters={
        "lufs_min_db": -32,
        "max_crest_factor_db": 18,
        "synthetic_tempo_bpm": [70, 150],
        "quantize_offsets": True,
    },
    augmentation_chain=[
        AugmentationStep(
            name="time_stretch",
            description="Formant-preserving stretch (0.7-1.3x general, 0.5-1.5x for synthetic renders).",
            parameters={"range": [0.7, 1.3], "synthetic_range": [0.5, 1.5]},
        ),
        AugmentationStep(
            name="pitch_shift",
            description="Random ±12 semitones; formant correction on leads, optional drift on bass; include octave drops with light saturation.",
            parameters={"semitones": 12, "bass_octave_drop_db": -12},
        ),
        AugmentationStep(
            name="coloration",
            description="Saturation/EQ to mimic DI + amp coloration before mixing.",
            parameters={"apply_to": ["bass", "lead"]},
        ),
        AugmentationStep(
            name="snr_mixing",
            description="Mix melody vs bass vs distractors at {+12, +6, 0, -6, -12} dB SNR with randomized phase alignment.",
            parameters={"snr_steps_db": [12, 6, 0, -6, -12], "phase_randomization": True},
        ),
        AugmentationStep(
            name="reverb_room",
            description="Convolve with small-room IRs and plate/club tails; vary pre-delay and wet mix; include early-reflection-only variants.",
            parameters={"pre_delay_ms": [0, 40], "wet_mix_percent": [5, 30]},
        ),
        AugmentationStep(
            name="logging",
            description="Record applied parameters per sample (pitch, stretch, SNR, IR, wet mix) for curriculum auditing.",
            parameters={},
        ),
    ],
    curriculum=[
        CurriculumPhase(
            name="Phase 1: Clean stems",
            epochs="0-2",
            focus="Train on unaugmented duet/counterpoint stems; freeze augmentation RNG for reproducibility.",
            notes="Stabilize pitch tracking and separation masks before heavy effects.",
        ),
        CurriculumPhase(
            name="Phase 2: Controlled mixtures",
            epochs="3-6",
            focus="Introduce SNR ladder with one distractor and mild pitch/time shifts.",
            notes="Monitor pitch F1 and onset MAE on held-out clean duets.",
        ),
        CurriculumPhase(
            name="Phase 3: Full augmentations",
            epochs="7+",
            focus="Enable wide-range shifts, multi-distractor mixes, and reverb variants with 30-40% clean quota.",
            notes="Prevent overfitting to heavy effects by mixing in earlier-phase data.",
        ),
    ],
    retraining_tasks={
        "separation_model": {
            "focus": "Fine-tune synthetic Demucs variant with higher LF weights and melody SI-SDR preservation.",
            "regularization": "Random low/high-pass perturbations to avoid timbre overfitting.",
        },
        "pitch_model": {
            "focus": "Retrain melody detector on augmented mixtures with octave-shifted bass and harmonic masking guidance.",
            "negatives": "Include hard-negative distractors where bass overlaps lead range.",
        },
    },
    benchmark={
        "level": "L2 poly-dominant",
        "command": "python -m backend.benchmarks.benchmark_runner --level L2",
        "metrics": "Track melody F1/onset MAE and detector coverage after each curriculum phase.",
    },
)
