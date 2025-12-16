from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


# ------------------------------------------------------------
# Stage A Config (Signal Conditioning)
# ------------------------------------------------------------

@dataclass
class StageAConfig:
    target_sample_rate: int = 44100
    channel_handling: str = "mono_sum"  # "mono_sum", "left_only", "right_only"
    dc_offset_removal: bool = True

    # Transient emphasis (hammer clicks)
    transient_pre_emphasis: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "alpha": 0.97}
    )

    # High-pass filter (protect C2 ≈ 65.4 Hz)
    high_pass_filter_cutoff: Dict[str, Any] = field(
        default_factory=lambda: {"value": 55.0}
    )
    high_pass_filter_order: Dict[str, Any] = field(
        default_factory=lambda: {"value": 4}
    )

    # Silence trimming (keep decay tails)
    silence_trimming: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "top_db": 50}
    )

    # Loudness normalization (EBU R128)
    loudness_normalization: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "target_lufs": -23.0}
    )

    # Peak limiter (Soft clip or -1 dB ceiling)
    peak_limiter: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,          # set True to use it
            "mode": "soft",            # "soft" | "hard"
            "ceiling_db": -1.0,        # max peak level
        }
    )

    # Noise floor estimation (percentile of RMS)
    noise_floor_estimation: Dict[str, Any] = field(
        default_factory=lambda: {"percentile": 30}
    )

    # BPM / beat grid detection
    bpm_detection: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True}
    )


# ------------------------------------------------------------
# Stage B Config (Detectors + Ensemble + ISS)
# ------------------------------------------------------------

@dataclass
class StageBConfig:
    # Source separation (HTDemucs)
    separation: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "model": "htdemucs",
            # If True, prefer the synthetic model fine-tuned on procedurally
            # generated L2 sine/saw/square/FM stems. Falls back to the
            # configured "model" when unavailable.
            "synthetic_model": False,
            "overlap": 0.25,  # Demucs overlap
            "shifts": 1,      # number of shifts (test-time augmentation)
            # Optional harmonic masking guided by a fast F0 prior
            "harmonic_masking": {
                "enabled": True,
                "mask_width": 0.02,
                "n_harmonics": 12,
            },
        }
    )

    # Global voicing threshold for ensemble F0
    confidence_voicing_threshold: float = 0.58

    # Additional relaxation applied when polyphonic context is detected
    polyphonic_voicing_relaxation: float = 0.07

    # SwiftF0 priority floor
    confidence_priority_floor: float = 0.5

    # Cross-detector disagreement tolerance (cents)
    pitch_disagreement_cents: float = 45.0

    # Ensemble weights (WI-aligned core)
    #   Piano: SwiftF0 dominates, SACF/CQT support.
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "swiftf0": 0.5,
            "sacf": 0.3,
            "cqt": 0.2,
            "yin": 0.3,
            "rmvpe": 0.5,   # Dominant for vocals / cello profiles
            "crepe": 0.5,   # Dominant for violin / flute profiles
            "fcpe": 0.4,    # FCPE and Supertone cover expressive L2 ladders
            "supertone": 0.4,
        }
    )

    # Tracker fusion strategy (especially for L2 ladder examples)
    tracker_fusion: Dict[str, Any] = field(
        default_factory=lambda: {
            "strategy": "confidence_vote",  # "confidence_vote" | "harmonicity_gate"
            "harmonicity_gate": {
                "enabled": True,
                "threshold": 0.08,
                "bandwidth": 0.04,
                "max_harmonics": 4,
            },
            "use_stability_fallback": True,
            "stability_window": 5,
        }
    )

    # Post-filtering after ensemble merge
    post_filters: Dict[str, Any] = field(
        default_factory=lambda: {
            "adaptive_median": {
                "enabled": True,
                "min_window": 3,
                "max_window": 7,
                "jitter_cents": 35.0,
            },
            "harmonic_salience": {
                "enabled": True,
                "threshold": 0.06,
                "bandwidth": 0.04,
                "max_harmonics": 4,
            },
            "octave_correction": {
                "enabled": True,
                "energy_margin": 1.1,
            },
        }
    )

    # Detector pools for L2 ladder tuning
    l2_tracker_pool: List[str] = field(
        default_factory=lambda: ["rmvpe", "fcpe", "supertone"]
    )

    # Polyphonic peeling (ISS) settings
    polyphonic_peeling: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_layers": 8,
            "mask_width": 0.03,  # Fractional bandwidth around harmonics
            "min_mask_width": 0.02,
            "max_mask_width": 0.08,
            "mask_growth": 1.1,
            "mask_shrink": 0.9,
            "harmonic_snr_stop_db": 3.0,
            "residual_rms_stop_ratio": 0.08,
            "residual_flatness_stop": 0.45,
            "validator_cents_tolerance": 50.0,
            "validator_agree_window": 5,
            "validator_disagree_decay": 0.6,
            "validator_min_agree_frames": 2,
            "validator_min_disagree_frames": 2,
            "max_harmonics": 12,
            "force_on_mix": True,
        }
    )

    # Voice tracking / skyline settings
    voice_tracking: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_alt_voices": 4,
            "max_jump_cents": 150.0,
            "hangover_frames": 2,
            "smoothing": 0.35,
            "confidence_bias": 5.0,
        }
    )

    # Global detector enable flags + defaults
    detectors: Dict[str, Any] = field(
        default_factory=lambda: {
            "rmvpe": {
                "enabled": False,
                "fmin": 50.0,
                "fmax": 1200.0,
                "hop_length": 160,
                "silence_threshold": 0.04,
            },
            "crepe": {
                "enabled": False,
                "model_capacity": "full",
                "fmin": 190.0,
                "fmax": 3500.0,
                "use_viterbi": False,
            },
            "swiftf0": {"enabled": True},
            "yin": {"enabled": True},
            "sacf": {"enabled": True},
            "cqt": {"enabled": True},
            "fcpe": {
                "enabled": False,
                "fmin": 50.0,
                "fmax": 2000.0,
                "hop_length": 160,
            },
            "supertone": {
                "enabled": False,
                "fmin": 40.0,
                "fmax": 1800.0,
                "hop_length": 160,
            },
        }
    )


# ------------------------------------------------------------
# Stage C Config (Note Segmentation)
# ------------------------------------------------------------

@dataclass
class StageCConfig:
    # Segmentation method selection + HMM defaults
    segmentation_method: Dict[str, Any] = field(
        default_factory=lambda: {
            "method": "hmm",  # "hmm" | "threshold" | "rms_gate"
            "states": ["attack", "sustain", "silence"],
        }
    )

    # Minimum note duration in milliseconds
    # WI: 30 ms for piano/guitar; captures fast grace notes/trills.
    min_note_duration_ms: float = 30.0

    # Polyphonic-specific minimum duration to suppress bass-induced flutter
    min_note_duration_ms_poly: float = 55.0

    # HMM frame stability (used in HMMProcessor)
    frame_stability: Dict[str, Any] = field(
        default_factory=lambda: {"stable_frames_required": 2}
    )

    # Pitch tolerance for merging (cents)
    pitch_tolerance_cents: float = 50.0

    # Gap filling (legato) in ms
    gap_filling: Dict[str, Any] = field(
        default_factory=lambda: {"max_gap_ms": 100.0}
    )

    # Confidence gates for polyphonic timelines (melody vs accompaniment)
    polyphonic_confidence: Dict[str, float] = field(
        default_factory=lambda: {"melody": 0.35, "accompaniment": 0.48}
    )

    # RMS → MIDI velocity mapping
    velocity_map: Dict[str, float] = field(
        default_factory=lambda: {
            "min_db": -40.0,
            "max_db": -4.0,
            "min_vel": 20.0,
            "max_vel": 105.0,
        }
    )

    # Polyphony filter mode ("skyline_top_voice" used as a hint to Stage D)
    polyphony_filter: Dict[str, str] = field(
        default_factory=lambda: {"mode": "skyline_top_voice"}
    )


# ------------------------------------------------------------
# Stage D Config (Rendering / MusicXML)
# ------------------------------------------------------------

@dataclass
class StageDConfig:
    # MusicXML divisions per quarter note (24 = 1/24 quarter)
    divisions_per_quarter: int = 24

    # Staff split point (MIDI pitch)
    staff_split_point: Dict[str, Any] = field(
        default_factory=lambda: {"pitch": 60}  # C4
    )

    # Staccato marking threshold (in beats)
    staccato_marking: Dict[str, Any] = field(
        default_factory=lambda: {"threshold_beats": 0.25}
    )

    # General glissando detection (disabled for piano by WI)
    glissando_threshold_general: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "min_semitones": 2.0,
            "max_time_ms": 500.0,
        }
    )

    # Piano-specific glissando handling (always discrete)
    glissando_handling_piano: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False}
    )


# ------------------------------------------------------------
# Instrument Profiles
# ------------------------------------------------------------

@dataclass
class InstrumentProfile:
    instrument: str
    recommended_algo: str
    fmin: float
    fmax: float
    # Arbitrary extra keys; Stage B currently uses:
    #   - "ensemble_smoothing_frames"
    #   - "viterbi"
    #   - "silence_threshold"
    special: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# Pipeline Config
# ------------------------------------------------------------

@dataclass
class PipelineConfig:
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b: StageBConfig = field(default_factory=StageBConfig)
    stage_c: StageCConfig = field(default_factory=StageCConfig)
    stage_d: StageDConfig = field(default_factory=StageDConfig)
    instrument_profiles: List[InstrumentProfile] = field(default_factory=list)

    def get_profile(self, instrument_name: str) -> Optional[InstrumentProfile]:
        """
        Robust instrument profile lookup with simple aliasing.
        """
        name = instrument_name.lower()

        # Simple aliases; extend as needed
        aliases = {
            "piano": "piano_61key",
            "keys": "piano_61key",
            "electric_guitar": "electric_guitar_clean",
            "electric-guitar": "electric_guitar_clean",
            "drums": "drums_percussive",
            "percussion": "drums_percussive",
        }
        canonical = aliases.get(name, name)

        for p in self.instrument_profiles:
            if p.instrument.lower() == canonical:
                return p
        return None


# ------------------------------------------------------------
# Default Instrument Profiles (WI-based)
# ------------------------------------------------------------

_profiles: List[InstrumentProfile] = [
    # 61-key piano (C2–C7), main target
    InstrumentProfile(
        instrument="piano_61key",
        recommended_algo="swiftf0",
        fmin=60.0,
        fmax=2200.0,
        special={
            # Piano: light ensemble smoothing per WI/master table
            "ensemble_smoothing_frames": 3,
            "viterbi": False,
        },
    ),

    # Vocals (singing) – RMVPE primary
    InstrumentProfile(
        instrument="vocals",
        recommended_algo="rmvpe",
        fmin=50.0,
        fmax=1200.0,
        special={
            "viterbi": True,
            "silence_threshold": 0.04,
        },
    ),

    # Violin – CREPE with Viterbi
    InstrumentProfile(
        instrument="violin",
        recommended_algo="crepe",
        fmin=190.0,
        fmax=3500.0,
        special={
            "viterbi": True,
        },
    ),

    # Bass Guitar – YIN, wide windows (low frequencies)
    InstrumentProfile(
        instrument="bass_guitar",
        recommended_algo="yin",
        fmin=30.0,
        fmax=400.0,
        special={
            "frame_length": 8192,
        },
    ),

    # Cello – RMVPE with smoothing
    InstrumentProfile(
        instrument="cello",
        recommended_algo="rmvpe",
        fmin=65.0,
        fmax=880.0,
        special={
            "viterbi": True,
            "silence_threshold": 0.04,
        },
    ),

    # Flute – CREPE, smaller windows (fast attacks)
    InstrumentProfile(
        instrument="flute",
        recommended_algo="crepe",
        fmin=261.0,
        fmax=3349.0,
        special={
            "small_window": True,
        },
    ),

    # Acoustic Guitar – SwiftF0, light smoothing
    InstrumentProfile(
        instrument="acoustic_guitar",
        recommended_algo="swiftf0",
        fmin=82.0,
        fmax=880.0,
        special={
            "ensemble_smoothing_frames": 3,
        },
    ),

    # Electric Guitar (clean) – YIN, low threshold
    InstrumentProfile(
        instrument="electric_guitar_clean",
        recommended_algo="yin",
        fmin=80.0,
        fmax=1200.0,
        special={
            "threshold": 0.05,
        },
    ),

    # Drums / percussive – no pitch
    InstrumentProfile(
        instrument="drums_percussive",
        recommended_algo="none",
        fmin=0.0,
        fmax=0.0,
        special={
            "ignore_pitch": True,
            "high_conf_threshold": 0.15,
        },
    ),
]


# ------------------------------------------------------------
# Default Pipeline Config instance
# ------------------------------------------------------------

PIANO_61KEY_CONFIG = PipelineConfig(
    instrument_profiles=_profiles
)
