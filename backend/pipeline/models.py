"""Dataclasses and enums used by the pipeline (namespaced version).

This file duplicates the definitions from the top‑level ``models.py``
module so that tests importing ``backend.pipeline.models`` will find
all expected classes and enumerations.  The ``ProcessingMode`` enum
has been added to mirror the usage in Stage A.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal, Tuple
from enum import Enum
import numpy as np


class AudioType(str, Enum):
    MONOPHONIC = "monophonic"
    POLYPHONIC_DOMINANT = "polyphonic_dominant"
    POLYPHONIC = "polyphonic"


class AudioQuality(str, Enum):
    LOSSLESS = "lossless"  # WAV, FLAC, AIFF
    LOSSY = "lossy"        # MP3, M4A, OGG


class ProcessingMode(str, Enum):
    """
    Processing modes describe the overall texture of the input audio.

    These values are used by Stage A to indicate whether the audio is
    strictly monophonic, contains a dominant melodic line atop a
    polyphonic accompaniment, or is fully polyphonic with no dominant
    melody.
    """
    MONOPHONIC = "monophonic"
    POLYPHONIC_DOMINANT = "polyphonic_dominant"
    POLYPHONIC = "polyphonic"


@dataclass
class MetaData:
    # Tuning / music-theory info
    tuning_offset: float = 0.0              # in semitones (fractional)
    detected_key: str = "C"                 # e.g. "C", "Gm"

    # Loudness / processing
    lufs: float = -14.0                     # integrated loudness in LUFS
    processing_mode: str = "mono"           # "mono" | "stereo" | "polyphonic"
    audio_type: AudioType = AudioType.MONOPHONIC
    audio_quality: AudioQuality = AudioQuality.LOSSLESS
    snr: float = 0.0                        # signal-to-noise estimate

    # Time–frequency analysis parameters
    window_size: int = 2048                 # analysis window size
    hop_length: int = 512                   # analysis hop length
    sample_rate: int = 44100                # effective working SR
    tempo_bpm: Optional[float] = 120.0      # global tempo estimate (assumed default if not detected)
    time_signature: str = "4/4"             # default, can be refined

    # Original IO info
    original_sr: Optional[int] = None
    target_sr: int = 44100
    duration_sec: float = 0.0

    # Optional beat grid (seconds)
    beats: List[float] = field(default_factory=list)

    # Extended robust fields
    audio_path: Optional[str] = None
    n_channels: int = 1
    normalization_gain_db: float = 0.0
    rms_db: float = -float("inf")
    noise_floor_rms: float = 0.0            # from Stage A percentile RMS
    noise_floor_db: float = -80.0           # log version of noise_floor_rms
    pipeline_version: str = "2.0.0"


@dataclass
class Stem:
    audio: np.ndarray   # Monophonic (or summed) audio array
    sr: int
    type: str           # 'vocals', 'bass', 'other', 'drums', or 'mix'


@dataclass
class StageAOutput:
    stems: Dict[str, Stem]                  # keys: 'vocals', 'bass', 'other', 'drums', 'mix', etc.
    meta: MetaData
    audio_type: AudioType

    # Extra Stage A diagnostics / helpers
    noise_floor_rms: float = 0.0
    noise_floor_db: float = -80.0
    beats: List[float] = field(default_factory=list)


@dataclass
class StageBOutput:
    time_grid: np.ndarray                   # Array of time stamps
    f0_main: np.ndarray                     # Main pitch track (or skyline)
    f0_layers: List[np.ndarray]             # Polyphonic layers
    # per_detector[stem_name][det_name] = (f0_array, conf_array)
    per_detector: Dict[str, Any]
    stem_timelines: Dict[str, List["FramePitch"]] = field(default_factory=dict)
    stems: Dict[str, Stem] = field(default_factory=dict)
    meta: Optional[MetaData] = None         # Passed through from Stage A
    diagnostics: Dict[str, Any] = field(default_factory=dict)  # Separation/masking/ISS flags
    resolved_stems: Dict[str, Stem] = field(default_factory=dict)

    def __iter__(self):
        """Enable tuple-style unpacking for legacy call sites/tests.

        Historic callers expected ``(timeline, f0_main, f0_layers, stem_timelines)``.
        We derive ``timeline`` from the first available stem timeline (or an empty
        list) to maintain that behaviour while keeping the rich dataclass shape.
        """

        timeline = []
        if self.stem_timelines:
            # Prefer a deterministic stem if available.
            for key in ("mix", "vocals", "other"):
                if key in self.stem_timelines:
                    timeline = self.stem_timelines[key]
                    break
            else:
                timeline = next(iter(self.stem_timelines.values()))

        yield timeline
        yield self.f0_main
        yield self.f0_layers
        yield self.stem_timelines


@dataclass
class FramePitch:
    time: float                             # seconds
    pitch_hz: float                         # 0.0 if unvoiced
    midi: Optional[int]                     # None if unvoiced
    confidence: float                       # 0–1
    rms: float = 0.0                        # Frame RMS energy (linear)
    active_pitches: List[Tuple[float, float]] = field(
        default_factory=list
    )  # List of (pitch_hz, confidence)


@dataclass
class AlternativePitch:
    midi: int
    confidence: float


@dataclass
class NoteEvent:
    # Raw timing
    start_sec: float
    end_sec: float

    # Pitch
    midi_note: int
    pitch_hz: float
    confidence: float = 0.0

    # Performance-ish info
    velocity: float = 0.8                   # normalized 0.0–1.0, NOT MIDI 0–127
    rms_value: float = 0.0                  # Raw RMS energy (linear)
    is_grace: bool = False
    dynamic: str = "mf"                     # "p", "mf", "f", etc.
    voice: int = 1                          # Voice index
    staff: str = "treble"                   # "treble" or "bass"

    # Musical grid (filled after quantization)
    measure: Optional[int] = None
    beat: Optional[float] = None            # beat in measure (1.0, 1.5, etc.)
    duration_beats: Optional[float] = None

    # Extra info
    alternatives: List[AlternativePitch] = field(default_factory=list)
    spec_thumb: Optional[str] = None        # optional spectrogram thumbnail id


@dataclass
class ChordEvent:
    time: float                             # seconds
    beat: float                             # global beat index
    symbol: str                             # e.g. "C", "G7", "Am"
    root: str = "C"
    quality: str = "M"                      # "M", "m", "7", etc.


@dataclass
class VexflowLayout:
    measures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    pitch_accuracy_score: float = 0.0
    rhythm_accuracy_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisData:
    meta: MetaData = field(default_factory=MetaData)
    timeline: List[FramePitch] = field(default_factory=list)
    events: List[NoteEvent] = field(default_factory=list)
    chords: List[ChordEvent] = field(default_factory=list)
    vexflow_layout: VexflowLayout = field(default_factory=VexflowLayout)

    # Newer fields (multi-stem + note pipeline)
    notes: List[NoteEvent] = field(default_factory=list)
    stem_timelines: Dict[str, List[FramePitch]] = field(default_factory=dict)
    stem_onsets: Dict[str, List[float]] = field(default_factory=dict)
    onsets: List[float] = field(default_factory=list)
    beats: List[float] = field(default_factory=list)  # Beat timestamps (seconds)

    # Extended robust fields
    pitch_tracker: str = "pyin"           # "pyin" | "crepe" | "swiftf0" etc.
    n_frames: int = 0
    frame_hop_seconds: float = 0.0
    notes_before_quantization: List[NoteEvent] = field(default_factory=list)
    benchmark: Optional[BenchmarkResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-serializable representation for debugging / API.
        """
        notes_to_use = self.notes if self.notes else self.events

        return {
            "meta": asdict(self.meta),
            "timeline": [asdict(f) for f in self.timeline],
            "notes": [
                {
                    "start_sec": e.start_sec,
                    "end_sec": e.end_sec,
                    "midi_note": e.midi_note,
                    "pitch_hz": e.pitch_hz,
                    "confidence": e.confidence,
                    "velocity": e.velocity,
                    "rms_value": e.rms_value,
                    "is_grace": e.is_grace,
                    "dynamic": e.dynamic,
                    "voice": e.voice,
                    "staff": e.staff,
                    "measure": e.measure,
                    "beat": e.beat,
                    "duration_beats": e.duration_beats,
                    "alternatives": [asdict(a) for a in e.alternatives],
                    "spec_thumb": e.spec_thumb,
                }
                for e in notes_to_use
            ],
            "chords": [
                {
                    "time": c.time,
                    "beat": c.beat,
                    "symbol": c.symbol,
                    "root": c.root,
                    "quality": c.quality,
                }
                for c in self.chords
            ],
            "vexflow_layout": self.vexflow_layout.measures,
            "beats": self.beats,
            # Extended / diagnostic fields
            "stem_timelines": {
                stem: [asdict(f) for f in frames]
                for stem, frames in self.stem_timelines.items()
            },
            "stem_onsets": self.stem_onsets,
            "onsets": self.onsets,
            "pitch_tracker": self.pitch_tracker,
            "n_frames": self.n_frames,
            "frame_hop_seconds": self.frame_hop_seconds,
            "notes_before_quantization": [
                asdict(e) for e in self.notes_before_quantization
            ],
            "benchmark": asdict(self.benchmark) if self.benchmark else None,
        }


@dataclass
class TranscriptionResult:
    musicxml: str
    analysis_data: AnalysisData
    midi_bytes: bytes = b""

    def __getitem__(self, key):
        """Allow dict-like access for compatibility."""
        # Direct attributes on this object
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
