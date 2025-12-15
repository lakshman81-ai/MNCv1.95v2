"""
High-level transcription orchestrator.

Pipeline:
    Stage A: load_and_preprocess
    Stage B: extract_features
    Stage C: apply_theory
    Stage D: quantize_and_render

Contract (as used in tests/test_pipeline_flow.py):

    from backend.pipeline import transcribe

    result = transcribe(audio_path, config=PIANO_61KEY_CONFIG)

    assert isinstance(result.musicxml, str)
    analysis = result.analysis_data

    assert analysis.meta.sample_rate == sr_target
    assert analysis.meta.target_sr == sr_target
    assert len(analysis.notes) > 0
"""

from typing import Optional

from .config import PIANO_61KEY_CONFIG, PipelineConfig
from .stage_a import load_and_preprocess
from .stage_b import extract_features
from .stage_c import apply_theory
from .stage_d import quantize_and_render
from .models import AnalysisData, StageAOutput, TranscriptionResult


def transcribe(
    audio_path: str,
    config: Optional[PipelineConfig] = None,
) -> TranscriptionResult:
    """
    High-level API: run the full pipeline on an audio file.

    Parameters
    ----------
    audio_path : str
        Path to the input audio file (e.g., .wav, .mp3).
    config : PipelineConfig, optional
        Full pipeline configuration. If None, uses PIANO_61KEY_CONFIG.

    Returns
    -------
    TranscriptionResult
        Object with `.musicxml` (Stage D output) and `.analysis_data`
        (meta + stem timelines + notes).
    """
    if config is None:
        config = PIANO_61KEY_CONFIG

    # --------------------------------------------------------
    # Stage A: Signal Conditioning
    # --------------------------------------------------------
    # Update: Pass full config to allow Stage A to resolve detector-based params
    stage_a_out: StageAOutput = load_and_preprocess(
        audio_path,
        config=config,
    )

    # --------------------------------------------------------
    # Stage B: Feature Extraction (Detectors + Ensemble)
    # --------------------------------------------------------
    stage_b_out = extract_features(
        stage_a_out,
        config=config,
    )

    # --------------------------------------------------------
    # Stage C: Note Event Extraction (Theory Application)
    # --------------------------------------------------------
    analysis_data = AnalysisData(
        meta=stage_a_out.meta,
        timeline=[],
        stem_timelines=stage_b_out.stem_timelines,
    )

    notes = apply_theory(
        analysis_data,
        config=config,
    )

    # --------------------------------------------------------
    # Stage D: Quantization + MusicXML Rendering
    # --------------------------------------------------------
    d_out: TranscriptionResult = quantize_and_render(
        notes,
        analysis_data,
        config=config,
    )

    # d_out contains musicxml, analysis_data, and midi_bytes.
    # Ensure we return a TranscriptionResult with all fields.
    # analysis_data was updated in-place by quantize_and_render (it adds beats, etc.)
    # but d_out.analysis_data is safer to use.

    return TranscriptionResult(
        musicxml=d_out.musicxml,
        analysis_data=d_out.analysis_data,
        midi_bytes=d_out.midi_bytes
    )
