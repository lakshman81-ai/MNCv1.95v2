from typing import Dict, Any, Optional
import io
import music21
import tempfile
import os
import shutil
import librosa
import numpy as np

from .pipeline.stage_a import load_and_preprocess
from .pipeline.stage_b import extract_features
from .pipeline.stage_c import apply_theory
from .pipeline.stage_d import quantize_and_render
from .pipeline.models import AnalysisData, TranscriptionResult, MetaData, AudioType

def transcribe_audio_pipeline(
    audio_path: str,
    *,
    stereo_mode: Optional[str] = None, # ignored
    use_mock: bool = False,
    start_offset: Optional[float] = None, # ignored
    max_duration: Optional[float] = None, # ignored
    use_crepe: bool = False,
    trim_silence: bool = True,
    mode: str = "quality", # "quality" or "fast"
    target_sample_rate: Optional[int] = None,
    window_size: Optional[int] = None,
    hop_length: Optional[int] = None,
    silence_top_db: Optional[float] = None,
    **kwargs,
) -> TranscriptionResult:
    """
    High-level API used by both FastAPI and the benchmark script.
    Orchestrates Stages A -> B -> C -> D.
    """
    if use_mock:
        # Legacy mock behavior for testing
        mock_xml_path = os.path.join(os.path.dirname(__file__), "mock_data", "happy_birthday.xml")
        if os.path.exists(mock_xml_path):
            with open(mock_xml_path, 'r', encoding='utf-8') as f:
                musicxml_str = f.read()
        else:
            musicxml_str = "<?xml version='1.0' encoding='utf-8'?><score-partwise><part><measure><note><rest/></note></measure></part></score-partwise>"

        return TranscriptionResult(
            musicxml=musicxml_str,
            analysis_data=AnalysisData(meta=MetaData(sample_rate=22050)),
            midi_bytes=b""
        )

    # Build a configurable pipeline config so callers can control the audio front end.
    from .pipeline.config import PipelineConfig

    pipeline_conf = PipelineConfig()

    # Stage A (audio front end)
    if target_sample_rate is not None:
        pipeline_conf.stage_a.target_sample_rate = int(target_sample_rate)

    pipeline_conf.stage_a.silence_trimming["enabled"] = bool(trim_silence)
    if silence_top_db is not None:
        pipeline_conf.stage_a.silence_trimming["top_db"] = float(silence_top_db)

    # Allow explicit front-end window/hop settings by propagating them to the detector
    # configs Stage A reads for frame sizing.
    if window_size is not None:
        pipeline_conf.stage_b.detectors["yin"]["frame_length"] = int(window_size)
        pipeline_conf.stage_b.detectors["swiftf0"]["n_fft"] = int(window_size)

    if hop_length is not None:
        pipeline_conf.stage_b.detectors["yin"]["hop_length"] = int(hop_length)
        pipeline_conf.stage_b.detectors["swiftf0"]["hop_length"] = int(hop_length)
        # Keep RMVPE hop aligned if enabled by callers
        pipeline_conf.stage_b.detectors["rmvpe"]["hop_length"] = int(hop_length)

    # 1. Stage A: Load and Preprocess (with Source Separation)
    # Returns StageAOutput
    stage_a_out = load_and_preprocess(audio_path, config=pipeline_conf)

    meta = stage_a_out.meta
    meta.processing_mode = mode

    # 2. Stage B: Extract Features (Segmentation)
    # Pitch tracking (SwiftF0/SACF) and Hysteresis segmentation
    # Now passing full StageAOutput to support stems
    conf_thresh = kwargs.get("confidence_threshold", 0.5)
    min_dur = kwargs.get("min_duration_ms", 0.0)

    stage_b_out = extract_features(
        stage_a_out,
        use_crepe=use_crepe,
        confidence_threshold=conf_thresh,
        min_duration_ms=min_dur
    )

    # Unpack from StageBOutput
    timeline = [] # Global timeline computed from main f0?
    # StageBOutput has f0_main, but timeline expects FramePitch objects.
    # stem_timelines is available.
    stem_timelines = stage_b_out.stem_timelines

    # We need a global timeline for AnalysisData.
    # In my previous implementation, I aggregated it.
    # Now I should aggregate it again or use what's available.
    if "vocals" in stem_timelines:
        timeline = stem_timelines["vocals"]
    elif "mix" in stem_timelines:
        timeline = stem_timelines["mix"]
    elif "other" in stem_timelines:
        timeline = stem_timelines["other"]

    notes = [] # Stage B doesn't produce notes yet (Stage C does)
    chords = [] # Stage B doesn't produce chords yet

    tracker_name = "swiftf0+sacf"
    print(f"Notes extracted using: {tracker_name}")

    # 3. Beat Tracking & Onsets (Stage B/Prep)
    # We need to extract beats and onsets for Stage C.

    beat_stem_audio = None
    if "drums" in stage_a_out.stems:
        beat_stem_audio = stage_a_out.stems["drums"].audio
    elif "other" in stage_a_out.stems: # If no drums (e.g. just other?)
        beat_stem_audio = stage_a_out.stems["other"].audio
    elif "vocals" in stage_a_out.stems: # Mono mix
        beat_stem_audio = stage_a_out.stems["vocals"].audio

    onsets = [] # Global fallback
    beats = [] # Global beats
    tempo = 120.0

    if beat_stem_audio is not None and len(beat_stem_audio) > 0:
        # Beat Track
        try:
            tempo_est, beat_frames = librosa.beat.beat_track(y=beat_stem_audio, sr=stage_a_out.meta.target_sr)
            tempo = float(tempo_est)
            beats = librosa.frames_to_time(beat_frames, sr=stage_a_out.meta.target_sr).tolist()
        except Exception as e:
            print(f"Beat tracking failed: {e}")

        # Global Onset Detect (fallback)
        try:
            onsets = sorted(list(librosa.onset.onset_detect(y=beat_stem_audio, sr=stage_a_out.meta.target_sr, units='time')))
        except:
            pass

    meta.tempo_bpm = tempo

    # Populate stem_onsets
    stem_onsets = {}
    for s_name in ["vocals", "bass", "other"]:
        if s_name in stage_a_out.stems:
             try:
                 y_s = stage_a_out.stems[s_name].audio
                 if len(y_s) > 0:
                     ons = librosa.onset.onset_detect(y=y_s, sr=stage_a_out.meta.target_sr, units='time')
                     stem_onsets[s_name] = sorted(list(ons))
                 else:
                     stem_onsets[s_name] = []
             except:
                 stem_onsets[s_name] = []

    # 3. Build AnalysisData
    analysis_data = AnalysisData(
        meta=meta,
        timeline=timeline,
        stem_timelines=stem_timelines,
        stem_onsets=stem_onsets,
        events=notes,
        chords=chords,
        notes=notes,
        pitch_tracker=tracker_name,
        n_frames=len(timeline),
        frame_hop_seconds=float(meta.hop_length) / float(meta.target_sr),
        onsets=onsets,
        beats=beats # Add beats
    )

    # Store pre-quantization notes
    from dataclasses import replace
    analysis_data.notes_before_quantization = [replace(n) for n in notes]

    # 4. Stage C: Apply Theory
    events_with_theory = apply_theory(notes, analysis_data)

    # 5. Stage D: Quantize and Render
    try:
        musicxml_str = quantize_and_render(events_with_theory, analysis_data)
    except Exception as e:
        print(f"Stage D (Rendering) failed: {e}. Returning placeholder XML.")
        musicxml_str = "<?xml version='1.0' encoding='utf-8'?><score-partwise><part><measure><note><rest/></note></measure></part></score-partwise>"

    # 6. Generate MIDI bytes
    midi_bytes = b""
    tmp_xml_path = None
    tmp_midi_path = None

    try:
        # Save XML to temp file to parse back for MIDI generation
        # (music21 allows parsing string, but temp file is safer for some backends)
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False, mode='w', encoding='utf-8') as tmp_xml:
            tmp_xml.write(musicxml_str)
            tmp_xml_path = tmp_xml.name

        try:
            s = music21.converter.parse(tmp_xml_path)

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
                tmp_midi_path = tmp_midi.name

            s.write('midi', fp=tmp_midi_path)

            with open(tmp_midi_path, 'rb') as f:
                midi_bytes = f.read()

        except Exception as e:
            print(f"MIDI generation failed: {e}")
            midi_bytes = b""

    except Exception as e:
        print(f"Failed to generate MIDI wrapper: {e}")
    finally:
        if tmp_xml_path and os.path.exists(tmp_xml_path):
            os.remove(tmp_xml_path)
        if tmp_midi_path and os.path.exists(tmp_midi_path):
            os.remove(tmp_midi_path)

    result = TranscriptionResult(
        musicxml=musicxml_str,
        analysis_data=analysis_data,
        midi_bytes=midi_bytes
    )

    return result

# wrapper for legacy calls
def transcribe_audio(
    file_path: str,
    use_mock: bool = False,
    stereo_mode: bool = False,
) -> str:
    """
    Legacy entry point.
    """
    res = transcribe_audio_pipeline(file_path, use_mock=use_mock, stereo_mode=str(stereo_mode))
    return res.musicxml
