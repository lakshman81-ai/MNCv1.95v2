import pytest
import numpy as np
import os
import scipy.signal
import torch
from unittest.mock import MagicMock, patch
from backend.pipeline.stage_a import load_and_preprocess, detect_audio_type, warped_linear_prediction, AudioType, StageAOutput

@pytest.fixture
def mock_audio_file(tmp_path):
    # Create a dummy audio file
    path = tmp_path / "test.wav"
    sr = 44100
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr)) # 1 sec sine
    import soundfile as sf
    sf.write(str(path), y, sr)
    return str(path), y, sr

@patch("backend.pipeline.stage_a.detect_audio_type")
@patch("backend.pipeline.stage_a.apply_model")
@patch("backend.pipeline.stage_a.pretrained.get_model")
def test_stage_a_monophonic_flow(mock_get_model, mock_apply_model, mock_detect, mock_audio_file):
    """
    Test that IF detected as Monophonic, it skips Demucs.
    We mock detect_audio_type to ensure we test the PATH, not the heuristic.
    """
    audio_path, y_orig, sr_orig = mock_audio_file

    # Force Monophonic detection
    mock_detect.return_value = AudioType.MONOPHONIC

    # Setup mock to ensure we don't actually load the heavy model if it were called
    mock_model = MagicMock()
    mock_model.samplerate = 44100
    mock_get_model.return_value = mock_model

    result = load_and_preprocess(audio_path, target_sr=22050)

    assert isinstance(result, StageAOutput)
    assert result.audio_type == AudioType.MONOPHONIC
    assert "vocals" in result.stems
    # Should be resampled to 16000
    assert result.stems["vocals"].sr == 16000

    # verify Demucs was NOT called
    mock_apply_model.assert_not_called()

@patch("backend.pipeline.stage_a.apply_model")
@patch("backend.pipeline.stage_a.pretrained.get_model")
def test_stage_a_polyphonic_flow(mock_get_model, mock_apply_model, mock_audio_file):
    """
    Test that a Polyphonic signal triggers Demucs and returns multiple stems.
    """
    # Create noisy audio (Polyphonic-ish)
    import soundfile as sf
    path = "poly_test.wav"
    sr = 44100
    # Mix sine + noise
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr)) + 0.5 * np.random.normal(size=sr)
    sf.write(path, y, sr)

    # Mock Demucs output
    # Demucs output is (Sources, Channels, Time)
    # 4 sources: vocals, drums, bass, other
    # Stereo output usually
    # Return shape: (1, 4, 2, N)
    mock_demucs_out = torch.zeros((1, 4, 2, sr))
    mock_apply_model.return_value = mock_demucs_out

    mock_model = MagicMock()
    mock_model.samplerate = 44100
    mock_model.sources = ["vocals", "drums", "bass", "other"]
    mock_get_model.return_value = mock_model

    try:
        # We manually patch detect_audio_type to force polyphonic
        with patch("backend.pipeline.stage_a.detect_audio_type", return_value=AudioType.POLYPHONIC):
            result = load_and_preprocess(path, target_sr=22050)

            assert result.audio_type == AudioType.POLYPHONIC
            assert "vocals" in result.stems
            assert "bass" in result.stems
            assert "other" in result.stems

            # Check Sampling Rates
            assert result.stems["vocals"].sr == 16000
            assert result.stems["other"].sr == 44100

            # Demucs SHOULD be called
            mock_apply_model.assert_called_once()

    finally:
        if os.path.exists(path):
            os.remove(path)

def test_whitening():
    """Test LPC whitening logic."""
    sr = 44100
    # Create a signal with a strong spectral envelope (e.g., filtered noise)
    noise = np.random.normal(size=sr)
    b, a = scipy.signal.butter(2, 0.1)
    y_colored = scipy.signal.lfilter(b, a, noise)

    y_white = warped_linear_prediction(y_colored, sr)

    # Whitened signal should have flatter spectrum (lower spectral contrast?)
    # or at least not be identical
    assert not np.array_equal(y_white, y_colored)
    assert len(y_white) == len(y_colored)
