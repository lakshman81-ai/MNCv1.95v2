import pytest
import numpy as np
import os
import soundfile as sf
from unittest.mock import MagicMock, patch
from backend.pipeline.stage_a import load_and_preprocess, TARGET_LUFS, SILENCE_THRESHOLD_DB

# Use a real file or create one for testing
@pytest.fixture
def temp_wav_file(tmp_path):
    path = tmp_path / "test_audio.wav"
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # A sine wave at -10dB
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), y, sr)
    return str(path)

@pytest.fixture
def silence_padded_wav_file(tmp_path):
    path = tmp_path / "silence_padded.wav"
    sr = 22050
    # 0.5s silence, 1s tone, 0.5s silence
    silence = np.zeros(int(0.5 * sr))
    t = np.linspace(0, 1.0, int(sr * 1.0))
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    y = np.concatenate([silence, tone, silence])
    sf.write(str(path), y, sr)
    return str(path)

def test_load_and_preprocess_success(temp_wav_file):
    stage_a_out = load_and_preprocess(temp_wav_file)
    # y, sr are now in stems['mix'] (Stage A returns 'mix' by default now)
    assert 'mix' in stage_a_out.stems
    y = stage_a_out.stems['mix'].audio
    sr = stage_a_out.stems['mix'].sr
    meta = stage_a_out.meta

    assert len(y) > 0
    assert meta.target_sr == 44100
    # Check normalization - allow flexibility if fallback used
    # The default target is -23 LUFS.
    assert np.isclose(meta.lufs, TARGET_LUFS, atol=2.0)

def test_load_fallback_soundfile(tmp_path):
    # Create a file that librosa might fail on if forced (mocking librosa fail)
    path = tmp_path / "fallback.wav"
    sf.write(str(path), np.random.uniform(-0.1, 0.1, 22050), 22050)

    # Patch librosa to be None or fail
    with patch('backend.pipeline.stage_a.librosa', None):
         stage_a_out = load_and_preprocess(str(path))
         y = stage_a_out.stems['mix'].audio
         meta = stage_a_out.meta
         assert len(y) > 0
         assert meta.audio_path == str(path)

def test_silence_trimming(silence_padded_wav_file):
    # The file has 0.5s silence at start/end.
    # trimming should remove most of it.
    stage_a_out = load_and_preprocess(silence_padded_wav_file)
    meta = stage_a_out.meta

    # Original length was 2.0s. Trimmed should be around 1.0s.
    # Allowing some margin for transitions
    assert meta.duration_sec < 1.8
    assert meta.duration_sec > 0.8

def test_audio_too_short(tmp_path):
    path = tmp_path / "short.wav"
    # Create 0.0s audio
    sf.write(str(path), np.array([], dtype=np.float32), 22050)

    # My implementation might return valid empty audio or raise.
    # If I want to enforce raising, I should add it to Stage A.
    # For now, let's just check it doesn't crash?
    # Or strict compliance: "Stage A must validate input".
    # I'll update Stage A to raise if empty.
    with pytest.raises(Exception): # ValueError or RuntimeError
        load_and_preprocess(str(path))

def test_file_not_found():
    # Expect RuntimeError or FileNotFoundError
    with pytest.raises((RuntimeError, FileNotFoundError)):
        load_and_preprocess("non_existent.wav")
