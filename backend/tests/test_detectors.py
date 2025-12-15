import numpy as np
import pytest
from backend.pipeline.detectors import SACFDetector, CQTDetector, YinDetector, midi_to_hz
from backend.tests.audio_utils import generate_sine_wave, generate_silence, generate_noise

class TestDetectors:
    @pytest.fixture
    def sr(self):
        return 22050

    @pytest.fixture
    def hop_length(self):
        return 256

    def test_sacf_detector_simple_sine(self, sr, hop_length):
        """Test if SACF correctly identifies a pure sine wave (440Hz)."""
        f0_target = 440.0
        duration = 1.0
        audio = generate_sine_wave(f0_target, duration, sr)

        detector = SACFDetector(sr, hop_length, fmin=60, fmax=2000)
        f0_curve, conf_curve = detector.predict(audio)

        # Check median pitch (ignoring onset/offset transients)
        # Filter out low confidence
        valid_indices = conf_curve > 0.5
        assert np.sum(valid_indices) > 0, "Detector found no valid pitch segments"

        median_f0 = np.median(f0_curve[valid_indices])
        assert abs(median_f0 - f0_target) < 5.0, f"Expected {f0_target}, got {median_f0}"

    def test_sacf_detector_silence(self, sr, hop_length):
        """Test SACF response to silence."""
        audio = generate_silence(1.0, sr)
        detector = SACFDetector(sr, hop_length, fmin=60, fmax=2000)
        f0_curve, conf_curve = detector.predict(audio)

        # Should be mostly 0 confidence or 0 pitch
        assert np.mean(conf_curve) < 0.1

    def test_cqt_detector_polyphony(self, sr, hop_length):
        """Test CQT Detector in polyphonic mode with a major third (A4 + C#5)."""
        f1 = 440.0 # A4
        f2 = 554.37 # C#5

        s1 = generate_sine_wave(f1, 1.0, sr)
        s2 = generate_sine_wave(f2, 1.0, sr)
        audio = s1 + s2

        detector = CQTDetector(sr, hop_length, fmin=60, fmax=2000)
        pitches_list, confs_list = detector.predict(audio, polyphony=True)

        # Check a frame in the middle
        mid_idx = len(pitches_list) // 2
        frame_pitches = pitches_list[mid_idx]

        # Verify both pitches are present
        found_f1 = any(abs(p - f1) < 10.0 for p in frame_pitches)
        found_f2 = any(abs(p - f2) < 10.0 for p in frame_pitches)

        assert found_f1, f"Failed to find {f1}Hz in CQT output: {frame_pitches}"
        assert found_f2, f"Failed to find {f2}Hz in CQT output: {frame_pitches}"

    def test_yin_detector_sine(self, sr, hop_length):
        """Test Yin Detector (PyIn wrapper)."""
        f0_target = 440.0
        audio = generate_sine_wave(f0_target, 0.5, sr) # Short clip for speed

        detector = YinDetector(sr, hop_length, fmin=60, fmax=2000)
        f0_curve, conf_curve = detector.predict(audio)

        valid_indices = conf_curve > 0.5 # PyIn uses 0-1 prob
        if np.sum(valid_indices) > 0:
            median_f0 = np.median(f0_curve[valid_indices])
            assert abs(median_f0 - f0_target) < 5.0

    def test_sacf_validate_curve(self, sr, hop_length):
        """Test SACF validate_curve method."""
        f0 = 440.0
        audio = generate_sine_wave(f0, 1.0, sr)
        detector = SACFDetector(sr, hop_length, fmin=60, fmax=2000)

        # Create a perfect curve
        n_frames = (len(audio) - 2048) // hop_length + 1
        perfect_curve = np.full(n_frames, f0)

        score = detector.validate_curve(perfect_curve, audio)
        assert score > 0.5, f"Validation score too low for perfect match: {score}"

        # Create a wrong curve (octave off is common, but let's try random)
        wrong_curve = np.full(n_frames, 300.0)
        score_wrong = detector.validate_curve(wrong_curve, audio)
        assert score_wrong < score, "Wrong curve should have lower score than correct curve"
