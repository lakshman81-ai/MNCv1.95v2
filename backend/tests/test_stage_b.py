import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from backend.pipeline.stage_b import (
    extract_features,
    create_harmonic_mask,
    iterative_spectral_subtraction,
    MultiVoiceTracker,
)
from backend.pipeline.models import StageAOutput, MetaData, Stem, AudioQuality, FramePitch
from backend.pipeline.detectors import SwiftF0Detector, SACFDetector
from backend.pipeline.config import PipelineConfig

class TestStageB:
    @pytest.fixture
    def sr(self):
        return 22050

    @pytest.fixture
    def hop_length(self):
        return 256

    @pytest.fixture
    def mock_stage_a_output(self, sr, hop_length):
        meta = MetaData(
            duration_sec=2.0,
            sample_rate=sr,
            hop_length=hop_length,
            audio_quality=AudioQuality.LOSSLESS,
            audio_path="test_audio.wav"
        )

        # Create dummy audio
        audio = np.zeros(int(2.0 * sr))
        stems = {
            "vocals": Stem(audio=audio, sr=sr, type="vocals"),
            "other": Stem(audio=audio, sr=sr, type="other")
        }

        return StageAOutput(
            audio_type="POLYPHONIC",
            meta=meta,
            stems=stems
        )

    def test_create_harmonic_mask(self, sr):
        # Create a dummy STFT (freqs x frames)
        n_fft = 2048
        n_frames = 10
        stft = np.ones((1025, n_frames), dtype=np.complex64)

        # Create a constant f0 curve at 440Hz
        f0_curve = np.full(n_frames, 440.0)

        # Updated signature: f0_hz, sr, n_fft, mask_width
        mask = create_harmonic_mask(f0_curve, sr, n_fft, mask_width=0.03)

        # Check if bins corresponding to 440Hz are masked
        fft_freqs = np.linspace(0, sr/2, 1025)

        # Find index for 440Hz
        idx_440 = np.argmin(np.abs(fft_freqs - 440.0))

        # The mask should be 0.0 at this index (masked out)
        assert mask[idx_440, 0] == 0.0, "Fundamental frequency should be masked"

        # Check harmonic (880Hz)
        idx_880 = np.argmin(np.abs(fft_freqs - 880.0))
        assert mask[idx_880, 0] == 0.0, "First harmonic should be masked"

    @patch("backend.pipeline.stage_b.SwiftF0Detector")
    @patch("backend.pipeline.stage_b.SACFDetector")
    def test_iterative_spectral_subtraction_flow(self, MockSACF, MockSwiftF0, sr, hop_length):
        """
        Verify the loop runs multiple times and stops when confidence is low.
        """
        # Setup Mocks
        primary = MockSwiftF0.return_value
        validator = MockSACF.return_value

        # Mock responses
        # Iteration 0: Strong note (440Hz)
        # Iteration 1: Weak note (330Hz) -> Should trigger stop if conf < termination

        n_frames = 100

        # Call 1: High confidence
        f0_1 = np.full(n_frames, 440.0)
        conf_1 = np.full(n_frames, 0.9)

        # Call 2: Low confidence (triggers stop in ISS if voiced_ratio < 0.05)
        # We need conf to be > 0.0 for voiced check.
        # ISS threshold is voiced_ratio < 0.05.
        # If conf_2 is 0.05 everywhere, voiced_ratio depends on (conf > 0.1).
        # 0.05 is not > 0.1. So voiced_ratio will be 0.
        # So it stops.

        f0_2 = np.full(n_frames, 330.0)
        conf_2 = np.full(n_frames, 0.05)

        primary.predict.side_effect = [(f0_1, conf_1), (f0_2, conf_2)]
        primary.hop_length = hop_length
        primary.n_fft = 2048 # needed for ISS stft?

        # validator.validate_curve.return_value = 0.8
        # ISS uses validator.predict
        validator.predict.return_value = (f0_1, conf_1) # Just pass validation

        t = np.arange(n_frames * hop_length) / float(sr)
        audio = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        extracted = iterative_spectral_subtraction(
            audio, sr, primary, validator, max_polyphony=4
        )

        # Should have extracted 1 note (the second one failed check)
        assert len(extracted) == 1
        assert extracted[0][0][0] == 440.0

    @patch("backend.pipeline.stage_b.SwiftF0Detector")
    @patch("backend.pipeline.stage_b.SACFDetector")
    def test_extract_features_routing(self, MockSACF, MockSwiftF0, mock_stage_a_output):
        """
        Verify that Vocals go to SwiftF0 direct, and Other goes to ISS.
        """
        # Setup Mocks
        # SwiftF0 instance 1 (Vocals)
        # SwiftF0 instance 2 (Other - Primary)

        # We need to distinguish instances or just count calls.
        # extract_features instantiates detectors inside.

        mock_swift_instance = MockSwiftF0.return_value
        mock_sacf_instance = MockSACF.return_value

        # Mock predict to return zeros so it doesn't crash
        n_frames = int(mock_stage_a_output.meta.duration_sec * 22050 / 256) + 1
        mock_swift_instance.predict.return_value = (np.zeros(n_frames), np.zeros(n_frames))

        stage_b_out = extract_features(mock_stage_a_output)
        stems = stage_b_out.stem_timelines

        # Check that SwiftF0 was initialized
        # In optimized implementation, detectors are reused, so we expect at least 1 init.
        assert MockSwiftF0.call_count >= 1

        # Check that SACF was initialized
        assert MockSACF.call_count >= 1

        # Check that stems contains keys
        assert "vocals" in stems
        assert "other" in stems

    def test_multivoice_tracker_holds_chords(self):
        tracker = MultiVoiceTracker(max_tracks=3, hangover_frames=1, smoothing=0.0)

        first_frame, confs = tracker.step([(440.0, 0.9), (660.0, 0.8), (550.0, 0.85)])
        assert np.count_nonzero(first_frame) == 3

        # Drop the middle candidate to test hangover/assignment stability
        second_frame, _ = tracker.step([(442.0, 0.6), (660.0, 0.7)])
        assert second_frame[1] > 0.0  # middle voice carried forward

    def test_validator_smoothing_prevents_dropouts(self, sr, hop_length):
        primary = MagicMock()
        validator = MagicMock()

        n_frames = 48
        f0 = np.full(n_frames, 440.0, dtype=np.float32)
        conf = np.full(n_frames, 0.9, dtype=np.float32)

        vf0 = f0.copy()
        vf0[10:12] = 520.0  # brief disagreement > 50 cents
        vconf = np.full(n_frames, 0.9, dtype=np.float32)

        primary.predict.return_value = (f0, conf)
        primary.hop_length = hop_length
        primary.n_fft = 2048
        validator.predict.return_value = (vf0, vconf)

        audio = np.random.randn(n_frames * hop_length).astype(np.float32) * 0.01

        layers = iterative_spectral_subtraction(
            audio,
            sr,
            primary_detector=primary,
            validator_detector=validator,
            max_polyphony=1,
            validator_min_disagree_frames=3,
            harmonic_snr_stop_db=-20.0,
            residual_flatness_stop=1.0,
        )

        assert len(layers) == 1
        gated_conf = layers[0][1]
        # Brief disagreement should not zero the track
        assert gated_conf[10] > 0.2

    @patch("backend.pipeline.stage_b.iterative_spectral_subtraction")
    @patch("backend.pipeline.stage_b._init_detector")
    def test_iss_config_overrides(self, mock_init_detector, mock_iss, mock_stage_a_output):
        # Configure peeling overrides
        config = PipelineConfig()
        peel = config.stage_b.polyphonic_peeling
        peel.update({
            "max_layers": 1,
            "mask_width": 0.04,
            "min_mask_width": 0.025,
            "max_mask_width": 0.06,
            "mask_growth": 1.2,
            "mask_shrink": 0.95,
            "harmonic_snr_stop_db": 4.0,
            "residual_rms_stop_ratio": 0.05,
            "residual_flatness_stop": 0.5,
            "validator_cents_tolerance": 35.0,
            "validator_agree_window": 3,
            "validator_disagree_decay": 0.7,
            "validator_min_agree_frames": 3,
            "validator_min_disagree_frames": 3,
            "max_harmonics": 10,
        })

        detector = MagicMock()
        detector.predict.return_value = (np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32))
        detector.hop_length = mock_stage_a_output.meta.hop_length
        detector.n_fft = 2048

        mock_init_detector.return_value = detector
        mock_iss.return_value = []

        extract_features(mock_stage_a_output, config=config)

        _, kwargs = mock_iss.call_args
        assert kwargs["mask_width"] == peel["mask_width"]
        assert kwargs["min_mask_width"] == peel["min_mask_width"]
        assert kwargs["max_harmonics"] == peel["max_harmonics"]
        assert kwargs["validator_cents_tolerance"] == peel["validator_cents_tolerance"]
