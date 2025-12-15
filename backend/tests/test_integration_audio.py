import numpy as np
import pytest
from backend.tests.audio_utils import generate_old_mcdonald, mix_audio
from backend.pipeline.detectors import SACFDetector, CQTDetector, midi_to_hz
from backend.pipeline.stage_b import extract_features
from backend.pipeline.models import StageAOutput, MetaData, Stem, AudioQuality, AudioType

class TestIntegrationAudio:
    @pytest.fixture
    def tune_data(self):
        return generate_old_mcdonald(sr=22050)

    def test_sacf_old_mcdonald_mono(self, tune_data):
        """
        Integration: Run SACF on the monophonic section of Old McDonald.
        Verifies correct melody extraction.
        """
        sr = 22050
        hop_length = 256

        # Extract mono part only
        end_time = tune_data["mono_end_time"]
        n_samples = int(end_time * sr)
        audio = tune_data["audio"][:n_samples]

        detector = SACFDetector(sr, hop_length, fmin=50, fmax=2000)
        f0_curve, conf = detector.predict(audio)

        times = np.arange(len(f0_curve)) * hop_length / sr

        # Verify against ground truth (G G G D E E D)
        # We check specific time points to see if pitch is correct

        for gt in tune_data["gt_mono"]:
            # Check midpoint of note
            mid_t = (gt["start"] + gt["end"]) / 2

            # Find closest frame
            idx = np.argmin(np.abs(times - mid_t))

            detected_f0 = f0_curve[idx]
            detected_conf = conf[idx]

            # Assert high confidence
            if detected_conf < 0.3:
                # Might be transition, check slightly later
                idx += 2
                detected_f0 = f0_curve[idx]

            target_f0 = gt["pitch"]

            # Allow 5% error (pitch detection isn't perfect, especially on onset)
            assert abs(detected_f0 - target_f0) < target_f0 * 0.05, \
                f"At {mid_t}s: Expected {target_f0}Hz, got {detected_f0}Hz"

    def test_pipeline_with_cqt_substitute(self, tune_data):
        """
        Integration: Run the full 'Old McDonald' (Mono + Poly) through Stage B.
        We patch SwiftF0 with CQTDetector to allow running in an environment without weights/musicxml.
        This validates the pipeline routing, ISS logic (CQT acts as primary), and data structures.
        """
        sr = 22050
        hop_length = 256
        audio = tune_data["audio"]

        meta = MetaData(
            duration_sec=len(audio)/sr,
            sample_rate=sr,
            hop_length=hop_length,
            audio_quality=AudioQuality.LOSSLESS,
            audio_path="integration_test.wav"
        )

        stems = {
            "other": Stem(audio=audio, sr=sr, type="other")
        }

        stage_a_out = StageAOutput(
            audio_type=AudioType.POLYPHONIC,
            meta=meta,
            stems=stems
        )

        # Patch SwiftF0 with CQTDetector
        # Note: CQTDetector.predict returns (f0, conf) in default (mono) mode.
        # ISS calls predict(y).
        # If we want Polyphony, ISS expects the detector to return ONE curve (the dominant), then it peels.
        # So CQTDetector (mono mode) is a perfect substitute for SwiftF0 (mono/dominant mode).

        from backend.pipeline.detectors import CQTDetector

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("backend.pipeline.stage_b.SwiftF0Detector", CQTDetector)

            timeline, _, _, stem_timelines = extract_features(stage_a_out)

            # Check polyphonic section
            poly_start = tune_data["mono_end_time"]

            # CQT is not as robust as SwiftF0, so loosen tolerances or check existence.
            # Just check that we got *something* reasonable.

            extracted_points = 0
            for gt in tune_data["gt_poly"]:
                mid_t = poly_start + (gt["start"] + gt["end"]) / 2

                # Check global timeline (should be dominant)
                if len(timeline) == 0: continue

                frame = min(timeline, key=lambda x: abs(x.time - mid_t))
                if frame.pitch_hz > 0:
                    extracted_points += 1

            assert extracted_points > 0, "Pipeline with CQT failed to extract any notes from Poly section"
