import argparse
import os
import sys
import numpy as np
import music21
import copy
from unittest.mock import patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.pipeline.models import AudioType
from backend.transcription import transcribe_audio_pipeline
from backend.benchmarks.utils import generate_test_data, load_ground_truth, calculate_poly_metrics

def generate_poly_xml(in_path: str, out_path: str):
    """Creates a polyphonic XML by adding a transposed part."""
    try:
        s = music21.converter.parse(in_path)
        # Assuming single part input
        if len(s.parts) > 0:
            p1 = s.parts[0]
            # Create harmony: Major 3rd up
            p2 = copy.deepcopy(p1)
            p2.id = 'Harmony'
            p2.transpose(4, inPlace=True)
            s.insert(0, p2)
            s.write('musicxml', fp=out_path)
            print(f"[INFO] Generated Polyphonic XML: {out_path}")
        else:
            print("[WARN] No parts found in XML.")
    except Exception as e:
        print(f"[ERROR] Failed to generate poly XML: {e}")

def run_poly_test():
    base_dir = os.path.dirname(__file__)
    mono_xml = os.path.join(base_dir, "../mock_data/happy_birthday.xml")
    poly_xml = os.path.join(base_dir, "poly_test.musicxml")
    poly_wav = os.path.join(base_dir, "poly_test.wav")

    # 1. Generate Poly XML
    if not os.path.exists(poly_xml):
        generate_poly_xml(mono_xml, poly_xml)

    # 2. Synthesize
    generate_test_data(poly_xml, poly_wav)

    # 3. Load GT (Polyphonic)
    print("[INFO] Loading Polyphonic Ground Truth...")
    gt_frames, gt_voiced = load_ground_truth(poly_xml, polyphony=True, hop_length=256)

    # 4. Run Pipeline
    # Force "other" stem or ensure it's treated as polyphonic?
    # Mode="quality" detects polyphony via stage_a.detect_audio_type.
    # Our synthetic poly audio should be detected as POLYPHONIC.
    # If not, we might need to force it.
    # But let's trust the detector or set params?
    # transcribe_audio_pipeline doesn't let us force AudioType easily without mocking Stage A.
    # However, spectral flatness of a mix should trigger Polyphonic.

    print("\n--- STEP 3: POLYPHONIC ADAPTATION (ISS) ---")

    # Force Polyphonic detection to ensure Demucs + ISS path is taken
    with patch('backend.pipeline.stage_a.detect_audio_type', return_value=AudioType.POLYPHONIC):
        res = transcribe_audio_pipeline(
            poly_wav,
            mode="quality",
            confidence_threshold=0.3, # Lower for polyphony as requested
            min_duration_ms=50.0
        )

    # Check what type was detected
    print(f"[INFO] Detected Audio Type: {res.analysis_data.meta.audio_type}")

    # Get Global Timeline (or stem timelines)
    # The pipeline merges everything into `timeline`.
    # FramePitch.active_pitches contains the candidates.
    # For Mock testing, "vocals" and "bass" also find notes because Mock reads the full XML.
    # To strictly test ISS (which runs on "other"), we should evaluate the "other" stem.

    timeline = res.analysis_data.stem_timelines.get("other", [])
    if not timeline:
        print("[WARN] No 'other' stem results found. Using global timeline.")
        timeline = res.analysis_data.timeline

    m = calculate_poly_metrics(timeline, gt_frames, gt_voiced)

    print("[METRICS]")
    print(f"F1 Score: {m['F1 Score']:.3f}")
    print(f"Precision: {m['Precision']:.3f}")
    print(f"Recall:    {m['Recall']:.3f}")

    if m['F1 Score'] > 0.80:
        print("[SUCCESS] Polyphonic Adaptation Passed (F1 > 0.80)")
    else:
        print("[FAIL] Polyphonic F1 < 0.80")

if __name__ == "__main__":
    run_poly_test()
