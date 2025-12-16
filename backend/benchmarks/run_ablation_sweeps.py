"""Small ablation sweeps for separator, tracker, and fusion strategies.

This script reuses the synthetic L2 fixture to keep runtime short while
exercising different Stage B frontends.  Results are written to CSV so
regressions or improvements are easy to spot in dashboards.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.benchmarks.benchmark_runner import BenchmarkSuite, run_pipeline_on_audio
from backend.pipeline.config import PipelineConfig
from backend.pipeline.models import AudioType


def _build_config(separator: str, tracker: str, fusion: str) -> PipelineConfig:
    config = PipelineConfig()

    # Separation toggle
    if separator == "none":
        config.stage_b.separation["enabled"] = False
    elif separator == "synthetic":
        config.stage_b.separation["enabled"] = True
        config.stage_b.separation["synthetic_model"] = True
        config.stage_b.separation["model"] = "synthetic_mdx"
    else:
        # lightweight fallback if an unknown string is passed
        config.stage_b.separation["enabled"] = False

    # Tracker choice
    for name in list(config.stage_b.detectors.keys()):
        config.stage_b.detectors[name]["enabled"] = False
    if tracker in config.stage_b.detectors:
        config.stage_b.detectors[tracker]["enabled"] = True
        if tracker == "crepe":
            config.stage_b.detectors[tracker]["model_capacity"] = "full"
    else:
        config.stage_b.detectors["swiftf0"]["enabled"] = True

    # Fusion / peeling strategy
    if fusion == "peeling":
        config.stage_b.polyphonic_peeling["max_layers"] = 2
    else:
        config.stage_b.polyphonic_peeling["max_layers"] = 0

    return config


def run_sweeps(output_dir: str = "results/ablation_sweeps") -> str:
    os.makedirs(output_dir, exist_ok=True)
    suite = BenchmarkSuite(output_dir)
    mix, sr, gt_melody, gt_stems, sections = suite._l2_fixture()

    separators = ["none", "synthetic"]
    trackers = ["swiftf0", "crepe", "rmvpe"]
    fusions = ["skyline", "peeling"]

    rows: List[Dict[str, str]] = []

    for sep in separators:
        for tracker in trackers:
            for fusion in fusions:
                config = _build_config(sep, tracker, fusion)
                res = run_pipeline_on_audio(
                    mix,
                    sr,
                    config,
                    AudioType.POLYPHONIC_DOMINANT,
                    extra_stems=gt_stems,
                    use_extra_stems_for_processing=False,
                )
                metrics = suite._compute_l2_metrics(res["stage_b_out"], res["notes"], gt_melody, gt_stems, sections)
                result = suite._save_run(
                    "L2",
                    f"ablate_sep-{sep}_trk-{tracker}_fusion-{fusion}",
                    res,
                    gt_melody,
                    extra_metrics=metrics,
                )
                rows.append({
                    "separator": sep,
                    "tracker": tracker,
                    "fusion": fusion,
                    "note_f1": f"{result.get('note_f1', float('nan')):.4f}",
                    "voicing_f1": f"{metrics.get('voicing_f1', float('nan'))}",
                    "octave_error_rate": f"{metrics.get('octave_error_rate', float('nan'))}",
                    "stem_si_sdr": str(metrics.get('stem_si_sdr', {})),
                })

    csv_path = os.path.join(output_dir, "ablation_sweeps.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


if __name__ == "__main__":
    output = run_sweeps()
    print(f"Ablation sweep results saved to {output}")
