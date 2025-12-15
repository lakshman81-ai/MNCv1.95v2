import os
import json
import traceback
import numpy as np
from typing import Dict, Any, List

# Imports from our new modules
from .levels import BENCHMARK_LEVELS
from .generators import generate_benchmark_example
from .synth import midi_to_wav_synth
from .metrics.stage_a import calculate_stage_a_metrics
from .metrics.stage_b import calculate_stage_b_metrics
from .metrics.stage_c import calculate_stage_c_metrics
from .metrics.stage_d import calculate_stage_d_metrics

# Pipeline imports (assuming these exist and are importable)
# We need to mock or ensure the pipeline is callable.
# Based on memory, the pipeline structure is `backend/pipeline/`.
import sys
# Ensure backend is in path if running from root
if "backend" not in sys.path:
    sys.path.append("backend")

# Dynamic imports to avoid toplevel errors if dependencies missing during init
# But we need them for the run.

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def run_full_benchmark(config: Any, output_dir: str = "benchmark_results") -> Dict[str, Any]:
    """
    Runs the full benchmark ladder.
    config: PipelineConfig object.
    """
    from pipeline.stage_a import load_and_preprocess
    from pipeline.stage_b import extract_features
    # We use apply_theory for Stage C and quantize_and_render for Stage D based on read_file
    from pipeline.stage_c import apply_theory
    from pipeline.stage_d import quantize_and_render

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for level in BENCHMARK_LEVELS:
        level_id = level["id"]
        print(f"Running Level: {level_id}")
        level_results = []

        for example_id in level["examples"]:
            print(f"  Example: {example_id}")
            example_res = {"id": example_id, "errors": []}

            try:
                # 1. Generate Ground Truth (MIDI)
                # We need a fresh Score object
                score = generate_benchmark_example(example_id)
                midi_path = os.path.join(output_dir, f"{example_id}.mid")
                score.write("midi", fp=midi_path)

                # 2. Synthesize Audio (WAV)
                wav_path = os.path.join(output_dir, f"{example_id}.wav")
                midi_to_wav_synth(score, wav_path)

                # 3. Stage A
                # Import inside loop to handle reloading if needed? No.
                stage_a_out = load_and_preprocess(wav_path, config=config.stage_a)

                # Stage A Metrics
                ma = calculate_stage_a_metrics(
                    stage_a_out.stems["mix"].audio if "mix" in stage_a_out.stems else list(stage_a_out.stems.values())[0].audio,
                    stage_a_out.meta.sample_rate,
                    stage_a_out.meta
                )
                example_res["stage_a_metrics"] = ma

                # 4. Stage B
                stage_b_out = extract_features(stage_a_out, config=config)

                # Stage B Metrics
                mb = calculate_stage_b_metrics(stage_b_out, midi_path)
                example_res["stage_b_metrics"] = mb

                # 5. Stage C
                from pipeline.models import AnalysisData

                # Convert StageBOutput to AnalysisData for Stage C
                analysis_data = AnalysisData(
                    meta=stage_b_out.meta,
                    stem_timelines=stage_b_out.stem_timelines,
                    # We might need to populate other fields if Stage C needs them
                )

                import pipeline.stage_c as stage_c_module
                if hasattr(stage_c_module, "apply_theory"):
                     notes = stage_c_module.apply_theory(analysis_data, config=config)
                else:
                     raise ImportError("Could not find Stage C apply_theory")

                # Stage C Metrics
                mc = calculate_stage_c_metrics(notes, midi_path)
                example_res["stage_c_metrics"] = mc

                # 6. Stage D
                # Stage D quantize_and_render expects List[NoteEvent], AnalysisData, Config
                # analysis_data was updated by Stage C (notes added)

                import pipeline.stage_d as stage_d_module
                if hasattr(stage_d_module, "quantize_and_render"):
                    xml_path = os.path.join(output_dir, f"{example_id}_out.musicxml")
                    xml_content = stage_d_module.quantize_and_render(notes, analysis_data, config=config)
                    with open(xml_path, "w") as f:
                        f.write(xml_content)
                else:
                    raise ImportError("Could not find Stage D entry point")

                # Stage D Metrics
                md = calculate_stage_d_metrics(xml_path, midi_path)
                example_res["stage_d_metrics"] = md

            except Exception as e:
                print(f"    Error: {e}")
                traceback.print_exc()
                example_res["errors"].append(str(e))

            level_results.append(example_res)

        results[level_id] = level_results

    # Write summary
    with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    return results
