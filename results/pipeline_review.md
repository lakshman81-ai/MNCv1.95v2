# Pipeline review

## Step 1 – Work-process review and potential disconnects
- **Unimplemented BPM detection hook.** `StageAConfig` exposes `bpm_detection` settings, but the pipeline never calls a beat/BPM estimator. Downstream stages (e.g., Stage C quantization and Stage D rendering) therefore rely on the default 120 BPM instead of an analyzed tempo, which disconnects the configured workflow from the actual behavior.
- **Rhythmic metadata never refreshed.** Stage D renders tempo and time signature from `analysis_data.meta`, yet those fields are never updated after Stage A creates them with defaults. The lack of propagation means outputs always reflect the initial placeholder values, regardless of the audio content.

## Step 2 – Parameter passing and code quality observations
- **Grid quantization receives an AnalysisData object through a `tempo_bpm` parameter.** `apply_theory` forwards the entire `AnalysisData` instance into `quantize_notes` via the `tempo_bpm` argument. Although the function type-checks for this special case, the signature suggests a numeric BPM, which can be confusing for callers and obscures intent.
- **Stage B configuration is only partially leveraged in Stage A.** `load_and_preprocess` adjusts hop length and window size based on Stage B detector settings when given a full `PipelineConfig`, but supplying only a `StageAConfig` silently skips those detector-informed adjustments. That divergence can lead to mismatched analysis parameters between stages when callers omit the full pipeline config.
