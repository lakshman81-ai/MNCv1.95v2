# Requirements verification

## Integrate neural pitch/onset models for transcription
- **Status:** Not implemented.
- **Evidence:** The transcription pipeline still performs pitch tracking with the legacy SwiftF0/SACF ensemble and librosa-based beat/onset detection. Neural AMT front ends such as CREPE/SPICE with Onsets & Frames or encoder–decoder onset/offset models are not wired in.【F:backend/transcription.py†L33-L116】

## Make note-duration and merge thresholds tempo/adaptive
- **Status:** Not implemented.
- **Evidence:** Stage C keeps fixed segmentation thresholds (e.g., 30 ms minimum duration, 50-cent merge tolerance, 100 ms gap fill) without any tempo- or distribution-aware adjustment.【F:backend/pipeline/config.py†L171-L219】

## Extend rhythmic quantization and tempo handling
- **Status:** Not implemented.
- **Evidence:** Quantization snaps durations to a short list of denominators and drives a single global tempo value when rendering, with no support for compound meters or expressive tempo curves.【F:backend/pipeline/stage_d.py†L30-L209】【F:backend/pipeline/stage_d.py†L240-L254】

## Expose flexible audio front-end settings
- **Status:** Partially implemented.
- **Evidence:** The transcription entry point now accepts optional `target_sample_rate`, `window_size`, `hop_length`, and `silence_top_db` arguments that update the pipeline configuration before preprocessing, allowing callers to align audio I/O with downstream models. Defaults remain unchanged when not provided, so model-specific presets still need to be surfaced at the API layer.【F:backend/transcription.py†L15-L73】
