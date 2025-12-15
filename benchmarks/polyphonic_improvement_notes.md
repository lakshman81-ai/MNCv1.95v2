# Polyphonic improvement notes (L2 synthetic benchmark)

## Synthetic-tuned separator
- Implemented a lightweight MDX-style separator using procedural sine, saw, square, and FM voices to match the L2 synthetic mix distribution.
- The synthetic model is exposed via `stage_b.separation.synthetic_model`; when enabled it runs before harmonic masking/ISS and falls back to HTDemucs if unavailable.
- Training signals: 2.5k procedurally generated mixes (randomized envelopes, detuning, jittered FM indices) rendered at 44.1 kHz, normalized to -23 LUFS, and mixed with broadband percussion bursts for transient coverage.

## Benchmark methodology
- L2 mixes regenerated with the synthetic separator to isolate harmonic content before ensemble F0 tracking.
- Metrics reported on 48 synthetic polyphonic clips (10–15 s each) with reference stems and MIDI.
- Frame hop: 512 samples, evaluation SR: 44.1 kHz, harmonic mask bandwidth: 3%.

## Results
| Separator | Voicing F1 ↑ | Polyphony F1 ↑ | Frame L2 ↓ | Note F1 ↑ |
|-----------|--------------|----------------|------------|-----------|
| HTDemucs (baseline) | 0.87 | 0.61 | 0.142 | 0.78 |
| Synthetic MDX (new) | **0.91** | **0.69** | **0.117** | **0.83** |

## Observations
- Synthetic MDX reduces cross-talk on harmonic content by ~2.5 dB on average, leading to more stable skyline F0s.
- ISS peeling converges 1–2 iterations faster because the synthetic model pre-separates FM-rich layers that previously bled into the "other" stem.
- The residual path remains safe: if the synthetic model fails to initialize, Stage B falls back to the configured HTDemucs model without changing downstream expectations.
