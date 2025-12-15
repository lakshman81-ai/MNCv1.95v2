# Scenario notes: Scales

Focus on isolated major/minor scales and arpeggios to validate pitch detection and timing stability.

## Fixtures to include
- At least one slow (<=90bpm) and one fast (>=140bpm) major scale in C and G.
- Minor scale variants (natural, harmonic) with clean single-instrument recordings.
- Optional arpeggio patterns to probe onset detection.

## Checklist
- [ ] Add audio files under `audio/`
- [ ] Add matching MIDI or MusicXML references under `references/`
- [ ] Run mock pipeline benchmark and log results in `results.md`
- [ ] Run full pipeline benchmark (when deps available) and log results in `results.md`
