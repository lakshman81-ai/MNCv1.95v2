# Scenario notes: Melody plus chords

Lead melodies accompanied by chord progressions to measure polyphonic extraction quality and chord labeling.

## Curated fixture plan
Use paired audio + MusicXML/MIDI so we can score both melody F1 and chord labeling. Target at least two tempos/keys per instrument family.

| Fixture | Tempo / Key | Texture | Audio | Reference |
|---------|-------------|---------|-------|-----------|
| Acoustic guitar lead + strummed triads | 92 BPM / C Major | Melody on top string, open-chord strums beneath | `audio/acoustic_lead_cmaj_92bpm.wav` | `references/acoustic_lead_cmaj_92bpm.musicxml` |
| Nylon arpeggios with high-register hook | 70 BPM / D Major | Broken chords plus short melody bursts | `audio/nylon_hook_dmaj_70bpm.wav` | `references/nylon_hook_dmaj_70bpm.musicxml` |
| Piano ballad melody + block chords | 110 BPM / F Major | Sustained left-hand triads, lyrical RH line | `audio/piano_ballad_fmaj_110bpm.wav` | `references/piano_ballad_fmaj_110bpm.mid` |
| Piano pop riff + syncopated chords | 128 BPM / G Major | Off-beat chords, octave melody jumps | `audio/piano_riff_gmaj_128bpm.wav` | `references/piano_riff_gmaj_128bpm.musicxml` |

## Checklist
- [ ] Add the audio files listed above under `audio/` (16-bit WAV preferred)
- [ ] Add matching MIDI or MusicXML references under `references/` using the filenames above
- [ ] Run mock pipeline benchmark and log results in `results.md`
- [ ] Run full pipeline benchmark (when deps available) and log results in `results.md`
