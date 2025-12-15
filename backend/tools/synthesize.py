import argparse
import music21
import numpy as np
import scipy.io.wavfile as wavfile
import os
import sys

def synthesize_xml(xml_path, output_path, sr=44100):
    print(f"Parsing {xml_path}...")
    try:
        score = music21.converter.parse(xml_path)
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return

    events = []

    # Use 120 BPM if we can't find one.
    # 120 BPM = 2 beats per second = 1 quarter note per 0.5 seconds.
    # seconds = quarterLength * (60 / BPM)
    current_bpm = 120.0

    # We iterate flat.notes but we also need to know tempo changes.
    # So we should iterate over all elements or just assume constant tempo for now.
    # For robust synthesis, let's use the offset to track time.

    flat_score = score.flatten()
    notes = flat_score.notes

    # Check for tempo marks
    tempos = flat_score.getElementsByClass(music21.tempo.MetronomeMark)
    if len(tempos) > 0:
        # Just take the first one for simplicity for now, or handle changes
        # music21 isn't giving us seconds, so we have to do it ourselves.
        current_bpm = tempos[0].getQuarterBPM()
        print(f"Found tempo: {current_bpm} BPM")
    else:
        print(f"No tempo found, using default {current_bpm} BPM")

    seconds_per_quarter = 60.0 / current_bpm

    max_end = 0.0

    print(f"Found {len(notes)} notes/chords.")

    for n in notes:
        try:
            # Calculate time from offsets (quarter notes)
            start_time = n.offset * seconds_per_quarter
            duration = n.duration.quarterLength * seconds_per_quarter
            end_time = start_time + duration

            if end_time > max_end:
                max_end = end_time

            freqs = []
            if isinstance(n, music21.note.Note):
                freqs.append(n.pitch.frequency)
            elif isinstance(n, music21.chord.Chord):
                for p in n.pitches:
                    freqs.append(p.frequency)

            if freqs:
                events.append({
                    'start': start_time,
                    'duration': duration,
                    'freqs': freqs
                })
        except Exception as e:
            print(f"Skipping element {n}: {e}")

    # Create audio buffer
    # Add 1s padding at the end
    num_samples = int(np.ceil(max_end * sr)) + sr
    audio = np.zeros(num_samples)

    print(f"Synthesizing {len(events)} events into {max_end:.2f}s audio...")

    for e in events:
        # Articulation: reduce duration slightly to create gaps (simulating key lift)
        # 90% of duration or minus 20ms, whichever is larger/safer
        actual_duration = max(0.01, e['duration'] - 0.02)
        start_idx = int(e['start'] * sr)
        dur_samples = int(actual_duration * sr)

        if dur_samples <= 0:
            continue

        t = np.linspace(0, actual_duration, dur_samples, endpoint=False)

        # Envelope (5ms attack, 20ms release) to avoid clicks
        attack_sec = 0.005
        release_sec = 0.02

        attack_samples = int(attack_sec * sr)
        release_samples = int(release_sec * sr)

        env = np.ones(dur_samples)

        # Apply attack
        if attack_samples > 0:
            count = min(attack_samples, dur_samples)
            env[:count] = np.linspace(0, 1, count)

        # Apply release
        if release_samples > 0 and dur_samples > attack_samples:
            count = min(release_samples, dur_samples - attack_samples)
            env[-count:] = np.linspace(1, 0, count)

        # Generate sound for each frequency
        note_audio = np.zeros(dur_samples)
        for f in e['freqs']:
            # Simple oscillator
            waveform = 0.6 * np.sin(2 * np.pi * f * t) + \
                       0.3 * np.sin(4 * np.pi * f * t) + \
                       0.1 * np.sin(6 * np.pi * f * t)
            note_audio += waveform

        # Mix into main buffer
        end_idx = start_idx + dur_samples
        if end_idx > len(audio):
            end_idx = len(audio)
            count = end_idx - start_idx
            note_audio = note_audio[:count]
            env = env[:count]

        audio[start_idx:end_idx] += note_audio * env

    # Normalize to -1..1
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95

    # Write
    wavfile.write(output_path, sr, (audio * 32767).astype(np.int16))
    print(f"Generated {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", help="Input MusicXML file")
    parser.add_argument("wav", help="Output WAV file")
    args = parser.parse_args()

    synthesize_xml(args.xml, args.wav)
