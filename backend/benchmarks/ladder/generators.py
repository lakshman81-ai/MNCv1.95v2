from music21 import stream, note, meter, key, tempo, chord, dynamics
import random
import copy

def create_happy_birthday_base():
    """Generates the base monophonic Happy Birthday theme in C Major."""
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=100))
    p.append(key.Key('C'))
    p.append(meter.TimeSignature('3/4'))

    # (pitch, quarter_length)
    melody_data = [
        # Upbeat? No, usually starts on beat 3. But for simplicity let's start on 1 or pickup.
        # Standard: G4(0.75), G4(0.25) -> A4(1) ...
        # The prompt provided a simplified version:
        # ("G4", 1), ("G4", 1), ("A4", 2) -> This is 4/4 timing disguised as 3/4 or just slow?
        # Prompt said: 3/4 time.
        # ("G4", 1), ("G4", 1), ("A4", 2) -> That's 4 beats. 3/4 has 3 beats.
        # The prompt's melody definition:
        # ("G4", 1), ("G4", 1), ("A4", 2), ("G4", 2), ("C5", 2), ("B4", 3)
        # Total: 1+1+2 = 4 beats? No, wait.
        # If it's 3/4, maybe these are eighths?
        # Let's stick to the prompt's provided list but ensure it fits measures if possible.
        # Prompt: ("G4", 1), ("G4", 1), ("A4", 2) -> 4 quarters. In 3/4, that's 1 bar + 1 beat.
        # Let's just use the prompt's sequence exactly as given, it might be loosely timed.

        # Phrase 1
        ("G4", 1), ("G4", 1), ("A4", 2),
        ("G4", 2), ("C5", 2), ("B4", 3),

        # Phrase 2
        ("G4", 1), ("G4", 1), ("A4", 2),
        ("G4", 2), ("D5", 2), ("C5", 3),

        # Phrase 3
        ("G4", 1), ("G4", 1), ("G5", 2),
        ("E5", 2), ("C5", 2), ("B4", 2), ("A4", 3),

        # Phrase 4
        ("F5", 1), ("F5", 1), ("E5", 2),
        ("C5", 2), ("D5", 2), ("C5", 3),
    ]

    for pitch_name, dur in melody_data:
        n = note.Note(pitch_name)
        n.quarterLength = dur
        p.append(n)

    s.append(p)
    return s

def create_old_macdonald_base():
    """Generates the base monophonic Old MacDonald in C Major."""
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=100))
    p.append(key.Key('C'))
    p.append(meter.TimeSignature('4/4'))

    melody_data = [
        # "Old MacDonald had a farm"
        ("C4", 1), ("C4", 1), ("C4", 1), ("G4", 1),
        ("A4", 1), ("A4", 1), ("G4", 2),
        # "E-I-E-I-O"
        ("E4", 1), ("E4", 1), ("D4", 1), ("D4", 1),
        ("C4", 2), # 2 beats

        # Repeat phrase (optional in prompt, but let's include for length)
        ("G4", 1), ("G4", 1), ("F4", 1), ("F4", 1),
        ("E4", 2), ("E4", 1), # Wait, prompt had ("E4", 2), ("E4", 1)?
        # Prompt: ("G4", 1), ("G4", 1), ("F4", 1), ("F4", 1), ("E4", 2), ("E4", 1), ("D4", 1), ("D4", 1), ("C4", 1), ("C4", 2)
        # That's a bit irregular at the end, but I will copy prompt exactly.
        ("D4", 1), ("D4", 1), ("C4", 1), ("C4", 2),
    ]

    for pitch_name, dur in melody_data:
        n = note.Note(pitch_name)
        n.quarterLength = dur
        p.append(n)

    s.append(p)
    return s

def apply_expressive_performance(score_in, intensity=1.0):
    """
    Applies velocity variations and micro-timing (humanization).
    Returns a new Score.
    """
    s = copy.deepcopy(score_in)

    # Iterate over all notes in all parts
    for p in s.parts:
        # Phrase arch: start soft, get louder, get soft.
        # Simple implementation: Sine wave volume over the part duration?
        # Or just random walk.

        # Base velocity
        base_vel = 90

        notes = list(p.flat.notes)
        total_notes = len(notes)

        for i, n in enumerate(notes):
            # 1. Velocity Dynamics (Phrasing + Jitter)
            # Simple arch: sin(pi * i / total)
            phrase_factor = 0.5 + 0.5 * 1.0 # Flat for now, maybe simple random walk better

            # Random jitter: +/- 10 * intensity
            jitter = random.uniform(-10, 10) * intensity
            vel = int(base_vel + jitter)
            vel = max(40, min(127, vel))

            n.volume.velocity = vel

            # 2. Timing Jitter (Micro-timing)
            # Offset shift. Note: shifting offsets in music21 flat stream doesn't always persist
            # if we don't manage it carefully, but modifying the object usually works if we write MIDI later.
            # Shift start time slightly
            time_shift = random.uniform(-0.05, 0.05) * intensity # quarter lengths
            # We can't easily change .offset directly in a stream iterator safely sometimes.
            # But for MIDI export, we might need to shift it.
            # Actually, `n.offset` is relative to measure or stream.
            # Safe way: We are just setting a property that our synthesizer or MIDI writer will use?
            # Music21 MIDI writer respects .offset.
            # But changing offset changes the grid.
            # Let's try to add a small deviation attribute that our synthesizer uses?
            # Or just modify offset.
            # Be careful not to swap order of notes.

            # For simplicity in this benchmark, let's rely mainly on Velocity for expression
            # and very subtle offset changes if our Synth supports it.
            # Our custom synth `midi_to_wav` will need to respect exact offsets.
            # Let's adjust offset by a tiny amount.
            new_offset = max(0.0, n.offset + time_shift)
            n.offset = new_offset

    return s

def get_harmony(song_name):
    """Returns a list of (start_beat, end_beat, chord_symbol) for accompaniment."""
    # Beat positions are cumulative quarter notes based on the generators above.

    if "happy_birthday" in song_name:
        # 3/4 time. Phrase 1 (6 beats): C... G...
        # Melody: G G A G C B (beats: 1+1+2+2+2+3 = 11? No)
        # Let's trace the melody accumulation:
        # P1: 1+1+2 + 2+2+3 = 11 beats.
        # P2: 1+1+2 + 2+2+3 = 11 beats.
        # P3: 1+1+2 + 2+2+2+3 = 13 beats.
        # P4: 1+1+2 + 2+2+3 = 11 beats.
        # This structure is irregular compared to standard 3/4.
        # I'll just map chords roughly to the timeline.

        # C Major context.
        return [
            (0, 4, "C"), (4, 8, "G"), (8, 11, "G"), # P1
            (11, 15, "G"), (15, 19, "C"), (19, 22, "C"), # P2
            (22, 26, "C"), (26, 30, "F"), (30, 35, "C"), # P3
            (35, 39, "G"), (39, 46, "C") # P4
        ]

    elif "old_macdonald" in song_name:
        # 4/4 time.
        # Melody: C C C G (1,1,1,1) -> 4 beats. Bar 1.
        # A A G (1,1,2) -> 4 beats. Bar 2.
        # E E D D (1,1,1,1) -> 4 beats. Bar 3.
        # C (2) -> 2 beats. Bar 4 (half).
        # Total ~14 beats + repeat?
        # Let's map simpler.
        return [
            (0, 4, "C"), # Old macdonald had a farm
            (4, 8, "C"), # E I E I O (Wait, A A G implies F C? or C F C?)
                         # A A G (IV IV I usually). Let's use F F C.
            (8, 12, "G"), # And on his farm... (chords vary)
            (12, 16, "C"),
            (16, 20, "C"),
            (20, 24, "C")
        ]
    return []

def apply_accompaniment(score_in, song_name, style="block"):
    """
    Adds a new Part with accompaniment.
    style: 'block' (L3) or 'broken' (L4).
    """
    s = copy.deepcopy(score_in)
    harmony = get_harmony(song_name)
    if not harmony:
        return s

    acc_part = stream.Part()
    # Copy key/tempo from first part
    src_part = s.parts[0]
    for el in src_part.getElementsByClass([key.Key, tempo.MetronomeMark, meter.TimeSignature]):
        acc_part.append(copy.deepcopy(el))

    # Generate chords
    for start, end, chord_sym in harmony:
        duration = end - start
        if duration <= 0: continue

        c = chord.Chord(chord_sym)
        # Shift down an octave for accompaniment
        c.transpose(-12, inPlace=True)

        if style == "block":
            # Simple block chords on the beat? Or whole note holds?
            # L3: "soft chordal accompaniment". Let's do sustained chords.
            c.quarterLength = duration
            c.offset = start
            c.volume.velocity = 50 # Soft
            acc_part.insert(start, c)

        elif style == "broken":
            # Arpeggiate: Root - Third - Fifth - Third pattern? or similar
            # Quarter note pulses
            pitches = [p for p in c.pitches]
            # Ensure we have at least 3
            while len(pitches) < 3:
                pitches.append(pitches[0])

            # Simple pattern: 0, 1, 2, 1
            pattern = [0, 1, 2, 1]
            step_len = 1.0 # Quarter note
            current_time = float(start)

            idx = 0
            while current_time < end:
                p_idx = pattern[idx % len(pattern)]
                n = note.Note(pitches[p_idx])
                n.quarterLength = min(step_len, end - current_time)
                n.volume.velocity = 60
                acc_part.insert(current_time, n)
                current_time += step_len
                idx += 1

    s.append(acc_part)
    return s

def generate_benchmark_example(example_id: str):
    """
    Dispatcher to create specific benchmark examples.
    """
    if "happy_birthday" in example_id:
        s = create_happy_birthday_base()
        base_name = "happy_birthday"
    elif "old_macdonald" in example_id:
        s = create_old_macdonald_base()
        base_name = "old_macdonald"
    else:
        raise ValueError(f"Unknown example base: {example_id}")

    # Modifications based on ID
    if "expressive" in example_id:
        s = apply_expressive_performance(s, intensity=1.0)

    if "poly_dominant" in example_id:
        # L3
        s = apply_accompaniment(s, base_name, style="block")

    if "poly_full" in example_id:
        # L4
        s = apply_accompaniment(s, base_name, style="broken")

    return s
