import numpy as np
import librosa

def generate_sine_wave(freq_hz: float, duration_sec: float, sr: int = 22050, amplitude: float = 1.0) -> np.ndarray:
    """Generates a pure sine wave."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return audio.astype(np.float32)

def generate_silence(duration_sec: float, sr: int = 22050) -> np.ndarray:
    """Generates silence."""
    return np.zeros(int(duration_sec * sr), dtype=np.float32)

def generate_noise(duration_sec: float, sr: int = 22050, amplitude: float = 0.1) -> np.ndarray:
    """Generates white noise."""
    return (np.random.rand(int(duration_sec * sr)) * 2 - 1).astype(np.float32) * amplitude

def mix_audio(signals: list[np.ndarray]) -> np.ndarray:
    """Mixes multiple signals of the same length together."""
    if not signals:
        return np.array([])

    # Ensure all are same length by padding
    max_len = max(len(s) for s in signals)
    output = np.zeros(max_len, dtype=np.float32)

    for s in signals:
        padded = np.pad(s, (0, max_len - len(s)))
        output += padded

    # Normalize to avoid clipping if needed, but for analysis we might want raw sum
    # Let's clip to -1.0 to 1.0 just in case it's used for playback simulation
    return np.clip(output, -1.0, 1.0)

def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def generate_old_mcdonald(sr: int = 22050) -> dict:
    """
    Generates an Old McDonald tune.
    Returns:
       {
           "audio": np.ndarray,
           "ground_truth_mono": List[(time, freq)],
           "ground_truth_poly": List[(time, [freqs])]
       }
    Tune: G G G D E E D (Old Mc-Don-ald had a farm)
    Notes (G4=67, D4=62, E4=64)
    """
    # Define Melody (Note, Duration)
    # Quarter note = 0.5s (120 BPM)
    q = 0.5
    h = 1.0

    melody_notes = [
        (67, q), (67, q), (67, q), (62, q), (64, q), (64, q), (62, h)
    ]

    # Part 1: Monophonic
    # Just the melody
    audio_segments = []

    # To track ground truth, we need start times
    current_time = 0.0
    gt_mono = []

    for midi, dur in melody_notes:
        freq = midi_to_hz(midi)
        seg = generate_sine_wave(freq, dur, sr, amplitude=0.8)
        audio_segments.append(seg)

        # Simple GT: mid-point of the segment
        gt_mono.append({
            "start": current_time,
            "end": current_time + dur,
            "pitch": freq
        })
        current_time += dur

    mono_audio = np.concatenate(audio_segments)

    # Part 2: Polyphonic (Dominant Melody + Accompaniment)
    # Melody repeats: G G G D E E D
    # Accompaniment: Sustained G3 (55) then C4 (60)?
    # Let's keep it simple: Constant G3 (55) drone

    poly_segments = []
    gt_poly = []

    # Reset time for part 2 relative start
    part2_start_time = current_time

    for midi, dur in melody_notes:
        freq_mel = midi_to_hz(midi)
        freq_acc = midi_to_hz(55) # G3

        # Melody is louder (0.8), Acc is quieter (0.3)
        s_mel = generate_sine_wave(freq_mel, dur, sr, amplitude=0.8)
        s_acc = generate_sine_wave(freq_acc, dur, sr, amplitude=0.3)

        mixed = s_mel + s_acc
        poly_segments.append(mixed)

        gt_poly.append({
            "start": current_time,
            "end": current_time + dur,
            "dominant": freq_mel,
            "others": [freq_mel, freq_acc]
        })
        current_time += dur

    poly_audio = np.concatenate(poly_segments)

    # Combine
    full_audio = np.concatenate([mono_audio, poly_audio])

    return {
        "audio": full_audio,
        "mono_end_time": part2_start_time,
        "total_duration": current_time,
        "gt_mono": gt_mono,
        "gt_poly": gt_poly
    }
