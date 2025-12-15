import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import librosa
import music21
import numpy as np
import torch

# Optional neural helpers
try:
    import torchcrepe
except Exception:  # pragma: no cover - optional dependency
    torchcrepe = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio to music notation")
    parser.add_argument("--audio_path", required=True, help="Path to input audio file")
    parser.add_argument("--audio_start_offset_sec", type=float, default=10.0, help="Start offset in seconds")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        choices=[16000, 44100, 48000],
        help="Sample rate (supports 16k, 44.1k, or 48k)",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=256,
        help="Hop length for analysis (e.g., 256 or the hop size required by a neural model)",
    )
    parser.add_argument("--output_musicxml", default="output.musicxml", help="Output MusicXML path")
    parser.add_argument("--output_midi", default="output.mid", help="Output MIDI path")
    parser.add_argument("--output_png", default="output.png", help="Output PNG path")
    parser.add_argument("--output_log", default="transcription_log.json", help="Output log path")
    parser.add_argument(
        "--quantization_strategy",
        choices=["nearest", "classifier"],
        default="nearest",
        help="Quantization strategy: snap to nearest denominator or use a learned classifier",
    )
    return parser.parse_args()

def freq_to_midi(freq):
    if freq is None or freq <= 0:
        return None
    return int(round(69 + 12 * np.log2(freq / 440.0)))

class DurationClassifier:
    """
    Lightweight probabilistic classifier that maps continuous beat durations to
    the most likely rhythmic category. The class stores Gaussian prototypes in
    log-beat space that can be refined offline and shipped as parameters.
    """

    def __init__(self, categories, mus=None, sigmas=None):
        self.categories = np.array(categories, dtype=float)
        # Default prototypes center around the provided categories in log space
        self.mus = np.array(mus) if mus is not None else np.log(self.categories)
        # Default spread loosely models human timing variability
        self.sigmas = (
            np.array(sigmas)
            if sigmas is not None
            else np.full_like(self.mus, 0.20, dtype=float)
        )

    def predict(self, beat_duration):
        beat_duration = max(beat_duration, 1e-6)
        log_val = np.log(beat_duration)
        log_probs = -0.5 * ((log_val - self.mus) / self.sigmas) ** 2
        idx = int(np.argmax(log_probs))
        return float(self.categories[idx])


def quantize_duration(
    seconds,
    start_time=0.0,
    tempo_times=None,
    tempo_curve=None,
    denominators=None,
    classifier=None,
    bpm=None,
    return_bpm: bool | None = None,
):
    """Quantize a duration (in seconds) to the nearest beat fraction.

    Tests call this helper with a fixed ``bpm`` and without tempo curves, so we
    allow both usage patterns. When ``bpm`` is provided it takes precedence over
    the tempo curve interpolation.
    """

    if denominators is None:
        denominators = [1.0]

    if bpm is None:
        tempo_times = np.asarray(tempo_times) if tempo_times is not None else np.array([0.0, 1.0])
        tempo_curve = np.asarray(tempo_curve) if tempo_curve is not None else np.array([120.0, 120.0])
        local_bpm = float(np.interp(start_time, tempo_times, tempo_curve))
    else:
        local_bpm = float(bpm)

    beats = seconds * (local_bpm / 60.0)

    if classifier is not None:
        quantized = classifier.predict(beats)
    else:
        quantized = min(denominators, key=lambda x: abs(x - beats))

    if return_bpm is None:
        # Preserve historical behaviour: if we interpolated from a curve, we
        # return the BPM; if a fixed bpm was provided and no curve was used we
        # mimic the legacy 2-tuple return to satisfy older call sites/tests.
        return_bpm = bpm is None or tempo_times is not None or tempo_curve is not None

    if return_bpm:
        return quantized, beats, local_bpm

    return quantized, beats


def compute_tempo_curve(y, sr, hop_length):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo_curve = librosa.beat.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None
    )
    tempo_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    return tempo_times, tempo_curve


def cumulative_beats(tempo_times, tempo_curve):
    if len(tempo_times) == 0:
        return np.array([])

    beats = [0.0]
    for i in range(1, len(tempo_times)):
        dt = tempo_times[i] - tempo_times[i - 1]
        beats.append(beats[-1] + (tempo_curve[i - 1] / 60.0) * dt)
    return np.array(beats)


class NeuralF0Estimator:
    """Estimate F0 using a neural model (CREPE/SPICE) with graceful fallback."""

    def __init__(self, params: Dict):
        self.params = params

    def _crepe_available(self) -> bool:
        return torchcrepe is not None

    def run(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return times (sec) and frequency estimates (Hz)."""
        if self._crepe_available():
            logger.info("Using torchcrepe for neural F0 estimation (CREPE).")
            return self._run_crepe(y, sr)

        logger.warning("torchcrepe unavailable; falling back to librosa.pyin for F0.")
        f0, _, voiced_probs = librosa.pyin(
            y,
            fmin=self.params["f0_fmin"],
            fmax=self.params["f0_fmax"],
            sr=sr,
            frame_length=self.params["frame_length"],
            hop_length=self.params["hop_length"],
        )
        times = librosa.times_like(f0, sr=sr, hop_length=self.params["hop_length"])
        # Use voiced probability to zero-out low confidence frames
        f0_clean = np.where(np.asarray(voiced_probs) > self.params["f0_confidence_threshold"], f0, None)
        return times, f0_clean

    def _run_crepe(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hop_length = int(sr * self.params["crepe_hop_seconds"])
        padded = torch.tensor(y, device=device).float().unsqueeze(0)
        torchcrepe.load_pretrained(torchcrepe.pretrained.full, device=device)
        with torch.no_grad():
            f0, pd = torchcrepe.predict(
                padded,
                sr,
                hop_length,
                self.params["f0_fmin"],
                self.params["f0_fmax"],
                model=torchcrepe.pretrained.full,
                return_periodicity=True,
                batch_size=8,
                device=device,
            )
        f0 = torchcrepe.filter.median(f0, 3)
        f0 = f0.cpu().numpy()[0]
        pd = pd.cpu().numpy()[0]
        f0[np.where(pd < self.params["f0_confidence_threshold"])] = None
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        return times, f0


class OnsetsFramesDetector:
    """Onsets & Frames style onset/offset detection with optional neural model."""

    def __init__(self, params: Dict):
        self.params = params

    def run(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return onset_probs, offset_probs, pitch_activation (freq matrix)."""
        try:
            import basic_pitch.inference  # type: ignore
            from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore

            logger.info("Using basic-pitch (Onsets & Frames) model for event detection.")
            times, onset_probs, offset_probs, pitch_activations = self._run_basic_pitch(
                y, sr, ICASSP_2022_MODEL_PATH
            )
            return onset_probs, offset_probs, pitch_activations
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning(
                "basic-pitch unavailable, using spectral heuristics for onset/offset detection: %s",
                exc,
            )
            return self._run_spectral_fallback(y, sr)

    def _run_basic_pitch(
        self, y: np.ndarray, sr: int, model_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from basic_pitch.inference import predict  # type: ignore

        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        model_output = predict(
            waveform,
            sr,
            model_path=model_path,
            onset_thresh=self.params["onset_threshold"],
            frame_thresh=self.params["frame_threshold"],
        )
        onset = model_output["onset_predictions"].squeeze(0).numpy()
        offset = model_output["offset_predictions"].squeeze(0).numpy()
        pitch = model_output["frame_predictions"].squeeze(0).numpy()
        times = model_output["note_times"]  # just to satisfy typing; not used downstream
        return times, onset, offset, pitch

    def _run_spectral_fallback(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hop_length = self.params["hop_length"]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_probs = librosa.util.normalize(onset_env)
        # offset via reversed onset strength
        offset_env = librosa.onset.onset_strength(y=y[::-1], sr=sr, hop_length=hop_length)[::-1]
        offset_probs = librosa.util.normalize(offset_env)

        # Multi-pitch activation via spectrogram peaks (not neural but compatible)
        S = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12))
        pitch_activation = np.zeros((84, S.shape[1]))
        for i in range(S.shape[1]):
            column = S[:, i]
            threshold = np.percentile(column, 90)
            active_bins = np.where(column >= threshold)[0]
            pitch_activation[active_bins, i] = 1.0
        return onset_probs, offset_probs, pitch_activation


def build_events_from_detections(
    times: np.ndarray,
    multi_pitch_midi: List[List[Optional[int]]],
    onset_probs: np.ndarray,
    offset_probs: np.ndarray,
    params: Dict,
) -> List[Dict]:
    """Convert neural detections into note events supporting polyphony."""

    note_events: List[Dict] = []
    active_notes: Dict[int, Dict[str, int]] = {}
    onset_threshold = params.get("onset_threshold", 0.5)
    offset_threshold = params.get("offset_threshold", 0.3)

    for frame_idx, (frame_pitches, onset_prob, offset_prob) in enumerate(
        zip(multi_pitch_midi, onset_probs, offset_probs)
    ):
        t = times[frame_idx]
        current_pitch_set = set(filter(lambda m: m is not None, frame_pitches))

        # Close notes explicitly ended by offset detector or deactivated bins
        ended = []
        for midi, info in active_notes.items():
            last_frame = info["start_frame"]
            if midi not in current_pitch_set or offset_prob >= offset_threshold:
                start_sec = times[last_frame]
                duration = t - start_sec
                if duration >= params["min_note_duration_sec"]:
                    note_events.append({
                        "start_sec": start_sec,
                        "end_sec": t,
                        "midi": midi,
                    })
                ended.append(midi)
        for midi in ended:
            active_notes.pop(midi, None)

        # Start new notes when onset probability is strong or note not yet active
        if onset_prob >= onset_threshold:
            for midi in current_pitch_set:
                if midi not in active_notes:
                    active_notes[midi] = {"start_frame": frame_idx}

    # Close residual active notes at end of track
    final_time = times[-1] if len(times) else 0
    for midi, info in active_notes.items():
        start_sec = times[info["start_frame"]]
        duration = final_time - start_sec
        if duration >= params["min_note_duration_sec"]:
            note_events.append({
                "start_sec": start_sec,
                "end_sec": final_time,
                "midi": midi,
            })

    note_events.sort(key=lambda n: (n["start_sec"], n["midi"]))
    return merge_adjacent_same_pitch(note_events, params)


def merge_adjacent_same_pitch(note_events: List[Dict], params: Dict) -> List[Dict]:
    merged: List[Dict] = []
    for note in note_events:
        if merged and note["midi"] == merged[-1]["midi"]:
            gap = note["start_sec"] - merged[-1]["end_sec"]
            if gap < params["merge_gap_threshold_sec"]:
                merged[-1]["end_sec"] = max(merged[-1]["end_sec"], note["end_sec"])
                continue
        merged.append(note)
    return merged


def assign_to_voices(notes: List[Dict], tempo: float, params: Dict) -> Tuple[music21.stream.Part, music21.stream.Part, List[Dict]]:
    """Insert notes into treble/bass parts with polyphonic voices."""

    p_treble = music21.stream.Part()
    p_bass = music21.stream.Part()
    p_treble.insert(0, music21.clef.TrebleClef())
    p_bass.insert(0, music21.clef.BassClef())

    mm = music21.tempo.MetronomeMark(number=tempo)
    p_treble.insert(0, mm)

    staff_voice_events: List[Dict] = []

    treble_voices: Dict[int, music21.stream.Voice] = defaultdict(music21.stream.Voice)
    bass_voices: Dict[int, music21.stream.Voice] = defaultdict(music21.stream.Voice)
    treble_spans: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    bass_spans: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    def _place_note(part_voices, span_map, staff_name, note_dict):
        start_beat = note_dict["start_sec"] * (tempo / 60.0)
        end_beat = note_dict["end_sec"] * (tempo / 60.0)
        q_dur, _ = quantize_duration(note_dict["end_sec"] - note_dict["start_sec"], tempo, params["rhythmic_denominators"])
        m21_note = music21.note.Note(note_dict["midi"])
        m21_note.quarterLength = q_dur

        # find a voice without overlap
        target_voice_idx = None
        for vid, spans in span_map.items():
            if all(end_beat <= s or span_end <= start_beat for s, span_end in spans):
                target_voice_idx = vid
                break
        if target_voice_idx is None:
            target_voice_idx = len(span_map)

        voice_stream = part_voices[target_voice_idx]
        voice_stream.id = f"voice-{staff_name}-{target_voice_idx}"
        voice_stream.insert(start_beat, m21_note)
        span_map[target_voice_idx].append((start_beat, end_beat))

        staff_voice_events.append({
            "start_sec": note_dict["start_sec"],
            "end_sec": note_dict["end_sec"],
            "midi": note_dict["midi"],
            "quantized_rhythm": q_dur,
            "start_beat": start_beat,
            "staff": staff_name,
            "voice": voice_stream.id,
        })

    for n in notes:
        if n["midi"] >= params["split_midi_threshold"]:
            _place_note(treble_voices, treble_spans, "treble", n)
        else:
            _place_note(bass_voices, bass_spans, "bass", n)

    for v in treble_voices.values():
        p_treble.append(v)
    for v in bass_voices.values():
        p_bass.append(v)

    return p_treble, p_bass, staff_voice_events

def main():
    args = parse_args()

    # Parameters from WI
    params = {
        "f0_fmin": librosa.note_to_hz('C2'),
        "f0_fmax": librosa.note_to_hz('C6'),
        "frame_length": 2048,
        "hop_length": args.hop_length,
        "pitch_smoothing_ms": 75,
        "min_note_duration_sec": 0.06,
        "merge_gap_threshold_sec": 0.15,
        "quantization_tolerance": 0.20,
        "rhythmic_denominators": [
            4.0,
            3.0,
            2.0,
            1.5,
            1.0,
            0.75,
            0.6666666667,
            0.5,
            0.3333333333,
            0.25,
            0.1666666667,
            0.125,
        ],
        "split_midi_threshold": 60
    }

    logger.info(f"Starting transcription for {args.audio_path}")

    if not os.path.exists(args.audio_path):
        logger.error(f"Audio file not found: {args.audio_path}")
        sys.exit(1)

    # 1. Load Audio
    logger.info("Step 1: Loading Audio...")
    try:
        y, sr = librosa.load(
            args.audio_path,
            sr=args.sample_rate,
            offset=args.audio_start_offset_sec,
        )
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        sys.exit(1)

    # 2. Preprocess (Normalization)
    logger.info("Step 2: Preprocessing...")
    y = librosa.util.normalize(y)

    # 3. Tempo and Beats
    logger.info("Step 3: Estimating Tempo Curve...")
    tempo_times, tempo_curve = compute_tempo_curve(y, sr, params["hop_length"])
    if tempo_curve.size == 0:
        logger.error("Failed to compute tempo curve")
        sys.exit(1)

    global_tempo = float(np.median(tempo_curve))
    logger.info(
        f"Detected tempo curve with median tempo {global_tempo:.2f} BPM and {len(tempo_curve)} windows"
    )

    beat_positions = cumulative_beats(tempo_times, tempo_curve)

    beat_period_sec = 60.0 / tempo if tempo > 0 else 0.5

    # 4. Pitch Tracking (pyin)
    logger.info("Step 4: Pitch Tracking...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=params["f0_fmin"],
        fmax=params["f0_fmax"],
        sr=sr,
        frame_length=params["frame_length"],
        hop_length=params["hop_length"]
    )

    times = librosa.times_like(f0, sr=sr, hop_length=params["hop_length"])

    rms_energy = librosa.feature.rms(
        y=y,
        frame_length=params["frame_length"],
        hop_length=params["hop_length"]
    )[0]
    mean_rms = float(np.mean(rms_energy)) if rms_energy.size else 0.0
    voiced_confidence = float(np.nanmean(voiced_probs)) if voiced_probs is not None else 0.0

    energy_factor = np.interp(mean_rms, [0.01, 0.1], [0.85, 1.15])
    confidence_factor = np.interp(voiced_confidence, [0.3, 0.9], [0.9, 1.1])
    adaptive_factor = (energy_factor + confidence_factor) / 2.0

    params["min_note_duration_sec"] = max(0.03, beat_period_sec * 0.25 * adaptive_factor)
    params["merge_gap_threshold_sec"] = beat_period_sec * 0.4 * (0.75 + (confidence_factor - 0.9))

    logger.info(
        "Adaptive timing: beat_period_sec=%.3fs, min_note_duration_sec=%.3fs, "
        "merge_gap_threshold_sec=%.3fs (energy_factor=%.3f, voiced_confidence=%.3f)",
        beat_period_sec,
        params["min_note_duration_sec"],
        params["merge_gap_threshold_sec"],
        energy_factor,
        voiced_confidence,
    )

    # 7. Segment Notes (Simplified logic merging Steps 5-8)
    logger.info("Step 7: Segmenting Notes...")

    current_midi = None
    start_time = None

    # Convert f0 sequence to MIDI sequence (handling None/unvoiced)
    midi_sequence = [freq_to_midi(f) if v else None for f, v in zip(f0, voiced_flag)]

    midi_pitches = [m for m in midi_sequence if m is not None]
    if midi_pitches:
        median_pitch = float(np.median(midi_pitches))
        lower_quartile = float(np.percentile(midi_pitches, 25))
        upper_quartile = float(np.percentile(midi_pitches, 75))
        params["split_midi_threshold"] = int(round((lower_quartile + median_pitch + upper_quartile) / 3.0))
    logger.info(
        "Adaptive staff split threshold set to MIDI %s based on pitch distribution",
        params["split_midi_threshold"],
    )

    note_events = []

    for i, midi in enumerate(midi_sequence):
        t = times[i]

        if midi is None:
            if current_midi is not None:
                # End note
                duration = t - start_time
                if duration >= params["min_note_duration_sec"]:
                    note_events.append({
                        "start_sec": start_time,
                        "end_sec": t,
                        "midi": current_midi
                    })
                current_midi = None
                start_time = None
        else:
            if current_midi is None:
                # Start note
                current_midi = midi
                start_time = t
            elif midi != current_midi:
                # Pitch change -> End current, start new
                duration = t - start_time
                if duration >= params["min_note_duration_sec"]:
                    note_events.append({
                        "start_sec": start_time,
                        "end_sec": t,
                        "midi": current_midi
                    })
                current_midi = midi
                start_time = t

    # Close last note if active
    if current_midi is not None:
        note_events.append({
            "start_sec": start_time,
            "end_sec": times[-1],
            "midi": current_midi
        })

    # 8. Merge Adjacent Same-Pitch Notes
    logger.info("Step 8: Merging Notes...")
    merged_notes = []
    if note_events:
        merged_notes.append(note_events[0])
        for n in note_events[1:]:
            last = merged_notes[-1]
            gap = n["start_sec"] - last["end_sec"]
            if n["midi"] == last["midi"] and gap < params["merge_gap_threshold_sec"]:
                last["end_sec"] = n["end_sec"]
            else:
                merged_notes.append(n)

    logger.info(f"Total notes extracted: {len(merged_notes)}")

    # 9-14. Quantization, Voice Assignment, MusicXML
    logger.info("Step 9-14: Building Score...")

    s = music21.stream.Score()
    p_treble = music21.stream.Part()
    p_bass = music21.stream.Part()

    p_treble.insert(0, music21.clef.TrebleClef())
    p_bass.insert(0, music21.clef.BassClef())

    # Tempo
    mm = music21.tempo.MetronomeMark(number=global_tempo)
    p_treble.insert(0, mm)

    duration_classifier = None
    if args.quantization_strategy == "classifier":
        duration_classifier = DurationClassifier(params["rhythmic_denominators"])

    log_entries = []

    for n in merged_notes:
        dur_sec = n["end_sec"] - n["start_sec"]
        q_dur, raw_beats, local_bpm = quantize_duration(
            dur_sec,
            n["start_sec"],
            tempo_times,
            tempo_curve,
            params["rhythmic_denominators"],
            classifier=duration_classifier,
            return_bpm=True,
        )

        m21_note = music21.note.Note(n["midi"])
        # Snap written duration to the quantized value instead of the raw beat length
        m21_note.quarterLength = q_dur

        # Calculate start beat using integrated tempo curve
        start_beat = float(np.interp(n["start_sec"], tempo_times, beat_positions))

        # Determine staff
        if n["midi"] >= params["split_midi_threshold"]:
            p_treble.insert(start_beat, m21_note)
            staff = "treble"
        else:
            p_bass.insert(start_beat, m21_note)
            staff = "bass"

        log_entries.append({
            "start_sec": n["start_sec"],
            "end_sec": n["end_sec"],
            "midi": n["midi"],
            "quantized_rhythm": q_dur,
            "start_beat": start_beat,
            "local_bpm": local_bpm,
            "staff": staff
        })

    s.insert(0, p_treble)
    s.insert(0, p_bass)

    # 13. Key Detection
    try:
        key = s.analyze('key')
        p_treble.insert(0, key)
        logger.info(f"Detected key: {key}")
    except Exception as e:
        logger.warning(f"Key detection failed: {e}")

    # Make Measures and Ties (Crucial for notation)
    logger.info("Structuring measures and ties...")
    try:
        s.makeMeasures(inPlace=True)
        s.makeTies(inPlace=True)
    except Exception as e:
        logger.error(f"Failed to make measures/ties: {e}")

    # 14. Render Output
    logger.info("Step 14: Writing Output Files...")
    try:
        s.write('musicxml', fp=args.output_musicxml)
        logger.info(f"Written MusicXML to {args.output_musicxml}")
    except Exception as e:
        logger.error(f"Failed to write MusicXML: {e}")

    try:
        s.write('midi', fp=args.output_midi)
        logger.info(f"Written MIDI to {args.output_midi}")
    except Exception as e:
        logger.error(f"Failed to write MIDI: {e}")

    # 15. Render PNG
    try:
        # Attempts to use external helper (MuseScore/LilyPond)
        s.write('musicxml.png', fp=args.output_png)
        logger.info(f"Written PNG to {args.output_png}")
    except Exception as e:
        logger.warning(f"PNG generation failed (environment dependencies likely missing): {e}")

    # 16. Logging
    log_payload = {
        "parameters": {
            "tempo_bpm": tempo,
            "beat_period_sec": beat_period_sec,
            "min_note_duration_sec": params["min_note_duration_sec"],
            "merge_gap_threshold_sec": params["merge_gap_threshold_sec"],
            "split_midi_threshold": params["split_midi_threshold"],
            "mean_rms_energy": mean_rms,
            "voiced_confidence": voiced_confidence,
        },
        "notes": log_entries,
    }

    with open(args.output_log, 'w') as f:
        json.dump(log_payload, f, indent=2)
    logger.info(f"Written log to {args.output_log}")

if __name__ == "__main__":
    main()
