import numpy as np
import music21
from typing import Dict, Any, List

def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Standard Levenshtein distance between two sequences of tokens.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x,y] = matrix[x-1, y-1]
            else:
                matrix[x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return int(matrix[size_x-1, size_y-1])

def tokenize_score(score_path: str) -> List[str]:
    """
    Converts a musicXML/MIDI file to a sequence of tokens for comparison.
    Token format: "PitchClass_DurationQuarter"
    """
    tokens = []
    try:
        s = music21.converter.parse(score_path)
        for n in s.flat.notes:
            if isinstance(n, music21.note.Note):
                dur = round(n.quarterLength, 2)
                pc = n.pitch.name
                tokens.append(f"{pc}_{dur}")
            elif isinstance(n, music21.chord.Chord):
                dur = round(n.quarterLength, 2)
                # Sort pitches for stability
                pcs = sorted([p.name for p in n.pitches])
                pc_str = "+".join(pcs)
                tokens.append(f"{pc_str}_{dur}")
    except Exception as e:
        print(f"Tokenization failed: {e}")
        return []
    return tokens

def calculate_stage_d_metrics(
    generated_score_path: str, # XML from Stage D
    ground_truth_midi_path: str
) -> Dict[str, float]:
    """
    Calculates Stage D (Score) metrics.
    Compares the generated MusicXML to the original MIDI.
    """
    metrics = {}

    # 1. Parse both
    try:
        ref_s = music21.converter.parse(ground_truth_midi_path)
        est_s = music21.converter.parse(generated_score_path)
    except Exception as e:
        return {"Score_Parse_Error": 1.0, "SymbolicDistance": 999.0}

    # 2. Tokenize and Levenshtein
    ref_tokens = tokenize_score(ground_truth_midi_path)
    est_tokens = tokenize_score(generated_score_path)

    dist = levenshtein_distance(ref_tokens, est_tokens)
    norm_dist = dist / max(len(ref_tokens), 1)

    metrics["SymbolicDistance"] = dist
    metrics["SymbolicDistanceNorm"] = norm_dist

    # 3. Pitch Class Accuracy (Bag of words style? Or aligned?)
    # Simple set overlap for now? Or just rely on Levenshtein which captures sequence.
    # Let's add a simple Pitch Class Histogram correlation?
    # Or just count correct tokens / total tokens (Match rate from Levenshtein)
    # Match Rate = 1 - NormDistance
    metrics["MatchRate"] = max(0.0, 1.0 - norm_dist)

    # 4. Bar Alignment
    # Check if total duration matches (in quarters)
    ref_dur = ref_s.duration.quarterLength
    est_dur = est_s.duration.quarterLength
    metrics["Duration_Diff_Quarters"] = abs(ref_dur - est_dur)

    return metrics
