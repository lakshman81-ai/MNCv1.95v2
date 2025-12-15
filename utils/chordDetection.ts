import { NoteEvent, ChordEvent } from '../types';

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const CHORD_SHAPES: { [key: string]: number[] } = {
    '': [0, 4, 7],       // Major
    'm': [0, 3, 7],      // Minor
    'dim': [0, 3, 6],    // Diminished
    'aug': [0, 4, 8],    // Augmented
    'sus4': [0, 5, 7],   // Suspended 4th
    'sus2': [0, 2, 7],   // Suspended 2nd
    '7': [0, 4, 7, 10],  // Dominant 7th
    'maj7': [0, 4, 7, 11],// Major 7th
    'm7': [0, 3, 7, 10], // Minor 7th
};

export const detectChords = (notes: NoteEvent[]): ChordEvent[] => {
    if (notes.length === 0) return [];

    const sortedNotes = [...notes].sort((a, b) => a.start_time - b.start_time);
    const endTime = sortedNotes[sortedNotes.length - 1].start_time + sortedNotes[sortedNotes.length - 1].duration;
    
    // Time window for chord analysis (e.g., every 0.5 seconds or 1 beat at 120bpm)
    const WINDOW_SIZE = 0.5; 
    const chords: ChordEvent[] = [];

    let lastChordText = '';

    for (let t = 0; t < endTime; t += WINDOW_SIZE) {
        // Find notes active in this window (overlapping significantly)
        const windowCenter = t + (WINDOW_SIZE / 2);
        const activeNotes = sortedNotes.filter(n => 
            n.start_time <= windowCenter && (n.start_time + n.duration) >= windowCenter
        );

        if (activeNotes.length < 2) {
            // Not enough notes for a chord, maybe extend previous or rest
            continue;
        }

        // Get unique pitch classes
        const pitchClasses = Array.from(new Set(activeNotes.map(n => Math.round(n.midi_pitch) % 12))).sort((a, b) => a - b);
        
        let bestMatch: { root: string, quality: string, text: string } | null = null;
        let maxScore = -1;

        // Try every pitch class present as a potential root
        for (const root of pitchClasses) {
            // Calculate intervals relative to root
            const intervals = pitchClasses.map(pc => (pc - root + 12) % 12).sort((a, b) => a - b);
            
            for (const [quality, shape] of Object.entries(CHORD_SHAPES)) {
                // Check how many intervals match the chord shape
                const matchCount = shape.filter(interval => intervals.includes(interval)).length;
                
                // Score based on % of chord tones present vs total notes present
                // We want to maximize the match of the shape, but also penalize non-chord tones slightly
                const shapeCoverage = matchCount / shape.length;
                const noisePenalty = (intervals.length - matchCount) * 0.1;
                
                // Bonus if the actual lowest note in the window matches this root (Bass note)
                const lowestNote = activeNotes.reduce((min, n) => n.midi_pitch < min.midi_pitch ? n : min, activeNotes[0]);
                const isBass = (Math.round(lowestNote.midi_pitch) % 12) === root;
                
                const finalScore = shapeCoverage - noisePenalty + (isBass ? 0.3 : 0);

                if (finalScore > maxScore && matchCount >= 2) { // Threshold: need at least 2 chord tones (e.g. root + 3rd)
                    maxScore = finalScore;
                    bestMatch = { root: NOTE_NAMES[root], quality, text: NOTE_NAMES[root] + quality };
                }
            }
        }

        if (bestMatch && bestMatch.text !== lastChordText) {
            chords.push({
                time: t,
                duration: WINDOW_SIZE,
                root: bestMatch.root,
                quality: bestMatch.quality,
                text: bestMatch.text
            });
            lastChordText = bestMatch.text;
        } else if (bestMatch && chords.length > 0) {
            // Extend previous chord
            chords[chords.length - 1].duration += WINDOW_SIZE;
        }
    }

    return chords;
};