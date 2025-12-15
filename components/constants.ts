

export const PIXELS_PER_SECOND = 80;

export const VOICES = [
    { id: 'piano', name: 'Grand Piano', category: 'Piano' },
    { id: 'bright_piano', name: 'Bright Piano', category: 'Piano' },
    { id: 'elec_piano', name: 'Electric Piano', category: 'Piano' },
    { id: 'harmonium_1', name: 'Harmonium 1', category: 'Indian' },
    { id: 'harmonium_2', name: 'Harmonium 2', category: 'Indian' },
    { id: 'sitar', name: 'Sitar', category: 'Indian' },
    { id: 'veena', name: 'Veena', category: 'Indian' },
    { id: 'shenai', name: 'Shehnai', category: 'Indian' },
    { id: 'sarod', name: 'Sarod', category: 'Indian' },
    { id: 'santur', name: 'Santur', category: 'Indian' },
    { id: 'tabla_voice', name: 'Tabla (Pitch)', category: 'Indian' },
    { id: 'bansuri', name: 'Bansuri', category: 'Indian' },
    { id: 'violin', name: 'Violin', category: 'Strings' },
    { id: 'strings', name: 'String Ensemble', category: 'Strings' },
    { id: 'guitar_nylon', name: 'Guitar (Nylon)', category: 'Guitar' },
    { id: 'guitar_steel', name: 'Guitar (Steel)', category: 'Guitar' },
    { id: 'synth_lead', name: 'Synth Lead', category: 'Synth' },
    { id: 'synth_pad', name: 'Synth Pad', category: 'Synth' }
];

export const STYLES = [
    { id: 'none', name: 'No Rhythm (Metronome)', timeSignature: '4/4' },
    { id: 'beat_8', name: '8 Beat', timeSignature: '4/4' },
    { id: 'beat_16', name: '16 Beat', timeSignature: '4/4' },
    { id: 'ballad', name: 'Ballad', timeSignature: '4/4' },
    { id: 'dance', name: 'Dance', timeSignature: '4/4' },
    { id: 'disco', name: 'Disco', timeSignature: '4/4' },
    { id: 'swing', name: 'Swing & Jazz', timeSignature: '4/4' },
    { id: 'r_n_b', name: 'R&B', timeSignature: '4/4' },
    { id: 'country', name: 'Country', timeSignature: '4/4' },
    { id: 'latin', name: 'Latin', timeSignature: '4/4' },
    { id: 'ballroom', name: 'Ballroom', timeSignature: '4/4' },
    { id: 'traditional', name: 'Traditional', timeSignature: '4/4' },
    { id: 'waltz', name: 'Waltz', timeSignature: '3/4' },
    { id: 'children', name: 'Children', timeSignature: '4/4' },
    { id: 'pianist', name: 'Pianist', timeSignature: '4/4' },
    // Specific Indian Styles
    { id: 'teen_taal', name: 'Teen Taal (Indian)', timeSignature: '16/4' },
    { id: 'dadra', name: 'Dadra (Indian)', timeSignature: '6/8' },
    { id: 'keherwa', name: 'Keherwa (Indian)', timeSignature: '4/4' },
    { id: 'rupak', name: 'Rupak (Indian)', timeSignature: '7/4' },
    { id: 'garba', name: 'Garba (Indian)', timeSignature: '6/8' },
    { id: 'bhangra', name: 'Bhangra (Indian)', timeSignature: '4/4' },
];

export const GENRES = [
    'Ballad',
    'Pop',
    'Rock',
    'Jazz',
    'Classical',
    'Electronic',
    'Ambient',
    'R&B',
    'Country',
    'Latin'
];

export interface RhythmPattern {
    length: number; // in beats
    steps: {
        beat: number; // 0-based
        sound: 'kick' | 'snare' | 'hihat_closed' | 'hihat_open' | 'tabla_dha' | 'tabla_tin' | 'tabla_na' | 'tabla_ge' | 'tabla_ke';
        velocity: number;
    }[];
}

export const RHYTHM_PATTERNS: Record<string, RhythmPattern> = {
    'teen_taal': {
        length: 16,
        steps: [
            { beat: 0, sound: 'tabla_dha', velocity: 1.0 }, // Dha
            { beat: 1, sound: 'tabla_dha', velocity: 0.8 }, // Dhin
            { beat: 2, sound: 'tabla_dha', velocity: 0.8 }, // Dhin
            { beat: 3, sound: 'tabla_dha', velocity: 1.0 }, // Dha
            { beat: 4, sound: 'tabla_dha', velocity: 1.0 }, // Dha
            { beat: 5, sound: 'tabla_dha', velocity: 0.8 }, // Dhin
            { beat: 6, sound: 'tabla_dha', velocity: 0.8 }, // Dhin
            { beat: 7, sound: 'tabla_dha', velocity: 1.0 }, // Dha
            { beat: 8, sound: 'tabla_dha', velocity: 1.0 }, // Dha
            { beat: 9, sound: 'tabla_tin', velocity: 0.8 }, // Tin
            { beat: 10, sound: 'tabla_tin', velocity: 0.8 }, // Tin
            { beat: 11, sound: 'tabla_na', velocity: 0.9 }, // Na
            { beat: 12, sound: 'tabla_na', velocity: 0.9 }, // Ta
            { beat: 13, sound: 'tabla_dha', velocity: 0.8 }, // Dhin
            { beat: 14, sound: 'tabla_dha', velocity: 0.8 }, // Dhin
            { beat: 15, sound: 'tabla_dha', velocity: 1.0 }, // Dha
        ]
    },
    'dadra': {
        length: 6,
        steps: [
            { beat: 0, sound: 'tabla_dha', velocity: 1.0 },
            { beat: 1, sound: 'tabla_dha', velocity: 0.7 }, // Dhin
            { beat: 2, sound: 'tabla_na', velocity: 0.8 },
            { beat: 3, sound: 'tabla_dha', velocity: 1.0 },
            { beat: 4, sound: 'tabla_tin', velocity: 0.7 },
            { beat: 5, sound: 'tabla_na', velocity: 0.8 },
        ]
    },
    'keherwa': {
        length: 8,
        steps: [
            { beat: 0, sound: 'tabla_dha', velocity: 1.0 },
            { beat: 1, sound: 'tabla_ge', velocity: 0.7 },
            { beat: 2, sound: 'tabla_na', velocity: 0.8 },
            { beat: 3, sound: 'tabla_tin', velocity: 0.7 },
            { beat: 4, sound: 'tabla_na', velocity: 0.8 },
            { beat: 5, sound: 'tabla_ke', velocity: 0.6 },
            { beat: 6, sound: 'tabla_dha', velocity: 0.9 },
            { beat: 7, sound: 'tabla_ge', velocity: 0.7 },
        ]
    },
    'beat_8': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 0.5, sound: 'hihat_closed', velocity: 0.6 },
            { beat: 1, sound: 'snare', velocity: 0.9 },
            { beat: 1.5, sound: 'hihat_closed', velocity: 0.6 },
            { beat: 2, sound: 'kick', velocity: 1.0 },
            { beat: 2.5, sound: 'hihat_closed', velocity: 0.6 },
            { beat: 3, sound: 'snare', velocity: 0.9 },
            { beat: 3.5, sound: 'hihat_open', velocity: 0.7 },
        ]
    },
    'beat_16': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 0.25, sound: 'hihat_closed', velocity: 0.5 },
            { beat: 0.5, sound: 'hihat_closed', velocity: 0.5 },
            { beat: 0.75, sound: 'hihat_closed', velocity: 0.5 },
            { beat: 1, sound: 'snare', velocity: 0.9 },
            { beat: 1.5, sound: 'hihat_closed', velocity: 0.5 },
            { beat: 2, sound: 'kick', velocity: 1.0 },
            { beat: 2.25, sound: 'kick', velocity: 0.8 },
            { beat: 3, sound: 'snare', velocity: 0.9 },
            { beat: 3.5, sound: 'hihat_open', velocity: 0.7 },
        ]
    },
    'ballad': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 0.9 },
            { beat: 1, sound: 'snare', velocity: 0.6 },
            { beat: 2, sound: 'kick', velocity: 0.8 },
            { beat: 3, sound: 'snare', velocity: 0.6 },
            { beat: 3.5, sound: 'kick', velocity: 0.5 },
        ]
    },
    'dance': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 0.5, sound: 'hihat_open', velocity: 0.7 },
            { beat: 1, sound: 'kick', velocity: 1.0 },
            { beat: 1.5, sound: 'hihat_open', velocity: 0.7 },
            { beat: 2, sound: 'kick', velocity: 1.0 },
            { beat: 2.5, sound: 'hihat_open', velocity: 0.7 },
            { beat: 3, sound: 'kick', velocity: 1.0 },
            { beat: 3.5, sound: 'hihat_open', velocity: 0.7 },
        ]
    },
    'disco': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 0.5, sound: 'hihat_open', velocity: 0.6 },
            { beat: 1, sound: 'kick', velocity: 1.0 },
            { beat: 1, sound: 'snare', velocity: 0.9 },
            { beat: 1.5, sound: 'hihat_open', velocity: 0.6 },
            { beat: 2, sound: 'kick', velocity: 1.0 },
            { beat: 2.5, sound: 'hihat_open', velocity: 0.6 },
            { beat: 3, sound: 'kick', velocity: 1.0 },
            { beat: 3, sound: 'snare', velocity: 0.9 },
            { beat: 3.5, sound: 'hihat_open', velocity: 0.6 },
        ]
    },
    'swing': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 0.8 },
            { beat: 1, sound: 'hihat_closed', velocity: 0.7 },
            { beat: 1.66, sound: 'hihat_closed', velocity: 0.5 }, // Swing feel
            { beat: 2, sound: 'kick', velocity: 0.7 },
            { beat: 3, sound: 'hihat_closed', velocity: 0.7 },
            { beat: 3.66, sound: 'hihat_closed', velocity: 0.5 },
        ]
    },
    'r_n_b': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 1, sound: 'snare', velocity: 0.9 },
            { beat: 1.5, sound: 'kick', velocity: 0.8 },
            { beat: 2.5, sound: 'kick', velocity: 0.8 },
            { beat: 3, sound: 'snare', velocity: 0.9 },
        ]
    },
    'country': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 0.5, sound: 'snare', velocity: 0.4 }, // Train beat shuffle
            { beat: 1, sound: 'snare', velocity: 0.8 },
            { beat: 1.5, sound: 'snare', velocity: 0.4 },
            { beat: 2, sound: 'kick', velocity: 1.0 },
            { beat: 2.5, sound: 'snare', velocity: 0.4 },
            { beat: 3, sound: 'snare', velocity: 0.8 },
            { beat: 3.5, sound: 'snare', velocity: 0.4 },
        ]
    },
    'latin': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 0.5, sound: 'snare', velocity: 0.6 }, // Rimshot feel
            { beat: 1.5, sound: 'kick', velocity: 0.9 },
            { beat: 2, sound: 'snare', velocity: 0.6 },
            { beat: 3, sound: 'kick', velocity: 0.9 },
            { beat: 3.5, sound: 'snare', velocity: 0.8 },
        ]
    },
    'ballroom': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 1, sound: 'snare', velocity: 0.8 },
            { beat: 2, sound: 'kick', velocity: 0.9 },
            { beat: 2.5, sound: 'kick', velocity: 0.7 },
            { beat: 3, sound: 'snare', velocity: 0.8 },
        ]
    },
    'traditional': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 1, sound: 'snare', velocity: 0.7 },
            { beat: 2, sound: 'kick', velocity: 1.0 },
            { beat: 3, sound: 'snare', velocity: 0.7 },
        ]
    },
    'waltz': {
        length: 3,
        steps: [
            { beat: 0, sound: 'kick', velocity: 1.0 },
            { beat: 1, sound: 'snare', velocity: 0.6 },
            { beat: 2, sound: 'snare', velocity: 0.6 },
        ]
    },
    'children': {
        length: 4,
        steps: [
            { beat: 0, sound: 'kick', velocity: 0.8 },
            { beat: 1, sound: 'hihat_closed', velocity: 0.6 },
            { beat: 2, sound: 'kick', velocity: 0.8 },
            { beat: 3, sound: 'hihat_closed', velocity: 0.6 },
        ]
    },
    'pianist': {
        length: 4,
        steps: [
            { beat: 0, sound: 'hihat_closed', velocity: 0.3 }, // Minimal metronome
            { beat: 1, sound: 'hihat_closed', velocity: 0.2 },
            { beat: 2, sound: 'hihat_closed', velocity: 0.3 },
            { beat: 3, sound: 'hihat_closed', velocity: 0.2 },
        ]
    }
};
