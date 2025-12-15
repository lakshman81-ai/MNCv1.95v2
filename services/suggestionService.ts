import { NoteEvent } from '../types';

export interface SuggestedSettings {
  voice: string;
  style: string;
  bpm: number;
}

export const SuggestionService = {
  generateSuggestions(notes: NoteEvent[]): SuggestedSettings | null {
    if (notes.length < 10) {
      return null;
    }

    const averagePitch = notes.reduce((sum, note) => sum + note.midi_pitch, 0) / notes.length;
    const averageDuration = notes.reduce((sum, note) => sum + note.duration, 0) / notes.length;
    const noteDensity = notes.length / (notes[notes.length - 1].start_time - notes[0].start_time);

    if (averagePitch < 48 && averageDuration > 0.4) {
      return {
        voice: 'synth_bass',
        style: 'funk',
        bpm: 95,
      };
    } else if (averagePitch > 65 && noteDensity > 5) {
      return {
        voice: 'piano',
        style: 'pop',
        bpm: 125,
      };
    } else if (noteDensity > 8 && averageDuration < 0.2) {
        return {
            voice: 'synth_lead',
            style: 'techno',
            bpm: 145,
        }
    } else {
      return {
        voice: 'grand_piano',
        style: 'ballad',
        bpm: 80,
      };
    }
  },
};
