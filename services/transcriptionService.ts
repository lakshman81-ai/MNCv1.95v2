
import { audioEngine } from './audioEngine';
import { NoteEvent, ChordEvent } from '../types';
import { detectChords } from '../utils/chordDetection';

// Helper to format Harmony XML
const getMusicXMLHarmony = (chord: ChordEvent): string => {
    let rootStep = chord.root.charAt(0);
    let rootAlter = 0;
    
    if (chord.root.includes('#')) rootAlter = 1;
    if (chord.root.includes('b')) rootAlter = -1;

    // Map quality to MusicXML kind
    let kind = 'major';
    if (chord.quality === 'm') kind = 'minor';
    if (chord.quality === 'dim') kind = 'diminished';
    if (chord.quality === 'aug') kind = 'augmented';
    if (chord.quality === '7') kind = 'dominant';
    if (chord.quality === 'maj7') kind = 'major-seventh';
    if (chord.quality === 'm7') kind = 'minor-seventh';
    if (chord.quality === 'sus4') kind = 'suspended-fourth';
    if (chord.quality === 'sus2') kind = 'suspended-second';

    let alterTag = '';
    if (rootAlter !== 0) {
        alterTag = `<root-alter>${rootAlter}</root-alter>`;
    }

    return `
      <harmony>
        <root>
          <root-step>${rootStep}</root-step>
          ${alterTag}
        </root>
        <kind>${kind}</kind>
      </harmony>`;
};

// Simple MusicXML Generator for Client-Side Fallback
const createMusicXML = (notes: NoteEvent[], chords: ChordEvent[]): string => {
    const BPM = 120;
    const SECONDS_PER_BEAT = 60 / BPM;
    const DIVISIONS = 24; 
    
    let xml = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <work><work-title>Analyzed Audio</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">`;

    const sortedNotes = [...notes].sort((a, b) => a.start_time - b.start_time);
    const sortedChords = [...chords].sort((a, b) => a.time - b.time);
    
    if (sortedNotes.length === 0) {
        return xml + `<measure number="1"><attributes><divisions>${DIVISIONS}</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>${DIVISIONS*4}</duration></note></measure></part></score-partwise>`;
    }

    const BEATS_PER_MEASURE = 4;
    const SECONDS_PER_MEASURE = SECONDS_PER_BEAT * BEATS_PER_MEASURE;
    const totalDuration = sortedNotes[sortedNotes.length - 1].start_time + sortedNotes[sortedNotes.length - 1].duration;
    const totalMeasures = Math.ceil(totalDuration / SECONDS_PER_MEASURE) || 1;

    let currentNoteIdx = 0;
    let currentChordIdx = 0;

    for (let m = 1; m <= totalMeasures; m++) {
        const measureTimeStart = (m - 1) * SECONDS_PER_MEASURE;
        const measureTimeEnd = m * SECONDS_PER_MEASURE;
        let measureCursor = measureTimeStart;
        
        xml += `<measure number="${m}">`;
        
        if (m === 1) {
            xml += `<attributes>
            <divisions>${DIVISIONS}</divisions>
            <key><fifths>0</fifths></key>
            <time><beats>${BEATS_PER_MEASURE}</beats><beat-type>4</beat-type></time>
            <clef><sign>G</sign><line>2</line></clef>
            </attributes>`;
        }

        while (currentNoteIdx < sortedNotes.length) {
            const note = sortedNotes[currentNoteIdx];
            if (note.start_time >= measureTimeEnd) break;
            
            // Insert Chord if it matches current time (approx)
            // We check if a chord starts within a small window of the current cursor position
            while (currentChordIdx < sortedChords.length) {
                const chord = sortedChords[currentChordIdx];
                // If chord is in the past relative to this measure, skip it
                if (chord.time < measureCursor - 0.1) {
                    currentChordIdx++;
                    continue;
                }
                // If chord is in the future beyond cursor, stop checking
                if (chord.time > measureCursor + 0.1) break;
                
                // If chord is at cursor, insert it
                xml += getMusicXMLHarmony(chord);
                currentChordIdx++;
                // Assuming only one chord per exact timestamp for simplicity
                break; 
            }

            // Rest before note
            if (note.start_time > measureCursor + 0.05) {
                const restSec = note.start_time - measureCursor;
                const restTicks = Math.round((restSec / SECONDS_PER_BEAT) * DIVISIONS);
                if (restTicks > 0) {
                    xml += `<note><rest/><duration>${restTicks}</duration><type>quarter</type></note>`;
                    measureCursor += restSec;
                }
            }

            // Re-check for chord after rest, before note
             while (currentChordIdx < sortedChords.length) {
                const chord = sortedChords[currentChordIdx];
                if (chord.time < measureCursor - 0.1) { currentChordIdx++; continue; }
                if (chord.time > measureCursor + 0.1) break;
                xml += getMusicXMLHarmony(chord);
                currentChordIdx++;
                break;
            }

            // Note
            const durationSec = Math.min(note.duration, measureTimeEnd - measureCursor); // Clip to measure
            const durationTicks = Math.max(1, Math.round((durationSec / SECONDS_PER_BEAT) * DIVISIONS));
            
            const stepNames = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B'];
            const alter = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0];
            const midi = Math.round(note.midi_pitch);
            const stepIndex = midi % 12;
            const octave = Math.floor(midi / 12) - 1;

            xml += `<note>
                <pitch>
                    <step>${stepNames[stepIndex]}</step>
                    <alter>${alter[stepIndex]}</alter>
                    <octave>${octave}</octave>
                </pitch>
                <duration>${durationTicks}</duration>
                <type>quarter</type>
            </note>`;
            
            measureCursor += durationSec;
            currentNoteIdx++;
        }
        
        // Fill rest of measure
        if (measureCursor < measureTimeEnd - 0.1) {
             const restSec = measureTimeEnd - measureCursor;
             const restTicks = Math.round((restSec / SECONDS_PER_BEAT) * DIVISIONS);
             if (restTicks > 0) xml += `<note><rest/><duration>${restTicks}</duration></note>`;
        }

        xml += `</measure>`;
    }

    xml += `</part></score-partwise>`;
    return xml;
};

export interface TranscriptionResult {
    xml: string;
    notes: NoteEvent[];
    chords: ChordEvent[];
}

export const TranscriptionService = {
  transcribeAudio: async (audioBlob: Blob): Promise<TranscriptionResult> => {
    // Client-Side Only flow for consistent Play Notes functionality
    // (Bypassing backend to ensure we have the NoteEvent objects needed for the sequencer)
    console.log("Using local transcription engine...");
    
    try {
        // Use a fresh context for decoding to avoid state issues
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        
        // Analyze audio to get notes using the existing engine logic
        const notes = audioEngine.analyzeAudioSegment(audioBuffer, 0, audioBuffer.duration);
        
        // Analyze Chords from notes
        const chords = detectChords(notes);
        
        // Convert notes to MusicXML
        const xml = createMusicXML(notes, chords);
        
        return { xml, notes, chords };
    } catch (clientError) {
        console.error("Local transcription failed:", clientError);
        // Return valid empty score to prevent UI crash
        return { 
            xml: `<?xml version="1.0" encoding="UTF-8"?><score-partwise><part-list><score-part id="P1"><part-name>Error</part-name></score-part></part-list><part id="P1"><measure number="1"><note><rest/><duration>4</duration></note></measure></part></score-partwise>`,
            notes: [],
            chords: []
        };
    }
  }
};