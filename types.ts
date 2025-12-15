
export enum NoteDuration {
  WHOLE = "1",
  HALF = "1/2",
  QUARTER = "1/4",
  EIGHTH = "1/8",
  SIXTEENTH = "1/16"
}

export interface NoteEvent {
  // Raw / Analysis Data
  id: string;
  start_time: number; // seconds
  duration: number; // seconds
  midi_pitch: number;
  velocity: number; // 0-1
  confidence: number; // 0-1
  selected?: boolean;

  // Canonical Notation Data
  pitch_label?: string; // e.g. "C4", "F#5"
  startBeat?: number;
  durationBeats?: number;
  staff?: 'treble' | 'bass';
  voice?: number; // 1 or 2
  fingering?: number | null;
  tie?: 'start' | 'stop' | 'continue' | null;
  slur?: boolean; // Legacy flag, prefer slurId
  slurId?: string | null; // Identifier for slur groups
  beamId?: string | null; // Identifier for beam groups
  isRest?: boolean; // explicit rest

  // Validation & QA
  isUncertain?: boolean;
  quantizeErrorBeats?: number;
  remediationFlags?: string[];
}

export interface ChordEvent {
  time: number;
  duration: number;
  root: string;
  quality: string;
  text: string;
}

export interface SlurValidationStats {
  totalSlursAttempted: number;
  slursKept: number;
  slursRemoved: number;
  reasonsSummary: Record<string, number>;
  examples: Array<{ slurId: string; reason: string; noteIds: string[] }>;
  collisionSafetySkipped: boolean;
}

export interface Diagnostics {
  slurValidation: SlurValidationStats;
  quantizeStats?: any;
  staffAssignment?: any;
}

export interface AudioState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  sourceUrl: string | null;
  sourceType: 'file' | 'youtube';
}

export interface AnalysisMetric {
  time: number;
  energy: number;
  pitchConfidence: number;
}

// History & logging types
export interface UserEdits {
  notes_modified: number;
  notes_deleted: number;
  notes_added: number;
}

export interface ExportStatus {
  musicxml: boolean;
  midi: boolean;
  pdf: boolean;
  csv: boolean;
}

export type RetentionPolicy = '10_items' | '50_items' | '100_items' | '7_days' | '30_days' | 'forever';

export interface HistoryEntry {
  id: string;
  timestamp: string; // ISO string
  title: string;
  source_type: 'youtube' | 'file' | 'mic';
  source_url: string | null;
  audio_duration_sec: number;
  notes_count: number;
  avg_confidence: number;
  bpm_detected: number;
  time_signature: string;
  instrument_estimate: string;
  tags: string[];
  user_edits: UserEdits;
  exports: ExportStatus;
  thumbnail?: string; // Data URL or placeholder
}

export interface LabelSettings {
  showLabels: boolean;
  format: 'scientific' | 'note_only' | 'solfege';
  accidentalStyle: 'sharp' | 'flat' | 'double_sharp';
  showOctave: boolean;
  showCentOffset: boolean;
  position: 'above' | 'inside' | 'below';
  minConfidence: number;
  keyboardSize: 37 | 49 | 54 | 61 | 76 | 88;
  selectedVoice: string;
  selectedStyle: string;
}