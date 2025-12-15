export type NoteDuration = 'w' | 'h' | 'q' | 'e' | 's';

export interface MelodyNote {
  pitch: string;
  duration: NoteDuration;
}

interface GeneratorOptions {
  seed?: number;
  tempo?: number;
}

const scale: string[] = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'];

const rhythmPatterns: NoteDuration[][] = [
  ['q', 'q', 'q', 'q'],
  ['q', 'q', 'h'],
  ['h', 'q', 'q'],
  ['h', 'h'],
  ['e', 'e', 'q', 'q', 'q'],
  ['q', 'e', 'e', 'q', 'q'],
];

function mulberry32(seed: number): () => number {
  return function random() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function pickStrongTone(random: () => number): string {
  const strongTones = ['C', 'E', 'G'];
  const octave = random() > 0.6 ? '5' : '4';
  const tone = strongTones[Math.floor(random() * strongTones.length)];
  return `${tone}${octave}`;
}

function clampIndex(index: number): number {
  if (index < 0) return 0;
  if (index >= scale.length) return scale.length - 1;
  return index;
}

function pickNextIndex(currentIndex: number, random: () => number): number {
  const stepWeights = [0, 1, -1, 2, -2];
  const choice = stepWeights[Math.floor(random() * stepWeights.length)];
  return clampIndex(currentIndex + choice);
}

function chooseRhythm(random: () => number): NoteDuration[] {
  return rhythmPatterns[Math.floor(random() * rhythmPatterns.length)];
}

function barToNotes(baseIndex: number, random: () => number): { notes: MelodyNote[]; nextIndex: number } {
  const rhythm = chooseRhythm(random);
  const notes: MelodyNote[] = [];
  let currentIndex = baseIndex;

  rhythm.forEach((duration, idx) => {
    if (idx === 0) {
      const strongPitch = pickStrongTone(random);
      currentIndex = clampIndex(scale.indexOf(strongPitch) >= 0 ? scale.indexOf(strongPitch) : baseIndex);
      notes.push({ pitch: scale[currentIndex], duration });
      return;
    }

    currentIndex = pickNextIndex(currentIndex, random);
    notes.push({ pitch: scale[currentIndex], duration });
  });

  return { notes, nextIndex: currentIndex };
}

export function generateChildMelody({ seed, tempo = 90 }: GeneratorOptions = {}): { melody: string; tempo: number } {
  const random = mulberry32(seed ?? Date.now());
  let currentIndex = 0;
  const melody: MelodyNote[] = [];

  for (let bar = 0; bar < 4; bar++) {
    const { notes, nextIndex } = barToNotes(currentIndex, random);
    melody.push(...notes);
    currentIndex = nextIndex;
  }

  melody[melody.length - 1] = { pitch: 'C5', duration: melody[melody.length - 1].duration };

  const melodyLine = melody.map((note) => `${note.pitch} ${note.duration}`).join(', ');
  return { melody: melodyLine, tempo };
}
