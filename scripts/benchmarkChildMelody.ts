import { performance } from 'node:perf_hooks';
import { generateChildMelody } from '../utils/childMelodyGenerator.ts';

const iterations = Number(process.argv[2] ?? 20);
const seeds = Array.from({ length: iterations }, (_, idx) => idx + 1);

const runs = seeds.map((seed) => {
  const start = performance.now();
  const { melody, tempo } = generateChildMelody({ seed });
  const durationMs = performance.now() - start;
  return { seed, melody, tempo, durationMs };
});

const min = Math.min(...runs.map((run) => run.durationMs));
const max = Math.max(...runs.map((run) => run.durationMs));
const avg = runs.reduce((sum, run) => sum + run.durationMs, 0) / runs.length;

console.log('Child melody generator benchmark');
console.log(`Iterations: ${iterations}`);
console.log(`Average duration: ${avg.toFixed(3)} ms (min ${min.toFixed(3)} ms, max ${max.toFixed(3)} ms)`);
console.log('Sample melodies (first five):');
console.table(
  runs.slice(0, 5).map((run) => ({
    seed: run.seed,
    tempo: `${run.tempo} bpm`,
    durationMs: run.durationMs.toFixed(3),
    melody: run.melody,
  })),
);
