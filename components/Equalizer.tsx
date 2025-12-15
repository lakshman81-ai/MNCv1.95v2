import React, { useEffect, useRef } from 'react';
import { audioEngine } from '../services/audioEngine';
import { formatPitch } from '../utils/pitchUtils';

interface EqualizerProps {
  isPlaying: boolean;
}

const Equalizer: React.FC<EqualizerProps> = ({ isPlaying }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);

  const getNoteName = (hz: number) => {
    if (hz === 0) return '';
    const midi = 69 + 12 * Math.log2(hz / 440);
    return formatPitch(midi, { format: 'note_only', accidentalStyle: 'sharp', showOctave: true }).display;
  };

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get real frequency data
    const data = audioEngine.getFrequencyData();
    const width = canvas.width;
    const height = canvas.height;
    const sampleRate = audioEngine.sampleRate;
    const bufferLength = data.length;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    const isIdle = !isPlaying || data.length === 0 || data.every(v => v === 0);

    // Simple visual style
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#a5b4fc'; // Indigo-300
    ctx.fillStyle = 'rgba(165, 180, 252, 0.05)'; // Very subtle fill

    ctx.beginPath();
    ctx.moveTo(0, height);

    let maxVal = -1;
    let maxIndex = -1;

    if (isIdle) {
        // Flat line when idle
        ctx.lineTo(width, height);
    } else {
        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = data[i] / 255.0;
            const y = height - (v * height);
            
            // Draw line
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);

            // Find dominant peak (ignoring low rumble < 50Hz)
            // Bin resolution approx sampleRate/2048 (e.g. 44100/2048 = ~21.5Hz)
            // Index 3 is approx 60Hz
            if (i > 3 && data[i] > maxVal) {
                maxVal = data[i];
                maxIndex = i;
            }

            x += sliceWidth;
        }
    }

    ctx.lineTo(width, height);
    ctx.stroke();
    ctx.fill();

    // Draw Dominant Peak Label Only
    if (!isIdle && maxVal > 80) { // Threshold to avoid noise labels
        const hz = maxIndex * sampleRate / 2048;
        // Limit label to musical range
        if (hz > 50 && hz < 10000) {
            const x = (maxIndex / bufferLength) * width;
            const y = height - (maxVal / 255.0 * height);
            const note = getNoteName(hz);
            
            // Simple Dot
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fillStyle = '#fff';
            ctx.fill();

            // Simple Text
            ctx.fillStyle = '#e0e7ff';
            ctx.font = '12px Inter, sans-serif';
            // Keep label within bounds
            const text = `${note} ${Math.round(hz)}Hz`;
            const textWidth = ctx.measureText(text).width;
            let textX = x + 8;
            if (textX + textWidth > width) textX = x - textWidth - 8;

            ctx.fillText(text, textX, y + 4);
        }
    }

    animationRef.current = requestAnimationFrame(draw);
  };

  useEffect(() => {
    draw();
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [isPlaying]);

  return (
    <div className="w-full h-full bg-zinc-900 rounded-lg overflow-hidden border border-zinc-800 shadow-inner relative">
      <div className="absolute top-3 left-4 text-xs font-medium text-zinc-500 z-10 select-none">
         Frequency Spectrum
      </div>
      <canvas 
        ref={canvasRef} 
        width={800} 
        height={300} 
        className="w-full h-full"
      />
    </div>
  );
};

export default Equalizer;