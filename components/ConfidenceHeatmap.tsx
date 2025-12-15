
import React, { useEffect, useMemo } from 'react';
import { 
  ComposedChart, 
  Line, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ReferenceLine,
  ResponsiveContainer
} from 'recharts';
import { NoteEvent } from '../types';
import { formatPitch } from '../utils/pitchUtils';
import { PIXELS_PER_SECOND } from './constants';

interface HeatmapProps {
  notes: NoteEvent[];
  currentTime?: number;
  totalDuration?: number;
  scrollRef?: React.RefObject<HTMLDivElement>;
  onScroll?: (e: React.UIEvent<HTMLDivElement>) => void;
}

const ConfidenceHeatmap: React.FC<HeatmapProps> = ({ notes, currentTime = 0, totalDuration = 60, scrollRef, onScroll }) => {

  // Sync scroll with playhead
  useEffect(() => {
    if (scrollRef && scrollRef.current) {
      // 120px is the left margin offset
      const offset = 120; 
      const playheadX = offset + (currentTime * PIXELS_PER_SECOND);
      const containerWidth = scrollRef.current.clientWidth;
      const targetScroll = playheadX - (containerWidth / 2);
      
      if (Math.abs(scrollRef.current.scrollLeft - targetScroll) > 50) {
        scrollRef.current.scrollTo({
            left: Math.max(0, targetScroll),
            behavior: 'smooth'
        });
      }
    }
  }, [currentTime, scrollRef]);

  const data = useMemo(() => {
      return notes.map(n => {
        const roundedPitch = Math.round(n.midi_pitch);
        const cents = (n.midi_pitch - roundedPitch) * 100; 
        const noteLabel = formatPitch(n.midi_pitch, { 
            format: 'scientific', 
            accidentalStyle: 'sharp', 
            showOctave: true 
        });

        return {
          time: n.start_time, 
          confidence: n.confidence,
          pitch: n.midi_pitch,
          noteName: noteLabel.display,
          cents: cents,
          deviationLabel: cents > 0 ? `+${Math.round(cents)}` : `${Math.round(cents)}`
        };
      });
  }, [notes]);

  if (notes.length === 0) return (
      <div className="h-64 w-full bg-zinc-900 rounded-b-lg p-4 flex items-center justify-center text-zinc-600 text-sm border-t-0 border border-zinc-800">
          <div className="text-center">
              <p className="mb-2">No pitch data analyzed yet.</p>
              <p className="text-xs text-zinc-500">Upload audio or load a video to see intonation analysis.</p>
          </div>
      </div>
  );

  // Calculate Y-Axis Domain safely
  const pitches = data.map(d => d.pitch).filter(p => p > 20);
  const minPitch = pitches.length ? Math.min(...pitches) : 48; // C3 default
  const maxPitch = pitches.length ? Math.max(...pitches) : 72; // C5 default
  const domainMin = Math.floor(minPitch) - 2;
  const domainMax = Math.ceil(maxPitch) + 2;

  // Layout Calculations to match Timeline Overlay
  // Chart needs to be wide enough so that 1s = PIXELS_PER_SECOND
  // Margin Left: 120, Margin Right: 20
  // Inner Width = totalDuration * PIXELS_PER_SECOND
  const chartWidth = 120 + (Math.max(1, totalDuration) * PIXELS_PER_SECOND) + 20;

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      const cents = d.cents;
      const absCents = Math.abs(Math.round(cents));
      
      let tuneStatus = "Perfect";
      let tuneColor = "text-emerald-400";
      let barColor = "bg-emerald-500";
      
      if (cents > 10) { 
          tuneStatus = "Sharp"; 
          tuneColor = "text-rose-400"; 
          barColor = "bg-rose-500";
      } else if (cents < -10) { 
          tuneStatus = "Flat"; 
          tuneColor = "text-sky-400"; 
          barColor = "bg-sky-500";
      } else if (absCents > 3) {
          tuneStatus = "Slightly " + (cents > 0 ? "Sharp" : "Flat");
          tuneColor = "text-yellow-400";
          barColor = "bg-yellow-500";
      }

      return (
        <div className="bg-zinc-950/95 backdrop-blur border border-zinc-700 p-3 rounded-xl shadow-2xl text-xs z-50 min-w-[160px]">
          <div className="flex justify-between items-center mb-3 border-b border-zinc-800 pb-2">
             <div className="flex items-baseline gap-1">
                 <span className="text-white font-bold text-base">{d.noteName}</span>
                 <span className="text-zinc-500 font-mono text-[10px]">{d.pitch.toFixed(2)}</span>
             </div>
             <span className="text-zinc-400 font-mono">T: {d.time.toFixed(2)}s</span>
          </div>
          
          <div className="space-y-3">
             {/* Intonation Section */}
             <div>
                 <div className="flex justify-between items-center mb-1">
                    <span className="text-zinc-400 text-[10px] uppercase tracking-wide">Intonation</span>
                    <span className={`font-bold ${tuneColor} text-[10px]`}>
                        {d.deviationLabel}c {tuneStatus}
                    </span>
                 </div>
                 {/* Intonation Bar */}
                 <div className="w-full h-2 bg-zinc-800 rounded-full relative overflow-hidden">
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-zinc-500 z-10"></div> {/* Center marker */}
                    <div 
                        className={`absolute top-0 bottom-0 h-full rounded-full ${barColor} opacity-80`} 
                        style={{ 
                            left: cents >= 0 ? '50%' : `calc(50% + ${Math.max(-50, cents)}%)`,
                            width: `${Math.min(50, Math.abs(cents))}%`
                        }}
                    ></div>
                 </div>
                 <div className="flex justify-between text-[9px] text-zinc-600 mt-0.5 font-mono">
                     <span>-50c</span>
                     <span>0</span>
                     <span>+50c</span>
                 </div>
             </div>

             {/* Confidence Section */}
             <div className="flex justify-between items-center pt-1 border-t border-zinc-800/50">
                <span className="text-zinc-400 text-[10px] uppercase tracking-wide">Reliability</span>
                <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                        <div className="h-full bg-emerald-500" style={{ width: `${d.confidence * 100}%` }}></div>
                    </div>
                    <span className={d.confidence > 0.8 ? "text-emerald-400 font-bold" : "text-zinc-400"}>
                        {Math.round(d.confidence * 100)}%
                    </span>
                </div>
             </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-64 w-full bg-zinc-900 rounded-b-xl p-0 flex flex-col relative overflow-hidden border border-t-0 border-zinc-800">
      
      {/* Header Legend */}
      <div className="absolute top-3 left-4 z-20 pointer-events-none">
        <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-1 shadow-black drop-shadow-md">
            Intonation & Pitch Analysis
        </h3>
        <div className="flex gap-3 text-[10px] text-zinc-500">
            <div className="flex items-center gap-1.5">
                <div className="w-2 h-0.5 bg-indigo-500"></div> 
                <span>Pitch Contour</span>
            </div>
            <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded bg-emerald-500/20 border border-emerald-500/30"></div> 
                <span>Confidence Signal</span>
            </div>
        </div>
      </div>
      
      {/* Scrollable Chart Area */}
      <div ref={scrollRef} onScroll={onScroll} className="flex-1 w-full overflow-x-auto overflow-y-hidden relative scrollbar-hide">
        <div style={{ width: chartWidth, height: '100%', position: 'relative' }}>
            
            {/* Playhead Overlay */}
            <div 
                className="absolute top-0 bottom-0 z-10 transition-all duration-75 ease-linear pointer-events-none"
                style={{ 
                    // Matches Chart margin-left (120)
                    left: `${120 + (currentTime * PIXELS_PER_SECOND)}px`,
                }}
            >
                <div className="h-full w-px bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]"></div>
                <div className="absolute -top-1 -left-[3px] w-2 h-2 bg-red-500 rounded-full shadow-md"></div>
            </div>

            <ComposedChart 
                width={chartWidth} 
                height={256} 
                data={data} 
                margin={{ top: 20, right: 20, left: 120, bottom: 0 }} 
            >
                <defs>
                    <linearGradient id="colorConf" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0.0}/>
                    </linearGradient>
                </defs>
                
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={true} horizontal={true} opacity={0.5} />
                
                <XAxis 
                    dataKey="time" 
                    type="number"
                    domain={[0, totalDuration]} 
                    tick={{ fontSize: 10, fill: '#52525b', fontFamily: 'monospace' }} 
                    tickFormatter={(val) => val.toFixed(0) + 's'}
                    tickCount={Math.ceil(totalDuration / 5)}
                    axisLine={{ stroke: '#3f3f46' }}
                    tickLine={false}
                    scale="linear"
                    allowDataOverflow={true}
                    interval={0} // Force all ticks if space allows, or let recharts decide
                    minTickGap={30}
                />
                
                <YAxis 
                    yAxisId="pitch"
                    domain={[domainMin, domainMax]} 
                    tick={{ fontSize: 10, fill: '#71717a', fontWeight: 500 }}
                    tickFormatter={(val) => {
                        try {
                            return formatPitch(val, { format: 'scientific', accidentalStyle: 'sharp', showOctave: true }).display;
                        } catch { return ''; }
                    }}
                    tickLine={false}
                    axisLine={false}
                    width={40}
                    allowDecimals={false}
                    interval={0} // Attempt to show more notes
                />
                
                <YAxis yAxisId="confidence" orientation="right" domain={[0, 1]} hide />

                <Tooltip 
                    content={<CustomTooltip />} 
                    cursor={{ stroke: '#a1a1aa', strokeWidth: 1, strokeDasharray: '4 4' }} 
                    offset={20}
                    isAnimationActive={false}
                />

                <Area 
                    yAxisId="confidence"
                    type="monotone" 
                    dataKey="confidence" 
                    stroke="none"
                    fill="url(#colorConf)" 
                    isAnimationActive={false}
                />

                <Line 
                    yAxisId="pitch"
                    type="monotone" 
                    dataKey="pitch" 
                    stroke="#6366f1" 
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 4, fill: '#fff', stroke: '#6366f1', strokeWidth: 2 }}
                    connectNulls={false}
                    isAnimationActive={false}
                />
            </ComposedChart>
        </div>
      </div>
    </div>
  );
};

export default ConfidenceHeatmap;
