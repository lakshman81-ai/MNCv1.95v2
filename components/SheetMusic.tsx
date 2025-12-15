
import React, { useEffect, useRef, useState } from 'react';
import * as OSMDModule from 'opensheetmusicdisplay';

// Robust import handling for OSMD to support various build environments (ESM/CJS)
const getOSMDClass = () => {
    // @ts-ignore
    if (OSMDModule.OpenSheetMusicDisplay) return OSMDModule.OpenSheetMusicDisplay;
    // @ts-ignore
    if (OSMDModule.default?.OpenSheetMusicDisplay) return OSMDModule.default.OpenSheetMusicDisplay;
    // @ts-ignore
    if (typeof OSMDModule.default === 'function') return OSMDModule.default;
    return OSMDModule;
};

const OpenSheetMusicDisplay = getOSMDClass();

interface SheetMusicProps {
  musicXML?: string; // Content string
  currentTime?: number;
  bpm?: number;
}

const SheetMusic: React.FC<SheetMusicProps> = ({ 
    musicXML, currentTime, bpm = 120
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const osmdRef = useRef<any>(null);
  const [isReady, setIsReady] = useState(false);

  // Initialize OSMD
  useEffect(() => {
    if (!containerRef.current) return;

    // Cleanup previous instance if re-initializing
    if (osmdRef.current) {
        try {
            osmdRef.current.clear();
        } catch (e) { console.warn("OSMD Clear error", e); }
        osmdRef.current = null;
    }

    try {
        if (!OpenSheetMusicDisplay) {
            console.error("OpenSheetMusicDisplay library not loaded correctly.");
            return;
        }

        const osmd = new OpenSheetMusicDisplay(containerRef.current, {
            autoResize: true,
            backend: 'svg',
            drawingParameters: 'compacttight', 
            drawTitle: true,
            drawSubtitle: true,
            drawComposer: true,
            drawCredits: false,
            // Enable cursor options
            followCursor: true,
        });
        
        // Custom styling for the cursor
        osmd.setOptions({
            cursorsOptions: [{
                type: 1, // Vertical Line
                color: "#EF4444", // Red-500
                alpha: 0.8,
                width: 3,
            }],
        });

        osmdRef.current = osmd;
        setIsReady(true);
    } catch (e) {
        console.error("Failed to initialize OpenSheetMusicDisplay", e);
    }
  }, []); 

  // Load XML
  useEffect(() => {
      if (isReady && osmdRef.current && musicXML) {
          const loadScore = async () => {
              try {
                  await osmdRef.current.load(musicXML);
                  osmdRef.current.render();
                  if (osmdRef.current.cursor) {
                      osmdRef.current.cursor.show();
                      osmdRef.current.cursor.reset();
                  }
              } catch (e) {
                  console.error("OSMD Load Error:", e);
              }
          };
          loadScore();
      }
  }, [isReady, musicXML]);

  // Update Cursor based on currentTime
  useEffect(() => {
      if (isReady && osmdRef.current && osmdRef.current.cursor && musicXML && currentTime !== undefined) {
          try {
              // Convert seconds to beats
              // Assuming 4/4 and BPM provided (default 120 in generator)
              const secondsPerBeat = 60 / bpm;
              const currentBeat = currentTime / secondsPerBeat;
              
              const cursor = osmdRef.current.cursor;
              // Only move if we are not at the end
              if (!cursor.iterator.endReached) {
                  // Reset if we jumped back (naive)
                  if (currentTime < 0.2) cursor.reset();
                  
                  // Helper: Convert OSMD timestamp to seconds
                  // OSMD uses "Measures". We need "RealValue" which is beats (usually quarter notes).
                  const iteratorTime = cursor.iterator.currentTimeStamp.RealValue * 4; // MusicXML Measures to Beats (assuming 4/4)
                  
                  // If our visual cursor is behind real time, advance it
                  // We limit loop to avoid freezing
                  let steps = 0;
                  while (iteratorTime < currentBeat && !cursor.iterator.endReached && steps < 50) {
                      cursor.next();
                      steps++;
                  }
              }
          } catch(e) {
              // Ignore cursor errors during seek
          }
      }
  }, [currentTime, isReady, musicXML, bpm]);

  return (
    <div className="w-full h-full min-h-[400px] overflow-auto bg-white rounded-xl shadow-sm p-4">
        <div ref={containerRef} className="w-full h-full" />
    </div>
  );
};

export default SheetMusic;
