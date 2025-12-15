
import React, { useState, useRef, useEffect } from 'react';
import { NoteEvent, AudioState, HistoryEntry, LabelSettings, ChordEvent } from './types';
import { PlayIcon, PauseIcon, UploadIcon, SettingsIcon, DownloadIcon, MusicIcon, HistoryIcon, TrashIcon, ActivityIcon, SegmentIcon, NextIcon, ChevronLeftIcon, ChevronRightIcon, MinusIcon, PlusIcon, LightBulbIcon, RefreshIcon, PianoIcon, SwatchIcon, StyleIcon } from './components/Icons';
import Equalizer from './components/Equalizer';
import SheetMusic from './components/SheetMusic';
import ConfidenceHeatmap from './components/ConfidenceHeatmap';
import SettingsModal from './components/SettingsModal';
import HistoryModal from './components/HistoryModal';
import SuggestionPopup from './components/SuggestionPopup';
import YouTubePlayer from './components/YouTubePlayer';
import { Toast, ToastType } from './components/Toast';
import { audioEngine } from './services/audioEngine';
import { HistoryService } from './services/historyService';
import { SuggestionService, SuggestedSettings } from './services/suggestionService';
import { TranscriptionService } from './services/transcriptionService';
import { RHYTHM_PATTERNS, STYLES, VOICES, GENRES } from './components/constants';

// --- Deterministic & Composition Engine ---

// Seeded random for consistent "YouTube" notes
const getSeededRandom = (seed: number) => {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
};

const generateId = (): string => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return 'sess_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
};

const generateThumbnail = (title: string): string => {
  const hash = title.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const hue = hash % 360;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(`
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
      <rect width="100" height="100" fill="hsl(${hue}, 20%, 20%)" />
      <path d="M0,50 Q25,${40 + (hash % 20)} 50,50 T100,50" stroke="hsl(${hue}, 70%, 60%)" stroke-width="3" fill="none" opacity="0.8"/>
    </svg>
  `)}`;
};

const getYoutubeId = (urlStr: string) => {
    try {
        const url = new URL(urlStr);
        if (url.hostname === 'youtu.be') {
            return url.pathname.slice(1);
        }
        if (url.hostname.includes('youtube.com')) {
            const v = url.searchParams.get('v');
            if (v) return v;
            if (url.pathname.startsWith('/embed/')) return url.pathname.split('/')[2];
            if (url.pathname.startsWith('/v/')) return url.pathname.split('/')[2];
        }
    } catch (e) {
        return null;
    }
    return null;
};

const App: React.FC = () => {
  // --- Refs ---
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioBufferRef = useRef<AudioBuffer | null>(null); // Store decoded audio

  // --- Scroll Synchronization Refs ---
  const sheetMusicScrollRef = useRef<HTMLDivElement>(null);

  // --- State ---
  const [audioState, setAudioState] = useState<AudioState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    volume: 1,
    sourceUrl: null,
    sourceType: 'youtube'
  });
  
  const [audioCrossOrigin, setAudioCrossOrigin] = useState<'anonymous' | undefined>('anonymous');
  
  // MusicXML Content
  const [musicXML, setMusicXML] = useState<string | undefined>(undefined);
  const [transcribedNotes, setTranscribedNotes] = useState<NoteEvent[]>([]); // Store notes for playback
  const [detectedChords, setDetectedChords] = useState<ChordEvent[]>([]);
  const [currentChord, setCurrentChord] = useState<string>('');

  // Synth Playback State
  const [isSynthPlaying, setIsSynthPlaying] = useState(false);

  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlayerReady, setIsPlayerReady] = useState(false);
  const [isRestricted, setIsRestricted] = useState(false); 
  
  const [ytUrl, setYtUrl] = useState('');
  const [ytVideoId, setYtVideoId] = useState<string | null>(null);
  const [seekTarget, setSeekTarget] = useState<number | null>(null);
  const [isBuffering, setIsBuffering] = useState(false);
  
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: ToastType } | null>(null);
  
  const [labelSettings, setLabelSettings] = useState<LabelSettings>({
    showLabels: true,
    format: 'scientific',
    accidentalStyle: 'sharp',
    showOctave: true,
    showCentOffset: false,
    position: 'above',
    minConfidence: 0.4,
    keyboardSize: 61,
    selectedVoice: 'piano',
    selectedStyle: 'none'
  });

  const [compositionGenre, setCompositionGenre] = useState('Ballad');

  // Rhythm State
  const [bpm, setBpm] = useState(120);

  // Suggestion State
  const [suggestedSettings, setSuggestedSettings] = useState<SuggestedSettings | null>(null);
  const [isSuggestionOpen, setIsSuggestionOpen] = useState(false);


  // Update current chord based on time
  useEffect(() => {
      if (detectedChords.length > 0) {
          const t = audioState.currentTime;
          const chord = detectedChords.find(c => t >= c.time && t < c.time + c.duration);
          setCurrentChord(chord ? chord.text : '');
      } else {
          setCurrentChord('');
      }
  }, [audioState.currentTime, detectedChords]);

  // Audio Playback Synchronization Effect
  useEffect(() => {
    if (audioState.sourceType === 'file' && audioRef.current) {
        if (audioState.isPlaying && !isSynthPlaying) {
            const playPromise = audioRef.current.play();
            if (playPromise !== undefined) {
                playPromise.catch(e => {
                    if (e.name !== 'AbortError') {
                        console.error("Playback failed:", e);
                        showToast("Playback failed", "error");
                        setAudioState(p => ({...p, isPlaying: false}));
                    }
                });
            }
        } else {
            audioRef.current.pause();
        }
    }
  }, [audioState.isPlaying, audioState.sourceType, isSynthPlaying]);

  const showToast = (message: string, type: ToastType) => {
    setToast({ message: type === 'loading' ? 'Loading...' : message, type });
    if (type === 'loading' && message) setToast({ message, type });
  };

  const resetSession = () => {
      audioEngine.stopAllTones();
      audioEngine.stopSequence();
      setIsSynthPlaying(false);
      setMusicXML(undefined);
      setTranscribedNotes([]);
      setDetectedChords([]);
      setCurrentChord('');
      setAudioState(prev => ({ ...prev, currentTime: 0, isPlaying: false, duration: 0 }));
      setIsPlayerReady(false); 
      setIsRestricted(false);
      setIsProcessing(false);
      audioBufferRef.current = null; // Clear buffer
      if (audioRef.current) audioRef.current.currentTime = 0;
      setSeekTarget(0);
  };

  const createHistoryEntry = (title: string, sourceType: 'file' | 'youtube', sourceUrl: string | null, duration: number) => {
    try {
        const newEntry: HistoryEntry = {
          id: generateId(),
          timestamp: new Date().toISOString(),
          title: title,
          source_type: sourceType,
          source_url: sourceUrl,
          audio_duration_sec: duration,
          notes_count: transcribedNotes.length,
          avg_confidence: 0,
          bpm_detected: 120,
          time_signature: "4/4",
          instrument_estimate: sourceType === 'youtube' ? "Composition" : "Audio Analysis",
          tags: ["transcription"],
          user_edits: { notes_modified: 0, notes_deleted: 0, notes_added: 0 },
          exports: { musicxml: true, midi: false, pdf: false, csv: false },
          thumbnail: generateThumbnail(title)
        };
        HistoryService.addEntry(newEntry);
    } catch (e) { console.warn("History error", e); }
  };

  // --- Handlers ---

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      showToast("Loading and transcribing audio...", "loading");
      resetSession(); 
      setIsProcessing(true);
      
      try {
        // 1. Load Audio for Playback
        const buffer = await audioEngine.loadAudioFile(file);
        audioBufferRef.current = buffer;
        const url = URL.createObjectURL(file);
        
        setAudioCrossOrigin(undefined); 
        setAudioState(prev => ({ 
            ...prev, 
            sourceUrl: url, 
            sourceType: 'file',
            duration: buffer.duration
        }));
        setYtVideoId(null);
        setIsPlayerReady(true);

        // 2. Transcribe
        const result = await TranscriptionService.transcribeAudio(file);
        setMusicXML(result.xml);
        setTranscribedNotes(result.notes);
        setDetectedChords(result.chords);

        showToast("Transcription Complete", "success");
        createHistoryEntry(file.name, 'file', null, buffer.duration);

      } catch (e) {
        console.error(e);
        showToast("Failed to process audio file", "error");
      } finally {
        setIsProcessing(false);
      }
    }
  };

  const handleYoutubeLoad = () => {
    const id = getYoutubeId(ytUrl);
    if (!id) {
        showToast("Invalid YouTube URL", "error");
        return;
    }
    resetSession();
    showToast("Loading Music...", "loading");
    setYtVideoId(id);
    setAudioCrossOrigin('anonymous');
    setAudioState(prev => ({ ...prev, sourceType: 'youtube', sourceUrl: ytUrl }));
  };

  const onYoutubePlayerReady = (duration: number) => {
      setAudioState(prev => ({ ...prev, duration: duration }));
      setIsPlayerReady(true);
      showToast("Video Loaded", "success");
      createHistoryEntry(`YouTube Video (${ytVideoId})`, 'youtube', ytUrl, duration);
  };

  const handleYoutubeError = (error: { code: number, message: string }) => {
      if (error.code === 150 || error.code === 101 || error.code === 153) {
          setIsRestricted(true);
          showToast("Playback restricted.", "info");
          setAudioState(prev => ({ ...prev, duration: prev.duration || 180 }));
      } else {
          showToast(error.message, "error");
          setIsPlayerReady(false);
          setIsProcessing(false);
      }
  };

  const togglePlay = async () => {
    // If synth is playing, stop it and switch to normal audio
    if (isSynthPlaying) {
        audioEngine.stopSequence();
        setIsSynthPlaying(false);
        setAudioState(prev => ({ ...prev, isPlaying: false }));
        return;
    }

    if (isRestricted) {
        showToast("Playback is disabled for this video (Copyright)", "error");
        return;
    }
    if (!isPlayerReady && audioState.sourceType !== 'file') {
        showToast("Please wait for music to load", "info");
        return;
    }
    if (isProcessing) {
        showToast("Processing...", "info");
        return;
    }

    const shouldPlay = !audioState.isPlaying;
    setAudioState(prev => ({ ...prev, isPlaying: shouldPlay }));

    if (audioState.sourceType === 'file' && shouldPlay) {
        try {
            await audioEngine.ensureContext();
            if (audioRef.current && audioRef.current.isConnected) {
                audioEngine.connectElement(audioRef.current);
            }
        } catch(e) {
            console.error("Audio Context Resume failed:", e);
        }
    }
  };

  const toggleSynthPlay = () => {
      if (isSynthPlaying) {
          audioEngine.stopSequence();
          setIsSynthPlaying(false);
      } else {
          if (transcribedNotes.length === 0 && !musicXML) {
              showToast("No notes to play", "info");
              return;
          }
          
          // Pause main audio if playing
          setAudioState(prev => ({ ...prev, isPlaying: false }));
          
          setIsSynthPlaying(true);
          audioEngine.playSequence(
              transcribedNotes, 
              labelSettings.selectedVoice,
              (time) => {
                  setAudioState(prev => ({ ...prev, currentTime: time }));
              },
              () => {
                  setIsSynthPlaying(false);
                  setAudioState(prev => ({ ...prev, currentTime: 0 }));
              }
          );
      }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setAudioState(prev => ({ ...prev, currentTime: time }));
    
    // Stop synth if seeking
    if (isSynthPlaying) {
        audioEngine.stopSequence();
        setIsSynthPlaying(false);
    }

    if (audioState.sourceType === 'file' && audioRef.current) {
        audioRef.current.currentTime = time;
    } else if (audioState.sourceType === 'youtube' && !isRestricted) {
        setSeekTarget(time);
        setTimeout(() => setSeekTarget(null), 100);
    }
  };

  const handleNativeTimeUpdate = () => {
    if (audioRef.current && !isSynthPlaying) {
      const time = audioRef.current.currentTime;
      setAudioState(prev => ({ ...prev, currentTime: time }));
    }
  };

  const handleYoutubeTimeUpdate = (time: number) => {
      if (!isSynthPlaying) {
          setAudioState(prev => ({ ...prev, currentTime: time }));
      }
  };

  const handleAcceptSuggestion = () => {
    if (suggestedSettings) {
      setLabelSettings(prev => ({
        ...prev,
        selectedVoice: suggestedSettings.voice,
        selectedStyle: suggestedSettings.style,
      }));
      setBpm(suggestedSettings.bpm);
      showToast("Settings applied", "success");
    }
    setIsSuggestionOpen(false);
  };

  const handleRejectSuggestion = () => {
    setIsSuggestionOpen(false);
  };

  // Check if player is strictly enabled
  const isPlayDisabled = 
    isProcessing || 
    (audioState.sourceType === 'file' && !audioState.sourceUrl) ||
    (!isPlayerReady && audioState.sourceType !== 'file' && !isRestricted);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200 flex flex-col font-sans selection:bg-indigo-500/30">
      
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      <SettingsModal 
        isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} 
        labelSettings={labelSettings} onLabelSettingsChange={setLabelSettings}
      />
      
      <HistoryModal 
        isOpen={isHistoryOpen} onClose={() => setIsHistoryOpen(false)} onLoadEntry={() => {}}
      />

      <SuggestionPopup
        isOpen={isSuggestionOpen}
        settings={suggestedSettings}
        onAccept={handleAcceptSuggestion}
        onReject={handleRejectSuggestion}
      />

      <audio 
        ref={audioRef} 
        src={audioState.sourceType === 'file' ? audioState.sourceUrl || undefined : undefined}
        crossOrigin={audioCrossOrigin}
        onTimeUpdate={handleNativeTimeUpdate}
        onEnded={() => setAudioState(prev => ({ ...prev, isPlaying: false }))}
        onPlay={() => setAudioState(prev => ({ ...prev, isPlaying: true }))}
        onPause={() => setAudioState(prev => ({ ...prev, isPlaying: false }))}
        onWaiting={() => setIsBuffering(true)}
        onPlaying={() => setIsBuffering(false)}
        onError={(e) => {
            if (audioState.sourceType === 'file') {
                console.error("Audio playback error", e);
                showToast("Audio playback error", "error");
            }
        }}
        className="hidden"
      />

      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
              <MusicIcon className="text-white w-5 h-5" />
            </div>
            <h1 className="font-bold text-lg tracking-tight text-white">Music Note Creator</h1>
          </div>
          <div className="flex items-center gap-3">
             <button title="Toggle Note Labels" onClick={() => setLabelSettings(s => ({ ...s, showLabels: !s.showLabels }))} className={`hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm font-medium border ${labelSettings.showLabels ? 'bg-indigo-900/30 text-indigo-300 border-indigo-500/30' : 'bg-zinc-800/50 text-zinc-400 border-zinc-700/50'}`}>
                <span className="font-bold font-serif italic">ABC</span>
             </button>
            <button title="Project History" onClick={() => setIsHistoryOpen(true)} className="p-2 text-zinc-400 hover:text-white bg-zinc-800/50 rounded-full hover:bg-zinc-700 transition-colors">
              <HistoryIcon className="w-5 h-5" />
            </button>
            <button title="Settings" onClick={() => setIsSettingsOpen(true)} className="p-2 text-zinc-400 hover:text-white bg-zinc-800/50 rounded-full hover:bg-zinc-700 transition-colors">
              <SettingsIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Sidebar: Controls */}
        <section className="lg:col-span-4 flex flex-col gap-6">
          
          {/* Audio Source */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4">Audio Source</h2>
            
            <div className="flex p-1 bg-zinc-950 rounded-full mb-4">
              <button 
                title="Use YouTube Source"
                className={`flex-1 py-1.5 text-sm rounded-full font-medium transition-all ${audioState.sourceType === 'youtube' ? 'bg-zinc-800 text-white shadow' : 'text-zinc-500 hover:text-zinc-300'}`}
                onClick={() => setAudioState(prev => ({ ...prev, sourceType: 'youtube' }))}
              >
                YouTube
              </button>
              <button 
                title="Upload Audio File"
                className={`flex-1 py-1.5 text-sm rounded-full font-medium transition-all ${audioState.sourceType === 'file' ? 'bg-zinc-800 text-white shadow' : 'text-zinc-500 hover:text-zinc-300'}`}
                onClick={() => setAudioState(prev => ({ ...prev, sourceType: 'file' }))}
              >
                Upload
              </button>
            </div>

            {audioState.sourceType === 'youtube' ? (
              <div className="space-y-4">
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      placeholder="Paste YouTube URL..." 
                      className="flex-1 bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 text-white"
                      value={ytUrl}
                      onChange={(e) => setYtUrl(e.target.value)}
                    />
                    <button 
                        onClick={handleYoutubeLoad}
                        disabled={isProcessing}
                        className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white px-3 py-2 rounded-lg transition-colors"
                        title="Load Video"
                    >
                        {isProcessing || (!isPlayerReady && ytVideoId && !isRestricted) ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <DownloadIcon className="w-4 h-4" />}
                    </button>
                  </div>
              </div>
            ) : (
              <div title="Click to upload audio file" onClick={() => fileInputRef.current?.click()} className="border-2 border-dashed border-zinc-700 hover:border-indigo-500 hover:bg-zinc-800/50 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all group mb-4">
                <UploadIcon className="w-8 h-8 text-zinc-500 group-hover:text-indigo-400 mb-2" />
                <span className="text-sm text-zinc-400 group-hover:text-zinc-200">Upload File</span>
                <input type="file" ref={fileInputRef} className="hidden" accept="audio/*,.mp3,.mpeg,.wav,.m4a" onChange={handleFileUpload} />
              </div>
            )}

            <div className="border-t border-zinc-800 pt-4 mt-4 flex flex-col gap-4">
                <div className="w-full flex flex-col gap-1 group">
                    <input 
                        type="range" min="0" max={audioState.duration || 1} step="0.1"
                        value={audioState.currentTime}
                        onChange={handleSeek}
                        className="w-full h-1 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                    />
                    <div className="flex justify-between text-xs text-zinc-500 font-mono group-hover:text-zinc-400">
                        <span>{Math.floor(audioState.currentTime / 60)}:{(Math.floor(audioState.currentTime) % 60).toString().padStart(2, '0')}</span>
                        <span>{Math.floor(audioState.duration / 60)}:{(Math.floor(audioState.duration) % 60).toString().padStart(2, '0')}</span>
                    </div>
                </div>
                
                <div className="flex items-center justify-between">
                    <button 
                        onClick={togglePlay}
                        disabled={isPlayDisabled}
                        className="flex items-center gap-2 bg-zinc-100 hover:bg-white text-black px-4 py-2 rounded-full font-bold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {audioState.isPlaying ? <PauseIcon className="w-4 h-4 fill-current" /> : <PlayIcon className="w-4 h-4 fill-current" />}
                        {audioState.isPlaying ? "Pause Audio" : "Play Audio"}
                    </button>
                    {isBuffering && <span className="text-xs text-indigo-400 animate-pulse">Buffering...</span>}
                </div>
            </div>
          </div>

          {/* Visualization Controls */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-0 overflow-hidden shadow-sm flex flex-col">
             {/* Header */}
             <div className="flex justify-between items-center p-4 pb-2 border-b border-zinc-800/50">
                 <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Visualization</h2>
             </div>
             <div className="p-4 space-y-4">
                <p className="text-xs text-zinc-500">
                    The sheet music below is generated by our advanced Python engine (Music21).
                </p>
             </div>
          </div>

        </section>

        {/* Right Content: Editors & Visuals */}
        <section className="lg:col-span-8 flex flex-col gap-4 relative">
            
            {/* Sheet Music Toolbar */}
            <div className="flex items-center justify-between -mb-2 px-1">
                <div className="flex items-center gap-3">
                    <h3 className="text-sm font-semibold text-zinc-400">Score Visualization</h3>
                    {currentChord && (
                        <div className="flex items-center gap-2 px-3 py-1 bg-indigo-900/30 border border-indigo-500/30 rounded-lg animate-in fade-in slide-in-from-left-2">
                            <span className="text-[10px] text-indigo-300 uppercase font-bold tracking-wider">CHORD</span>
                            <span className="text-sm font-bold text-white">{currentChord}</span>
                        </div>
                    )}
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setIsSettingsOpen(true)}
                        className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-full transition-colors border bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border-zinc-700 hover:border-zinc-500 hover:text-zinc-100"
                        title="Visualization Settings"
                    >
                        <SettingsIcon className="w-3 h-3" />
                        <span>Display</span>
                    </button>
                    <button
                        onClick={toggleSynthPlay}
                        className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-full transition-colors border ${isSynthPlaying ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border-zinc-700 hover:border-indigo-500 hover:text-indigo-400'}`}
                        title="Play transcribed notes synthetically"
                    >
                        {isSynthPlaying ? <PauseIcon className="w-3 h-3 fill-current" /> : <PlayIcon className="w-3 h-3 fill-current" />}
                        {isSynthPlaying ? "Stop Notes" : "Play Notes"}
                    </button>
                </div>
            </div>

            {/* Sheet Music Editor */}
            <div className="bg-white rounded-2xl shadow-lg overflow-hidden border border-zinc-200 relative min-h-[500px]">
                <SheetMusic 
                    musicXML={musicXML}
                    currentTime={audioState.currentTime}
                    bpm={bpm}
                />
                {!musicXML && !isProcessing && (
                    <div className="absolute inset-0 flex items-center justify-center text-zinc-400 text-sm">
                        Upload an audio file to generate sheet music
                    </div>
                )}
                 {isProcessing && (
                    <div className="absolute inset-0 flex items-center justify-center bg-white/80 backdrop-blur-sm z-10">
                        <div className="flex flex-col items-center gap-2">
                             <div className="w-8 h-8 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
                             <span className="text-indigo-600 font-medium">Transcribing...</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Visualizer */}
            <div className="h-32 mt-auto">
                <Equalizer isPlaying={audioState.isPlaying || isSynthPlaying} />
            </div>

            {/* Youtube Player Hidden Overlay */}
            {audioState.sourceType === 'youtube' && ytVideoId && (
                <div className="fixed bottom-6 left-6 w-48 h-28 rounded-xl overflow-hidden shadow-2xl border border-zinc-700 z-50 opacity-90 hover:opacity-100 transition-opacity">
                    <YouTubePlayer 
                        videoId={ytVideoId}
                        isPlaying={audioState.isPlaying}
                        onReady={onYoutubePlayerReady}
                        onStateChange={(isPlaying) => setAudioState(p => ({...p, isPlaying}))}
                        onTimeUpdate={handleYoutubeTimeUpdate}
                        seekTo={seekTarget}
                        onError={handleYoutubeError}
                    />
                </div>
            )}

        </section>

      </main>
    </div>
  );
};

export default App;