import React, { useState } from 'react';
import { XIcon, PlayIcon } from './Icons';
import { LabelSettings } from '../types';
import { VOICES, STYLES } from './constants';
import { audioEngine } from '../services/audioEngine';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  labelSettings: LabelSettings;
  onLabelSettingsChange: (s: LabelSettings) => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, labelSettings, onLabelSettingsChange }) => {
  const [activeTab, setActiveTab] = useState<'keyboard'|'notation'|'sound'>('notation');

  if (!isOpen) return null;

  const handleChange = (key: keyof LabelSettings, value: any) => {
    onLabelSettingsChange({ ...labelSettings, [key]: value });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal Content */}
      <div className="relative bg-zinc-900 w-full max-w-3xl max-h-[85vh] rounded-xl border border-zinc-800 shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-800 bg-zinc-900/50">
          <div className="flex gap-4">
             <button 
                onClick={() => setActiveTab('notation')}
                className={`text-sm font-semibold transition-colors ${activeTab === 'notation' ? 'text-white border-b-2 border-indigo-500' : 'text-zinc-500 hover:text-zinc-300'}`}
             >
                Notation
             </button>
             <button 
                onClick={() => setActiveTab('keyboard')}
                className={`text-sm font-semibold transition-colors ${activeTab === 'keyboard' ? 'text-white border-b-2 border-indigo-500' : 'text-zinc-500 hover:text-zinc-300'}`}
             >
                Keyboard
             </button>
             <button
                onClick={() => setActiveTab('sound')}
                className={`text-sm font-semibold transition-colors ${activeTab === 'sound' ? 'text-white border-b-2 border-indigo-500' : 'text-zinc-500 hover:text-zinc-300'}`}
             >
                Sound & Rhythm
             </button>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors">
            <XIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          
          {activeTab === 'notation' && (
            <section className="space-y-6">
                {/* Preview Section - Moved to Top & Sticky */}
                <div className="sticky top-0 z-20 -mx-2 p-4 bg-zinc-950/95 backdrop-blur border-b border-zinc-800 rounded-lg flex items-center justify-between shadow-lg">
                    <div className="flex flex-col">
                        <span className="text-xs text-zinc-500 uppercase font-bold">Live Preview</span>
                        <span className="text-[10px] text-zinc-600">See how your notes will appear</span>
                    </div>
                    <div className="flex gap-6 items-center pr-4">
                        <div className="relative w-16 h-24 border-b border-t border-zinc-600 flex items-center justify-center bg-zinc-900 rounded-sm">
                            <div className="w-4 h-3 bg-white rounded-full transform -rotate-12"></div>
                            <div className="absolute w-0.5 h-12 bg-white right-5 top-[-15px]"></div>
                            {labelSettings.showLabels && (
                                <span className={`absolute text-xs text-indigo-400 font-bold whitespace-nowrap ${
                                    labelSettings.position === 'above' ? '-top-6' : 
                                    labelSettings.position === 'below' ? '-bottom-6' : 
                                    'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 mix-blend-difference'
                                }`}>
                                    {labelSettings.format === 'solfege' ? 'Do' : `C${labelSettings.accidentalStyle === 'sharp' ? '♯' : labelSettings.accidentalStyle === 'flat' ? '♭' : 'x'}4`}
                                    {labelSettings.showOctave && labelSettings.format !== 'note_only' ? '' : ''}
                                    {labelSettings.showCentOffset ? <span className="text-[8px] text-zinc-500 ml-0.5">+5c</span> : ''}
                                </span>
                            )}
                        </div>
                        
                        {/* Play Icon to audition the note */}
                        <button 
                            type="button"
                            onClick={() => audioEngine.playTone(60, 0.5, labelSettings.selectedVoice)}
                            className="flex items-center justify-center w-12 h-12 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white transition-all shadow-lg shadow-indigo-500/20 border border-indigo-400/50 group"
                            title={`Play ${VOICES.find(v => v.id === labelSettings.selectedVoice)?.name || 'Note'}`}
                        >
                            <PlayIcon className="w-5 h-5 fill-current ml-0.5 group-active:scale-95 transition-transform" />
                        </button>
                    </div>
                </div>

                <div className="flex items-center justify-between pt-2">
                    <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-wider">Note Labels / Pitch Names</h3>
                    <label className="flex items-center space-x-2 cursor-pointer">
                        <span className="text-sm text-zinc-300 mr-2">Show Labels</span>
                        <div className="relative inline-block w-10 h-6 align-middle select-none transition duration-200 ease-in">
                            <input 
                                type="checkbox" 
                                checked={labelSettings.showLabels}
                                onChange={(e) => handleChange('showLabels', e.target.checked)}
                                className="toggle-checkbox absolute block w-4 h-4 rounded-full bg-white border-4 appearance-none cursor-pointer translate-x-1 checked:translate-x-5 checked:border-indigo-500 transition-transform top-1"
                            />
                            <div className={`block overflow-hidden h-6 rounded-full cursor-pointer ${labelSettings.showLabels ? 'bg-indigo-600' : 'bg-zinc-700'}`}></div>
                        </div>
                    </label>
                </div>

                <div className={`grid grid-cols-1 md:grid-cols-2 gap-6 transition-opacity ${labelSettings.showLabels ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
                    
                    {/* Format */}
                    <div className="space-y-3">
                        <label className="text-xs font-medium text-zinc-400">Label Format</label>
                        <div className="space-y-2">
                            <label className="flex items-center space-x-2 p-2 rounded hover:bg-zinc-800/50 cursor-pointer">
                                <input 
                                    type="radio" 
                                    checked={labelSettings.format === 'scientific'}
                                    onChange={() => handleChange('format', 'scientific')}
                                    className="text-indigo-500 bg-zinc-900 border-zinc-700" 
                                />
                                <span className="text-sm text-zinc-300">Scientific (C4, C#4)</span>
                            </label>
                            <label className="flex items-center space-x-2 p-2 rounded hover:bg-zinc-800/50 cursor-pointer">
                                <input 
                                    type="radio" 
                                    checked={labelSettings.format === 'note_only'}
                                    onChange={() => handleChange('format', 'note_only')}
                                    className="text-indigo-500 bg-zinc-900 border-zinc-700" 
                                />
                                <span className="text-sm text-zinc-300">Note only (C, C#)</span>
                            </label>
                            <label className="flex items-center space-x-2 p-2 rounded hover:bg-zinc-800/50 cursor-pointer">
                                <input 
                                    type="radio" 
                                    checked={labelSettings.format === 'solfege'}
                                    onChange={() => handleChange('format', 'solfege')}
                                    className="text-indigo-500 bg-zinc-900 border-zinc-700" 
                                />
                                <span className="text-sm text-zinc-300">Solfège (Do, Re)</span>
                            </label>
                        </div>
                    </div>

                    {/* Accidental Style */}
                    <div className="space-y-3">
                        <label className="text-xs font-medium text-zinc-400">Accidental Style</label>
                        <select 
                            value={labelSettings.accidentalStyle}
                            onChange={(e) => handleChange('accidentalStyle', e.target.value)}
                            className="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-indigo-500"
                        >
                            <option value="sharp">♯ / Sharp</option>
                            <option value="flat">♭ / Flat</option>
                            <option value="double_sharp">x (Classic)</option>
                        </select>

                        <div className="pt-2 space-y-3">
                            <label className="flex items-center space-x-2">
                                <input 
                                    type="checkbox" 
                                    checked={labelSettings.showOctave}
                                    onChange={(e) => handleChange('showOctave', e.target.checked)}
                                    className="rounded border-zinc-700 text-indigo-500 bg-zinc-900" 
                                />
                                <span className="text-sm text-zinc-300">Include Octave Number</span>
                            </label>
                            <label className="flex items-center space-x-2">
                                <input 
                                    type="checkbox" 
                                    checked={labelSettings.showCentOffset}
                                    onChange={(e) => handleChange('showCentOffset', e.target.checked)}
                                    className="rounded border-zinc-700 text-indigo-500 bg-zinc-900" 
                                />
                                <span className="text-sm text-zinc-300">Show Cent Offset (+12c)</span>
                            </label>
                        </div>
                    </div>

                    {/* Position */}
                    <div className="space-y-3">
                        <label className="text-xs font-medium text-zinc-400">Label Position</label>
                        <div className="flex gap-2">
                            {['above', 'inside', 'below'].map(pos => (
                                <button 
                                    key={pos}
                                    onClick={() => handleChange('position', pos)}
                                    className={`flex-1 py-2 text-xs rounded border capitalize transition-colors ${labelSettings.position === pos ? 'bg-indigo-600 border-indigo-500 text-white' : 'bg-zinc-900 border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}
                                >
                                    {pos}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Confidence */}
                    <div className="space-y-2">
                        <div className="flex justify-between">
                            <label className="text-xs font-medium text-zinc-400">Min Confidence Threshold</label>
                            <span className="text-xs text-zinc-500">{Math.round(labelSettings.minConfidence * 100)}%</span>
                        </div>
                        <input 
                            type="range" 
                            min="0" max="1" step="0.05"
                            value={labelSettings.minConfidence}
                            onChange={(e) => handleChange('minConfidence', parseFloat(e.target.value))}
                            className="w-full accent-indigo-500 h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer" 
                        />
                        <p className="text-[10px] text-zinc-500">Only show labels for notes with high ML confidence.</p>
                    </div>

                </div>
            </section>
          )}

          {activeTab === 'keyboard' && (
             <section className="space-y-6">
                 <div className="space-y-4">
                    <label className="text-sm font-medium text-zinc-300">Keyboard Size & Range</label>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        {[
                          { size: 37, label: 'Mini', range: 'F3 - F6' },
                          { size: 49, label: 'Synth', range: 'C2 - C6' },
                          { size: 54, label: 'Toy / Entry', range: 'C2 - F6' },
                          { size: 61, label: 'Standard', range: 'C2 - C7' },
                          { size: 76, label: 'Semi-Weighted', range: 'E1 - G7' },
                          { size: 88, label: 'Full Piano', range: 'A0 - C8' }
                        ].map((opt) => (
                            <button
                                key={opt.size}
                                onClick={() => handleChange('keyboardSize', opt.size)}
                                className={`flex flex-col items-start py-3 px-4 rounded-lg border text-left transition-all relative overflow-hidden ${
                                    labelSettings.keyboardSize === opt.size
                                    ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-500/20'
                                    : 'bg-zinc-900 border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:border-zinc-600'
                                }`}
                            >
                                <span className="text-sm font-bold">{opt.size} Keys</span>
                                <span className={`text-[11px] mt-0.5 ${labelSettings.keyboardSize === opt.size ? 'text-indigo-200' : 'text-zinc-500'}`}>{opt.label}</span>
                                <span className={`text-[10px] font-mono mt-1 opacity-60`}>{opt.range}</span>
                            </button>
                        ))}
                    </div>
                    <p className="text-xs text-zinc-500 mt-2">
                        Generated notes will be constrained to fit within the selected instrument's range.
                    </p>
                 </div>
             </section>
          )}

          {activeTab === 'sound' && (
             <section className="space-y-6">
                 <div className="space-y-6">
                    <div>
                        <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-wider mb-4">Instrument Voice</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {['Piano', 'Indian', 'Strings', 'Guitar', 'Synth'].map(cat => (
                                <div key={cat} className="space-y-2">
                                    <h4 className="text-xs font-semibold text-zinc-500 uppercase">{cat}</h4>
                                    <div className="space-y-1">
                                        {VOICES.filter(v => v.category === cat).map(voice => (
                                            <label key={voice.id} className="flex items-center justify-between p-2 rounded hover:bg-zinc-800/50 cursor-pointer group">
                                                <div className="flex items-center space-x-3">
                                                    <input
                                                        type="radio"
                                                        name="voice_select"
                                                        checked={labelSettings.selectedVoice === voice.id}
                                                        onChange={() => handleChange('selectedVoice', voice.id)}
                                                        className="text-indigo-500 bg-zinc-900 border-zinc-700 focus:ring-indigo-500"
                                                    />
                                                    <span className={`text-sm group-hover:text-zinc-200 ${labelSettings.selectedVoice === voice.id ? 'text-white font-medium' : 'text-zinc-400'}`}>
                                                        {voice.name}
                                                    </span>
                                                </div>
                                                <button 
                                                    onClick={(e) => {
                                                        e.preventDefault();
                                                        e.stopPropagation();
                                                        audioEngine.playTone(60, 0.5, voice.id);
                                                    }}
                                                    className="p-1.5 rounded-full hover:bg-zinc-700 text-zinc-500 hover:text-white transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
                                                    title="Preview Voice"
                                                >
                                                    <PlayIcon className="w-3 h-3 fill-current" />
                                                </button>
                                            </label>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="border-t border-zinc-800 pt-6">
                        <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-wider mb-4">Rhythm / Style</h3>
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                             {STYLES.map(style => (
                                 <button
                                    key={style.id}
                                    onClick={() => handleChange('selectedStyle', style.id)}
                                    className={`text-left px-3 py-2 rounded-lg text-sm border transition-all ${
                                        labelSettings.selectedStyle === style.id
                                        ? 'bg-indigo-900/40 border-indigo-500 text-indigo-200'
                                        : 'bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-300'
                                    }`}
                                 >
                                     <div className="font-medium">{style.name}</div>
                                     <div className="text-[10px] opacity-60 font-mono mt-0.5">{style.timeSignature}</div>
                                 </button>
                             ))}
                        </div>
                    </div>
                 </div>
             </section>
          )}

        </div>
        
        {/* Footer */}
        <div className="p-4 border-t border-zinc-800 bg-zinc-900/50 flex justify-end gap-3">
            <button onClick={onClose} className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors">Close</button>
        </div>

      </div>
    </div>
  );
};

export default SettingsModal;