import React, { useEffect, useState } from 'react';
import { 
  XIcon, FilterIcon, CalendarIcon, RefreshIcon, TrashIcon, 
  MusicIcon, DownloadIcon, EditIcon, CopyIcon, TagIcon, SaveIcon 
} from './Icons';
import { HistoryService } from '../services/historyService';
import { HistoryEntry, RetentionPolicy } from '../types';

interface HistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoadEntry: (entry: HistoryEntry) => void;
}

const HistoryModal: React.FC<HistoryModalProps> = ({ isOpen, onClose, onLoadEntry }) => {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [selectedEntry, setSelectedEntry] = useState<HistoryEntry | null>(null);
  const [retentionPolicy, setRetentionPolicy] = useState<RetentionPolicy>('50_items');

  // Filter States
  const [filterText, setFilterText] = useState('');
  const [filterSource, setFilterSource] = useState<'all' | 'youtube' | 'file'>('all');
  const [filterMinConfidence, setFilterMinConfidence] = useState(0);
  const [filterBPM, setFilterBPM] = useState<[number, number]>([0, 300]);
  const [showFilters, setShowFilters] = useState(false);

  // Edit States
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState('');
  const [newTag, setNewTag] = useState('');
  const [isAddingTag, setIsAddingTag] = useState(false);

  const refreshData = () => {
    setHistory(HistoryService.getHistory());
    setRetentionPolicy(HistoryService.getRetentionPolicy());
  };

  useEffect(() => {
    if (isOpen) {
      refreshData();
      setSelectedEntry(null);
    }
  }, [isOpen]);

  // Actions
  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this history entry?')) {
      HistoryService.deleteEntry(id);
      refreshData();
      if (selectedEntry?.id === id) setSelectedEntry(null);
    }
  };

  const handleClearAll = () => {
    if (confirm('Clear entire history? This cannot be undone.')) {
      HistoryService.clearHistory();
      refreshData();
      setSelectedEntry(null);
    }
  };

  const handleDuplicate = () => {
    if (selectedEntry) {
        const copy = HistoryService.duplicateEntry(selectedEntry.id);
        refreshData();
        if (copy) setSelectedEntry(copy);
    }
  };

  const handleRename = () => {
    if (selectedEntry && renameValue.trim()) {
        HistoryService.renameEntry(selectedEntry.id, renameValue);
        refreshData();
        setSelectedEntry({ ...selectedEntry, title: renameValue });
        setIsRenaming(false);
    }
  };

  const handleAddTag = () => {
    if (selectedEntry && newTag.trim()) {
        HistoryService.addTag(selectedEntry.id, newTag.trim());
        refreshData();
        // Update local state to reflect immediately. Ensure tags exists.
        const currentTags = Array.isArray(selectedEntry.tags) ? selectedEntry.tags : [];
        setSelectedEntry({ ...selectedEntry, tags: [...currentTags, newTag.trim()] });
        setNewTag('');
        setIsAddingTag(false);
    }
  };

  const handleRetentionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value as RetentionPolicy;
    HistoryService.setRetentionPolicy(val);
    refreshData();
  };

  // Filter Logic
  const filteredHistory = history.filter(entry => {
    // Safe accessors
    const title = entry.title || '';
    const tags = Array.isArray(entry.tags) ? entry.tags : [];
    const instrument = entry.instrument_estimate || '';

    const matchesText = title.toLowerCase().includes(filterText.toLowerCase()) || 
                        tags.some(t => t && t.toLowerCase().includes(filterText.toLowerCase())) ||
                        instrument.toLowerCase().includes(filterText.toLowerCase());
    
    const matchesSource = filterSource === 'all' || entry.source_type === filterSource;
    const matchesConf = (entry.avg_confidence || 0) >= filterMinConfidence;
    const matchesBPM = (entry.bpm_detected || 0) >= filterBPM[0] && (entry.bpm_detected || 0) <= filterBPM[1];

    return matchesText && matchesSource && matchesConf && matchesBPM;
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      
      <div className="relative bg-zinc-900 w-full max-w-6xl h-[85vh] rounded-xl border border-zinc-800 shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-800 bg-zinc-900/50">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-white">Project History</h2>
            <span className="bg-zinc-800 text-zinc-400 text-xs px-2 py-0.5 rounded-full">{history.length} items</span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={HistoryService.exportHistoryAsJSON} className="text-xs text-zinc-400 hover:text-indigo-400 flex items-center gap-1 px-2 py-1 rounded hover:bg-zinc-800">
                <SaveIcon className="w-3 h-3" /> Export JSON
            </button>
            <button onClick={onClose} className="p-1 hover:bg-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors">
              <XIcon className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden">
          
          {/* Left: Sidebar (List & Filters) */}
          <div className="w-full md:w-[400px] border-r border-zinc-800 flex flex-col bg-zinc-950/30">
            
            {/* Search & Filter Toggle */}
            <div className="p-3 border-b border-zinc-800 space-y-3">
              <div className="relative">
                <FilterIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                <input 
                  type="text" 
                  placeholder="Search projects..." 
                  className="w-full bg-zinc-900 border border-zinc-700 rounded-md pl-9 pr-3 py-2 text-sm focus:outline-none focus:border-indigo-500 text-zinc-200"
                  value={filterText}
                  onChange={(e) => setFilterText(e.target.value)}
                />
              </div>
              
              <button 
                onClick={() => setShowFilters(!showFilters)}
                className="w-full flex items-center justify-between text-xs text-zinc-400 hover:text-zinc-200 px-1"
              >
                <span>{showFilters ? 'Hide Filters' : 'Show Advanced Filters'}</span>
                <span>{showFilters ? '▲' : '▼'}</span>
              </button>

              {showFilters && (
                <div className="space-y-3 pt-2 bg-zinc-900/50 p-2 rounded-lg border border-zinc-800">
                    <div>
                        <label className="text-[10px] text-zinc-500 uppercase font-bold">Source</label>
                        <select 
                            value={filterSource} 
                            onChange={(e) => setFilterSource(e.target.value as any)}
                            className="w-full mt-1 bg-zinc-950 border border-zinc-700 rounded text-xs px-2 py-1 text-zinc-300"
                        >
                            <option value="all">All Sources</option>
                            <option value="youtube">YouTube</option>
                            <option value="file">Upload</option>
                        </select>
                    </div>
                    <div>
                        <label className="text-[10px] text-zinc-500 uppercase font-bold flex justify-between">
                            <span>Min Confidence</span>
                            <span>{(filterMinConfidence * 100).toFixed(0)}%</span>
                        </label>
                        <input 
                            type="range" 
                            min="0" max="1" step="0.1" 
                            value={filterMinConfidence} 
                            onChange={(e) => setFilterMinConfidence(parseFloat(e.target.value))}
                            className="w-full mt-1 h-1 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                    </div>
                    <div>
                        <label className="text-[10px] text-zinc-500 uppercase font-bold">BPM Range ({filterBPM[0]} - {filterBPM[1]})</label>
                        <div className="flex gap-2 mt-1">
                            <input type="number" value={filterBPM[0]} onChange={e => setFilterBPM([+e.target.value, filterBPM[1]])} className="w-1/2 bg-zinc-950 border border-zinc-700 rounded text-xs px-2 py-1" />
                            <input type="number" value={filterBPM[1]} onChange={e => setFilterBPM([filterBPM[0], +e.target.value])} className="w-1/2 bg-zinc-950 border border-zinc-700 rounded text-xs px-2 py-1" />
                        </div>
                    </div>
                </div>
              )}
            </div>

            {/* Project List */}
            <div className="flex-1 overflow-y-auto">
              {filteredHistory.length === 0 ? (
                <div className="p-8 text-center text-zinc-500 text-sm">
                  No projects match your filters.
                </div>
              ) : (
                filteredHistory.map(entry => (
                  <div 
                    key={entry.id}
                    onClick={() => { setSelectedEntry(entry); setIsRenaming(false); setIsAddingTag(false); }}
                    className={`p-3 border-b border-zinc-800 cursor-pointer hover:bg-zinc-800/50 transition-colors flex items-start gap-3 group ${selectedEntry?.id === entry.id ? 'bg-indigo-900/10 border-l-4 border-l-indigo-500' : 'border-l-4 border-l-transparent'}`}
                  >
                    <div className="w-12 h-12 bg-zinc-800 rounded overflow-hidden shrink-0 flex items-center justify-center border border-zinc-700">
                      {entry.thumbnail ? (
                        <img src={entry.thumbnail} alt="thumb" className="w-full h-full object-cover" />
                      ) : (
                        <MusicIcon className="w-5 h-5 text-zinc-500" />
                      )}
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between items-start">
                        <h4 className="text-sm font-medium text-zinc-200 truncate pr-2">{entry.title || 'Untitled'}</h4>
                        {/* Inline delete for quick action */}
                        <button 
                             onClick={(e) => handleDelete(e, entry.id)}
                             className="text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                            <TrashIcon className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <div className="flex items-center gap-2 text-[10px] text-zinc-500 mt-1">
                        <span>{new Date(entry.timestamp).toLocaleDateString()}</span>
                        <span>•</span>
                        <span>{Math.floor(entry.audio_duration_sec / 60)}:{(entry.audio_duration_sec % 60).toString().padStart(2, '0')}</span>
                        {entry.source_type === 'youtube' && <span className="text-red-400/80">YT</span>}
                      </div>
                      {/* Mini Confidence Bar */}
                      <div className="mt-2 h-1 w-full bg-zinc-800 rounded-full overflow-hidden">
                        <div 
                            className={`h-full ${entry.avg_confidence > 0.8 ? 'bg-green-500' : entry.avg_confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`} 
                            style={{ width: `${(entry.avg_confidence || 0) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Retention Settings Footer */}
            <div className="p-3 border-t border-zinc-800 bg-zinc-900 flex justify-between items-center text-xs">
              <select 
                value={retentionPolicy} 
                onChange={handleRetentionChange}
                className="bg-transparent text-zinc-500 border-none focus:ring-0 cursor-pointer hover:text-zinc-300"
              >
                <option value="10_items">Keep 10 Items</option>
                <option value="50_items">Keep 50 Items</option>
                <option value="100_items">Keep 100 Items</option>
                <option value="7_days">Keep 7 Days</option>
                <option value="30_days">Keep 30 Days</option>
                <option value="forever">Keep Forever</option>
              </select>
              <button onClick={handleClearAll} className="text-zinc-600 hover:text-red-400">Clear All</button>
            </div>
          </div>

          {/* Right: Detail View */}
          <div className="flex-1 overflow-y-auto bg-zinc-900/50 p-6 md:p-8">
            {selectedEntry ? (
              <div className="max-w-3xl mx-auto space-y-8">
                
                {/* Title & Main Actions */}
                <div className="flex justify-between items-start pb-6 border-b border-zinc-800">
                    <div className="space-y-2 flex-1">
                        {isRenaming ? (
                            <div className="flex items-center gap-2">
                                <input 
                                    autoFocus
                                    className="bg-zinc-800 text-2xl font-bold text-white px-2 py-1 rounded border border-indigo-500 outline-none w-full"
                                    value={renameValue}
                                    onChange={(e) => setRenameValue(e.target.value)}
                                    onBlur={handleRename}
                                    onKeyDown={(e) => e.key === 'Enter' && handleRename()}
                                />
                            </div>
                        ) : (
                            <div className="flex items-center gap-2 group">
                                <h1 className="text-3xl font-bold text-white tracking-tight">{selectedEntry.title || 'Untitled'}</h1>
                                <button onClick={() => { setIsRenaming(true); setRenameValue(selectedEntry.title || ''); }} className="opacity-0 group-hover:opacity-100 p-1 text-zinc-500 hover:text-indigo-400">
                                    <EditIcon className="w-4 h-4" />
                                </button>
                            </div>
                        )}
                        
                        <div className="flex items-center gap-4 text-sm text-zinc-400">
                            <div className="flex items-center gap-1.5">
                                <CalendarIcon className="w-4 h-4" />
                                <span>{new Date(selectedEntry.timestamp).toLocaleString()}</span>
                            </div>
                            <span className="bg-zinc-800 px-2 py-0.5 rounded text-xs uppercase font-medium tracking-wide">{selectedEntry.source_type}</span>
                            {selectedEntry.source_url && (
                                <a href={selectedEntry.source_url} target="_blank" rel="noreferrer" className="text-indigo-400 hover:underline truncate max-w-[200px]">
                                    Source Link
                                </a>
                            )}
                        </div>
                    </div>

                    <div className="flex gap-2">
                         <button 
                             onClick={handleDuplicate}
                             className="p-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-md transition-colors border border-zinc-700"
                             title="Duplicate Project"
                         >
                             <CopyIcon className="w-5 h-5" />
                         </button>
                         <button 
                             onClick={(e) => handleDelete(e, selectedEntry.id)}
                             className="p-2 bg-zinc-800 hover:bg-red-900/20 text-zinc-300 hover:text-red-400 rounded-md transition-colors border border-zinc-700"
                             title="Delete Project"
                         >
                             <TrashIcon className="w-5 h-5" />
                         </button>
                    </div>
                </div>

                {/* KPI Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800/60 shadow-sm relative overflow-hidden group">
                        <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500/50"></div>
                        <div className="text-xs text-zinc-500 uppercase font-bold mb-1">Confidence</div>
                        <div className="text-2xl font-bold text-white group-hover:text-indigo-400 transition-colors">{((selectedEntry.avg_confidence || 0) * 100).toFixed(0)}<span className="text-sm text-zinc-600">%</span></div>
                    </div>
                    <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800/60 shadow-sm relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-purple-500/50"></div>
                        <div className="text-xs text-zinc-500 uppercase font-bold mb-1">Detected BPM</div>
                        <div className="text-2xl font-bold text-white">{selectedEntry.bpm_detected || 0}</div>
                    </div>
                    <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800/60 shadow-sm relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-emerald-500/50"></div>
                        <div className="text-xs text-zinc-500 uppercase font-bold mb-1">Note Count</div>
                        <div className="text-2xl font-bold text-white">{selectedEntry.notes_count || 0}</div>
                    </div>
                    <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800/60 shadow-sm relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-orange-500/50"></div>
                        <div className="text-xs text-zinc-500 uppercase font-bold mb-1">Instrument</div>
                        <div className="text-2xl font-bold text-white truncate">{selectedEntry.instrument_estimate || '-'}</div>
                    </div>
                </div>

                {/* Tags Section */}
                <div>
                    <h3 className="text-sm font-semibold text-zinc-300 mb-3 flex items-center gap-2">
                        <TagIcon className="w-4 h-4" /> Tags
                    </h3>
                    <div className="flex flex-wrap gap-2">
                        {Array.isArray(selectedEntry.tags) && selectedEntry.tags.map(tag => (
                            <span key={tag} className="px-3 py-1 bg-zinc-800 text-zinc-300 text-sm rounded-full border border-zinc-700">
                                {tag}
                            </span>
                        ))}
                        {isAddingTag ? (
                            <input 
                                autoFocus
                                className="px-3 py-1 bg-zinc-900 border border-indigo-500 text-sm rounded-full text-white outline-none w-24"
                                placeholder="New tag..."
                                value={newTag}
                                onChange={(e) => setNewTag(e.target.value)}
                                onBlur={handleAddTag}
                                onKeyDown={(e) => e.key === 'Enter' && handleAddTag()}
                            />
                        ) : (
                            <button 
                                onClick={() => setIsAddingTag(true)}
                                className="px-3 py-1 bg-zinc-900 border border-zinc-700 border-dashed text-zinc-500 hover:text-white hover:border-zinc-500 text-sm rounded-full transition-all"
                            >
                                + Add Tag
                            </button>
                        )}
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* User Edits Log */}
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold text-zinc-300">Session Activity</h3>
                        <div className="bg-zinc-950 rounded-lg border border-zinc-800 p-5 space-y-3">
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-zinc-500">Notes Modified</span>
                                <span className="font-mono text-zinc-300">{selectedEntry.user_edits?.notes_modified || 0}</span>
                            </div>
                            <div className="w-full h-px bg-zinc-900"></div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-zinc-500">Notes Added</span>
                                <span className="font-mono text-green-400">+{selectedEntry.user_edits?.notes_added || 0}</span>
                            </div>
                            <div className="w-full h-px bg-zinc-900"></div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-zinc-500">Notes Deleted</span>
                                <span className="font-mono text-red-400">-{selectedEntry.user_edits?.notes_deleted || 0}</span>
                            </div>
                        </div>
                    </div>

                    {/* Export Status */}
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold text-zinc-300">Export Formats</h3>
                        <div className="grid grid-cols-2 gap-3">
                            {selectedEntry.exports && Object.entries(selectedEntry.exports).map(([key, val]) => (
                                <div key={key} className={`p-3 rounded-lg border flex items-center justify-between ${val ? 'bg-indigo-900/10 border-indigo-500/30' : 'bg-zinc-950 border-zinc-800 opacity-50'}`}>
                                    <span className="text-sm font-medium uppercase text-zinc-400">{key}</span>
                                    {val && <div className="w-2 h-2 rounded-full bg-indigo-500"></div>}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Primary Action Button */}
                <div className="pt-6 border-t border-zinc-800">
                    <button 
                        onClick={() => { onLoadEntry(selectedEntry); onClose(); }}
                        className="w-full py-4 bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white rounded-xl shadow-lg shadow-indigo-500/20 font-bold text-lg flex items-center justify-center gap-3 transition-all transform hover:scale-[1.01]"
                    >
                        <RefreshIcon className="w-5 h-5" />
                        Re-open Project Session
                    </button>
                </div>

              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-zinc-600 opacity-60">
                <MusicIcon className="w-24 h-24 mb-6 opacity-20" />
                <p className="text-xl font-medium text-zinc-400">Select a project</p>
                <p className="text-sm mt-2 max-w-xs text-center">View full details, editing history, and restore previous sessions.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HistoryModal;