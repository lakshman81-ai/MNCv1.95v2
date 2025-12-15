import { HistoryEntry, ExportStatus, UserEdits, RetentionPolicy } from '../types';

const STORAGE_KEY = 'mnc_history_v1';
const RETENTION_KEY = 'mnc_retention_policy';

// Safe ID generator that works in non-secure contexts
const generateId = (): string => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return 'hist_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
};

export const HistoryService = {
  getHistory: (): HistoryEntry[] => {
    if (typeof window === 'undefined') return [];
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      const data = stored ? JSON.parse(stored) : [];
      
      // Sanitize data structure to ensure arrays exist
      if (Array.isArray(data)) {
        return data.map((item: any) => ({
          ...item,
          tags: Array.isArray(item.tags) ? item.tags : [],
          user_edits: item.user_edits || { notes_modified: 0, notes_deleted: 0, notes_added: 0 },
          exports: item.exports || { musicxml: false, midi: false, pdf: false, csv: false }
        }));
      }
      return [];
    } catch (e) {
      console.error("Failed to load history", e);
      return [];
    }
  },

  getRetentionPolicy: (): RetentionPolicy => {
    if (typeof window === 'undefined') return '50_items';
    return (localStorage.getItem(RETENTION_KEY) as RetentionPolicy) || '50_items';
  },

  setRetentionPolicy: (policy: RetentionPolicy) => {
    if (typeof window === 'undefined') return;
    localStorage.setItem(RETENTION_KEY, policy);
    HistoryService.enforceRetentionPolicy();
  },

  enforceRetentionPolicy: () => {
    const history = HistoryService.getHistory();
    const policy = HistoryService.getRetentionPolicy();
    let newHistory = [...history];

    const now = new Date().getTime();
    
    switch (policy) {
      case '10_items':
        newHistory = newHistory.slice(0, 10);
        break;
      case '50_items':
        newHistory = newHistory.slice(0, 50);
        break;
      case '100_items':
        newHistory = newHistory.slice(0, 100);
        break;
      case '7_days':
        newHistory = newHistory.filter(h => (now - new Date(h.timestamp).getTime()) < 7 * 24 * 60 * 60 * 1000);
        break;
      case '30_days':
        newHistory = newHistory.filter(h => (now - new Date(h.timestamp).getTime()) < 30 * 24 * 60 * 60 * 1000);
        break;
      case 'forever':
      default:
        break;
    }

    // Always sort by date descending
    newHistory.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    localStorage.setItem(STORAGE_KEY, JSON.stringify(newHistory));
  },

  addEntry: (entry: HistoryEntry) => {
    if (typeof window === 'undefined') return;
    const history = HistoryService.getHistory();
    const newHistory = [entry, ...history];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newHistory));
    HistoryService.enforceRetentionPolicy();
  },

  updateEntry: (id: string, updates: Partial<Omit<HistoryEntry, 'user_edits' | 'exports'>> & { user_edits?: Partial<UserEdits>; exports?: Partial<ExportStatus> }) => {
    if (typeof window === 'undefined') return;
    const history = HistoryService.getHistory();
    const index = history.findIndex(h => h.id === id);
    if (index !== -1) {
      const current = history[index];
      
      const updatedEntry = {
        ...current,
        ...updates,
        user_edits: updates.user_edits ? { ...current.user_edits, ...updates.user_edits } : current.user_edits,
        exports: updates.exports ? { ...current.exports, ...updates.exports } : current.exports,
        tags: updates.tags || current.tags
      };

      history[index] = updatedEntry as HistoryEntry;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    }
  },

  duplicateEntry: (id: string): HistoryEntry | null => {
    if (typeof window === 'undefined') return null;
    const history = HistoryService.getHistory();
    const entry = history.find(h => h.id === id);
    if (!entry) return null;

    const copy: HistoryEntry = {
      ...entry,
      id: generateId(),
      title: `${entry.title} (Copy)`,
      timestamp: new Date().toISOString(),
      exports: { musicxml: false, midi: false, pdf: false, csv: false },
      user_edits: { notes_modified: 0, notes_deleted: 0, notes_added: 0 }, // Reset edit history for new copy
      tags: [...entry.tags] // Clone tags
    };

    HistoryService.addEntry(copy);
    return copy;
  },

  renameEntry: (id: string, newTitle: string) => {
    HistoryService.updateEntry(id, { title: newTitle });
  },

  addTag: (id: string, tag: string) => {
    const history = HistoryService.getHistory();
    const entry = history.find(h => h.id === id);
    // Safe check for tags array
    if (entry) {
        const currentTags = Array.isArray(entry.tags) ? entry.tags : [];
        if (!currentTags.includes(tag)) {
            HistoryService.updateEntry(id, { tags: [...currentTags, tag] });
        }
    }
  },

  deleteEntry: (id: string) => {
    if (typeof window === 'undefined') return;
    const history = HistoryService.getHistory();
    const newHistory = history.filter(h => h.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newHistory));
  },

  clearHistory: () => {
    if (typeof window === 'undefined') return;
    localStorage.removeItem(STORAGE_KEY);
  },

  exportHistoryAsJSON: () => {
    const history = HistoryService.getHistory();
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "music_note_creator_history.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  }
};