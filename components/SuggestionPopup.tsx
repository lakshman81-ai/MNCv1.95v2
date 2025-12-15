import React from 'react';
import { SuggestedSettings } from '../services/suggestionService';
import { CheckIcon, CloseIcon, LightBulbIcon } from './Icons';

interface SuggestionPopupProps {
  isOpen: boolean;
  settings: SuggestedSettings | null;
  onAccept: () => void;
  onReject: () => void;
}

const SuggestionPopup: React.FC<SuggestionPopupProps> = ({ isOpen, settings, onAccept, onReject }) => {
  if (!isOpen || !settings) {
    return null;
  }

  return (
    <div className="fixed bottom-5 right-5 z-[110] bg-zinc-800 border border-zinc-700 rounded-xl shadow-2xl max-w-sm w-full animate-in slide-in-from-bottom-5">
      <div className="p-5">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 bg-yellow-500 rounded-full flex items-center justify-center flex-shrink-0">
            <LightBulbIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white mb-1">Style Suggestion</h3>
            <p className="text-zinc-400 text-sm mb-4">
              Based on the audio, we suggest the following settings:
            </p>
            <div className="grid grid-cols-3 gap-3 text-center mb-5">
              <div>
                <span className="text-xs text-zinc-500 uppercase">Voice</span>
                <p className="font-semibold text-white capitalize">{settings.voice.replace('_', ' ')}</p>
              </div>
              <div>
                <span className="text-xs text-zinc-500 uppercase">Style</span>
                <p className="font-semibold text-white capitalize">{settings.style}</p>
              </div>
              <div>
                <span className="text-xs text-zinc-500 uppercase">BPM</span>
                <p className="font-semibold text-white">{settings.bpm}</p>
              </div>
            </div>
          </div>
        </div>
        <div className="flex gap-3 justify-end">
          <button
            onClick={onReject}
            className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded-lg transition-colors flex items-center gap-2"
          >
            <CloseIcon className="w-4 h-4" />
            Reject
          </button>
          <button
            onClick={onAccept}
            className="px-4 py-2 bg-green-600 hover:bg-green-500 text-white font-medium rounded-lg shadow-lg flex items-center gap-2"
          >
            <CheckIcon className="w-4 h-4" />
            Accept
          </button>
        </div>
      </div>
    </div>
  );
};

export default SuggestionPopup;
