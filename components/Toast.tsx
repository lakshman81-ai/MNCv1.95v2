import React, { useEffect, useState } from 'react';
import { XIcon } from './Icons';

export type ToastType = 'success' | 'info' | 'loading' | 'error';

interface ToastProps {
  message: string;
  type: ToastType;
  onClose: () => void;
}

export const Toast: React.FC<ToastProps> = ({ message, type, onClose }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Trigger animation
    requestAnimationFrame(() => setIsVisible(true));

    if (type !== 'loading') {
      const timer = setTimeout(() => {
        setIsVisible(false);
        setTimeout(onClose, 300); // Wait for exit animation
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [type, onClose]);

  const getStyles = () => {
    switch (type) {
      case 'success': return 'bg-emerald-600 border-emerald-500 text-white';
      case 'error': return 'bg-red-600 border-red-500 text-white';
      case 'loading': return 'bg-indigo-600 border-indigo-500 text-white';
      default: return 'bg-zinc-800 border-zinc-700 text-zinc-200';
    }
  };

  return (
    <div 
      className={`fixed bottom-6 right-6 px-4 py-3 rounded-lg shadow-2xl border flex items-center gap-3 z-[100] transition-all duration-300 transform ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'} ${getStyles()}`}
    >
      {type === 'loading' && (
        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
      )}
      <span className="text-sm font-medium pr-2">{message}</span>
      {type !== 'loading' && (
        <button onClick={() => { setIsVisible(false); setTimeout(onClose, 300); }} className="opacity-60 hover:opacity-100">
          <XIcon className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};