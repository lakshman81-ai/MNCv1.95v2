import React, { useEffect, useRef } from 'react';

interface YouTubePlayerProps {
  videoId: string;
  isPlaying: boolean;
  onReady: (duration: number) => void;
  onStateChange: (isPlaying: boolean) => void;
  onTimeUpdate: (currentTime: number) => void;
  seekTo?: number | null;
  onError?: (error: { code: number; message: string }) => void;
}

declare global {
  interface Window {
    YT: any;
    onYouTubeIframeAPIReady: () => void;
  }
}

const YouTubePlayer: React.FC<YouTubePlayerProps> = ({ 
  videoId, 
  isPlaying, 
  onReady, 
  onStateChange,
  onTimeUpdate,
  seekTo,
  onError
}) => {
  const playerRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const timeIntervalRef = useRef<number | undefined>(undefined);

  // Initialize Player
  useEffect(() => {
    const loadAPI = () => {
      if (!window.YT) {
        const tag = document.createElement('script');
        tag.src = "https://www.youtube.com/iframe_api";
        const firstScriptTag = document.getElementsByTagName('script')[0];
        firstScriptTag.parentNode?.insertBefore(tag, firstScriptTag);
        
        window.onYouTubeIframeAPIReady = initPlayer;
      } else {
        initPlayer();
      }
    };

    const initPlayer = () => {
      if (!containerRef.current) return;
      
      // If player exists, just cue new video
      if (playerRef.current && playerRef.current.cueVideoById) {
         playerRef.current.cueVideoById(videoId);
         return;
      }

      const origin = typeof window !== 'undefined' ? window.location.origin : '';

      playerRef.current = new window.YT.Player(containerRef.current, {
        height: '100%',
        width: '100%',
        videoId: videoId,
        playerVars: {
          'playsinline': 1,
          'controls': 0, // We use custom controls
          'disablekb': 1,
          'modestbranding': 1,
          'rel': 0,
          'origin': origin // Critical for fixing Error 150/153
        },
        events: {
          'onReady': onPlayerReady,
          'onStateChange': onPlayerStateChange,
          'onError': onPlayerError
        }
      });
    };

    loadAPI();

    return () => {
      if (timeIntervalRef.current) clearInterval(timeIntervalRef.current);
      if (playerRef.current && playerRef.current.destroy) {
          try {
             // playerRef.current.destroy(); // Keeping instance often better for React
          } catch(e) {}
      }
    };
  }, []);

  // Handle Video ID Change
  useEffect(() => {
    if (playerRef.current && playerRef.current.cueVideoById) {
        playerRef.current.cueVideoById(videoId);
    }
  }, [videoId]);

  // Handle Play/Pause Prop
  useEffect(() => {
    if (!playerRef.current || !playerRef.current.playVideo) return;
    
    const playerState = playerRef.current.getPlayerState();
    // 1 = Playing, 2 = Paused, 5 = Cued
    if (isPlaying && (playerState !== 1 && playerState !== 3)) {
      playerRef.current.playVideo();
    } else if (!isPlaying && playerState === 1) {
      playerRef.current.pauseVideo();
    }
  }, [isPlaying]);

  // Handle Seek Prop
  useEffect(() => {
    if (seekTo !== null && seekTo !== undefined && playerRef.current && playerRef.current.seekTo) {
      playerRef.current.seekTo(seekTo, true);
    }
  }, [seekTo]);

  const onPlayerReady = (event: any) => {
    const duration = event.target.getDuration();
    onReady(duration);
    
    // If isPlaying prop is true (user clicked play before ready), start now.
    if (isPlaying) {
        event.target.playVideo();
    }
  };

  const onPlayerError = (event: any) => {
      console.error("YouTube Player Error Code:", event.data);
      let msg = "An error occurred with the video player.";
      const code = event.data;
      
      // 2 – The request contains an invalid parameter value.
      // 5 – The requested content cannot be played in an HTML5 player.
      // 100 – The video requested was not found.
      // 101 – The owner of the requested video does not allow it to be played in embedded players.
      // 150 – This error is the same as 101. It's just a 101 error in disguise!
      // 153 - Related to Mnet/Copyright/Embed restrictions
      
      if (code === 100) msg = "Video not found or removed.";
      if (code === 101 || code === 150 || code === 153) {
          msg = "Playback restricted by copyright owner.";
      }
      
      if (onError) onError({ code, message: msg });
  };

  const onPlayerStateChange = (event: any) => {
    const isNowPlaying = event.data === 1;
    onStateChange(isNowPlaying);

    if (isNowPlaying) {
      startTimePolling();
    } else {
      stopTimePolling();
    }
  };

  const startTimePolling = () => {
    stopTimePolling();
    timeIntervalRef.current = window.setInterval(() => {
      if (playerRef.current && playerRef.current.getCurrentTime) {
        const time = playerRef.current.getCurrentTime();
        onTimeUpdate(time);
      }
    }, 100); // 10Hz update
  };

  const stopTimePolling = () => {
    if (timeIntervalRef.current) {
      clearInterval(timeIntervalRef.current);
    }
  };

  return (
    <div className="w-full h-full rounded-xl overflow-hidden bg-black">
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
};

export default YouTubePlayer;