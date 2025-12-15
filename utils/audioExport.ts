
// Utility to convert AudioBuffer to WAV Blob for API transmission

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function floatTo16BitPCM(output: DataView, offset: number, input: Float32Array) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
}

export const audioBufferToWav = (buffer: AudioBuffer, startTime: number, duration: number): Blob => {
  // Use 24kHz to balance quality ("not a large downsample") vs payload size to prevent API 500 errors.
  // 24kHz is sufficient for clear pitch detection up to 12kHz.
  const targetSampleRate = 24000;
  const sourceSampleRate = buffer.sampleRate;
  
  const startSample = Math.floor(startTime * sourceSampleRate);
  const endSample = Math.min(Math.floor((startTime + duration) * sourceSampleRate), buffer.length);
  
  // Extract channel data (Mono) - we always use mono to save 50% size
  const sourceData = buffer.getChannelData(0).slice(startSample, endSample);
  
  if (sourceData.length === 0) return new Blob([]);

  // Resample using Linear Interpolation
  const ratio = sourceSampleRate / targetSampleRate;
  const newLength = Math.floor(sourceData.length / ratio);
  const resampledData = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const sourceIndex = i * ratio;
    const indexFloor = Math.floor(sourceIndex);
    const indexCeil = Math.min(sourceData.length - 1, indexFloor + 1);
    const fraction = sourceIndex - indexFloor;
    
    // Linear interpolation: y = y0 + (y1 - y0) * fraction
    resampledData[i] = sourceData[indexFloor] * (1 - fraction) + sourceData[indexCeil] * fraction;
  }

  const numChannels = 1; 
  const format = 1; // PCM
  const bitDepth = 16;
  const blockAlign = numChannels * (bitDepth / 8);
  const byteRate = targetSampleRate * blockAlign;
  const dataSize = newLength * blockAlign;
  const headerSize = 44;
  const totalSize = headerSize + dataSize;

  const arrayBuffer = new ArrayBuffer(totalSize);
  const view = new DataView(arrayBuffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, targetSampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  floatTo16BitPCM(view, 44, resampledData);

  return new Blob([view], { type: 'audio/wav' });
};

export const blobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Remove data URL prefix (e.g. "data:audio/wav;base64,")
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};
