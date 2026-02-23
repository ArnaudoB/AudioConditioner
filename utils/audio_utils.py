import numpy as np
import librosa

def audio_to_waveform(audio_path: str, sr: int = 48000) -> np.ndarray:
    """
    Convert an audio file to a waveform numpy array, resampled to the specified sampling rate.
    The waveform is normalized to be between -1 and 1.
    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sampling rate for the output waveform.
    Returns:
        np.ndarray: Normalized waveform array.
    """
    # Load the audio file and resample to the target sampling rate
    waveform, original_sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Normalize the waveform to be between -1 and 1
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    waveform = waveform.astype(np.float32)
    
    return waveform