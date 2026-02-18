import numpy as np
import librosa

def audio_to_waveform(audio_path: str, sr: int = 48000) -> np.ndarray:
    """
    Charge un audio et le transforme en waveform lisible par CLAP.
    
    Args:
        audio_path: Chemin vers le fichier audio
        sr: Sample rate (48000 Hz par défaut pour CLAP)
    
    Returns:
        Waveform en tant que numpy array (mono, normalized)
    """
    # Charger l'audio
    waveform, original_sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Normaliser entre -1 et 1
    waveform = waveform.astype(np.float32)
    
    return waveform