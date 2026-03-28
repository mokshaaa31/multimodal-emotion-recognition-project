"""
Audio Processing Utilities for RAVDESS Emotion Recognition

Functions for extracting audio from video and computing MFCC features.
MFCC (Mel-Frequency Cepstral Coefficients) are standard features
for speech and emotion recognition tasks.
"""

import subprocess
import numpy as np
import os


def extract_audio(video_path, output_path="temp.wav", sample_rate=22050):
    """
    Extract audio track from video using FFmpeg.
    
    Args:
        video_path: Path to input video file
        output_path: Path for output WAV file (default: temp.wav)
        sample_rate: Audio sample rate (default: 22050 Hz)
    
    Returns:
        Path to extracted audio file
    
    Raises:
        RuntimeError: If FFmpeg fails or is not installed
    """
    # Remove existing file if present
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # FFmpeg command to extract audio
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # PCM 16-bit format
        "-ar", str(sample_rate),  # Sample rate
        "-ac", "1",               # Mono channel
        "-y",                     # Overwrite output
        "-loglevel", "error",     # Suppress verbose output
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"FFmpeg did not create audio file. Error: {result.stderr}")
        
        return output_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out while extracting audio")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: download from https://ffmpeg.org"
        )


def extract_audio_features(audio_path, n_mfcc=40, sample_rate=22050):
    """
    Extract MFCC features from audio file.
    
    MFCCs capture the spectral envelope of speech, which contains
    information about emotion, speaker identity, and phonetic content.
    
    Args:
        audio_path: Path to audio file (WAV format)
        n_mfcc: Number of MFCC coefficients to extract (default: 40)
        sample_rate: Expected sample rate (default: 22050 Hz)
    
    Returns:
        numpy array of shape (n_mfcc,) - mean MFCC features across time
    """
    import librosa
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    if len(y) == 0:
        # Return zeros if audio is empty/silent
        return np.zeros(n_mfcc, dtype=np.float32)
    
    # Extract MFCCs
    # Shape: (n_mfcc, time_frames)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Average across time to get fixed-size representation
    # Shape: (n_mfcc,)
    mfcc_mean = np.mean(mfccs, axis=1).astype(np.float32)
    
    return mfcc_mean


def extract_audio_features_full(audio_path, n_mfcc=40, sample_rate=22050):
    """
    Extract full MFCC features without averaging (for more detailed analysis).
    
    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        sample_rate: Sample rate
    
    Returns:
        numpy array of shape (n_mfcc, time_frames)
    """
    import librosa
    
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    if len(y) == 0:
        return np.zeros((n_mfcc, 1), dtype=np.float32)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    return mfccs.astype(np.float32)


def get_audio_duration(audio_path):
    """
    Get duration of audio file in seconds.
    """
    import librosa
    
    y, sr = librosa.load(audio_path, sr=None)
    return len(y) / sr
