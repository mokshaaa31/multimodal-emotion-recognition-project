"""
Utilities package for RAVDESS Emotion Recognition

Contains:
- video_utils: Frame extraction and preprocessing
- audio_utils: Audio extraction and MFCC computation
"""

from .video_utils import get_frames, preprocess_frame, get_video_info
from .audio_utils import extract_audio, extract_audio_features

__all__ = [
    'get_frames',
    'preprocess_frame', 
    'get_video_info',
    'extract_audio',
    'extract_audio_features'
]
