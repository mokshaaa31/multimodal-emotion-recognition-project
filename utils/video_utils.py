"""
Video Processing Utilities for RAVDESS Emotion Recognition

Functions for extracting and preprocessing video frames.
"""

import cv2
import numpy as np


def get_frames(video_path, max_frames=5):
    """
    Extract evenly-spaced frames from a video file.
    
    For emotion recognition, we sample multiple frames to capture
    facial expressions throughout the video, then aggregate features.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract (default: 5)
    
    Returns:
        List of frames as numpy arrays (H, W, C) in RGB format
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print("Error: Video has 0 frames")
        cap.release()
        return []
    
    # Calculate frame indices to extract (evenly spaced)
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames


def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess a single frame for the video model.
    
    Args:
        frame: RGB frame as numpy array (H, W, C)
        target_size: Target size as (height, width)
    
    Returns:
        Preprocessed frame as numpy array (C, H, W) normalized to [0, 1]
    """
    # Resize
    frame = cv2.resize(frame, target_size)
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Convert HWC to CHW (channels first for PyTorch)
    frame = np.transpose(frame, (2, 0, 1))
    
    return frame


def get_video_info(video_path):
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info
