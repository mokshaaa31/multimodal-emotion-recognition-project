"""
🎭 Multimodal Emotion Recognition - RAVDESS Dataset
====================================================

Models are configured to EXACTLY match the pre-trained checkpoints:

video_model.pth:
  - MobileNetV3 Small with conv_head
  - Output: 1024 features

audio_model.pth:
  - self.net = Sequential(Linear(40,128), ReLU, Linear(128,256), ReLU)
  - Output: 256 features

fusion_model.pth:
  - video_proj: 1024 → 256
  - audio_proj: 256 → 256  
  - CrossAttention (4 heads)
  - fc: 256 → 128 → 4 classes
"""

import streamlit as st
import tempfile
import torch
import numpy as np
import cv2
import os

# ===============================
# ⚙️ PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="RAVDESS Emotion Recognition",
    page_icon="🎭",
    layout="centered"
)

# ===============================
# 📦 IMPORTS
# ===============================
from utils.video_utils import get_frames
from utils.audio_utils import extract_audio, extract_audio_features
from models.model import VideoTransformer
from models.audio_model import AudioEncoder
from models.fusion_model import CrossAttentionModel

# ===============================
# 🎨 HEADER
# ===============================
st.title("🎭 Multimodal Emotion Recognition")
st.markdown("Detect emotions from video using **facial expressions** and **speech**")

# ===============================
# 📥 DOWNLOAD MODELS
# ===============================
MODEL_IDS = {
    "video_model.pth": "1SMZL7V-nnaOItT-CvI2rm_Oyeg5zkS1i",
    "audio_model.pth": "18grYKARGAt0qRz1qNui3tnWyzGCYdiu_",
    "fusion_model.pth": "1RSPfbQToHUzRDOrlQse7kNXQ5TAGucAm"
}

@st.cache_resource
def download_models():
    import gdown
    for filename, file_id in MODEL_IDS.items():
        if not os.path.exists(filename):
            st.info(f"📥 Downloading {filename}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)

download_models()

# ===============================
# ⚡ LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    """Load models with architectures matching the checkpoints exactly."""
    
    # Initialize with correct architectures
    video_model = VideoTransformer()
    audio_model = AudioEncoder(input_dim=40, hidden_dim=128, output_dim=256)
    fusion_model = CrossAttentionModel(video_dim=1024, audio_dim=256, hidden_dim=256, num_classes=4)
    
    # Load weights
    video_model.load_state_dict(torch.load("video_model.pth", map_location="cpu"))
    audio_model.load_state_dict(torch.load("audio_model.pth", map_location="cpu"))
    fusion_model.load_state_dict(torch.load("fusion_model.pth", map_location="cpu"))
    
    # Evaluation mode
    video_model.eval()
    audio_model.eval()
    fusion_model.eval()
    
    return video_model, audio_model, fusion_model

try:
    video_model, audio_model, fusion_model = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.exception(e)
    st.stop()

# ===============================
# 🎯 EMOTION LABELS
# ===============================
EMOTIONS = ["😊 Happy", "😢 Sad", "😠 Angry", "😐 Neutral"]

# ===============================
# 📤 FILE UPLOAD
# ===============================
st.markdown("---")
video_file = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi", "mkv"])

if video_file:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close()
    
    st.video(video_file)
    
    if st.button("🔍 Analyze Emotion", type="primary"):
        try:
            with st.spinner("Analyzing..."):
                
                # === VIDEO PROCESSING ===
                frames = get_frames(tfile.name, max_frames=5)
                if not frames:
                    st.error("❌ Could not extract frames")
                    st.stop()
                
                video_features = []
                for frame in frames:
                    # Preprocess
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frame = np.transpose(frame, (2, 0, 1))  # HWC → CHW
                    tensor = torch.tensor(frame).unsqueeze(0)
                    
                    with torch.no_grad():
                        feat = video_model(tensor)
                    video_features.append(feat)
                
                video_feat = torch.mean(torch.stack(video_features), dim=0)
                
                # === AUDIO PROCESSING ===
                audio_path = extract_audio(tfile.name)
                mfcc = extract_audio_features(audio_path)
                audio_tensor = torch.tensor(mfcc).float().unsqueeze(0)
                
                with torch.no_grad():
                    audio_feat = audio_model(audio_tensor)
                
                # === FUSION & PREDICTION ===
                with torch.no_grad():
                    logits = fusion_model(video_feat, audio_feat)
                
                probs = torch.softmax(logits, dim=1).numpy()[0]
                pred = int(np.argmax(probs))
                conf = probs[pred] * 100
            
            # === RESULTS ===
            st.markdown("---")
            st.subheader("🎯 Results")
            
            col1, col2 = st.columns(2)
            col1.metric("Emotion", EMOTIONS[pred])
            col2.metric("Confidence", f"{conf:.1f}%")
            
            st.markdown("**All Probabilities:**")
            for i, (label, prob) in enumerate(zip(EMOTIONS, probs)):
                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
            
            # Cleanup
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)
        
        finally:
            if os.path.exists(tfile.name):
                os.remove(tfile.name)

st.markdown("---")
st.caption("🎭 RAVDESS Emotion Recognition | Capstone Project")
