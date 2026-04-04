"""
Multimodal Emotion Recognition - Streamlit App
Uses the NEW trained models
"""

import streamlit as st
import tempfile
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import timm

from utils.video_utils import get_frames
from utils.audio_utils import extract_audio, extract_audio_features


# ==================== MODELS (Same as train.py) ====================

class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_small_100", pretrained=False)
        self.backbone.classifier = nn.Identity()
        self.proj = nn.Linear(1024, 256)
    
    def forward(self, x):
        x = self.backbone(x)
        return self.proj(x)


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    
    def forward(self, video_feat, audio_feat):
        combined = torch.cat([video_feat, audio_feat], dim=1)
        return self.fc(combined)


# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Emotion Recognition", page_icon="🎭", layout="centered")
st.title("🎭 Multimodal Emotion Recognition")
st.write("Upload a video to detect emotion using **Audio + Video** analysis")


# ==================== LOAD NEW TRAINED MODELS ====================
@st.cache_resource
def load_models():
    video_model = VideoModel()
    audio_model = AudioModel()
    fusion_model = FusionModel()
    
    # Check if NEW models exist
    if not os.path.exists("video_model_new.pth"):
        st.error("❌ Trained models not found! Run `python train.py` first.")
        return None, None, None
    
    try:
        video_model.load_state_dict(torch.load("video_model_new.pth", map_location="cpu"))
        audio_model.load_state_dict(torch.load("audio_model_new.pth", map_location="cpu"))
        fusion_model.load_state_dict(torch.load("fusion_model_new.pth", map_location="cpu"))
        st.success("✅ Trained models loaded! (73% accuracy)")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None
    
    video_model.eval()
    audio_model.eval()
    fusion_model.eval()
    
    return video_model, audio_model, fusion_model


video_model, audio_model, fusion_model = load_models()


# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app detects emotions from videos:
    - 🎥 **Video**: Facial expressions (MobileNetV3)
    - 🎧 **Audio**: Voice tone (MFCC features)
    - 🔗 **Fusion**: Combined analysis
    """)
    
    st.header("😊 Emotions")
    st.write("- 😊 Happy\n- 😢 Sad\n- 😠 Angry\n- 😐 Neutral")
    
    st.header("📊 Model Info")
    st.write("Trained on RAVDESS dataset\n\nAccuracy: ~73%")


# ==================== FILE UPLOAD ====================
st.header("📤 Upload Video")
video_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi"],
    help="Upload a video with a person speaking or showing emotion"
)


# ==================== PROCESS VIDEO ====================
if video_file is not None and video_model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close()
    
    st.video(video_file)
    
    if st.button("🔍 Analyze Emotion", type="primary"):
        try:
            with st.spinner("🔍 Analyzing..."):
                
                # ===== VIDEO PROCESSING =====
                frames = get_frames(tfile.name, max_frames=5)
                
                if len(frames) == 0:
                    st.error("❌ Could not extract frames")
                    st.stop()
                
                video_features = []
                for frame in frames:
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frame = np.transpose(frame, (2, 0, 1))
                    frame_tensor = torch.tensor(frame).unsqueeze(0)
                    
                    with torch.no_grad():
                        feat = video_model(frame_tensor)
                    video_features.append(feat)
                
                video_feat = torch.mean(torch.stack(video_features), dim=0)
                
                # ===== AUDIO PROCESSING =====
                audio_path = extract_audio(tfile.name)
                audio_np = extract_audio_features(audio_path)
                audio_tensor = torch.tensor(audio_np).float().unsqueeze(0)
                
                with torch.no_grad():
                    audio_feat = audio_model(audio_tensor)
                
                # ===== FUSION & PREDICTION =====
                with torch.no_grad():
                    output = fusion_model(video_feat, audio_feat)
                
                probs = torch.softmax(output, dim=1).numpy()[0]
                pred = int(np.argmax(probs))
                confidence = probs[pred] * 100
            
            # ===== DISPLAY RESULTS =====
            st.header("🎯 Results")
            
            labels = ["😊 Happy", "😢 Sad", "😠 Angry", "😐 Neutral"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Emotion", labels[pred])
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            st.subheader("📊 All Predictions")
            for label, prob in zip(labels, probs):
                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
            
            st.success("✅ Analysis complete!")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        
        finally:
            if os.path.exists(tfile.name):
                os.remove(tfile.name)
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")


# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using PyTorch & Streamlit</p>", unsafe_allow_html=True)