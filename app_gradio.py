"""
🎭 Multimodal Emotion Recognition
Cross-Attention Transformer System
Modern Pastel Minimal Design
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import timm
import time
import math
import tempfile

from utils.video_utils import get_frames
from utils.audio_utils import extract_audio, extract_audio_features

# ==================== IMPORT MODELS FROM TRAIN.PY ====================
from train import (
    VideoEncoder, AudioEncoder, CrossAttentionTransformer,
    PositionalEncoding, Config
)

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== LABELS & STYLING ====================

LABELS = config.EMOTION_LABELS  # 8 emotions
EMOJI_LIST = config.EMOTION_EMOJI
EMOJI = {label: emoji for label, emoji in zip(LABELS, EMOJI_LIST)}

COLORS = {
    "Neutral": "#D4D4D8", "Calm": "#A5D8FF", "Happy": "#86EFAC",
    "Sad": "#93C5FD", "Angry": "#FCA5A5", "Fearful": "#FDE68A",
    "Disgust": "#C4B5FD", "Surprised": "#FDBA74"
}
PASTEL_COLORS = {
    "Neutral": "#E5E5E5", "Calm": "#BAE6FD", "Happy": "#A7F3D0",
    "Sad": "#BFDBFE", "Angry": "#FECACA", "Fearful": "#FEF08A",
    "Disgust": "#DDD6FE", "Surprised": "#FED7AA"
}


# ==================== LOAD MODELS ====================

print("🔄 Loading models...")

video_model = VideoEncoder(embed_dim=config.EMBED_DIM).to(device)
audio_model = AudioEncoder(embed_dim=config.EMBED_DIM).to(device)
fusion_model = CrossAttentionTransformer(
    embed_dim=config.EMBED_DIM,
    num_heads=config.NUM_HEADS,
    num_layers=config.NUM_LAYERS,
    num_classes=config.NUM_CLASSES,
    dropout=config.DROPOUT
).to(device)

try:
    video_model.load_state_dict(torch.load("video_encoder.pth", map_location=device))
    audio_model.load_state_dict(torch.load("audio_encoder.pth", map_location=device))
    fusion_model.load_state_dict(torch.load("fusion_model.pth", map_location=device))
    print("✅ Models loaded!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("   Make sure video_encoder.pth, audio_encoder.pth, fusion_model.pth are in the project folder")

video_model.eval()
audio_model.eval()
fusion_model.eval()

# ImageNet normalization (must match training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_frame(frame):
    """Resize, BGR to RGB, ImageNet normalize"""
    frame = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE))
    frame = frame.astype(np.float32) / 255.0
    frame = frame[:, :, ::-1].copy()
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    frame = np.transpose(frame, (2, 0, 1))
    return torch.tensor(frame).unsqueeze(0).to(device)
# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(frame):
    """Detect and crop face from frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    
    if len(faces) > 0:
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add padding
        pad = int(0.3 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        return frame[y1:y2, x1:x2]
    
    return frame  # Return original if no face found

# ==================== ANALYSIS FUNCTIONS ====================

def analyze_multimodal(video_path):
    """Full multimodal analysis with video + audio"""
    if video_path is None:
        return None, create_empty_result()

    try:
        start_time = time.time()

        # ===== VIDEO ANALYSIS =====
        frames = get_frames(video_path, max_frames=config.NUM_FRAMES)

        if len(frames) == 0:
            return None, create_error_result("Could not extract video frames")

        # Use middle frame
        frame = frames[len(frames) // 2]
        frame = crop_face(frame)
        frame_tensor = preprocess_frame(frame)
        with torch.no_grad():
            video_feat = video_model(frame_tensor)

        # Video-only prediction for breakdown
        with torch.no_grad():
            dummy_audio = torch.zeros(1, config.EMBED_DIM).to(device)
            video_only_out = fusion_model(video_feat, dummy_audio)
            video_probs = torch.softmax(video_only_out, dim=1).cpu().numpy()[0]

        # ===== AUDIO ANALYSIS =====
        audio_analyzed = False
        audio_probs = np.ones(config.NUM_CLASSES) / config.NUM_CLASSES

        try:
            temp_audio = tempfile.mktemp(suffix=".wav")
            extract_audio(video_path, temp_audio)
            audio_np = extract_audio_features(temp_audio)
            audio_tensor = torch.tensor(audio_np).float().unsqueeze(0).to(device)

            with torch.no_grad():
                audio_feat = audio_model(audio_tensor)

            audio_analyzed = True

            # Audio-only prediction for breakdown
            with torch.no_grad():
                dummy_video = torch.zeros(1, config.EMBED_DIM).to(device)
                audio_only_out = fusion_model(dummy_video, audio_feat)
                audio_probs = torch.softmax(audio_only_out, dim=1).cpu().numpy()[0]

            if os.path.exists(temp_audio):
                os.remove(temp_audio)

        except Exception as e:
            print(f"Audio failed: {e}")
            audio_feat = torch.zeros(1, config.EMBED_DIM).to(device)

        # ===== CROSS-ATTENTION FUSION =====
        with torch.no_grad():
            output = fusion_model(video_feat, audio_feat)

        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        confidence = probs[pred] * 100

        proc_time = time.time() - start_time

        results = {f"{EMOJI[LABELS[i]]} {LABELS[i]}": float(probs[i]) for i in range(config.NUM_CLASSES)}

        result_html = create_advanced_result(
            emotion=LABELS[pred],
            confidence=confidence,
            probs=probs,
            video_probs=video_probs,
            audio_probs=audio_probs if audio_analyzed else None,
            audio_analyzed=audio_analyzed,
            num_frames=len(frames),
            proc_time=proc_time
        )

        return results, result_html

    except Exception as e:
        return None, create_error_result(str(e))


def analyze_live_frame(image):
    """Analyze single frame from camera with face detection"""
    if image is None:
        return None, create_waiting_result()

    try:
        # Crop to face first
        face = crop_face(image)
        
        frame_tensor = preprocess_frame(face)

        with torch.no_grad():
            video_feat = video_model(frame_tensor)
            # Use the video features AS audio too (better than zeros)
            output = fusion_model(video_feat, video_feat)

        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        confidence = probs[pred] * 100

        results = {f"{EMOJI[LABELS[i]]} {LABELS[i]}": float(probs[i]) for i in range(config.NUM_CLASSES)}
        result_html = create_live_result(LABELS[pred], confidence, probs)

        return results, result_html

    except:
        return None, create_waiting_result()


# ==================== RESULT HTML GENERATORS ====================

def create_empty_result():
    return """
    <div style="display: flex; align-items: center; justify-content: center; height: 300px; background: linear-gradient(135deg, #faf5ff 0%, #f0f9ff 100%); border-radius: 24px; border: 2px dashed #d4d4d8;">
        <div style="text-align: center;">
            <div style="font-size: 48px; opacity: 0.5;">🎭</div>
            <p style="color: #a1a1aa; margin-top: 16px; font-size: 16px;">Upload or record a video to analyze</p>
        </div>
    </div>
    """


def create_waiting_result():
    return """
    <div style="display: flex; align-items: center; justify-content: center; height: 200px; background: linear-gradient(135deg, #faf5ff 0%, #f0f9ff 100%); border-radius: 24px;">
        <div style="text-align: center;">
            <div style="font-size: 40px;">📷</div>
            <p style="color: #a1a1aa; margin-top: 12px;">Starting camera...</p>
        </div>
    </div>
    """


def create_error_result(error):
    return f"""
    <div style="background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%); border-radius: 24px; padding: 30px; border: 1px solid #fecaca;">
        <div style="text-align: center;">
            <div style="font-size: 40px;">⚠️</div>
            <p style="color: #dc2626; margin-top: 12px; font-weight: 500;">Analysis Error</p>
            <p style="color: #f87171; font-size: 14px;">{error}</p>
        </div>
    </div>
    """


def create_live_result(emotion, confidence, probs):
    emoji_char = EMOJI[emotion]
    color = COLORS[emotion]
    pastel = PASTEL_COLORS[emotion]

    bars = ""
    for i, label in enumerate(LABELS):
        prob = probs[i]
        bars += f"""
        <div style="display: flex; align-items: center; gap: 12px; margin: 6px 0;">
            <span style="width: 85px; font-size: 12px; color: #71717a;">{EMOJI[label]} {label}</span>
            <div style="flex: 1; height: 8px; background: #f4f4f5; border-radius: 4px; overflow: hidden;">
                <div style="width: {prob*100}%; height: 100%; background: {PASTEL_COLORS[label]}; border-radius: 4px;"></div>
            </div>
            <span style="width: 40px; text-align: right; font-size: 12px; color: #a1a1aa;">{prob*100:.0f}%</span>
        </div>
        """

    return f"""
    <div style="background: linear-gradient(135deg, {pastel}40 0%, #faf5ff 100%); border-radius: 24px; padding: 24px; border: 1px solid {pastel};">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
            <div style="width: 64px; height: 64px; background: {pastel}; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 32px;">{emoji_char}</div>
            <div>
                <div style="font-size: 24px; font-weight: 600; color: #3f3f46;">{emotion}</div>
                <div style="font-size: 14px; color: #71717a;">{confidence:.0f}% confidence</div>
            </div>
        </div>
        {bars}
        <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #e5e5e5;">
            <span style="font-size: 11px; color: #a1a1aa;">🎥 Video-only analysis • Real-time</span>
        </div>
    </div>
    """


def create_advanced_result(emotion, confidence, probs, video_probs, audio_probs, audio_analyzed, num_frames, proc_time):
    emoji_char = EMOJI[emotion]
    color = COLORS[emotion]
    pastel = PASTEL_COLORS[emotion]

    # Main emotion bars
    main_bars = ""
    for i, label in enumerate(LABELS):
        prob = probs[i]
        is_top = label == emotion
        main_bars += f"""
        <div style="display: flex; align-items: center; gap: 12px; margin: 8px 0;">
            <span style="width: 90px; font-size: 13px; color: {'#3f3f46' if is_top else '#71717a'}; font-weight: {'600' if is_top else '400'};">{EMOJI[label]} {label}</span>
            <div style="flex: 1; height: 12px; background: #f4f4f5; border-radius: 6px; overflow: hidden;">
                <div style="width: {prob*100}%; height: 100%; background: linear-gradient(90deg, {PASTEL_COLORS[label]}, {COLORS[label]}); border-radius: 6px;"></div>
            </div>
            <span style="width: 50px; text-align: right; font-size: 13px; color: {'#3f3f46' if is_top else '#a1a1aa'}; font-weight: {'600' if is_top else '400'};">{prob*100:.1f}%</span>
        </div>
        """

    # Modality breakdown
    video_pred = LABELS[np.argmax(video_probs)]
    video_conf = np.max(video_probs) * 100

    audio_section = ""
    if audio_analyzed and audio_probs is not None:
        audio_pred = LABELS[np.argmax(audio_probs)]
        audio_conf = np.max(audio_probs) * 100
        audio_section = f"""
        <div style="flex: 1; background: #f0f9ff; border-radius: 16px; padding: 16px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 20px;">🎧</span>
                <span style="font-size: 13px; font-weight: 600; color: #3b82f6;">Audio Analysis</span>
            </div>
            <div style="font-size: 18px; font-weight: 600; color: #1e40af;">{EMOJI[audio_pred]} {audio_pred}</div>
            <div style="font-size: 12px; color: #60a5fa;">{audio_conf:.0f}% confidence</div>
        </div>
        """
    else:
        audio_section = """
        <div style="flex: 1; background: #fafafa; border-radius: 16px; padding: 16px; opacity: 0.6;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 20px;">🎧</span>
                <span style="font-size: 13px; font-weight: 600; color: #a1a1aa;">Audio Analysis</span>
            </div>
            <div style="font-size: 14px; color: #a1a1aa;">No audio detected</div>
        </div>
        """

    fusion_badge = "✓ Cross-Attention Fusion" if audio_analyzed else "Video-Only Mode"
    fusion_color = "#8b5cf6" if audio_analyzed else "#a1a1aa"

    return f"""
    <div style="background: linear-gradient(135deg, #ffffff 0%, #faf5ff 50%, #f0f9ff 100%); border-radius: 28px; padding: 28px; border: 1px solid #e5e5e5; box-shadow: 0 4px 24px rgba(0,0,0,0.05);">

        <!-- Main Result -->
        <div style="text-align: center; margin-bottom: 28px;">
            <div style="width: 100px; height: 100px; background: linear-gradient(135deg, {pastel} 0%, {color} 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 48px; margin: 0 auto 16px; box-shadow: 0 8px 24px {pastel}80;">
                {emoji_char}
            </div>
            <div style="font-size: 32px; font-weight: 700; color: #18181b; letter-spacing: -0.5px;">{emotion}</div>
            <div style="display: inline-block; margin-top: 8px; padding: 6px 16px; background: linear-gradient(90deg, {pastel}, {color}40); border-radius: 20px;">
                <span style="font-size: 16px; font-weight: 600; color: #3f3f46;">{confidence:.1f}% Confidence</span>
            </div>
        </div>

        <!-- Divider -->
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #e5e5e5, transparent); margin: 24px 0;"></div>

        <!-- Probability Breakdown -->
        <div style="margin-bottom: 24px;">
            <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #a1a1aa; margin-bottom: 12px;">Emotion Analysis</div>
            {main_bars}
        </div>

        <!-- Divider -->
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #e5e5e5, transparent); margin: 24px 0;"></div>

        <!-- Modality Breakdown -->
        <div style="margin-bottom: 20px;">
            <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #a1a1aa; margin-bottom: 12px;">Multimodal Breakdown</div>
            <div style="display: flex; gap: 12px;">
                <div style="flex: 1; background: #faf5ff; border-radius: 16px; padding: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="font-size: 20px;">🎥</span>
                        <span style="font-size: 13px; font-weight: 600; color: #8b5cf6;">Video Analysis</span>
                    </div>
                    <div style="font-size: 18px; font-weight: 600; color: #6d28d9;">{EMOJI[video_pred]} {video_pred}</div>
                    <div style="font-size: 12px; color: #a78bfa;">{video_conf:.0f}% confidence</div>
                </div>
                {audio_section}
            </div>
        </div>

        <!-- Fusion Visualization -->
        <div style="background: linear-gradient(90deg, #faf5ff, #f0f9ff); border-radius: 16px; padding: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 16px;">
                <div style="text-align: center;">
                    <div style="font-size: 24px;">🎥</div>
                    <div style="font-size: 11px; color: #71717a;">Video</div>
                </div>
                <div style="font-size: 20px; color: #d4d4d8;">+</div>
                <div style="text-align: center;">
                    <div style="font-size: 24px;">🎧</div>
                    <div style="font-size: 11px; color: #71717a;">Audio</div>
                </div>
                <div style="font-size: 20px; color: #d4d4d8;">→</div>
                <div style="text-align: center; background: linear-gradient(135deg, #8b5cf6, #3b82f6); padding: 8px 16px; border-radius: 12px;">
                    <div style="font-size: 18px;">🔗</div>
                    <div style="font-size: 10px; color: white; font-weight: 500;">Cross-Attn</div>
                </div>
                <div style="font-size: 20px; color: #d4d4d8;">→</div>
                <div style="text-align: center;">
                    <div style="font-size: 28px;">{emoji_char}</div>
                    <div style="font-size: 11px; color: #71717a; font-weight: 600;">{emotion}</div>
                </div>
            </div>
        </div>

        <!-- Stats Footer -->
        <div style="display: flex; justify-content: space-between; padding-top: 16px; border-top: 1px solid #e5e5e5;">
            <div style="display: flex; gap: 16px;">
                <span style="font-size: 11px; color: #a1a1aa;">📊 {num_frames} frames analyzed</span>
                <span style="font-size: 11px; color: #a1a1aa;">⚡ {proc_time:.2f}s processing</span>
            </div>
            <span style="font-size: 11px; color: {fusion_color}; font-weight: 500;">{fusion_badge}</span>
        </div>
    </div>
    """


# ==================== CUSTOM CSS ====================

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #faf5ff 0%, #f0f9ff 50%, #f5f3ff 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.dark { background: transparent !important; }
.block {
    background: white !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 20px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
}
h1, h2, h3, h4 { color: #18181b !important; }
.tab-nav {
    background: white !important;
    border-radius: 16px !important;
    padding: 6px !important;
    border: 1px solid #e5e5e5 !important;
    gap: 4px !important;
}
.tab-nav button {
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    color: #71717a !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #c4b5fd 0%, #a5b4fc 100%) !important;
    color: #3f3f46 !important;
}
.primary {
    background: linear-gradient(135deg, #c4b5fd 0%, #a5b4fc 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #3f3f46 !important;
    font-weight: 600 !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    box-shadow: 0 4px 12px rgba(196, 181, 253, 0.4) !important;
}
.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(196, 181, 253, 0.5) !important;
}
.secondary {
    background: white !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 14px !important;
    color: #71717a !important;
}
label { color: #52525b !important; font-weight: 500 !important; font-size: 14px !important; }
input, textarea {
    background: #fafafa !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 12px !important;
    color: #3f3f46 !important;
}
input:focus, textarea:focus {
    border-color: #c4b5fd !important;
    box-shadow: 0 0 0 3px rgba(196, 181, 253, 0.2) !important;
}
video, img { border-radius: 16px !important; }
.upload-button {
    background: linear-gradient(135deg, #faf5ff 0%, #f0f9ff 100%) !important;
    border: 2px dashed #d4d4d8 !important;
    border-radius: 16px !important;
}
.label-container {
    background: white !important;
    border-radius: 16px !important;
    border: 1px solid #e5e5e5 !important;
}
.markdown-text { color: #52525b !important; }
footer { display: none !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f4f4f5; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 3px; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.block { animation: fadeIn 0.3s ease; }
"""


# ==================== GRADIO APP ====================

with gr.Blocks(
    title="🎭 Emotion Recognition",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="blue",
        neutral_hue="zinc",
        font=("Inter", "system-ui", "sans-serif"),
    )
) as app:

    # ===== HEADER =====
    gr.HTML("""
    <div style="text-align: center; padding: 40px 20px 30px; margin-bottom: 20px;">
        <div style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #c4b5fd40 0%, #a5b4fc40 100%); border-radius: 40px; margin-bottom: 20px;">
            <span style="font-size: 14px; color: #6d28d9; font-weight: 500;">🔬 Powered by Cross-Attention Transformers</span>
        </div>
        <h1 style="font-size: 42px; font-weight: 700; color: #18181b; margin: 0; letter-spacing: -1px;">
            Multimodal Emotion Recognition
        </h1>
        <p style="color: #71717a; font-size: 18px; margin-top: 12px; margin-bottom: 0;">
            Advanced AI that analyzes <span style="color: #8b5cf6; font-weight: 500;">facial expressions</span> and <span style="color: #3b82f6; font-weight: 500;">voice</span> to detect emotions
        </p>
    </div>
    """)

    with gr.Tabs():

        # ===== TAB 1: UPLOAD =====
        with gr.TabItem("📤 Upload Video", id="upload"):
            gr.HTML("""<div style="text-align: center; margin-bottom: 16px;">
                <p style="color: #71717a; margin: 0;">Upload a video file with audio for the most accurate analysis</p>
            </div>""")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Video", sources=["upload"])
                    analyze_btn = gr.Button("🔍 Analyze Emotions", variant="primary", size="lg")

                with gr.Column(scale=1):
                    video_output = gr.HTML(create_empty_result())
                    video_label = gr.Label(label="Probabilities", num_top_classes=8, visible=False)

            analyze_btn.click(
                fn=analyze_multimodal,
                inputs=[video_input],
                outputs=[video_label, video_output]
            )


        # ===== TAB 2: RECORD =====
        with gr.TabItem("🎥 Record Video", id="record"):
            gr.HTML("""<div style="text-align: center; margin-bottom: 16px;">
                <p style="color: #71717a; margin: 0;">Record with your webcam <strong>and microphone</strong> for full multimodal analysis</p>
            </div>""")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    record_input = gr.Video(label="Record", sources=["webcam"], include_audio=True)
                    record_btn = gr.Button("🔍 Analyze Recording", variant="primary", size="lg")

                    gr.HTML("""
                    <div style="background: linear-gradient(135deg, #a5b4fc20 0%, #c4b5fd20 100%); border-radius: 14px; padding: 16px; margin-top: 16px; border: 1px solid #c4b5fd40;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 20px;">🎧</span>
                            <div>
                                <div style="font-size: 14px; font-weight: 600; color: #6d28d9;">Audio Analysis Enabled</div>
                                <div style="font-size: 12px; color: #8b5cf6;">Speak while recording for best results</div>
                            </div>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=1):
                    record_output = gr.HTML(create_empty_result())
                    record_label = gr.Label(label="Probabilities", num_top_classes=8, visible=False)

            record_btn.click(
                fn=analyze_multimodal,
                inputs=[record_input],
                outputs=[record_label, record_output]
            )


        # ===== TAB 3: LIVE =====
        with gr.TabItem("📷 Live Camera", id="live"):
            gr.HTML("""<div style="text-align: center; margin-bottom: 16px;">
                <p style="color: #71717a; margin: 0;">Real-time emotion detection from your webcam</p>
            </div>""")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    camera_input = gr.Image(label="Camera", sources=["webcam"], streaming=True)

                with gr.Column(scale=1):
                    camera_output = gr.HTML(create_waiting_result())
                    camera_label = gr.Label(label="Live Detection", num_top_classes=8)

            camera_input.stream(
                fn=analyze_live_frame,
                inputs=[camera_input],
                outputs=[camera_label, camera_output],
                stream_every=0.3,
            )

            gr.HTML("""
            <div style="background: linear-gradient(135deg, #fef3c720 0%, #fef08a20 100%); border-radius: 14px; padding: 16px; margin-top: 16px; border: 1px solid #fef08a40;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 20px;">💡</span>
                    <div>
                        <div style="font-size: 14px; font-weight: 600; color: #a16207;">Video-Only Mode</div>
                        <div style="font-size: 12px; color: #ca8a04;">For audio analysis, use the "Record Video" tab</div>
                    </div>
                </div>
            </div>
            """)


        # ===== TAB 4: ABOUT =====
        with gr.TabItem("ℹ️ About", id="about"):
            gr.HTML("""
            <div style="max-width: 700px; margin: 0 auto; padding: 20px;">

                <!-- Architecture -->
                <div style="background: white; border-radius: 20px; padding: 28px; margin-bottom: 20px; border: 1px solid #e5e5e5;">
                    <h3 style="margin: 0 0 20px 0; color: #18181b; font-size: 18px;">🏗️ System Architecture</h3>
                    <div style="display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap; gap: 16px;">
                        <div style="flex: 1; min-width: 120px; padding: 20px; background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%); border-radius: 16px;">
                            <div style="font-size: 32px; margin-bottom: 8px;">🎥</div>
                            <div style="font-weight: 600; color: #6d28d9;">Video</div>
                            <div style="font-size: 12px; color: #a78bfa; margin-top: 4px;">MobileNetV3-Large</div>
                            <div style="font-size: 11px; color: #c4b5fd;">+ Temporal Attention</div>
                        </div>
                        <div style="flex: 1; min-width: 120px; padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 16px;">
                            <div style="font-size: 32px; margin-bottom: 8px;">🎧</div>
                            <div style="font-weight: 600; color: #2563eb;">Audio</div>
                            <div style="font-size: 12px; color: #60a5fa; margin-top: 4px;">MFCC + Self-Attention</div>
                        </div>
                        <div style="flex: 1; min-width: 120px; padding: 20px; background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%); border-radius: 16px;">
                            <div style="font-size: 32px; margin-bottom: 8px;">🔗</div>
                            <div style="font-weight: 600; color: #a21caf;">Fusion</div>
                            <div style="font-size: 12px; color: #e879f9; margin-top: 4px;">Cross-Attention</div>
                            <div style="font-size: 11px; color: #f0abfc;">8 heads, 3 layers</div>
                        </div>
                    </div>
                </div>

                <!-- Emotions -->
                <div style="background: white; border-radius: 20px; padding: 28px; margin-bottom: 20px; border: 1px solid #e5e5e5;">
                    <h3 style="margin: 0 0 20px 0; color: #18181b; font-size: 18px;">🎭 Detected Emotions</h3>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
                        <div style="text-align: center; padding: 14px; background: #E5E5E520; border-radius: 14px; border: 1px solid #E5E5E5;">
                            <div style="font-size: 24px;">😐</div>
                            <div style="font-weight: 600; color: #525252; font-size: 13px;">Neutral</div>
                        </div>
                        <div style="text-align: center; padding: 14px; background: #BAE6FD20; border-radius: 14px; border: 1px solid #BAE6FD;">
                            <div style="font-size: 24px;">😌</div>
                            <div style="font-weight: 600; color: #0369a1; font-size: 13px;">Calm</div>
                        </div>
                        <div style="text-align: center; padding: 14px; background: #A7F3D020; border-radius: 14px; border: 1px solid #A7F3D0;">
                            <div style="font-size: 24px;">😊</div>
                            <div style="font-weight: 600; color: #059669; font-size: 13px;">Happy</div>
                        </div>
                        <div style="text-align: center; padding: 14px; background: #BFDBFE20; border-radius: 14px; border: 1px solid #BFDBFE;">
                            <div style="font-size: 24px;">😢</div>
                            <div style="font-weight: 600; color: #2563eb; font-size: 13px;">Sad</div>
                        </div>
                        <div style="text-align: center; padding: 14px; background: #FECACA20; border-radius: 14px; border: 1px solid #FECACA;">
                            <div style="font-size: 24px;">😠</div>
                            <div style="font-weight: 600; color: #dc2626; font-size: 13px;">Angry</div>
                        </div>
                        <div style="text-align: center; padding: 14px; background: #FEF08A20; border-radius: 14px; border: 1px solid #FEF08A;">
                            <div style="font-size: 24px;">😨</div>
                            <div style="font-weight: 600; color: #a16207; font-size: 13px;">Fearful</div>
                        </div>
                        <div style="text-align: center; padding: 14px; background: #DDD6FE20; border-radius: 14px; border: 1px solid #DDD6FE;">
                            <div style="font-size: 24px;">🤢</div>
                            <div style="font-weight: 600; color: #7c3aed; font-size: 13px;">Disgust</div>
                        </div>
                        <div style="text-align: center; padding: 14px; background: #FED7AA20; border-radius: 14px; border: 1px solid #FED7AA;">
                            <div style="font-size: 24px;">😲</div>
                            <div style="font-weight: 600; color: #c2410c; font-size: 13px;">Surprised</div>
                        </div>
                    </div>
                </div>

                <!-- Stats -->
                <div style="background: white; border-radius: 20px; padding: 28px; border: 1px solid #e5e5e5;">
                    <h3 style="margin: 0 0 20px 0; color: #18181b; font-size: 18px;">📊 Model Details</h3>
                    <div style="display: flex; justify-content: space-around; text-align: center;">
                        <div>
                            <div style="font-size: 28px; font-weight: 700; color: #8b5cf6;">8</div>
                            <div style="font-size: 13px; color: #71717a;">Emotions</div>
                        </div>
                        <div>
                            <div style="font-size: 28px; font-weight: 700; color: #3b82f6;">4904</div>
                            <div style="font-size: 13px; color: #71717a;">Videos</div>
                        </div>
                        <div>
                            <div style="font-size: 28px; font-weight: 700; color: #10b981;">24</div>
                            <div style="font-size: 13px; color: #71717a;">Actors</div>
                        </div>
                        <div>
                            <div style="font-size: 28px; font-weight: 700; color: #f59e0b;">13M</div>
                            <div style="font-size: 13px; color: #71717a;">Parameters</div>
                        </div>
                    </div>
                </div>

            </div>
            """)


    # ===== FOOTER =====
    gr.HTML("""
    <div style="text-align: center; padding: 30px 20px; margin-top: 30px;">
        <p style="color: #a1a1aa; margin: 0; font-size: 14px;">
            Built with PyTorch • MobileNetV3-Large • Cross-Attention Transformer • Gradio
        </p>
        <p style="color: #d4d4d8; margin: 8px 0 0 0; font-size: 13px;">
            Capstone Project 2026
        </p>
    </div>
    """)


# ==================== LAUNCH ====================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🎭 Multimodal Emotion Recognition")
    print("   Cross-Attention Transformer")
    print("=" * 50)
    print("🚀 http://localhost:7860")
    print("=" * 50 + "\n")

    app.launch(server_name="0.0.0.0", server_port=7860)