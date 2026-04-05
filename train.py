"""
================================================================================
MULTIMODAL EMOTION RECOGNITION USING CROSS-ATTENTION TRANSFORMERS
================================================================================
- 8 Emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- Cross-Attention Transformer Fusion
- Data Augmentation for Maximum Accuracy
- Checkpoint Saving for Resume Training
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
import timm
import math
import random

from utils.video_utils import get_frames
from utils.audio_utils import extract_audio, extract_audio_features


# ================================================================================
# CONFIGURATION
# ================================================================================

class Config:
    # Model
    EMBED_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 3
    DROPOUT = 0.2
    
    # Training
    BATCH_SIZE = 8
    EPOCHS = 25
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 0.01
    
    # Data
    NUM_FRAMES = 8
    IMG_SIZE = 224
    NUM_CLASSES = 8  # 8 emotions
    
    # Augmentation
    USE_AUGMENTATION = True
    
    # Labels
    EMOTION_LABELS = [
        "Neutral", "Calm", "Happy", "Sad", 
        "Angry", "Fearful", "Disgust", "Surprised"
    ]
    EMOTION_EMOJI = ["😐", "😌", "😊", "😢", "😠", "😨", "🤢", "😲"]


config = Config()


# ================================================================================
# POSITIONAL ENCODING
# ================================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ================================================================================
# VIDEO ENCODER (MobileNetV3-Large + Temporal Attention)
# ================================================================================

class VideoEncoder(nn.Module):
    """
    Extracts spatial features from video frames using MobileNetV3-Large
    Then applies temporal attention across frames
    """
    def __init__(self, embed_dim=256, num_frames=8):
        super().__init__()
        
        # Pretrained CNN backbone
        self.backbone = timm.create_model("mobilenetv3_large_100", pretrained=True)
        self.backbone.classifier = nn.Identity()
        backbone_dim = 1280  # MobileNetV3-Large output — FIXED (was 960)
        
        # Freeze early layers for transfer learning
        for i, param in enumerate(self.backbone.parameters()):
            if i < 100:  # Freeze first 100 layers
                param.requires_grad = False
        
        # Project to embedding dimension
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Temporal attention across frames
        self.temporal_pos = PositionalEncoding(embed_dim, max_len=num_frames)
        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
    
    def forward(self, x, num_frames=1):
        """
        x: (batch * num_frames, 3, H, W) or (batch, 3, H, W)
        Returns: (batch, embed_dim)
        """
        batch_size = x.size(0) // num_frames if num_frames > 1 else x.size(0)
        
        # Extract features from each frame
        features = self.backbone(x)       # (batch * frames, 1280)
        features = self.proj(features)    # (batch * frames, embed_dim)
        
        if num_frames > 1:
            # Reshape to (batch, frames, embed_dim)
            features = features.view(batch_size, num_frames, -1)
            
            # Add temporal positional encoding
            features = self.temporal_pos(features)
            
            # Temporal attention
            features = self.temporal_attn(features)
            
            # Pool across time
            features = features.mean(dim=1)  # (batch, embed_dim)
        
        return features


# ================================================================================
# AUDIO ENCODER (MFCC + Transformer)
# ================================================================================

class AudioEncoder(nn.Module):
    """
    Processes MFCC features using MLP + Self-Attention
    """
    def __init__(self, input_dim=40, embed_dim=256):
        super().__init__()
        
        # MLP to project MFCC
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Self-attention for audio features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x):
        """
        x: (batch, 40) MFCC features
        Returns: (batch, embed_dim)
        """
        # Project input
        x = self.input_proj(x)  # (batch, embed_dim)
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(x)
        
        return x.squeeze(1)  # (batch, embed_dim)


# ================================================================================
# CROSS-ATTENTION TRANSFORMER (The Core of Your Project!)
# ================================================================================

class CrossAttentionLayer(nn.Module):
    """
    Single Cross-Attention Layer
    Query from one modality attends to Key/Value from another modality
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Multi-head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        """
        query: (batch, seq_len, embed_dim) - Modality being updated
        key_value: (batch, seq_len, embed_dim) - Modality being attended to
        
        Returns: Updated query, attention weights
        """
        # Pre-norm cross-attention
        q_norm = self.norm1(query)
        attn_output, attn_weights = self.cross_attn(
            query=q_norm,
            key=key_value,
            value=key_value
        )
        
        # Residual connection
        query = query + self.dropout1(attn_output)
        
        # Pre-norm FFN
        query = query + self.dropout2(self.ffn(self.norm2(query)))
        
        return query, attn_weights


class CrossAttentionTransformer(nn.Module):
    """
    =========================================================================
    CROSS-ATTENTION TRANSFORMER FOR MULTIMODAL FUSION
    =========================================================================
    
    This is the CORE of your project!
    
    How it works:
    1. Video features (face) and Audio features (voice) are separate
    2. Cross-attention allows each modality to "look at" the other
    3. Video asks: "What in the audio matches my visual expression?"
    4. Audio asks: "What in the video matches my voice tone?"
    5. Bidirectional fusion creates rich multimodal representation
    
    Architecture:
    
        Video Feat ─────┐         ┌───── Audio Feat
             │          │         │          │
             │    ┌─────▼─────────▼─────┐    │
             │    │   CROSS-ATTENTION   │    │
             │    │                     │    │
             │    │  V→A: Video attends │    │
             │    │       to Audio      │    │
             │    │                     │    │
             │    │  A→V: Audio attends │    │
             │    │       to Video      │    │
             │    └─────────┬───────────┘    │
             │              │                │
             │        ┌─────▼─────┐          │
             │        │  FUSION   │          │
             │        │ ATTENTION │          │
             │        └─────┬─────┘          │
             │              │                │
             │        ┌─────▼─────┐          │
             │        │ CLASSIFIER│          │
             │        │ 8 Emotions│          │
             │        └───────────┘          │
    
    =========================================================================
    """
    
    def __init__(self, embed_dim=256, num_heads=8, num_layers=3, num_classes=8, dropout=0.2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Learnable modality embeddings
        self.video_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.audio_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Cross-Attention Layers: Video → Audio
        self.video_cross_attn = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-Attention Layers: Audio → Video
        self.audio_cross_attn = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Self-Attention for Fusion
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Store attention weights for visualization
        self.attention_weights = {}
    
    def forward(self, video_feat, audio_feat):
        """
        video_feat: (batch, embed_dim) - From VideoEncoder
        audio_feat: (batch, embed_dim) - From AudioEncoder
        
        Returns: (batch, num_classes) - Emotion logits
        """
        batch_size = video_feat.size(0)
        
        # Add sequence dimension: (batch, embed_dim) → (batch, 1, embed_dim)
        video_seq = video_feat.unsqueeze(1)
        audio_seq = audio_feat.unsqueeze(1)
        
        # Add learnable modality tokens
        video_seq = video_seq + self.video_token.expand(batch_size, -1, -1)
        audio_seq = audio_seq + self.audio_token.expand(batch_size, -1, -1)
        
        # Apply positional encoding
        video_seq = self.pos_encoder(video_seq)
        audio_seq = self.pos_encoder(audio_seq)
        
        # =============================================
        # BIDIRECTIONAL CROSS-ATTENTION
        # =============================================
        
        # Video attends to Audio (multiple layers)
        video_enhanced = video_seq
        for layer in self.video_cross_attn:
            video_enhanced, v2a_weights = layer(video_enhanced, audio_seq)
        
        # Audio attends to Video (multiple layers)
        audio_enhanced = audio_seq
        for layer in self.audio_cross_attn:
            audio_enhanced, a2v_weights = layer(audio_enhanced, video_seq)
        
        # Store attention weights for visualization
        self.attention_weights['video_to_audio'] = v2a_weights
        self.attention_weights['audio_to_video'] = a2v_weights
        
        # =============================================
        # FUSION
        # =============================================
        
        # Concatenate enhanced features
        combined = torch.cat([video_enhanced, audio_enhanced], dim=1)  # (batch, 2, embed_dim)
        
        # Self-attention over combined representation
        fused = self.fusion_transformer(combined)  # (batch, 2, embed_dim)
        
        # Global pooling - both mean and max for richer representation
        fused_mean = fused.mean(dim=1)  # (batch, embed_dim)
        fused_max = fused.max(dim=1)[0]  # (batch, embed_dim)
        
        # Concatenate pooled features
        final_repr = torch.cat([fused_mean, fused_max], dim=1)  # (batch, embed_dim * 2)
        
        # Classify
        logits = self.classifier(final_repr)
        
        return logits
    
    def get_attention_weights(self):
        """Return stored attention weights for visualization"""
        return self.attention_weights


# ================================================================================
# DATA AUGMENTATION
# ================================================================================

class VideoAugmentation:
    """Data augmentation for video frames"""
    
    @staticmethod
    def random_horizontal_flip(frame, p=0.5):
        if random.random() < p:
            return cv2.flip(frame, 1)
        return frame
    
    @staticmethod
    def random_brightness(frame, delta=30):
        value = random.randint(-delta, delta)
        frame = frame.astype(np.int16) + value
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    
    @staticmethod
    def random_contrast(frame, lower=0.8, upper=1.2):
        factor = random.uniform(lower, upper)
        mean = frame.mean()
        frame = (frame - mean) * factor + mean
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    
    @staticmethod
    def random_rotation(frame, max_angle=15):
        angle = random.uniform(-max_angle, max_angle)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h))
    
    @staticmethod
    def random_crop(frame, scale=(0.85, 1.0)):
        h, w = frame.shape[:2]
        s = random.uniform(scale[0], scale[1])
        new_h, new_w = int(h * s), int(w * s)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        frame = frame[top:top+new_h, left:left+new_w]
        return cv2.resize(frame, (w, h))
    
    @staticmethod
    def augment(frame):
        """Apply random augmentations"""
        frame = VideoAugmentation.random_horizontal_flip(frame)
        frame = VideoAugmentation.random_brightness(frame)
        frame = VideoAugmentation.random_contrast(frame)
        if random.random() < 0.3:
            frame = VideoAugmentation.random_rotation(frame)
        if random.random() < 0.3:
            frame = VideoAugmentation.random_crop(frame)
        return frame


class AudioAugmentation:
    """Data augmentation for audio features"""
    
    @staticmethod
    def add_noise(mfcc, noise_level=0.05):
        noise = np.random.randn(*mfcc.shape) * noise_level
        return mfcc + noise
    
    @staticmethod
    def time_shift(mfcc, shift_max=0.1):
        shift = int(len(mfcc) * random.uniform(-shift_max, shift_max))
        return np.roll(mfcc, shift)
    
    @staticmethod
    def augment(mfcc):
        """Apply random augmentations"""
        if random.random() < 0.5:
            mfcc = AudioAugmentation.add_noise(mfcc)
        return mfcc


# ================================================================================
# DATASET WITH 8 EMOTIONS
# ================================================================================

class RAVDESSDataset(Dataset):
    """
    RAVDESS Dataset with 8 Emotions
    
    Filename format: XX-XX-EMOTION-XX-XX-XX-XX.mp4
    
    Emotion codes:
    01 = neutral, 02 = calm, 03 = happy, 04 = sad
    05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    """
    
    def __init__(self, data_dir, augment=False, num_frames=8):
        self.samples = []
        self.augment = augment
        self.num_frames = num_frames
        
        print(f"📁 Scanning: {data_dir}")
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov")):
                    path = os.path.join(root, file)
                    
                    try:
                        parts = file.replace(".mp4", "").replace(".avi", "").replace(".mov", "").split("-")
                        if len(parts) >= 3:
                            emotion_code = int(parts[2])
                            
                            # Map to 0-7 (8 emotions)
                            if 1 <= emotion_code <= 8:
                                self.samples.append({
                                    'path': path,
                                    'label': emotion_code - 1,  # 0-indexed
                                    'emotion': config.EMOTION_LABELS[emotion_code - 1]
                                })
                    except:
                        continue
        
        print(f"✅ Found {len(self.samples)} videos")
        
        # Show distribution
        labels = [s['label'] for s in self.samples]
        dist = Counter(labels)
        print("📊 Distribution:")
        for k in sorted(dist.keys()):
            emoji = config.EMOTION_EMOJI[k]
            name = config.EMOTION_LABELS[k]
            print(f"   {emoji} {name}: {dist[k]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['path']
        label = sample['label']
        
        # ===== VIDEO =====
        frames = get_frames(video_path, max_frames=self.num_frames)
        
        if len(frames) == 0:
            frame = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)
        else:
            # Get middle frame
            frame = frames[len(frames) // 2]
        
        # Detect and crop face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            pad = int(0.3 * w)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            frame = frame[y1:y2, x1:x2]
        
        frame = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE))
        
        # Apply augmentation
        if self.augment:
            frame = VideoAugmentation.augment(frame)
        
        # Normalize with ImageNet stats
        frame = frame.astype(np.float32) / 255.0
        frame = frame[:, :, ::-1].copy()  # BGR to RGB
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame = (frame - mean) / std
        frame = np.transpose(frame, (2, 0, 1))  # HWC → CHW
        frame_tensor = torch.tensor(frame)
        
        # ===== AUDIO =====
        try:
            temp_audio = f"temp_train_{idx}.wav"
            extract_audio(video_path, temp_audio)
            audio_feat = extract_audio_features(temp_audio)
            
            # Apply augmentation
            if self.augment:
                audio_feat = AudioAugmentation.augment(audio_feat)
            
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        except:
            audio_feat = np.zeros(40, dtype=np.float32)
        
        audio_tensor = torch.tensor(audio_feat).float()
        
        return frame_tensor, audio_tensor, label


# ================================================================================
# CHECKPOINT FUNCTIONS
# ================================================================================

CHECKPOINT_PATH = "checkpoint_crossattn.pth"

def save_checkpoint(epoch, video_enc, audio_enc, fusion, optimizer, scheduler, best_acc, history):
    checkpoint = {
        'epoch': epoch,
        'video_encoder': video_enc.state_dict(),
        'audio_encoder': audio_enc.state_dict(),
        'fusion_model': fusion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc,
        'history': history,
        'config': {
            'embed_dim': config.EMBED_DIM,
            'num_heads': config.NUM_HEADS,
            'num_layers': config.NUM_LAYERS,
            'num_classes': config.NUM_CLASSES
        }
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"   💾 Checkpoint saved (Epoch {epoch + 1})")


def load_checkpoint(video_enc, audio_enc, fusion, optimizer, scheduler, device):
    if os.path.exists(CHECKPOINT_PATH):
        print("📂 Found checkpoint! Resuming...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        video_enc.load_state_dict(checkpoint['video_encoder'])
        audio_enc.load_state_dict(checkpoint['audio_encoder'])
        fusion.load_state_dict(checkpoint['fusion_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint.get('history', {'train_acc': [], 'val_acc': [], 'train_loss': []})
        
        print(f"   ✅ Resuming from Epoch {start_epoch + 1}")
        print(f"   🏆 Best accuracy: {best_acc:.1f}%")
        
        return start_epoch, best_acc, history
    else:
        print("🆕 Starting fresh training...")
        return 0, 0, {'train_acc': [], 'val_acc': [], 'train_loss': []}


# ================================================================================
# TRAINING FUNCTION
# ================================================================================

def train():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # Print architecture
    print("\n" + "=" * 70)
    print("🧠 MULTIMODAL EMOTION RECOGNITION USING CROSS-ATTENTION TRANSFORMERS")
    print("=" * 70)
    print(f"""
    ┌────────────────────────────────────────────────────────────────┐
    │                      ARCHITECTURE                              │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │   Video Input              Audio Input                         │
    │   (Frames)                 (MFCC)                              │
    │       │                        │                               │
    │       ▼                        ▼                               │
    │  ┌──────────┐            ┌──────────┐                          │
    │  │MobileNet │            │   MLP    │                          │
    │  │  V3-L    │            │ Encoder  │                          │
    │  └────┬─────┘            └────┬─────┘                          │
    │       │                       │                                │
    │       ▼                       ▼                                │
    │   [256-dim]               [256-dim]                            │
    │       │                       │                                │
    │       │    ┌─────────────┐    │                                │
    │       └───►│   CROSS     │◄───┘                                │
    │            │  ATTENTION  │                                     │
    │            │ TRANSFORMER │                                     │
    │            │             │                                     │
    │            │ • {config.NUM_LAYERS} Layers    │                                     │
    │            │ • {config.NUM_HEADS} Heads     │                                     │
    │            │ • Bidirect. │                                     │
    │            └──────┬──────┘                                     │
    │                   │                                            │
    │                   ▼                                            │
    │            ┌─────────────┐                                     │
    │            │  CLASSIFIER │                                     │
    │            │ 8 Emotions  │                                     │
    │            └─────────────┘                                     │
    │                                                                │
    │   Emotions: 😐 😌 😊 😢 😠 😨 🤢 😲                              │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
    """)
    print("=" * 70 + "\n")
    
    # Dataset
    dataset = RAVDESSDataset("data", augment=config.USE_AUGMENTATION, num_frames=config.NUM_FRAMES)
    
    if len(dataset) == 0:
        print("❌ No data found in 'data/' folder!")
        return
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    # Val set without augmentation
    val_dataset = RAVDESSDataset("data", augment=False, num_frames=config.NUM_FRAMES)
    _, val_set = torch.utils.data.random_split(val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Train: {len(train_set)}, Val: {len(val_set)}")
    
    # Models
    video_encoder = VideoEncoder(embed_dim=config.EMBED_DIM).to(device)
    audio_encoder = AudioEncoder(embed_dim=config.EMBED_DIM).to(device)
    fusion_model = CrossAttentionTransformer(
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in video_encoder.parameters()) + \
                   sum(p.numel() for p in audio_encoder.parameters()) + \
                   sum(p.numel() for p in fusion_model.parameters())
    trainable_params = sum(p.numel() for p in video_encoder.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in audio_encoder.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    
    print(f"📐 Total parameters: {total_params:,}")
    print(f"📐 Trainable parameters: {trainable_params:,}")
    
    # Optimizer with different learning rates
    params = [
        {'params': video_encoder.parameters(), 'lr': config.LEARNING_RATE * 0.1},  # Lower LR for pretrained
        {'params': audio_encoder.parameters(), 'lr': config.LEARNING_RATE},
        {'params': fusion_model.parameters(), 'lr': config.LEARNING_RATE}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.LEARNING_RATE * 0.1, config.LEARNING_RATE, config.LEARNING_RATE],
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Load checkpoint
    start_epoch, best_acc, history = load_checkpoint(
        video_encoder, audio_encoder, fusion_model, optimizer, scheduler, device
    )
    
    print(f"\n🚀 Training: Epoch {start_epoch + 1} to {config.EPOCHS}\n")
    
    # Training loop
    for epoch in range(start_epoch, config.EPOCHS):
        # ===== TRAINING =====
        video_encoder.train()
        audio_encoder.train()
        fusion_model.train()
        
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for frames, audio, labels in pbar:
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            video_feat = video_encoder(frames)
            audio_feat = audio_encoder(audio)
            output = fusion_model(video_feat, audio_feat)
            
            # Loss
            loss = criterion(output, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(video_encoder.parameters()) + 
                list(audio_encoder.parameters()) + 
                list(fusion_model.parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            scheduler.step()
            
            # Stats
            train_loss += loss.item()
            _, pred = output.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100. * train_correct / train_total:.1f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # ===== VALIDATION =====
        video_encoder.eval()
        audio_encoder.eval()
        fusion_model.eval()
        
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for frames, audio, labels in val_loader:
                frames = frames.to(device)
                audio = audio.to(device)
                labels = labels.to(device)
                
                video_feat = video_encoder(frames)
                audio_feat = audio_encoder(audio)
                output = fusion_model(video_feat, audio_feat)
                
                _, pred = output.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
                
                val_predictions.extend(pred.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        train_acc = 100. * train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        
        # Update history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_loss)
        
        print(f"📈 Epoch {epoch + 1}: Train {train_acc:.1f}%, Val {val_acc:.1f}%, Loss {avg_loss:.3f}")
        
        # Per-class accuracy
        if (epoch + 1) % 5 == 0:
            print("   Per-class accuracy:")
            for i in range(config.NUM_CLASSES):
                class_correct = sum(1 for p, l in zip(val_predictions, val_labels) if p == l == i)
                class_total = sum(1 for l in val_labels if l == i)
                if class_total > 0:
                    class_acc = 100. * class_correct / class_total
                    print(f"      {config.EMOTION_EMOJI[i]} {config.EMOTION_LABELS[i]}: {class_acc:.1f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(video_encoder.state_dict(), "video_encoder.pth")
            torch.save(audio_encoder.state_dict(), "audio_encoder.pth")
            torch.save(fusion_model.state_dict(), "fusion_model.pth")
            
            # Save config for inference
            torch.save({
                'embed_dim': config.EMBED_DIM,
                'num_heads': config.NUM_HEADS,
                'num_layers': config.NUM_LAYERS,
                'num_classes': config.NUM_CLASSES,
                'emotion_labels': config.EMOTION_LABELS,
                'emotion_emoji': config.EMOTION_EMOJI
            }, "model_config.pth")
            
            print(f"   ✅ New best! Accuracy: {best_acc:.1f}%")
        
        # Save checkpoint
        save_checkpoint(epoch, video_encoder, audio_encoder, fusion_model, optimizer, scheduler, best_acc, history)
    
    print(f"\n{'='*70}")
    print(f"🎉 TRAINING COMPLETE!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.1f}%")
    print(f"💾 Models saved: video_encoder.pth, audio_encoder.pth, fusion_model.pth")
    print(f"{'='*70}")


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    train()