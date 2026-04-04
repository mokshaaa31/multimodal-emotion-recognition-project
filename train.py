"""
Training script with CHECKPOINT SAVING
- Saves progress after every epoch
- Resume anytime by running: python train.py
"""

import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
import timm

from utils.video_utils import get_frames
from utils.audio_utils import extract_audio, extract_audio_features


# ==================== MODELS ====================

class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_small_100", pretrained=True)
        self.backbone.classifier = nn.Identity()
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
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


# ==================== DATASET ====================

class EmotionDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        
        print(f"📁 Scanning: {data_dir}")
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov")):
                    path = os.path.join(root, file)
                    
                    try:
                        parts = file.replace(".mp4", "").replace(".avi", "").replace(".mov", "").split("-")
                        if len(parts) >= 3:
                            emotion_code = int(parts[2])
                            
                            emotion_map = {
                                1: 3,   # neutral
                                2: 3,   # calm -> neutral
                                3: 0,   # happy
                                4: 1,   # sad
                                5: 2,   # angry
                                6: 1,   # fearful -> sad
                                7: 2,   # disgust -> angry
                                8: 0,   # surprised -> happy
                            }
                            
                            if emotion_code in emotion_map:
                                self.samples.append({
                                    'path': path,
                                    'label': emotion_map[emotion_code]
                                })
                    except:
                        continue
        
        print(f"✅ Found {len(self.samples)} videos")
        
        labels = [s['label'] for s in self.samples]
        dist = Counter(labels)
        label_names = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Neutral"}
        print("📊 Distribution:")
        for k in sorted(dist.keys()):
            print(f"   {label_names[k]}: {dist[k]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['path']
        label = sample['label']
        
        frames = get_frames(video_path, max_frames=5)
        if len(frames) == 0:
            frame = np.zeros((224, 224, 3), dtype=np.float32)
        else:
            frame = frames[len(frames)//2]
        
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame_tensor = torch.tensor(frame)
        
        try:
            temp_audio = f"temp_train_{idx}.wav"
            extract_audio(video_path, temp_audio)
            audio_feat = extract_audio_features(temp_audio)
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        except:
            audio_feat = np.zeros(40, dtype=np.float32)
        
        audio_tensor = torch.tensor(audio_feat).float()
        
        return frame_tensor, audio_tensor, label


# ==================== CHECKPOINT FUNCTIONS ====================

CHECKPOINT_PATH = "checkpoint.pth"

def save_checkpoint(epoch, video_model, audio_model, fusion_model, optimizer, scheduler, best_acc):
    """Save training progress"""
    checkpoint = {
        'epoch': epoch,
        'video_model': video_model.state_dict(),
        'audio_model': audio_model.state_dict(),
        'fusion_model': fusion_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"   💾 Checkpoint saved (Epoch {epoch+1})")


def load_checkpoint(video_model, audio_model, fusion_model, optimizer, scheduler, device):
    """Load training progress if exists"""
    if os.path.exists(CHECKPOINT_PATH):
        print("📂 Found checkpoint! Resuming training...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        video_model.load_state_dict(checkpoint['video_model'])
        audio_model.load_state_dict(checkpoint['audio_model'])
        fusion_model.load_state_dict(checkpoint['fusion_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        
        print(f"   ✅ Resuming from Epoch {start_epoch + 1}")
        print(f"   🏆 Best accuracy so far: {best_acc:.1f}%")
        
        return start_epoch, best_acc
    else:
        print("🆕 Starting fresh training...")
        return 0, 0


# ==================== TRAINING ====================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    dataset = EmotionDataset("data")
    
    if len(dataset) == 0:
        print("❌ No data found in 'data/' folder!")
        return
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use fixed seed for consistent splits when resuming
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"📊 Train: {len(train_set)}, Val: {len(val_set)}")
    
    # Initialize models
    video_model = VideoModel().to(device)
    audio_model = AudioModel().to(device)
    fusion_model = FusionModel().to(device)
    
    # Optimizer & Scheduler
    params = list(video_model.parameters()) + list(audio_model.parameters()) + list(fusion_model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if exists
    start_epoch, best_acc = load_checkpoint(video_model, audio_model, fusion_model, optimizer, scheduler, device)
    
    epochs = 20
    
    print(f"\n🚀 Training: Epoch {start_epoch + 1} to {epochs}\n")
    
    for epoch in range(start_epoch, epochs):
        # ===== TRAINING =====
        video_model.train()
        audio_model.train()
        fusion_model.train()
        
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for frames, audio, labels in pbar:
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            video_feat = video_model(frames)
            audio_feat = audio_model(audio)
            output = fusion_model(video_feat, audio_feat)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = output.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*train_correct/train_total:.1f}%'})
        
        # ===== VALIDATION =====
        video_model.eval()
        audio_model.eval()
        fusion_model.eval()
        
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, audio, labels in val_loader:
                frames = frames.to(device)
                audio = audio.to(device)
                labels = labels.to(device)
                
                video_feat = video_model(frames)
                audio_feat = audio_model(audio)
                output = fusion_model(video_feat, audio_feat)
                
                _, pred = output.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        train_acc = 100. * train_correct / train_total
        
        print(f"📈 Epoch {epoch+1}: Train {train_acc:.1f}%, Val {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(video_model.state_dict(), "video_model_new.pth")
            torch.save(audio_model.state_dict(), "audio_model_new.pth")
            torch.save(fusion_model.state_dict(), "fusion_model_new.pth")
            print(f"   ✅ New best! Accuracy: {best_acc:.1f}%")
        
        # Save checkpoint after every epoch
        save_checkpoint(epoch, video_model, audio_model, fusion_model, optimizer, scheduler, best_acc)
        
        scheduler.step()
    
    print(f"\n🎉 Training complete!")
    print(f"🏆 Best accuracy: {best_acc:.1f}%")
    print(f"💾 Models saved: video_model_new.pth, audio_model_new.pth, fusion_model_new.pth")
    
    # Clean up checkpoint after successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("🧹 Checkpoint cleaned up")


if __name__ == "__main__":
    train()