"""
Cross-Attention Fusion Model - Matches the pre-trained checkpoint exactly.

Checkpoint structure:
- video_proj.weight → [256, 1024]
- video_proj.bias   → [256]
- audio_proj.weight → [256, 256]
- audio_proj.bias   → [256]
- attn.in_proj_weight  → [768, 256]
- attn.in_proj_bias    → [768]
- attn.out_proj.weight → [256, 256]
- attn.out_proj.bias   → [256]
- fc.0.weight → [128, 256]
- fc.0.bias   → [128]
- fc.2.weight → [4, 128]
- fc.2.bias   → [4]

Output: 4 emotion classes
"""

import torch
import torch.nn as nn


class CrossAttentionModel(nn.Module):
    """
    Multimodal fusion using cross-attention.
    
    Layer names MUST match checkpoint exactly:
    - video_proj, audio_proj (not video_projection, etc.)
    - attn (MultiheadAttention)
    - fc (Sequential with indices 0, 2 for Linear layers)
    """
    
    def __init__(self, video_dim=1024, audio_dim=256, hidden_dim=256, num_classes=4):
        super().__init__()
        
        # Projection layers - names must match checkpoint
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Classifier - Sequential with Linear at indices 0 and 2
        # fc.0 = Linear(256, 128)
        # fc.1 = ReLU
        # fc.2 = Linear(128, 4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # fc.0
            nn.ReLU(),                    # fc.1
            nn.Linear(128, num_classes)   # fc.2
        )
    
    def forward(self, video_feat, audio_feat):
        """
        Args:
            video_feat: (batch_size, 1024)
            audio_feat: (batch_size, 256)
        
        Returns:
            logits: (batch_size, 4)
        """
        # Project to common dimension
        v = self.video_proj(video_feat).unsqueeze(1)  # (B, 1, 256)
        a = self.audio_proj(audio_feat).unsqueeze(1)  # (B, 1, 256)
        
        # Cross-attention
        fused, _ = self.attn(v, a, a)
        fused = fused.squeeze(1)  # (B, 256)
        
        # Classify
        return self.fc(fused)
