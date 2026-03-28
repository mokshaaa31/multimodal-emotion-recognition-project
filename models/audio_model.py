"""
Audio Feature Encoder - Matches the pre-trained checkpoint exactly.

Checkpoint structure:
- net.0.weight → [128, 40]   (Linear: 40 → 128)
- net.0.bias   → [128]
- net.2.weight → [256, 128]  (Linear: 128 → 256)
- net.2.bias   → [256]

This means: Sequential(Linear, ReLU, Linear, ReLU) with indices 0, 1, 2, 3
Output: 256 features
"""

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Audio feature encoder.
    
    IMPORTANT: Uses self.net (not self.encoder) to match checkpoint keys.
    """
    
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=256):
        super().__init__()
        
        # MUST be named "net" to match checkpoint keys: net.0, net.2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # net.0
            nn.ReLU(),                          # net.1
            nn.Linear(hidden_dim, output_dim),  # net.2
            nn.ReLU()                           # net.3
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 40) - MFCC features
        
        Returns:
            features: Tensor of shape (batch_size, 256)
        """
        return self.net(x)
