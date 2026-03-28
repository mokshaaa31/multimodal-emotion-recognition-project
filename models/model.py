"""
Video Feature Extractor - Matches the pre-trained checkpoint exactly.

Checkpoint structure:
- model.conv_stem, model.bn1, model.blocks.* → MobileNetV3 backbone
- model.conv_head → [1024, 576, 1, 1] projects to 1024 features

Output: 1024 features
"""

import torch
import torch.nn as nn
import timm


class VideoTransformer(nn.Module):
    """
    Video feature extractor using MobileNetV3 Small.
    
    The checkpoint includes conv_head which projects 576 → 1024.
    We use timm's model but keep the conv_head layer.
    """
    
    def __init__(self):
        super().__init__()
        
        # Create MobileNetV3 Small with conv_head included
        # num_classes=0 removes the final classifier but keeps conv_head
        self.model = timm.create_model(
            "mobilenetv3_small_100", 
            pretrained=False,
            num_classes=0,  # Remove classifier, keep conv_head
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            features: Tensor of shape (batch_size, 1024)
        """
        # This outputs 1024 features (after conv_head + pooling)
        return self.model(x)
