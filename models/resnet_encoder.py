"""
ResNet50-based encoder for MVF-Net.

Alternative architecture using ResNet50 backbone for improved performance.
"""

import torch
import torch.nn as nn
import torchvision.models as tvmodel
from typing import Optional


class ResNetEncoder(nn.Module):
    """
    ResNet50-based encoder for 3D face reconstruction.
    
    Alternative to VggEncoder with ResNet50 backbone for better feature extraction.
    
    Architecture:
    - ResNet50 backbone (without final FC layers)
    - Adaptive average pooling
    - Per-view pose regression heads
    - Fused shape + expression regression head
    
    Input: (B, 9, 224, 224) - 3 concatenated RGB images
    Output: (B, 249) - [shape(199) + exp(29) + pose1(7) + pose2(7) + pose3(7)]
    """
    
    def __init__(self, weights_path: Optional[str] = None):
        """
        Initialize ResNet encoder.
        
        Args:
            weights_path: Optional path to pretrained ResNet50 weights
        """
        super().__init__()
        
        # Load or initialize ResNet50 backbone
        if weights_path:
            base_model = tvmodel.resnet50(weights=None)
            state_dict = torch.load(weights_path)
            base_model.load_state_dict(state_dict)
        else:
            base_model = tvmodel.resnet50(weights=None)
        
        # Remove classification head, keep feature extraction
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = 2048
        
        # Pose regression heads for each view (7 params each)
        self.fc_pose_front = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7)
        )
        
        self.fc_pose_left = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7)
        )
        
        self.fc_pose_right = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7)
        )
        
        # Shape + expression regression head (228 params from concatenated features)
        self.fc_params = nn.Sequential(
            nn.Linear(self.feat_dim * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 228)  # 199 shape + 29 expression
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize linear layer weights using Xavier uniform."""
        for module in [
            self.fc_pose_front,
            self.fc_pose_left,
            self.fc_pose_right,
            self.fc_params
        ]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, 9, 224, 224) - 3 concatenated RGB images (front, left, right)
        
        Returns:
            output: (B, 249) - Shape(199) + Exp(29) + Pose_front(7) + Pose_left(7) + Pose_right(7)
        """
        B = x.size(0)
        
        # Extract features for each view
        front = x[:, 0:3, :, :]
        left = x[:, 3:6, :, :]
        right = x[:, 6:9, :, :]
        
        feat_front = self.backbone(front)
        feat_left = self.backbone(left)
        feat_right = self.backbone(right)
        
        feat_front = self.avgpool(feat_front).view(B, -1)
        feat_left = self.avgpool(feat_left).view(B, -1)
        feat_right = self.avgpool(feat_right).view(B, -1)
        
        # Regress poses per view
        pose_front = self.fc_pose_front(feat_front)  # (B, 7)
        pose_left = self.fc_pose_left(feat_left)
        pose_right = self.fc_pose_right(feat_right)
        
        # Regress shape + expression from concatenated features
        feat_concat = torch.cat([feat_front, feat_left, feat_right], dim=1)
        params = self.fc_params(feat_concat)  # (B, 228)
        
        # Concatenate all outputs
        output = torch.cat([params, pose_front, pose_left, pose_right], dim=1)  # (B, 249)
        
        return output
