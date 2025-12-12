"""
VGG-based encoder for MVF-Net.

Original architecture from MVF-Net paper using VGG16-BN backbone.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import torchvision.models as tvmodel


def reset_params(net: nn.Module) -> None:
    """
    Initialize network parameters.
    
    Args:
        net: Neural network module to initialize
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, 0.0, 0.0001)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.normal(m.weight, 1.0, 0.02)
            nn.init.constant(m.bias, 0)


class VggEncoder(nn.Module):
    """
    VGG16-BN based encoder for 3D face reconstruction.
    
    Takes multi-view face images and regresses morphable model parameters
    and head pose for each view.
    
    Architecture:
    - VGG16-BN feature extraction
    - Per-view pose regression (7 params each)
    - Fused shape + expression regression (228 params)
    
    Input: (B, 9, 224, 224) - 3 concatenated RGB images
    Output: (B, 249) - [shape(199) + exp(29) + pose1(7) + pose2(7) + pose3(7)]
    """
    
    def __init__(self):
        """Initialize VGG encoder."""
        super(VggEncoder, self).__init__()
        
        self.featChannel = 512
        
        # Feature extraction backbone
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
            
            ('conv2', nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))),
            ('bn2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
            
            ('conv3', nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))),
            ('bn3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(True)),
            
            ('conv4', nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))),
            ('bn4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
            
            ('conv5', nn.Conv2d(256, 512, (3, 3), (1, 1), 1)),
            ('bn5', nn.BatchNorm2d(512)),
            ('relu5', nn.ReLU(True)),
            ('pool4', nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
            
            ('conv6', nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)),
            ('bn6', nn.BatchNorm2d(512)),
            ('relu6', nn.ReLU(True)),
            
            ('conv7', nn.Conv2d(512, 512, (3, 3), (1, 1), 1)),
            ('bn7', nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU(True)),
            ('pool5', nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
        ]))
        
        # 3D morphable model (shape + expression) regression head
        self.fc_3dmm = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.featChannel * 3, 256 * 3)),
            ('relu1', nn.ReLU(True)),
            ('fc2', nn.Linear(256 * 3, 228))  # 199 shape + 29 expression
        ]))
        
        # Pose regression head (shared across views)
        self.fc_pose = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(512, 256)),
            ('relu2', nn.ReLU(True)),
            ('fc4', nn.Linear(256, 7))  # rotation(3) + translation(2) + scale(1) + focal(1)
        ]))
        
        # Initialize parameters
        reset_params(self.fc_3dmm)
        reset_params(self.fc_pose)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, 9, 224, 224) - 3 concatenated RGB images (front, left, right)
        
        Returns:
            output: (B, 249) - Shape(199) + Exp(29) + Pose_front(7) + Pose_left(7) + Pose_right(7)
        """
        # Extract features for each view
        imga = x[:, 0:3, :, :]
        feata = self.layer1(imga)
        feata = F.avg_pool2d(feata, feata.size()[2:]).view(feata.size(0), feata.size(1))
        posea = self.fc_pose(feata)
        
        imgb = x[:, 3:6, :, :]
        featb = self.layer1(imgb)
        featb = F.avg_pool2d(featb, featb.size()[2:]).view(featb.size(0), featb.size(1))
        poseb = self.fc_pose(featb)
        
        imgc = x[:, 6:9, :, :]
        featc = self.layer1(imgc)
        featc = F.avg_pool2d(featc, featc.size()[2:]).view(featc.size(0), featc.size(1))
        posec = self.fc_pose(featc)
        
        # Regress shape + expression from concatenated features
        para = self.fc_3dmm(torch.cat([feata, featb, featc], dim=1))
        
        # Concatenate all outputs
        out = torch.cat([para, posea, poseb, posec], dim=1)
        
        return out
