import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import torchvision.models as tvmodel

weights_path = "resnet50-11ad3fa6.pth"

def reset_params(net):
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
    def __init__(self):
        super(VggEncoder, self).__init__()

        self.featChannel = 512
        self.layer1 = tvmodel.vgg16_bn(pretrained=True).features
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1',  nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))),
            ('bn1',  nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(True)),
            ('pool1',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
       
            ('conv2',  nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))),
            ('bn2',  nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(True)),
            ('pool2',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
        
            ('conv3',  nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))),
            ('bn3',  nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(True)),
        
            ('conv4', nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))),
            ('bn4',  nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(True)),
            ('pool3',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
        
            ('conv5',  nn.Conv2d(256, 512, (3, 3), (1, 1), 1)),
            ('bn5',  nn.BatchNorm2d(512)),
            ('relu5', nn.ReLU(True)),
            ('pool4',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
        
            ('conv6',  nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)),
            ('bn6',  nn.BatchNorm2d(512)),
            ('relu6', nn.ReLU(True)),
        
            ('conv7',  nn.Conv2d(512, 512, (3, 3), (1, 1), 1)),
            ('bn7',  nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU(True)),
            ('pool5',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
            ]))

        
            
        self.fc_3dmm = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.featChannel*3, 256*3)),
            ('relu1', nn.ReLU(True)),
            ('fc2', nn.Linear(256*3, 228))]))
        
        self.fc_pose = nn.Sequential(OrderedDict([
           ('fc3', nn.Linear(512, 256)),
           ('relu2', nn.ReLU(True)),
           ('fc4', nn.Linear(256, 7))]))
        reset_params(self.fc_3dmm)
        reset_params(self.fc_pose)

    def forward(self, x):
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
        para = self.fc_3dmm(torch.cat([feata, featb, featc], dim=1))
        out = torch.cat([para, posea, poseb, posec], dim=1)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, feat_dim=512, num_shape=199, num_exp=29, num_pose=7, weights_path=None):
        super().__init__()

        self.num_shape = num_shape
        self.num_exp = num_exp
        self.num_pose = num_pose
        
        # Load ResNet50
        if weights_path is not None:
            # Tạo model và load state_dict từ file local
            base_model = tvmodel.resnet50(weights=None)
            state_dict = torch.load(weights_path)
            base_model.load_state_dict(state_dict)
        else:
            # Không dùng pretrained
            base_model = tvmodel.resnet50(weights=None)
        
        # Chỉ giữ convolutional layers
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # output: (batch, 2048, 1, 1)
        self.feat_dim = 2048  # ResNet50 output channel

        # Cải tiến: Learnable fusion weights
        self.w = nn.Parameter(torch.ones(3, self.feat_dim))
        
        # Multi-view fusion for shape and expression
        self.fc_shape = nn.Sequential(
            nn.Linear(self.feat_dim * 3, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_shape)
        )
        
        self.fc_exp = nn.Sequential(
            nn.Linear(self.feat_dim * 3, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_exp)
        )

        # View-specific pose prediction
        self.fc_pose = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_pose)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for newly added layers"""
        for m in [self.fc_shape, self.fc_exp, self.fc_pose]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """
        x: (B, 9, H, W) -> front|left|right concatenated along channel
        output: concatenated [3DMM, poseA, poseB, poseC]
        """

        batch_size = x.size(0)
        
        # Tách 3 view
        front = x[:, 0:3, :, :]
        left = x[:, 3:6, :, :]
        right = x[:, 6:9, :, :]

        # Trích xuất feature backbone
        feat_a = self.backbone(front)
        feat_a = self.avgpool(feat_a).view(batch_size, -1)
        feat_b = self.backbone(left)
        feat_b = self.avgpool(feat_b).view(batch_size, -1)
        feat_c = self.backbone(right)
        feat_c = self.avgpool(feat_c).view(batch_size, -1)

        # Weighted sum fusion
        feat_a = feat_a * self.w[0]   # (B, feat_dim)
        feat_b = feat_b * self.w[1]
        feat_c = feat_c * self.w[2]
    
        # Concatenate weighted features
        feat_fused = torch.cat([feat_a, feat_b, feat_c], dim=1)  # (B, 6144)
        
        # Predict view-invariant parameters
        shape_params = self.fc_shape(feat_fused)  # (B, 199)
        exp_params = self.fc_exp(feat_fused)      # (B, 29)
        
        # Predict view-specific pose parameters (use original unweighted features)
        # Re-extract features without weighting for pose prediction
        feat_a_orig = self.backbone(front)
        feat_a_orig = self.avgpool(feat_a_orig).view(batch_size, -1)
        pose_a = self.fc_pose(feat_a_orig)
        
        feat_b_orig = self.backbone(left)
        feat_b_orig = self.avgpool(feat_b_orig).view(batch_size, -1)
        pose_b = self.fc_pose(feat_b_orig)
        
        feat_c_orig = self.backbone(right)
        feat_c_orig = self.avgpool(feat_c_orig).view(batch_size, -1)
        pose_c = self.fc_pose(feat_c_orig)
        
        # Concatenate all outputs
        output = torch.cat([shape_params, exp_params, pose_a, pose_b, pose_c], dim=1)
        
        return output