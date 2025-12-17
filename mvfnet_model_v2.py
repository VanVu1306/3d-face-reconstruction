"""
MVF-Net Model Architecture
Implements the multi-view 3D face morphable model regression network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from scipy.io import loadmat

# ============================================================================
# 3DMM Model Definition
# ============================================================================

class BFM_Model:
    """Basel Face Model (BFM) wrapper"""
    
    def __init__(self, model_path='BFM/BFM_model_front.mat'):
        """
        Load BFM 2009 model and FaceWarehouse expression basis
        Download from: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads
        """
        # Load BFM model
        model = loadmat(model_path)
        
        # Mean shape (107127,) -> (35709, 3)
        self.mean_shape = model['meanshape'].reshape(-1, 3)
        
        # Identity basis (107127, 199)
        self.id_basis = model['idBase'].reshape(-1, 199, 3)
        
        # Expression basis (107127, 29) from FaceWarehouse
        self.exp_basis = model['exBase'].reshape(-1, 29, 3) if 'exBase' in model else None
        
        # Triangles
        self.triangles = model['tri']
        
        # Keypoints (68 landmark indices)
        self.keypoints = model['keypoints'].flatten() - 1  # MATLAB 1-indexed to Python 0-indexed
        
        self.num_vertices = self.mean_shape.shape[0]
        self.num_id_params = 199
        self.num_exp_params = 29
        
    def generate_vertices(self, shape_params, exp_params):
        """
        Generate 3D vertices from parameters
        
        Args:
            shape_params: (batch_size, 199) identity parameters
            exp_params: (batch_size, 29) expression parameters
            
        Returns:
            vertices: (batch_size, num_vertices, 3)
        """
        batch_size = shape_params.shape[0]
        device = shape_params.device
        
        # Convert to torch tensors if needed
        mean_shape = torch.from_numpy(self.mean_shape).float().to(device)
        id_basis = torch.from_numpy(self.id_basis).float().to(device)
        
        # Generate shape: mean + identity_basis @ shape_params
        # (num_vertices, 3) + (num_vertices, 199, 3) @ (batch, 199, 1)
        vertices = mean_shape.unsqueeze(0).expand(batch_size, -1, -1)
        shape_offset = torch.einsum('vdc,bc->bvd', id_basis, shape_params)
        vertices = vertices + shape_offset
        
        # Add expression offset if available
        if exp_params is not None and self.exp_basis is not None:
            exp_basis = torch.from_numpy(self.exp_basis).float().to(device)
            exp_offset = torch.einsum('vec,bc->bvd', exp_basis, exp_params)
            vertices = vertices + exp_offset
        
        return vertices

# ============================================================================
# Differentiable Renderer
# ============================================================================

class DifferentiableRenderer(nn.Module):
    """Differentiable rendering for texture sampling and projection"""
    
    def __init__(self, img_size=224):
        super().__init__()
        self.img_size = img_size
        
    def projection(self, vertices, pose_params):
        """
        Weak perspective projection
        
        Args:
            vertices: (batch, num_vertices, 3)
            pose_params: (batch, 6) [f, pitch, yaw, roll, tx, ty]
            
        Returns:
            points_2d: (batch, num_vertices, 2)
        """
        batch_size = vertices.shape[0]
        
        # Extract pose parameters
        scale = pose_params[:, 0:1]  # (batch, 1)
        pitch = pose_params[:, 1:2]
        yaw = pose_params[:, 2:3]
        roll = pose_params[:, 3:4]
        tx = pose_params[:, 4:5]
        ty = pose_params[:, 5:6]
        
        # Build rotation matrix from Euler angles
        rotation = self.euler_to_rotation(pitch, yaw, roll)  # (batch, 3, 3)
        
        # Apply rotation
        vertices_rot = torch.bmm(vertices, rotation.transpose(1, 2))  # (batch, num_vertices, 3)
        
        # Weak perspective projection
        x = vertices_rot[:, :, 0]
        y = vertices_rot[:, :, 1]
        
        # Scale and translate
        x_proj = scale * x + tx
        y_proj = scale * y + ty
        
        points_2d = torch.stack([x_proj, y_proj], dim=2)  # (batch, num_vertices, 2)
        
        # Convert to image coordinates [0, img_size]
        points_2d = (points_2d + 1) * self.img_size / 2
        
        return points_2d
    
    def euler_to_rotation(self, pitch, yaw, roll):
        """Convert Euler angles to rotation matrix"""
        batch_size = pitch.shape[0]
        device = pitch.device
        
        # Rotation around X axis (pitch)
        cos_p = torch.cos(pitch)
        sin_p = torch.sin(pitch)
        Rx = torch.zeros(batch_size, 3, 3, device=device)
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1] = cos_p.squeeze()
        Rx[:, 1, 2] = -sin_p.squeeze()
        Rx[:, 2, 1] = sin_p.squeeze()
        Rx[:, 2, 2] = cos_p.squeeze()
        
        # Rotation around Y axis (yaw)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        Ry = torch.zeros(batch_size, 3, 3, device=device)
        Ry[:, 0, 0] = cos_y.squeeze()
        Ry[:, 0, 2] = sin_y.squeeze()
        Ry[:, 1, 1] = 1
        Ry[:, 2, 0] = -sin_y.squeeze()
        Ry[:, 2, 2] = cos_y.squeeze()
        
        # Rotation around Z axis (roll)
        cos_r = torch.cos(roll)
        sin_r = torch.sin(roll)
        Rz = torch.zeros(batch_size, 3, 3, device=device)
        Rz[:, 0, 0] = cos_r.squeeze()
        Rz[:, 0, 1] = -sin_r.squeeze()
        Rz[:, 1, 0] = sin_r.squeeze()
        Rz[:, 1, 1] = cos_r.squeeze()
        Rz[:, 2, 2] = 1
        
        # Combined rotation: R = Rz @ Ry @ Rx
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))
        return R
    
    def sample_texture(self, image, points_2d):
        """
        Sample texture from image using 2D projected points
        
        Args:
            image: (batch, 3, H, W)
            points_2d: (batch, num_vertices, 2) in pixel coordinates
            
        Returns:
            colors: (batch, num_vertices, 3)
        """
        batch_size, _, H, W = image.shape
        
        # Normalize to [-1, 1] for grid_sample
        grid = points_2d.clone()
        grid[:, :, 0] = 2 * grid[:, :, 0] / W - 1
        grid[:, :, 1] = 2 * grid[:, :, 1] / H - 1
        grid = grid.unsqueeze(2)  # (batch, num_vertices, 1, 2)
        
        # Sample texture
        colors = F.grid_sample(image, grid, align_corners=True, mode='bilinear')
        colors = colors.squeeze(3).permute(0, 2, 1)  # (batch, num_vertices, 3)
        
        return colors

# ============================================================================
# MVF-Net Architecture
# ============================================================================

class MVFNet(nn.Module):
    """Multi-View 3D Face Morphable Model Regression Network"""
    
    def __init__(self, bfm_model, img_size=224):
        super().__init__()
        
        self.bfm_model = bfm_model
        self.img_size = img_size
        self.renderer = DifferentiableRenderer(img_size)
        
        # Feature extraction (VGG-Face backbone)
        vgg_face = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg_face.features.children()))
        
        # Adaptive pooling to get fixed size features
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Feature dimension after VGG conv layers
        self.feat_dim = 512 * 7 * 7
        
        # Per-view feature extraction
        self.fc_feat = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(inplace=True)
        )
        
        # Per-view pose regression
        self.fc_pose = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)  # [scale, pitch, yaw, roll, tx, ty]
        )
        
        # Multi-view 3DMM parameter regression
        self.fc_3dmm = nn.Sequential(
            nn.Linear(512 * 3, 1024),  # Concatenate 3 views
            nn.ReLU(inplace=True),
            nn.Linear(1024, 228)  # 199 identity + 29 expression
        )
        
    def forward(self, img_A, img_B, img_C):
        """
        Forward pass
        
        Args:
            img_A: (batch, 3, 224, 224) left view
            img_B: (batch, 3, 224, 224) frontal view
            img_C: (batch, 3, 224, 224) right view
            
        Returns:
            dict with:
                - shape_params: (batch, 199)
                - exp_params: (batch, 29)
                - pose_A, pose_B, pose_C: (batch, 6)
        """
        # Extract features for each view
        feat_A = self.extract_features(img_A)
        feat_B = self.extract_features(img_B)
        feat_C = self.extract_features(img_C)
        
        # Regress pose for each view
        pose_A = self.fc_pose(feat_A)
        pose_B = self.fc_pose(feat_B)
        pose_C = self.fc_pose(feat_C)
        
        # Concatenate features from all views
        feat_all = torch.cat([feat_A, feat_B, feat_C], dim=1)
        
        # Regress 3DMM parameters
        params_3dmm = self.fc_3dmm(feat_all)
        shape_params = params_3dmm[:, :199]
        exp_params = params_3dmm[:, 199:]
        
        return {
            'shape_params': shape_params,
            'exp_params': exp_params,
            'pose_A': pose_A,
            'pose_B': pose_B,
            'pose_C': pose_C
        }
    
    def extract_features(self, img):
        """Extract features from single image"""
        x = self.features(img)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_feat(x)
        return x
    
    def generate_mesh(self, shape_params, exp_params):
        """Generate 3D mesh from parameters"""
        return self.bfm_model.generate_vertices(shape_params, exp_params)
    
    def render_view(self, vertices, pose_params, texture_img):
        """
        Render 3D mesh to 2D image
        
        Args:
            vertices: (batch, num_vertices, 3)
            pose_params: (batch, 6)
            texture_img: (batch, 3, H, W)
            
        Returns:
            rendered: (batch, 3, H, W)
        """
        # Project vertices to 2D
        points_2d = self.renderer.projection(vertices, pose_params)
        
        # Sample texture from source image
        colors = self.renderer.sample_texture(texture_img, points_2d)
        
        # In full implementation, you would rasterize here
        # For simplicity, we return the sampled colors
        return colors, points_2d

# ============================================================================
# Model Factory
# ============================================================================

def create_mvfnet(bfm_path='BFM/BFM_model_front.mat', img_size=224):
    """Create MVF-Net model"""
    bfm_model = BFM_Model(bfm_path)
    model = MVFNet(bfm_model, img_size)
    return model

# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Test model creation
    print("Creating MVF-Net model...")
    model = create_mvfnet()
    model.eval()
    
    # Test forward pass
    batch_size = 2
    img_A = torch.randn(batch_size, 3, 224, 224)
    img_B = torch.randn(batch_size, 3, 224, 224)
    img_C = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(img_A, img_B, img_C)
    
    print("\nOutput shapes:")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")
    
    print("\nModel created successfully!")
