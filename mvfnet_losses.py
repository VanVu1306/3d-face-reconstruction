"""
MVF-Net Loss Functions
Implements supervised and self-supervised losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# Supervised Losses (for pretraining on 300W-LP)
# ============================================================================

class LandmarkLoss(nn.Module):
    """2D landmark alignment loss"""
    
    def __init__(self, bfm_model):
        super().__init__()
        self.bfm_model = bfm_model
        self.landmark_indices = torch.from_numpy(bfm_model.keypoints).long()
        
    def forward(self, pred_vertices, pose_params, gt_landmarks, renderer):
        """
        Args:
            pred_vertices: (batch, num_vertices, 3)
            pose_params: (batch, 6)
            gt_landmarks: (batch, 68, 2)
            renderer: DifferentiableRenderer instance
            
        Returns:
            loss: scalar
        """
        # Project 3D vertices to 2D
        points_2d = renderer.projection(pred_vertices, pose_params)
        
        # Extract landmark points
        pred_landmarks = points_2d[:, self.landmark_indices, :]  # (batch, 68, 2)
        
        # Compute L2 loss
        loss = F.mse_loss(pred_landmarks, gt_landmarks)
        
        return loss


class PoseLoss(nn.Module):
    """Pose parameter L2 loss"""
    
    def forward(self, pred_pose, gt_pose):
        """
        Args:
            pred_pose: (batch, 6)
            gt_pose: (batch, 6)
            
        Returns:
            loss: scalar
        """
        return F.mse_loss(pred_pose, gt_pose)


class Params3DMMLoss(nn.Module):
    """3DMM parameters L2 loss"""
    
    def forward(self, pred_shape, pred_exp, gt_shape, gt_exp):
        """
        Args:
            pred_shape: (batch, 199)
            pred_exp: (batch, 29)
            gt_shape: (batch, 199)
            gt_exp: (batch, 29)
            
        Returns:
            loss: scalar
        """
        shape_loss = F.mse_loss(pred_shape, gt_shape)
        exp_loss = F.mse_loss(pred_exp, gt_exp)
        return shape_loss + exp_loss


class RegularizationLoss(nn.Module):
    """Regularization on 3DMM parameters"""
    
    def __init__(self, weight_shape=1e-4, weight_exp=1e-4):
        super().__init__()
        self.weight_shape = weight_shape
        self.weight_exp = weight_exp
        
    def forward(self, shape_params, exp_params):
        """
        Args:
            shape_params: (batch, 199)
            exp_params: (batch, 29)
            
        Returns:
            loss: scalar
        """
        shape_reg = self.weight_shape * torch.mean(shape_params ** 2)
        exp_reg = self.weight_exp * torch.mean(exp_params ** 2)
        return shape_reg + exp_reg


class SupervisedLoss(nn.Module):
    """Combined supervised loss for pretraining"""
    
    def __init__(self, bfm_model, lambda1=0.1, lambda2=10, lambda3=1, lambda4=1):
        super().__init__()
        
        self.landmark_loss = LandmarkLoss(bfm_model)
        self.pose_loss = PoseLoss()
        self.params_loss = Params3DMMLoss()
        self.reg_loss = RegularizationLoss()
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        
    def forward(self, pred, gt, renderer):
        """
        Args:
            pred: dict with predicted parameters
            gt: dict with ground truth
            renderer: DifferentiableRenderer
            
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        # Generate vertices from predicted parameters
        vertices = renderer.bfm_model.generate_vertices(
            pred['shape_params'], pred['exp_params']
        )
        
        # Compute individual losses for each view
        landmark_loss_A = self.landmark_loss(
            vertices, pred['pose_A'], gt['landmarks_A'], renderer
        )
        landmark_loss_B = self.landmark_loss(
            vertices, pred['pose_B'], gt['landmarks_B'], renderer
        )
        landmark_loss_C = self.landmark_loss(
            vertices, pred['pose_C'], gt['landmarks_C'], renderer
        )
        landmark_loss = (landmark_loss_A + landmark_loss_B + landmark_loss_C) / 3
        
        # Pose losses
        pose_loss_A = self.pose_loss(pred['pose_A'], gt['pose_A'])
        pose_loss_B = self.pose_loss(pred['pose_B'], gt['pose_B'])
        pose_loss_C = self.pose_loss(pred['pose_C'], gt['pose_C'])
        pose_loss = (pose_loss_A + pose_loss_B + pose_loss_C) / 3
        
        # 3DMM parameter loss
        params_loss = self.params_loss(
            pred['shape_params'], pred['exp_params'],
            gt['shape_params'], gt['exp_params']
        )
        
        # Regularization
        reg_loss = self.reg_loss(pred['shape_params'], pred['exp_params'])
        
        # Total loss
        total_loss = (
            self.lambda1 * landmark_loss +
            self.lambda2 * pose_loss +
            self.lambda3 * params_loss +
            self.lambda4 * reg_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'landmark': landmark_loss.item(),
            'pose': pose_loss.item(),
            'params': params_loss.item(),
            'reg': reg_loss.item()
        }
        
        return total_loss, loss_dict

# ============================================================================
# Self-Supervised Losses (for training on Multi-PIE)
# ============================================================================

class PhotometricLoss(nn.Module):
    """Photometric consistency loss between observed and rendered images"""
    
    def forward(self, img_observed, img_rendered, mask_observed=None, mask_rendered=None):
        """
        Args:
            img_observed: (batch, 3, H, W)
            img_rendered: (batch, 3, H, W)
            mask_observed: (batch, 1, H, W) or None
            mask_rendered: (batch, 1, H, W) or None
            
        Returns:
            loss: scalar
        """
        # Combine masks
        if mask_observed is not None and mask_rendered is not None:
            mask = (mask_observed * mask_rendered) > 0.5
        elif mask_observed is not None:
            mask = mask_observed > 0.5
        elif mask_rendered is not None:
            mask = mask_rendered > 0.5
        else:
            mask = torch.ones_like(img_observed[:, :1, :, :]) > 0
        
        # Compute pixel-wise L2 loss
        diff = (img_observed - img_rendered) ** 2
        diff = diff * mask
        
        loss = diff.sum() / (mask.sum() * 3 + 1e-8)
        
        return loss


class AlignmentLoss(nn.Module):
    """
    View alignment loss using optical flow
    Uses a pretrained flow network (PWC-Net)
    """
    
    def __init__(self, flow_network):
        super().__init__()
        self.flow_network = flow_network
        
        # Freeze flow network parameters
        for param in self.flow_network.parameters():
            param.requires_grad = False
        
    def forward(self, img1, img2, mask1=None, mask2=None):
        """
        Args:
            img1: (batch, 3, H, W) observed image
            img2: (batch, 3, H, W) rendered image
            mask1: (batch, 1, H, W) or None
            mask2: (batch, 1, H, W) or None
            
        Returns:
            loss: scalar
        """
        # Fill non-face regions with gray to help flow estimation
        if mask1 is not None:
            img1_filled = img1 * mask1 + 0.5 * (1 - mask1)
        else:
            img1_filled = img1
            
        if mask2 is not None:
            img2_filled = img2 * mask2 + 0.5 * (1 - mask2)
        else:
            img2_filled = img2
        
        # Compute bidirectional optical flow
        with torch.no_grad():
            flow_forward = self.flow_network(img1_filled, img2_filled)
            flow_backward = self.flow_network(img2_filled, img1_filled)
        
        # Compute flow magnitude
        flow_mag_forward = torch.sqrt(
            flow_forward[:, 0:1, :, :] ** 2 + flow_forward[:, 1:2, :, :] ** 2
        )
        flow_mag_backward = torch.sqrt(
            flow_backward[:, 0:1, :, :] ** 2 + flow_backward[:, 1:2, :, :] ** 2
        )
        
        # Apply masks
        if mask1 is not None and mask2 is not None:
            mask = (mask1 * mask2) > 0.5
            flow_mag_forward = flow_mag_forward * mask
            flow_mag_backward = flow_mag_backward * mask
            loss = (flow_mag_forward.sum() + flow_mag_backward.sum()) / (mask.sum() * 2 + 1e-8)
        else:
            loss = flow_mag_forward.mean() + flow_mag_backward.mean()
        
        return loss


class SelfSupervisedLoss(nn.Module):
    """Combined self-supervised loss"""
    
    def __init__(self, bfm_model, flow_network, lambda5=1, lambda6=10, lambda7=0.1):
        super().__init__()
        
        self.landmark_loss = LandmarkLoss(bfm_model)
        self.photo_loss = PhotometricLoss()
        self.align_loss = AlignmentLoss(flow_network)
        
        self.lambda5 = lambda5
        self.lambda6 = lambda6
        self.lambda7 = lambda7
        
    def forward(self, pred, images, renderer, landmarks=None, masks=None):
        """
        Args:
            pred: dict with predicted parameters
            images: dict with 'A', 'B', 'C' views
            renderer: DifferentiableRenderer
            landmarks: dict with detected landmarks (optional)
            masks: dict with visibility masks (optional)
            
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        # Generate 3D mesh
        vertices = renderer.bfm_model.generate_vertices(
            pred['shape_params'], pred['exp_params']
        )
        
        # Render cross-view projections
        # A -> B: Sample texture from A, render to B
        colors_A, points_A = renderer.render_view(
            vertices, pred['pose_A'], images['A']
        )
        img_A_to_B = self._rasterize(
            colors_A, points_A, pred['pose_B'], renderer.img_size
        )
        
        # C -> B: Sample texture from C, render to B
        colors_C, points_C = renderer.render_view(
            vertices, pred['pose_C'], images['C']
        )
        img_C_to_B = self._rasterize(
            colors_C, points_C, pred['pose_B'], renderer.img_size
        )
        
        # B -> A: Sample texture from B, render to A
        colors_B_A, points_B_A = renderer.render_view(
            vertices, pred['pose_B'], images['B']
        )
        img_B_to_A = self._rasterize(
            colors_B_A, points_B_A, pred['pose_A'], renderer.img_size
        )
        
        # B -> C: Sample texture from B, render to C
        colors_B_C, points_B_C = renderer.render_view(
            vertices, pred['pose_B'], images['B']
        )
        img_B_to_C = self._rasterize(
            colors_B_C, points_B_C, pred['pose_C'], renderer.img_size
        )
        
        # Photometric losses
        photo_A_B = self.photo_loss(
            images['B'], img_A_to_B,
            masks.get('B_from_A') if masks else None,
            masks.get('A_to_B') if masks else None
        )
        photo_C_B = self.photo_loss(
            images['B'], img_C_to_B,
            masks.get('B_from_C') if masks else None,
            masks.get('C_to_B') if masks else None
        )
        photo_B_A = self.photo_loss(
            images['A'], img_B_to_A,
            masks.get('A') if masks else None,
            masks.get('B_to_A') if masks else None
        )
        photo_B_C = self.photo_loss(
            images['C'], img_B_to_C,
            masks.get('C') if masks else None,
            masks.get('B_to_C') if masks else None
        )
        photo_loss = (photo_A_B + photo_C_B + photo_B_A + photo_B_C) / 4
        
        # Alignment losses
        align_A_B = self.align_loss(
            images['B'], img_A_to_B,
            masks.get('B_from_A') if masks else None,
            masks.get('A_to_B') if masks else None
        )
        align_C_B = self.align_loss(
            images['B'], img_C_to_B,
            masks.get('B_from_C') if masks else None,
            masks.get('C_to_B') if masks else None
        )
        align_B_A = self.align_loss(
            images['A'], img_B_to_A,
            masks.get('A') if masks else None,
            masks.get('B_to_A') if masks else None
        )
        align_B_C = self.align_loss(
            images['C'], img_B_to_C,
            masks.get('C') if masks else None,
            masks.get('B_to_C') if masks else None
        )
        align_loss = (align_A_B + align_C_B + align_B_A + align_B_C) / 4
        
        # Landmark loss (if landmarks provided)
        if landmarks is not None:
            landmark_loss_A = self.landmark_loss(
                vertices, pred['pose_A'], landmarks['A'], renderer
            )
            landmark_loss_B = self.landmark_loss(
                vertices, pred['pose_B'], landmarks['B'], renderer
            )
            landmark_loss_C = self.landmark_loss(
                vertices, pred['pose_C'], landmarks['C'], renderer
            )
            landmark_loss = (landmark_loss_A + landmark_loss_B + landmark_loss_C) / 3
        else:
            landmark_loss = torch.tensor(0.0, device=vertices.device)
        
        # Total loss
        total_loss = (
            self.lambda5 * landmark_loss +
            self.lambda6 * photo_loss +
            self.lambda7 * align_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'landmark': landmark_loss.item(),
            'photo': photo_loss.item(),
            'align': align_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _rasterize(self, colors, points_2d, pose_params, img_size):
        """
        Simplified rasterization (placeholder)
        In practice, use a proper differentiable renderer
        """
        batch_size = colors.shape[0]
        # This is a simplified version - use proper rasterization in practice
        img = torch.zeros(batch_size, 3, img_size, img_size, device=colors.device)
        return img

# ============================================================================
# Loss Factory
# ============================================================================

def create_supervised_loss(bfm_model, **kwargs):
    """Create supervised loss for pretraining"""
    return SupervisedLoss(bfm_model, **kwargs)


def create_selfsupervised_loss(bfm_model, flow_network, **kwargs):
    """Create self-supervised loss for training"""
    return SelfSupervisedLoss(bfm_model, flow_network, **kwargs)
