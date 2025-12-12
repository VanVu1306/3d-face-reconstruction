"""
MVF-Net inference pipeline.

Handles model loading, image preprocessing, and 3D reconstruction.
Wraps the original MVF-Net architecture with clean APIs.
"""

import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, List, Dict, Optional

from models import VggEncoder, ResNetEncoder
from preprocessing import crop_image, FaceDetector
from reconstruction import ShapeReconstructor, write_ply

# Import for backward compatibility with old code
# tools module no longer required here; reconstruction and models are used directly


class MVFNetInference:
    """
    MVF-Net inference wrapper.
    
    Handles:
    - Model loading from checkpoint
    - Image preprocessing and face detection
    - 3D reconstruction from predictions
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "VggEncoder",
        device: str = "cpu",
        crop: bool = True,
    ):
        """
        Initialize MVF-Net model.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            model_type: "VggEncoder" or "ResNetEncoder"
            device: "cpu" or "cuda"
            crop: Crop face using face-alignment library
        """
        self.device = device
        self.crop = crop
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model(model_type, checkpoint_path)
    
    def _load_model(
        self,
        model_type: str,
        checkpoint_path: str,
    ) -> torch.nn.Module:
        """Load model from checkpoint."""
        if model_type == "VggEncoder":
            model = VggEncoder()
        elif model_type == "ResNetEncoder":
            model = ResNetEncoder()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location=torch.device(self.device))
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        
        # Remove "module." prefix if present (from DataParallel)
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        
        model.load_state_dict(new_state)
        model.eval()
        model.to(self.device)
        
        print(f"✓ Model loaded from {checkpoint_path}")
        return model
    
    def preprocess_images(
        self,
        image_front: Image.Image,
        image_left: Image.Image,
        image_right: Image.Image,
        resolution: int = 224,
    ) -> torch.Tensor:
        """
        Preprocess images for model input.
        
        Args:
            image_front: Front view image (PIL Image)
            image_left: Left view image (PIL Image)
            image_right: Right view image (PIL Image)
            resolution: Input resolution
        
        Returns:
            input_tensor: (1, 9, resolution, resolution) batched tensor
        """
        images = [image_front, image_left, image_right]
        
        # Crop if requested
        if self.crop:
            print("Cropping images using face-alignment...")
            images = [crop_image(img, res=resolution) for img in images]
        else:
            images = [img.resize((resolution, resolution), Image.BICUBIC) for img in images]
        
        # Convert to tensors
        tensors = [transforms.functional.to_tensor(img) for img in images]
        
        # Stack: (1, 9, resolution, resolution)
        input_tensor = torch.cat(tensors, dim=0).unsqueeze(0)
        
        return input_tensor
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run model forward pass.
        
        Args:
            input_tensor: (1, 9, 224, 224)
        
        Returns:
            predictions: (1, 249)
        """
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        return predictions
    
    def reconstruct_face(
        self,
        predictions: torch.Tensor,
        model_shape_path: str = "data/3dmm/Model_Shape.mat",
        model_expression_path: str = "data/3dmm/Model_Expression.mat",
        sigma_exp_path: str = "data/3dmm/sigma_exp.mat",
    ) -> Dict:
        """
        Convert model predictions to 3D face shape.
        
        Args:
            predictions: (1, 249) model output
            model_shape_path: Path to morphable model shape
            model_exp_path: Path to morphable model expression
            sigma_exp_path: Path to expression standard deviation
        
        Returns:
            result: Dict with keys:
                - vertices: (N, 3) 3D face shape
                - faces: (M, 3) face connectivity
                - keypoints_front: (68, 2) facial landmarks
                - keypoints_left: (68, 2)
                - keypoints_right: (68, 2)
        """
        preds_np = predictions[0].cpu().numpy()
        
        # Use ShapeReconstructor from new modular structure
        reconstructor = ShapeReconstructor(
            model_shape_path=model_shape_path,
            model_expression_path=model_expression_path,
            sigma_exp_path=sigma_exp_path,
        )
        
        face_data = reconstructor.reconstruct(preds_np)

        return {
            "vertices": face_data["vertices"],
            "faces": face_data["faces"],
            "keypoints_front": face_data["keypoints_front"],
            "keypoints_left": face_data["keypoints_left"],
            "keypoints_right": face_data["keypoints_right"],
        }
    
    def inference(
        self,
        image_front: Image.Image,
        image_left: Image.Image,
        image_right: Image.Image,
        resolution: int = 224,
    ) -> Dict:
        """
        Complete inference pipeline: preprocessing → forward → reconstruction.
        
        Args:
            image_front: Front view image
            image_left: Left view image
            image_right: Right view image
            resolution: Input resolution
        
        Returns:
            result: Dict with 3D face data
        """
        print("\n" + "=" * 70)
        print("MVF-Net Inference")
        print("=" * 70)
        
        # Preprocess
        print("\n[1/3] Preprocessing images...")
        input_tensor = self.preprocess_images(
            image_front, image_left, image_right,
            resolution=resolution
        )
        print(f"  Input tensor shape: {input_tensor.shape}")
        
        # Forward
        print("\n[2/3] Running model forward pass...")
        predictions = self.forward(input_tensor)
        print(f"  Predictions shape: {predictions.shape}")
        
        # Reconstruction
        print("\n[3/3] Reconstructing 3D face...")
        result = self.reconstruct_face(predictions)
        print(f"  Vertices: {result['vertices'].shape}")
        print(f"  Faces: {result['faces'].shape}")
        print(f"  Keypoints: {result['keypoints_front'].shape}")
        print("=" * 70)
        
        return result


def run_inference(
    image_front_path: str,
    image_left_path: str,
    image_right_path: str,
    checkpoint_path: str,
    device: str = "cpu",
    crop: bool = True,
) -> Dict:
    """
    High-level function for MVF-Net inference.
    
    Args:
        image_front_path: Path to front view image
        image_left_path: Path to left view image
        image_right_path: Path to right view image
        checkpoint_path: Path to model checkpoint
        device: "cpu" or "cuda"
        crop: Whether to crop faces
    
    Returns:
        result: 3D face reconstruction result
    """
    # Load images
    img_front = Image.open(image_front_path).convert('RGB')
    img_left = Image.open(image_left_path).convert('RGB')
    img_right = Image.open(image_right_path).convert('RGB')
    
    # Initialize and run inference
    inference = MVFNetInference(
        checkpoint_path=checkpoint_path,
        device=device,
        crop=crop,
    )
    
    result = inference.inference(img_front, img_left, img_right)
    
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference from three-view images in a folder")
    parser.add_argument('--image_path', type=str, default='./data/imgs', help='folder containing front.jpg,left.jpg,right.jpg')
    parser.add_argument('--save_dir', type=str, default='./result', help='directory to save output PLY')
    parser.add_argument('--checkpoint', type=str, default='./data/weights/net.pth', help='model checkpoint path')
    parser.add_argument('--model_type', type=str, default='VggEncoder', choices=['VggEncoder', 'ResNetEncoder'], help='model type')
    parser.add_argument('--device', type=str, default='cpu', help='device: cpu or cuda')
    parser.add_argument('--crop', action='store_true', help='crop faces using face-alignment')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    front = os.path.join(args.image_path, 'front.jpg')
    left = os.path.join(args.image_path, 'left.jpg')
    right = os.path.join(args.image_path, 'right.jpg')

    result = run_inference(front, left, right, checkpoint_path=args.checkpoint, device=args.device, crop=args.crop)

    # write PLY
    out_path = os.path.join(args.save_dir, 'shape_result.ply')
    write_ply(out_path, result['vertices'], result['faces'])
    print(f"Wrote result to {out_path}")
