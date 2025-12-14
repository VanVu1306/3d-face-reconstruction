"""
Face detection and image cropping utilities.

Uses face-alignment library for robust face detection.
"""

import numpy as np
from PIL import Image
import face_alignment
from typing import Tuple


class FaceDetector:
    """
    Face detection and automatic cropping using face-alignment library.
    
    Detects facial landmarks and crops to face region with padding.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize face detector.
        
        Args:
            device: 'cpu' or 'cuda' for face-alignment library
        """
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.THREE_D,
            flip_input=False,
            device=device
        )
    
    def crop_image(
        self,
        image: Image.Image,
        resolution: int = 224,
    ) -> Image.Image:
        """
        Detect face and crop image to face region.
        
        Args:
            image: Input PIL Image
            resolution: Output image resolution
        
        Returns:
            Cropped face image
        
        Raises:
            RuntimeError: If no face detected
        """
        # Detect landmarks
        pts = self.fa.get_landmarks(np.array(image))
        if len(pts) < 1:
            raise RuntimeError("No face detected in image!")
        
        pts = np.array(pts[0]).astype(np.int32)
        
        h = image.size[1]
        w = image.size[0]
        
        # Get face bounding box from landmarks
        x_max = np.max(pts[:68, 0])
        x_min = np.min(pts[:68, 0])
        y_max = np.max(pts[:68, 1])
        y_min = np.min(pts[:68, 1])
        bbox = [y_min, x_min, y_max, x_max]
        
        # Compute crop box with padding
        c = [
            bbox[2] - (bbox[2] - bbox[0]) / 2,
            bbox[3] - (bbox[3] - bbox[1]) / 2.0
        ]
        c[0] = c[0] - (bbox[2] - bbox[0]) * 0.12
        
        s = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.5)
        old_bb = np.array([
            c[0] - s / 2,
            c[1] - s / 2,
            c[0] + s / 2,
            c[1] + s / 2
        ]).astype(np.int32)
        
        # Create crop image
        crop_img = Image.new('RGB', (s, s))
        
        # Compute shifts and clipping
        shift_x = 0 - old_bb[1]
        shift_y = 0 - old_bb[0]
        old_bb_clipped = np.array([
            max(0, old_bb[0]),
            max(0, old_bb[1]),
            min(h, old_bb[2]),
            min(w, old_bb[3])
        ]).astype(np.int32)
        
        hb = old_bb_clipped[2] - old_bb_clipped[0]
        wb = old_bb_clipped[3] - old_bb_clipped[1]
        new_bb = np.array([
            max(0, shift_y),
            max(0, shift_x),
            max(0, shift_y) + hb,
            max(0, shift_x) + wb
        ]).astype(np.int32)
        
        # Paste cropped region
        cache = image.crop((
            old_bb_clipped[1],
            old_bb_clipped[0],
            old_bb_clipped[3],
            old_bb_clipped[2]
        ))
        crop_img.paste(cache, (new_bb[1], new_bb[0], new_bb[3], new_bb[2]))
        
        # Resize to target resolution
        crop_img = crop_img.resize((resolution, resolution), Image.BICUBIC)
        
        return crop_img
    
    def get_landmarks(self, image: Image.Image) -> np.ndarray:
        """
        Get facial landmarks from image.
        
        Args:
            image: Input PIL Image
        
        Returns:
            landmarks: (68, 2) facial landmark coordinates
        
        Raises:
            RuntimeError: If no face detected
        """
        pts = self.fa.get_landmarks(np.array(image))
        if len(pts) < 1:
            raise RuntimeError("No face detected in image!")
        
        return np.array(pts[0]).astype(np.int32)


def crop_image(image: Image.Image, res: int = 224) -> Image.Image:
    """
    Convenience function for face detection and cropping.
    
    Args:
        image: Input PIL Image
        res: Output resolution
    
    Returns:
        Cropped face image
    
    Raises:
        RuntimeError: If no face detected
    """
    detector = FaceDetector(device='cpu')
    return detector.crop_image(image, resolution=res)
