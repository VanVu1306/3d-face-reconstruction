"""
MVF-Net Data Preparation Notebook
Prepares 300W-LP and Multi-PIE datasets for training
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from pathlib import Path
import dlib
from scipy.io import loadmat

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    data_root = "./data"
    w300lp_path = os.path.join(data_root, "300W_LP")
    multipie_path = os.path.join(data_root, "Multi-PIE")
    output_path = "./processed_data"
    
    # Image settings
    img_size = 224
    
    # Dataset splits
    train_split = 0.9
    random_seed = 42

# ============================================================================
# Landmark Detection and Face Cropping
# ============================================================================

class FacePreprocessor:
    def __init__(self):
        # Initialize dlib face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    def detect_landmarks(self, image):
        """Detect 68 facial landmarks"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        landmarks = self.predictor(gray, faces[0])
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        return points
    
    def get_crop_bbox(self, landmarks, scale=1.2):
        """Get bounding box from landmarks with padding"""
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        size = max(width, height) * scale
        
        bbox = [
            int(center_x - size/2),
            int(center_y - size/2),
            int(center_x + size/2),
            int(center_y + size/2)
        ]
        return bbox
    
    def crop_and_resize(self, image, bbox, target_size=224):
        """Crop face region and resize to target size"""
        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (target_size, target_size))
        return resized

# ============================================================================
# 300W-LP Dataset Processing
# ============================================================================

class W300LPProcessor:
    """Process 300W-LP dataset for supervised pretraining"""
    
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.preprocessor = FacePreprocessor()
        
    def load_mat_labels(self, mat_file):
        """Load ground truth from .mat file"""
        data = loadmat(mat_file)
        
        # Extract 3DMM parameters
        shape_params = data['Shape_Para'].flatten()  # 199-dim
        exp_params = data['Exp_Para'].flatten()  # 29-dim
        
        # Extract pose parameters
        pose_params = data['Pose_Para'].flatten()  # 6-dim [f, pitch, yaw, roll, tx, ty]
        
        # Extract landmarks
        landmarks_2d = data['pt2d'].T  # 68x2
        
        return {
            'shape_params': shape_params,
            'exp_params': exp_params,
            'pose_params': pose_params,
            'landmarks': landmarks_2d
        }
    
    def create_triplets(self):
        """Create training triplets (left, frontal, right views)"""
        triplets = []
        
        # Iterate through subjects
        for subject_dir in Path(self.data_path).iterdir():
            if not subject_dir.is_dir():
                continue
                
            images = {}
            labels = {}
            
            # Group images by yaw angle
            for img_file in subject_dir.glob("*.jpg"):
                mat_file = img_file.with_suffix('.mat')
                if not mat_file.exists():
                    continue
                
                # Parse yaw angle from filename
                yaw = self._parse_yaw(img_file.name)
                
                images[yaw] = str(img_file)
                labels[yaw] = self.load_mat_labels(str(mat_file))
            
            # Create triplets: left (-30 to -60), frontal (-15 to 15), right (30 to 60)
            left_views = [y for y in images.keys() if -60 <= y <= -30]
            frontal_views = [y for y in images.keys() if -15 <= y <= 15]
            right_views = [y for y in images.keys() if 30 <= y <= 60]
            
            for left in left_views:
                for frontal in frontal_views:
                    for right in right_views:
                        triplets.append({
                            'left': (images[left], labels[left]),
                            'frontal': (images[frontal], labels[frontal]),
                            'right': (images[right], labels[right])
                        })
        
        return triplets
    
    def _parse_yaw(self, filename):
        """Parse yaw angle from filename"""
        # Filename format: xxx_yaw.jpg
        parts = filename.split('_')
        for part in parts:
            if 'yaw' in part.lower():
                return int(part.replace('yaw', '').replace('.jpg', ''))
        return 0
    
    def process_and_save(self):
        """Process all data and save"""
        print("Creating triplets from 300W-LP...")
        triplets = self.create_triplets()
        
        print(f"Found {len(triplets)} triplets")
        
        # Save triplet information
        os.makedirs(self.output_path, exist_ok=True)
        
        processed_triplets = []
        for idx, triplet in enumerate(triplets):
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(triplets)}...")
            
            processed = self._process_triplet(triplet, idx)
            if processed is not None:
                processed_triplets.append(processed)
        
        # Save metadata
        with open(os.path.join(self.output_path, '300wlp_triplets.json'), 'w') as f:
            json.dump(processed_triplets, f)
        
        print(f"Processed {len(processed_triplets)} triplets successfully")
    
    def _process_triplet(self, triplet, idx):
        """Process single triplet"""
        processed = {}
        
        for view in ['left', 'frontal', 'right']:
            img_path, labels = triplet[view]
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Crop and resize
            landmarks = labels['landmarks']
            bbox = self.preprocessor.get_crop_bbox(landmarks)
            cropped = self.preprocessor.crop_and_resize(img, bbox, Config.img_size)
            
            # Save processed image
            save_name = f"{idx}_{view}.jpg"
            save_path = os.path.join(self.output_path, '300wlp_images', save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cropped)
            
            processed[view] = {
                'image_path': save_path,
                'shape_params': labels['shape_params'].tolist(),
                'exp_params': labels['exp_params'].tolist(),
                'pose_params': labels['pose_params'].tolist(),
                'landmarks': labels['landmarks'].tolist()
            }
        
        return processed

# ============================================================================
# Multi-PIE Dataset Processing
# ============================================================================

class MultiPIEProcessor:
    """Process Multi-PIE dataset for self-supervised training"""
    
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.preprocessor = FacePreprocessor()
        
        # Camera IDs for different views
        self.camera_views = {
            'left': ['05_1', '04_1', '08_0'],  # Left side views
            'frontal': ['05_0'],  # Frontal view
            'right': ['05_0', '19_0', '20_0']  # Right side views
        }
    
    def create_triplets(self):
        """Create training triplets from Multi-PIE"""
        triplets = []
        
        # Iterate through sessions and subjects
        for session_dir in Path(self.data_path).glob("session*"):
            for subject_dir in session_dir.glob("*"):
                if not subject_dir.is_dir():
                    continue
                
                # Find images for each view
                views = self._find_views(subject_dir)
                
                # Create triplets
                for left_img in views.get('left', []):
                    for frontal_img in views.get('frontal', []):
                        for right_img in views.get('right', []):
                            # Check if they have similar expressions
                            if self._check_consistency(left_img, frontal_img, right_img):
                                triplets.append({
                                    'left': left_img,
                                    'frontal': frontal_img,
                                    'right': right_img
                                })
        
        return triplets
    
    def _find_views(self, subject_dir):
        """Find images for different views"""
        views = {'left': [], 'frontal': [], 'right': []}
        
        for img_file in subject_dir.rglob("*.png"):
            # Parse camera ID from path
            camera_id = self._parse_camera_id(str(img_file))
            
            for view, camera_list in self.camera_views.items():
                if camera_id in camera_list:
                    views[view].append(str(img_file))
                    break
        
        return views
    
    def _parse_camera_id(self, path):
        """Parse camera ID from image path"""
        parts = Path(path).parts
        for part in parts:
            if 'cam' in part or any(c in part for c in ['05_0', '05_1', '04_1', '08_0', '19_0', '20_0']):
                return part
        return None
    
    def _check_consistency(self, img1, img2, img3):
        """Check if images have consistent expressions (simplified)"""
        # In practice, you'd check lighting codes and recording IDs
        # For now, we assume images from the same recording session are consistent
        return True
    
    def process_and_save(self):
        """Process and save Multi-PIE triplets"""
        print("Creating triplets from Multi-PIE...")
        triplets = self.create_triplets()
        
        print(f"Found {len(triplets)} triplets")
        
        os.makedirs(self.output_path, exist_ok=True)
        
        processed_triplets = []
        for idx, triplet in enumerate(triplets):
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(triplets)}...")
            
            processed = self._process_triplet(triplet, idx)
            if processed is not None:
                processed_triplets.append(processed)
        
        # Save metadata
        with open(os.path.join(self.output_path, 'multipie_triplets.json'), 'w') as f:
            json.dump(processed_triplets, f)
        
        print(f"Processed {len(processed_triplets)} triplets successfully")
    
    def _process_triplet(self, triplet, idx):
        """Process single triplet"""
        processed = {}
        
        for view in ['left', 'frontal', 'right']:
            img_path = triplet[view]
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Detect landmarks
            landmarks = self.preprocessor.detect_landmarks(img)
            if landmarks is None:
                return None
            
            # Crop and resize
            bbox = self.preprocessor.get_crop_bbox(landmarks)
            cropped = self.preprocessor.crop_and_resize(img, bbox, Config.img_size)
            
            # Save processed image
            save_name = f"{idx}_{view}.jpg"
            save_path = os.path.join(self.output_path, 'multipie_images', save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cropped)
            
            processed[view] = {
                'image_path': save_path,
                'landmarks': landmarks.tolist()
            }
        
        return processed

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Process 300W-LP dataset
    print("=" * 80)
    print("Processing 300W-LP Dataset")
    print("=" * 80)
    w300lp_processor = W300LPProcessor(
        Config.w300lp_path,
        os.path.join(Config.output_path, '300wlp')
    )
    w300lp_processor.process_and_save()
    
    # Process Multi-PIE dataset
    print("\n" + "=" * 80)
    print("Processing Multi-PIE Dataset")
    print("=" * 80)
    multipie_processor = MultiPIEProcessor(
        Config.multipie_path,
        os.path.join(Config.output_path, 'multipie')
    )
    multipie_processor.process_and_save()
    
    print("\n" + "=" * 80)
    print("Data preparation completed!")
    print("=" * 80)
