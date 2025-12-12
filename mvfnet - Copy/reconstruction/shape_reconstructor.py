"""
3D face shape reconstruction from model predictions.

Converts MVF-Net model predictions (shape + expression parameters + poses)
to 3D face vertices and facial landmarks using the 3D morphable model.
"""

import numpy as np
import scipy.io as io
import math
from typing import List, Tuple, Dict


class ShapeReconstructor:
    """
    Reconstructs 3D face shape from morphable model parameters.
    
    Loads pre-computed 3D morphable model and converts model predictions
    to 3D vertices and 2D projected landmarks.
    """
    
    # Default paths for morphable model files (moved to data/3dmm)
    DEFAULT_MODEL_SHAPE_PATH = 'data/3dmm/Model_Shape.mat'
    DEFAULT_MODEL_EXPRESSION_PATH = 'data/3dmm/Model_Expression.mat'
    DEFAULT_SIGMA_EXP_PATH = 'data/3dmm/sigma_exp.mat'
    
    def __init__(
        self,
        model_shape_path: str = DEFAULT_MODEL_SHAPE_PATH,
        model_expression_path: str = DEFAULT_MODEL_EXPRESSION_PATH,
        sigma_exp_path: str = DEFAULT_SIGMA_EXP_PATH,
    ):
        """
        Initialize shape reconstructor with morphable model files.
        
        Args:
            model_shape_path: Path to Model_Shape.mat
            model_expression_path: Path to Model_Expression.mat
            sigma_exp_path: Path to sigma_exp.mat
        """
        # Load morphable model files
        self.model_shape = io.loadmat(model_shape_path)
        self.model_exp = io.loadmat(model_expression_path)
        self.sigma_exp = io.loadmat(sigma_exp_path)
        
        # Keypoint indices for 68 facial landmarks
        self.kpt_index = np.reshape(
            self.model_shape['keypoints'],
            68
        ).astype(np.int32) - 1  # Convert to 0-based indexing
        
        # Normalization parameters for pose
        self.pose_mean = np.array(
            [0, 0, 0, 112, 112, 0, 0],
            dtype=np.float32
        )
        self.pose_std = np.array([
            math.pi / 2.0,
            math.pi / 2.0,
            math.pi / 2.0,
            56,
            56,
            1,
            224.0 / (2 * 180000.0)
        ], dtype=np.float32)
    
    @staticmethod
    def _angle_to_rotation(angles: np.ndarray) -> np.ndarray:
        """
        Convert rotation angles to rotation matrix.
        
        Args:
            angles: [phi, gamma, theta] rotation angles
        
        Returns:
            R: (3, 3) rotation matrix
        """
        phi = angles[0]
        gamma = angles[1]
        theta = angles[2]
        
        # X-axis rotation
        R_x = np.eye(3)
        R_x[1, 1] = math.cos(phi)
        R_x[1, 2] = math.sin(phi)
        R_x[2, 1] = -math.sin(phi)
        R_x[2, 2] = math.cos(phi)
        
        # Y-axis rotation
        R_y = np.eye(3)
        R_y[0, 0] = math.cos(gamma)
        R_y[0, 2] = -math.sin(gamma)
        R_y[2, 0] = math.sin(gamma)
        R_y[2, 2] = math.cos(gamma)
        
        # Z-axis rotation
        R_z = np.eye(3)
        R_z[0, 0] = math.cos(theta)
        R_z[0, 1] = math.sin(theta)
        R_z[1, 0] = -math.sin(theta)
        R_z[1, 1] = math.cos(theta)
        
        return np.matmul(np.matmul(R_x, R_y), R_z)
    
    def _preds_to_pose(
        self,
        preds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract pose parameters from predictions.
        
        Args:
            preds: (7,) pose predictions (normalized)
        
        Returns:
            R: (3, 3) rotation matrix
            t: (2,) 2D translation
            s: float scale
        """
        pose = preds * self.pose_std + self.pose_mean
        R = self._angle_to_rotation(pose[:3])
        t2d = pose[3:5]
        s = pose[6]
        return R, t2d, s
    
    def reconstruct(
        self,
        preds: np.ndarray,
        image_size: int = 224,
    ) -> Dict[str, np.ndarray]:
        """
        Reconstruct 3D face from model predictions.
        
        Args:
            preds: (249,) model predictions
                   [shape(199) + exp(29) + pose_front(7) + pose_left(7) + pose_right(7)]
            image_size: Size of input images (for landmark coordinate system)
        
        Returns:
            Dict with keys:
                - vertices: (N, 3) 3D face vertices
                - faces: (M, 3) face connectivity
                - keypoints_front: (68, 2) front view landmarks
                - keypoints_left: (68, 2) left view landmarks
                - keypoints_right: (68, 2) right view landmarks
        """
        # Extract shape and expression parameters
        alpha = np.reshape(preds[:199], [199, 1]) * np.reshape(
            self.model_shape['sigma'],
            [199, 1]
        )
        beta = np.reshape(preds[199:228], [29, 1]) * 1.0 / (
            1000.0 * np.reshape(self.sigma_exp['sigma_exp'], [29, 1])
        )
        
        # Reconstruct 3D face
        face_shape = (
            np.matmul(self.model_shape['w'], alpha) +
            np.matmul(self.model_exp['w_exp'], beta) +
            self.model_shape['mu_shape']
        )
        face_shape = face_shape.reshape(-1, 3)
        
        # Extract poses and project landmarks
        R_front, t_front, s_front = self._preds_to_pose(preds[228:228 + 7])
        keypoints_front = np.matmul(
            face_shape[self.kpt_index],
            s_front * R_front[:2].transpose()
        ) + np.repeat(np.reshape(t_front, [1, 2]), 68, axis=0)
        keypoints_front[:, 1] = image_size - keypoints_front[:, 1]
        
        R_left, t_left, s_left = self._preds_to_pose(preds[228 + 7:228 + 14])
        keypoints_left = np.matmul(
            face_shape[self.kpt_index],
            s_left * R_left[:2].transpose()
        ) + np.repeat(np.reshape(t_left, [1, 2]), 68, axis=0)
        keypoints_left[:, 1] = image_size - keypoints_left[:, 1]
        
        R_right, t_right, s_right = self._preds_to_pose(preds[228 + 14:])
        keypoints_right = np.matmul(
            face_shape[self.kpt_index],
            s_right * R_right[:2].transpose()
        ) + np.repeat(np.reshape(t_right, [1, 2]), 68, axis=0)
        keypoints_right[:, 1] = image_size - keypoints_right[:, 1]
        
        # Get face connectivity
        faces = self.model_shape['tri'].astype(np.int64).transpose() - 1
        
        return {
            'vertices': face_shape,
            'faces': faces,
            'keypoints_front': keypoints_front,
            'keypoints_left': keypoints_left,
            'keypoints_right': keypoints_right,
        }
