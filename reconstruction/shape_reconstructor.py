# reconstruction/shape_reconstructor.py
import os
import math
import numpy as np
import scipy.io as io
from PIL import Image

from postprocessing.improvements import enhance_front_colors

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
_DATA_3DMM = os.path.join(_BASE_DIR, "data", "3dmm")

print("Loading 3DMM models...")

model_loaded = False
for shape_path in [
    os.path.join(_DATA_3DMM, "Model_Shape.mat"),
    os.path.join(_BASE_DIR, "Model_Shape.mat"),
]:
    if os.path.exists(shape_path):
        try:
            model_shape = io.loadmat(shape_path)
            print(f"✓ Model_Shape.mat loaded from: {shape_path}")
            model_loaded = True
            break
        except Exception:
            continue

if not model_loaded:
    raise FileNotFoundError("Cannot load Model_Shape.mat")

model_exp = io.loadmat(os.path.join(_DATA_3DMM, "Model_Expression.mat"))
data = io.loadmat(os.path.join(_DATA_3DMM, "sigma_exp.mat"))

kpt_index = np.reshape(model_shape['keypoints'], 68).astype(np.int32) - 1
pose_mean = np.array([0, 0, 0, 112, 112, 0, 0]).astype(np.float32)
pose_std  = np.array([math.pi/2.0, math.pi/2.0, math.pi/2.0, 56, 56, 1, 224.0 / (2 * 180000.0)]).astype(np.float32)

def angle_to_rotation(angles):
    phi, gamma, theta = angles[0], angles[1], angles[2]

    R_x = np.eye(3)
    R_x[1, 1] = math.cos(phi)
    R_x[1, 2] = math.sin(phi)
    R_x[2, 1] = -math.sin(phi)
    R_x[2, 2] = math.cos(phi)

    R_y = np.eye(3)
    R_y[0, 0] = math.cos(gamma)
    R_y[0, 2] = -math.sin(gamma)
    R_y[2, 0] = math.sin(gamma)
    R_y[2, 2] = math.cos(gamma)

    R_z = np.eye(3)
    R_z[0, 0] = math.cos(theta)
    R_z[0, 1] = math.sin(theta)
    R_z[1, 0] = -math.sin(theta)
    R_z[1, 1] = math.cos(theta)

    return np.matmul(np.matmul(R_x, R_y), R_z)

def preds_to_pose(preds):
    pose = preds * pose_std + pose_mean
    R = angle_to_rotation(pose[:3])
    t2d = pose[3:5]
    s = pose[6]
    return R, t2d, s

def preds_to_shape(preds):
    alpha = np.reshape(preds[:199], [199, 1]) * np.reshape(model_shape['sigma'], [199, 1])
    beta  = np.reshape(preds[199:228], [29, 1]) * 1.0/(1000.0 * np.reshape(data['sigma_exp'], [29, 1]))

    face_shape = np.matmul(model_shape['w'], alpha) + np.matmul(model_exp['w_exp'], beta) + model_shape['mu_shape']
    face_shape = face_shape.reshape(-1, 3)

    R, t, s = preds_to_pose(preds[228:228+7])

    kptA = np.matmul(face_shape[kpt_index], s*R[:2].transpose()) + np.repeat(np.reshape(t, [1, 2]), 68, axis=0)
    kptA[:, 1] = 224 - kptA[:, 1]

    triangles = model_shape['tri'].astype(np.int64).transpose() - 1
    return [face_shape, triangles, kptA]

def get_front_texture_colors(vertex_3d, pose_pred, original_image):
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    else:
        img_np = original_image

    h, w = img_np.shape[:2]

    R, t2d, s = preds_to_pose(pose_pred)

    vertex_2d = np.matmul(vertex_3d, s * R[:2].T) + np.tile(t2d, (vertex_3d.shape[0], 1))
    vertex_2d[:, 1] = h - vertex_2d[:, 1]

    vertex_2d[:, 0] = np.clip(vertex_2d[:, 0], 0, w - 1)
    vertex_2d[:, 1] = np.clip(vertex_2d[:, 1], 0, h - 1)

    colors = []
    for pt in vertex_2d:
        x, y = pt[0], pt[1]
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)

        if 0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h:
            dx, dy = x - x0, y - y0
            c00 = img_np[y0, x0]
            c10 = img_np[y0, x1]
            c01 = img_np[y1, x0]
            c11 = img_np[y1, x1]

            color = (c00 * (1-dx) * (1-dy) +
                    c10 * dx * (1-dy) +
                    c01 * (1-dx) * dy +
                    c11 * dx * dy)
            colors.append(color)
        else:
            colors.append(np.array([0, 0, 0]))

    colors = np.array(colors).astype(np.uint8)
    colors = enhance_front_colors(colors)  
    return colors
