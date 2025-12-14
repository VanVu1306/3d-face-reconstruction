# """
# Configuration settings for MVF-Net and post-processing.
# Centralized configuration management for paths, hyperparameters, and model settings.
# """

# import os
# from pathlib import Path

# # ============================================================================
# # PATHS CONFIGURATION
# # ============================================================================

# # Project root
# PROJECT_ROOT = Path(__file__).parent.absolute()

# # Data directories
# DATA_DIR = PROJECT_ROOT / "data"
# RESULT_DIR = PROJECT_ROOT / "result"
# IMG_DIR = DATA_DIR / "imgs"

# # Model checkpoint paths
# MODEL_CHECKPOINT = DATA_DIR / "weights" / "net.pth"
# RESNET50_WEIGHTS = DATA_DIR / "weights" / "resnet50-11ad3fa6.pth"

# # 3D Morphable Model files (bundled by the author)
# MODEL_SHAPE_PATH = DATA_DIR / "3dmm" / "Model_Shape.mat"
# MODEL_EXPRESSION_PATH = DATA_DIR / "3dmm" / "Model_Expression.mat"
# SIGMA_EXP_PATH = DATA_DIR / "3dmm" / "sigma_exp.mat"

# # ============================================================================
# # MODEL CONFIGURATION
# # ============================================================================

# # MVF-Net Model
# MODEL_CONFIG = {
#     "architecture": "VggEncoder",  # or "ResNetEncoder"
#     "input_shape": (1, 9, 224, 224),  # (batch, 3_views * 3_channels, H, W)
#     "output_shape": (1, 249),  # [shape(199) + exp(29) + pose_front(7) + pose_left(7) + pose_right(7)]
# }

# # ResNet50 backbone (if using ResNetEncoder)
# RESNET_CONFIG = {
#     "use_pretrained": False,
#     "weights_path": str(RESNET50_WEIGHTS),
# }

# # Image processing
# IMAGE_CONFIG = {
#     "resolution": 224,
#     "crop": True,  # Crop face using face alignment
#     "device": "cpu",  # "cpu" or "cuda"
# }

# # ============================================================================
# # POST-PROCESSING CONFIGURATION
# # ============================================================================

# # Bilateral filtering
# BILATERAL_CONFIG = {
#     "enabled": True,
#     "iterations": 5,
#     "sigma_spatial": 0.1,
#     "sigma_range": 0.1,
# }

# # Adaptive Laplacian smoothing
# SMOOTHING_CONFIG = {
#     "enabled": True,
#     "iterations": 3,
#     "lambda_smooth": 0.5,
# }

# # Self-intersection removal
# INTERSECTION_CONFIG = {
#     "enabled": True,
#     "iterations": 3,
#     "threshold": 0.005,
#     "chunk_size": 300,
# }

# # Normal enhancement from image
# NORMAL_ENHANCEMENT_CONFIG = {
#     "enabled": False,  # Requires camera parameters
#     "enhancement_scale": 0.002,
# }

# # Overall pipeline
# PIPELINE_CONFIG = {
#     "enable_bilateral": BILATERAL_CONFIG["enabled"],
#     "enable_smoothing": SMOOTHING_CONFIG["enabled"],
#     "enable_normal_enhancement": NORMAL_ENHANCEMENT_CONFIG["enabled"],
#     "bilateral_iterations": BILATERAL_CONFIG["iterations"],
#     "smoothing_iterations": SMOOTHING_CONFIG["iterations"],
#     "output_format": "ply",
#     "output_as_text": True,  # ASCII PLY format (easier to debug)
# }

# # ============================================================================
# # QUALITY METRICS CONFIGURATION
# # ============================================================================

# METRICS_CONFIG = {
#     "compute_edge_uniformity": True,
#     "compute_triangle_quality": True,
#     "compute_laplacian_smoothness": True,
#     "compute_volume": True,
#     "compute_self_intersections": True,
# }

# # ============================================================================
# # INFERENCE CONFIGURATION
# # ============================================================================

# INFERENCE_CONFIG = {
#     "batch_size": 1,
#     "device": IMAGE_CONFIG["device"],
#     "deterministic": True,
# }

# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def ensure_dirs_exist():
#     """Create necessary directories if they don't exist."""
#     for directory in [DATA_DIR, RESULT_DIR, IMG_DIR]:
#         directory.mkdir(parents=True, exist_ok=True)


# def get_config_dict():
#     """Return all configuration as a dictionary."""
#     return {
#         "model": MODEL_CONFIG,
#         "image": IMAGE_CONFIG,
#         "bilateral": BILATERAL_CONFIG,
#         "smoothing": SMOOTHING_CONFIG,
#         "intersection": INTERSECTION_CONFIG,
#         "pipeline": PIPELINE_CONFIG,
#         "metrics": METRICS_CONFIG,
#         "inference": INFERENCE_CONFIG,
#     }


# if __name__ == "__main__":
#     # Print configuration
#     import json
#     config = get_config_dict()
#     print(json.dumps(config, indent=2, default=str))
