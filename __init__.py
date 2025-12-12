"""
MVF-Net package.

Multi-View 3D Face Morphable Model Regression.
Original paper: MVF-Net: Multi-View 3D Face Morphable Model Regression (CVPR 2019)

Authors: Fanzi Wu, Linchao Bao, Yajing Chen, Yonggen Ling, Yibing Song, 
         Songnan Li, King Ngi Ngan, Wei Liu

This package provides inference capabilities and preprocessing utilities
for the MVF-Net model architecture.

Modules:
    - inference: Model loading and inference pipeline
"""

from .inference import MVFNetInference, run_inference

__all__ = [
    "MVFNetInference",
    "run_inference",
]

__version__ = "1.0.0"
