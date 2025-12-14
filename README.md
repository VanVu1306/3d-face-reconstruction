# Multi-View Face 3D Reconstruction

A developing implementation for reconstructing 3D face geometry from multiple 2D images using deep learning and morphable face models.

## Quick Start

### 1. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download Required Data

Download required files and place them in the specified folders:

- **3D Morphable Model files** (`Model_Shape.mat`, `Model_Expression.mat`, `sigma_exp.mat`)

  - Source: [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
  - Destination: `data/weights/`

- **Model checkpoint** (`net.pth` - VGG-based model)
  - Source: [Dropbox](https://www.dropbox.com/s/7ds3aesjjmybjh9/net.pth?dl=0)
  - Destination: `data/weights/`

### 3. Run Inference

Place three-view images (`front.jpg`, `left.jpg`, `right.jpg`) in a folder, then run:

```bash
python inference.py --imgs .\data\imgs --save_dir .\result
```

Output: `result/shape_texture.ply`

## Structure

```
mvfnet/
├── models/                  # Neural network backbones
│   ├── vgg_encoder.py       # VGG16-BN encoder (default)
│   └── resnet_encoder.py    # ResNet50 encoder (optional)
├── preprocessing/           # (Reserved) image preprocessing
│   └── face_detector.py
├── reconstruction/          # Geometry & texture reconstruction
│   ├── shape_reconstructor.py   # 3DMM decoding + projection
│   └── ply_io.py                # PLY mesh writer
├── postprocessing/          # Mesh & texture postprocessing
│   └── improvements.py      # Texture enhancement (CLAHE, smoothing)
├── inference.py             # Main inference entry point (CLI)
├── config.py                # Configuration (optional)
└── data/
    ├── weights/             # Model checkpoints
    ├── 3dmm/                # 3D Morphable Model assets
    └── imgs/                # Input images

```

## Usage

### Command Line

```bash
python inference.py --imgs .\data\imgs --save_dir .\result
```

### Optional flag

```bash
--enable_enhance # enable texture postprocessing (default behavior)
```

### Pipeline Overview

1. Multi-view input

- Three RGB images (front / left / right)
- Resized to 224 × 224

2. Deep regression

- CNN encoder (VGG16-BN by default)
- Predicts a 249-D parameter vector

3. 3DMM decoding

- Shape coefficients (199)
- Expression coefficients (29)
- Pose parameters (3 views × 7)

4. Geometry reconstruction

- Dense 3D face mesh
- Fixed topology from 3DMM

5. Texture projection

- Front-view color sampling
- Bilinear interpolation
- Optional enhancement (CLAHE + smoothing)

6. Mesh export

- ASCII PLY with per-vertex RGB colors

## Technical Details

**Input:** Three-view RGB images (224×224)

**Output:** Textured 3D face mesh (.ply)

**Predicted parameter layout (249D):**

- Shape parameters: 199 dimensions
- Expression parameters: 29 dimensions
- Per-view pose (3 views × 7 params): 21 dimensions

## Notes

- Meshes are saved in ASCII PLY format by default.
- All model weights are in `data/weights/`; all 3DMM data in `data/3dmm/`
- GPU support requires a CUDA-enabled PyTorch installation

## References

This is a refactored and modularized implementation of MVF-Net:

> **MVF-Net: Multi-View 3D Face Morphable Model Regression**  
> Fanzi Wu, Linchao Bao, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, Wei Liu  
> CVPR 2019  
> [![Paper](https://img.shields.io/badge/CVPR-2019-blue)](https://arxiv.org/abs/1904.04473)

**Original repository:** [fwu11/mvf-net](https://github.com/Fanziapril/mvfnet)
