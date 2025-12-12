# Multi-View Face 3D Reconstruction

A developing implementation for reconstructing 3D face geometry from multiple 2D images using deep learning and morphable face models.

## Quick Start

### 1. Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Data

Download required files and place them in the specified folders:

- **3D Morphable Model files** (`Model_Shape.mat`, `Model_Expression.mat`, `sigma_exp.mat`) 
  - Source: [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
  - Destination: `data/3dmm/`

- **Model checkpoint** (`net.pth` - VGG-based model)
  - Source: [Dropbox](https://www.dropbox.com/s/7ds3aesjjmybjh9/net.pth?dl=0)
  - Destination: `data/weights/`

### 3. Run Inference

Place three-view images (`front.jpg`, `left.jpg`, `right.jpg`) in a folder, then run:

```bash
python inference.py --image_path ./data/imgs --save_dir ./result
```

Output: `result/shape_result.ply` (3D face mesh in PLY format)

**Optional parameters:**
```bash
# Different checkpoint
--checkpoint ./data/weights/resnet50_mvfnet.pth

# GPU acceleration
--device cuda

# Automatic face cropping
--crop
```

## Structure

```
mvfnet/
├── models/                  # Neural network architectures
│   ├── vgg_encoder.py      # VGG16-BN encoder
│   └── resnet_encoder.py   # ResNet50 encoder
├── preprocessing/           # Image processing
│   └── face_detector.py    # Face detection and cropping
├── reconstruction/          # 3D shape reconstruction
│   ├── shape_reconstructor.py
│   └── ply_io.py
├── postprocessing/          # Optional, still in-the-work mesh improvements
│   ├── improvements.py
│   ├── metrics.py
│   ├── pipeline.py
│   └── utils.py
├── inference.py            # Main inference pipeline (CLI + API)
├── config.py               # Configuration
└── data/
    ├── weights/            # Model checkpoints
    ├── 3dmm/              # Morphable model files
    └── imgs/              # Input images
```

## Usage

### Command Line

```bash
python inference.py --image_path ./data/imgs --save_dir ./result
```

### Python API

```python
from inference import MVFNetInference
from PIL import Image

# Initialize model
model = MVFNetInference(
    checkpoint_path="./data/weights/net.pth",
    model_type="VggEncoder",
    device="cpu"
)

# Load images
front = Image.open("./data/imgs/front.jpg").convert('RGB')
left = Image.open("./data/imgs/left.jpg").convert('RGB')
right = Image.open("./data/imgs/right.jpg").convert('RGB')

# Run inference
result = model.inference(front, left, right)

# Access 3D data
vertices = result["vertices"]       # (N, 3) vertex positions
faces = result["faces"]             # (M, 3) face indices
keypoints = result["keypoints_front"]  # (68, 2) landmarks
```

### Saving Results

```python
from reconstruction import write_ply

write_ply("output.ply", vertices=result["vertices"], mesh=result["faces"])
```

## Technical Details

**Input:** Three-view RGB images (224×224 each)

**Output:** 249-dimensional parameter vector decomposed as:
- Shape parameters: 199 dimensions
- Expression parameters: 29 dimensions
- Per-view pose (3 views × 7 params): 21 dimensions

**Model variants:**
- `VggEncoder`: Uses VGG16-BN backbone, lighter, faster
- `ResNetEncoder`: Uses ResNet50 backbone, potentially more accurate

## Notes

- Output meshes are saved in ASCII PLY format by default
- All model weights are in `data/weights/`; all 3DMM data in `data/3dmm/`
- GPU support requires a CUDA-enabled PyTorch installation
- Face detection uses the `face-alignment` library (included in requirements)

## References

This is a refactored and modularized implementation of MVF-Net:

> **MVF-Net: Multi-View 3D Face Morphable Model Regression**  
> Fanzi Wu, Linchao Bao, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, Wei Liu  
> CVPR 2019  
> [![Paper](https://img.shields.io/badge/CVPR-2019-blue)](https://arxiv.org/abs/1904.04473)

**Original repository:** [fwu11/mvf-net](https://github.com/Fanziapril/mvfnet)

