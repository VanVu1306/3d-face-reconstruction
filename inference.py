# inference.py
import torch
import argparse
import os
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models import VggEncoder
from reconstruction.shape_reconstructor import preds_to_shape, get_front_texture_colors
from reconstruction.ply_io import write_textured_ply

def main():
    parser = argparse.ArgumentParser(description='3D Face Reconstruction (Front Texture Only)')
    parser.add_argument('--imgs', type=str, default='./data/imgs',
                        help='path containing front.jpg, left.jpg, right.jpg')
    parser.add_argument('--save_dir', type=str, default='./result',
                        help='directory to save output')
    parser.add_argument('--enable_enhance', action='store_true',
                        help='enable color enhancement (handled in postprocessing)')
    options = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    imgs_dir = options.imgs
    if not os.path.isabs(imgs_dir):
        imgs_dir = os.path.join(BASE_DIR, imgs_dir)

    save_dir = options.save_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(BASE_DIR, save_dir)

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 50)
    print("3D FACE - FRONT TEXTURE ONLY")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("\n1. Loading images...")
    required_images = ['front.jpg', 'left.jpg', 'right.jpg']
    image_paths = [os.path.join(imgs_dir, img) for img in required_images]

    for i, path in enumerate(image_paths):
        if os.path.exists(path):
            print(f"Found: {required_images[i]}")
        else:
            print(f"Missing: {required_images[i]}")
            print(f"Path: {path}")
            raise FileNotFoundError(path)

    img_front_orig = Image.open(image_paths[0]).convert('RGB')
    img_left_orig  = Image.open(image_paths[1]).convert('RGB')
    img_right_orig = Image.open(image_paths[2]).convert('RGB')

    print(f"  Image sizes: {img_front_orig.size}")

    print("\n2. Loading model...")
    model = VggEncoder()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    model_paths = [
        os.path.join(BASE_DIR, 'data/weights', 'net.pth'),  
        os.path.join(BASE_DIR, 'net.pth'),
    ]

    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"  Loading from: {model_path}")
            ckpt = torch.load(model_path, map_location=device)
            sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=True)
            model.eval()
            model_loaded = True
            print("  ✓ Model loaded")
            break

    if not model_loaded:
        raise FileNotFoundError("Model file not found in: " + str(model_paths))

    print("\n3. Processing images...")
    img_front = img_front_orig.resize((224, 224), Image.BICUBIC)
    img_left  = img_left_orig.resize((224, 224), Image.BICUBIC)
    img_right = img_right_orig.resize((224, 224), Image.BICUBIC)

    img_front_tensor = transforms.functional.to_tensor(img_front)
    img_left_tensor  = transforms.functional.to_tensor(img_left)
    img_right_tensor = transforms.functional.to_tensor(img_right)

    print("\n4. Running 3D reconstruction...")
    input_tensor = torch.cat([img_front_tensor, img_left_tensor, img_right_tensor], 0)
    input_tensor = input_tensor.view(1, 9, 224, 224).to(device)

    start = time.time()
    with torch.no_grad():
        preds = model(input_tensor)
    inference_time = time.time() - start
    print(f"  ✓ Completed in {inference_time:.3f}s")

    print("\n5. Creating 3D mesh...")
    preds_np = preds[0].detach().cpu().numpy()

    face_shape, triangles, _kptA = preds_to_shape(preds_np)
    vertices = face_shape

    print(f" Mesh created: {len(vertices)} vertices, {len(triangles)} faces")

    print("\n6. Generating texture (front view only)...")
    pose_A = preds_np[228:228+7]

    start_tex = time.time()
    vertex_colors = get_front_texture_colors(vertices, pose_A, img_front_orig)
    tex_time = time.time() - start_tex
    print(f"  ✓ Texture generated in {tex_time:.3f}s")

    print("\n7. Saving final model...")
    output_file = os.path.join(save_dir, 'shape_texture.ply')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write_textured_ply(output_file, vertices, triangles, vertex_colors)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / 1024
        print(f"Output file: {output_file}")
        print(f"File size: {file_size:.1f} KB")

        print(f"\nModel info:")
        print(f"  Vertices: {len(vertices):,}")
        print(f"  Faces: {len(triangles):,}")

        color_min = np.min(vertex_colors)
        color_max = np.max(vertex_colors)
        color_mean = np.mean(vertex_colors)

        print(f"\nColor info:")
        print(f"  Range: {color_min}-{color_max}")
        print(f"  Mean: {color_mean:.1f}")

        black_count = np.sum(np.all(vertex_colors == [0, 0, 0], axis=1))
        print(f"   Black vertices: {black_count}" if black_count > 0 else "   No black vertices")

        print(f"\nPerformance:")
        print(f"  Total time: {inference_time + tex_time:.2f}s")
        print(f"  - 3D reconstruction: {inference_time:.2f}s")
        print(f"  - Texture generation: {tex_time:.2f}s")

        print("\n" + "=" * 50)
        print("Successfully created shape_texture.ply")
        print("=" * 50)
    else:
        print(f"ERROR: File not created: {output_file}")

if __name__ == "__main__":
    main()
