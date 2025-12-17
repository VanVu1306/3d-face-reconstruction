"""
MVF-Net Evaluation Script
Evaluate model on MICC Florence dataset and visualize results
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import json
import argparse

from model_definition import create_mvfnet

# ============================================================================
# Evaluation Metrics
# ============================================================================

class MICCEvaluator:
    """Evaluator for MICC Florence dataset"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def load_ground_truth_scan(self, scan_path):
        """Load ground truth 3D scan"""
        # Load PLY or OBJ file containing ground truth scan
        # This is a placeholder - implement based on actual file format
        vertices = np.load(scan_path)  # Assuming .npy format
        return vertices
    
    def point_to_plane_distance(self, pred_vertices, gt_vertices):
        """
        Compute point-to-plane distance between predicted and GT meshes
        
        Args:
            pred_vertices: (N, 3) predicted vertices
            gt_vertices: (M, 3) ground truth vertices
            
        Returns:
            mean_error: scalar
            std_error: scalar
        """
        # For each predicted vertex, find nearest GT vertex
        # and compute distance to its plane
        
        from scipy.spatial import KDTree
        
        # Build KD-tree for GT vertices
        tree = KDTree(gt_vertices)
        
        errors = []
        for pred_v in pred_vertices:
            # Find k nearest neighbors
            distances, indices = tree.query(pred_v, k=3)
            
            # Estimate plane from 3 nearest points
            neighbors = gt_vertices[indices]
            
            # Compute plane normal using cross product
            v1 = neighbors[1] - neighbors[0]
            v2 = neighbors[2] - neighbors[0]
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            
            # Compute point-to-plane distance
            point_on_plane = neighbors[0]
            distance = abs(np.dot(pred_v - point_on_plane, normal))
            errors.append(distance)
        
        errors = np.array(errors)
        return np.mean(errors), np.std(errors)
    
    def evaluate_triplet(self, img_A, img_B, img_C, gt_scan_path):
        """Evaluate on single triplet"""
        with torch.no_grad():
            # Preprocess images
            img_A_tensor = self._preprocess(img_A).unsqueeze(0).to(self.device)
            img_B_tensor = self._preprocess(img_B).unsqueeze(0).to(self.device)
            img_C_tensor = self._preprocess(img_C).unsqueeze(0).to(self.device)
            
            # Forward pass
            pred = self.model(img_A_tensor, img_B_tensor, img_C_tensor)
            
            # Generate 3D mesh
            vertices = self.model.generate_mesh(
                pred['shape_params'], 
                pred['exp_params']
            )
            vertices = vertices.cpu().numpy()[0]  # (num_vertices, 3)
            
            # Load ground truth
            gt_vertices = self.load_ground_truth_scan(gt_scan_path)
            
            # Compute error
            mean_error, std_error = self.point_to_plane_distance(
                vertices, gt_vertices
            )
            
            return mean_error, std_error, vertices
    
    def _preprocess(self, img):
        """Preprocess image for model input"""
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img
    
    def evaluate_dataset(self, test_triplets):
        """Evaluate on entire dataset"""
        results = []
        
        print("Evaluating on MICC dataset...")
        for triplet in tqdm(test_triplets):
            # Load images
            img_A = cv2.imread(triplet['left'])
            img_B = cv2.imread(triplet['frontal'])
            img_C = cv2.imread(triplet['right'])
            
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
            img_C = cv2.cvtColor(img_C, cv2.COLOR_BGR2RGB)
            
            # Evaluate
            mean_error, std_error, vertices = self.evaluate_triplet(
                img_A, img_B, img_C, triplet['gt_scan']
            )
            
            results.append({
                'subject_id': triplet['subject_id'],
                'mean_error': mean_error,
                'std_error': std_error,
                'vertices': vertices
            })
        
        # Compute overall statistics
        mean_errors = [r['mean_error'] for r in results]
        std_errors = [r['std_error'] for r in results]
        
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"Mean Error: {np.mean(mean_errors):.4f} ± {np.std(mean_errors):.4f}")
        print(f"Std Error:  {np.mean(std_errors):.4f} ± {np.std(std_errors):.4f}")
        
        return results

# ============================================================================
# Visualization
# ============================================================================

class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def visualize_reconstruction(img, vertices, triangles, save_path=None):
        """Visualize 3D reconstruction"""
        fig = plt.figure(figsize=(15, 5))
        
        # Plot input image
        ax1 = fig.add_subplot(131)
        ax1.imshow(img)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # Plot 3D mesh (front view)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles,
            cmap='viridis',
            alpha=0.8
        )
        ax2.set_title('Reconstructed 3D Face (Front)')
        ax2.view_init(elev=0, azim=90)
        
        # Plot 3D mesh (side view)
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles,
            cmap='viridis',
            alpha=0.8
        )
        ax3.set_title('Reconstructed 3D Face (Side)')
        ax3.view_init(elev=0, azim=180)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_multi_view(img_A, img_B, img_C, vertices, triangles, save_path=None):
        """Visualize multi-view inputs and reconstruction"""
        fig = plt.figure(figsize=(20, 5))
        
        # Plot input images
        ax1 = fig.add_subplot(141)
        ax1.imshow(img_A)
        ax1.set_title('Left View')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(142)
        ax2.imshow(img_B)
        ax2.set_title('Frontal View')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(143)
        ax3.imshow(img_C)
        ax3.set_title('Right View')
        ax3.axis('off')
        
        # Plot 3D reconstruction
        ax4 = fig.add_subplot(144, projection='3d')
        ax4.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles,
            cmap='viridis',
            alpha=0.8
        )
        ax4.set_title('Reconstructed 3D Face')
        ax4.view_init(elev=10, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_error_map(vertices, errors, triangles, save_path=None):
        """Visualize error distribution on 3D mesh"""
        fig = plt.figure(figsize=(15, 5))
        
        # Front view
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles,
            cmap='jet',
            alpha=0.8
        )
        surf1.set_array(errors)
        ax1.set_title('Error Map (Front)')
        ax1.view_init(elev=0, azim=90)
        
        # Side view
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles,
            cmap='jet',
            alpha=0.8
        )
        surf2.set_array(errors)
        ax2.set_title('Error Map (Side)')
        ax2.view_init(elev=0, azim=180)
        
        # Top view
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles,
            cmap='jet',
            alpha=0.8
        )
        surf3.set_array(errors)
        ax3.set_title('Error Map (Top)')
        ax3.view_init(elev=90, azim=0)
        
        # Add colorbar
        fig.colorbar(surf3, ax=[ax1, ax2, ax3], shrink=0.5, label='Error (mm)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def export_obj(vertices, triangles, texture_coords=None, save_path='output.obj'):
        """Export mesh to OBJ file"""
        with open(save_path, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
            
            # Write texture coordinates if available
            if texture_coords is not None:
                for vt in texture_coords:
                    f.write(f'vt {vt[0]:.6f} {vt[1]:.6f}\n')
            
            # Write faces
            for tri in triangles:
                if texture_coords is not None:
                    f.write(f'f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}\n')
                else:
                    f.write(f'f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n')
        
        print(f"Mesh exported to {save_path}")

# ============================================================================
# Demo Inference
# ============================================================================

def demo_inference(model, img_A_path, img_B_path, img_C_path, output_dir='./results'):
    """Run inference on custom images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    img_C = cv2.imread(img_C_path)
    
    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
    img_C = cv2.cvtColor(img_C, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    def preprocess(img):
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img
    
    img_A_tensor = preprocess(img_A).unsqueeze(0).cuda()
    img_B_tensor = preprocess(img_B).unsqueeze(0).cuda()
    img_C_tensor = preprocess(img_C).unsqueeze(0).cuda()
    
    # Inference
    model.eval()
    with torch.no_grad():
        pred = model(img_A_tensor, img_B_tensor, img_C_tensor)
        
        # Generate 3D mesh
        vertices = model.generate_mesh(
            pred['shape_params'],
            pred['exp_params']
        )
        vertices = vertices.cpu().numpy()[0]
    
    # Get triangles from BFM model
    triangles = model.bfm_model.triangles
    
    # Visualize
    visualizer = Visualizer()
    visualizer.visualize_multi_view(
        img_A, img_B, img_C, vertices, triangles,
        save_path=os.path.join(output_dir, 'reconstruction.png')
    )
    
    # Export mesh
    visualizer.export_obj(
        vertices, triangles,
        save_path=os.path.join(output_dir, 'mesh.obj')
    )
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - reconstruction.png: Visualization")
    print(f"  - mesh.obj: 3D mesh file")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate MVF-Net')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['evaluate', 'demo'],
                        help='Evaluation mode')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data JSON (for evaluate mode)')
    parser.add_argument('--img_A', type=str, default=None,
                        help='Path to left view image (for demo mode)')
    parser.add_argument('--img_B', type=str, default=None,
                        help='Path to frontal view image (for demo mode)')
    parser.add_argument('--img_C', type=str, default=None,
                        help='Path to right view image (for demo mode)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_mvfnet()
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {args.checkpoint}")
    
    if args.mode == 'evaluate':
        # Load test data
        with open(args.test_data, 'r') as f:
            test_triplets = json.load(f)
        
        # Evaluate
        evaluator = MICCEvaluator(model, device)
        results = evaluator.evaluate_dataset(test_triplets)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
    elif args.mode == 'demo':
        # Run demo
        demo_inference(
            model,
            args.img_A,
            args.img_B,
            args.img_C,
            args.output_dir
        )

if __name__ == "__main__":
    main()
