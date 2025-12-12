"""
Post-processing utility functions.

Helper functions for mesh consistency validation, visualization, and I/O.
"""

import os
import numpy as np
import trimesh
import open3d as o3d
from typing import Tuple


def verify_mesh_consistency(
    vertices_before: np.ndarray,
    faces_before: np.ndarray,
    vertices_after: np.ndarray,
    faces_after: np.ndarray,
    step_name: str = "",
) -> bool:
    """
    Verify mesh dimensions remain consistent.
    
    Debug function to catch dimension changes between processing steps.
    Checks vertex/face counts and face index validity.
    
    Args:
        vertices_before: Previous vertex array
        faces_before: Previous face array
        vertices_after: Current vertex array
        faces_after: Current face array
        step_name: Label for this step (for logging)
    
    Returns:
        is_consistent: True if mesh is valid, False otherwise
    """
    print(f"\n[{step_name}] Mesh consistency check:")
    print(f"  Vertices: {len(vertices_before)} → {len(vertices_after)}")
    print(f"  Faces: {len(faces_before)} → {len(faces_after)}")
    
    if len(vertices_before) != len(vertices_after):
        print(f"  ⚠️  WARNING: Vertex count changed!")
    
    if len(faces_before) != len(faces_after):
        print(f"  ⚠️  WARNING: Face count changed!")
    
    # Check for invalid face indices
    if len(faces_after) > 0:
        max_idx = faces_after.max()
        if max_idx >= len(vertices_after):
            print(
                f"  ⚠️  ERROR: Invalid faces "
                f"(max index {max_idx} >= {len(vertices_after)} vertices)"
            )
            return False
    
    print(f"  ✓ Mesh is consistent")
    return True


def visualize_mesh(
    mesh_path: str,
    title: str = "3D Mesh",
) -> None:
    """
    Visualize mesh using Open3D.
    
    Args:
        mesh_path: Path to PLY file
        title: Window title
    """
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        o3d.visualization.draw_geometries([mesh], window_name=title)
    except Exception as e:
        print(f"Error visualizing mesh: {e}")


def visualize_comparison(
    mesh_before_path: str,
    mesh_after_path: str,
    offset: float = 0.3,
) -> None:
    """
    Visualize original and improved meshes side-by-side.
    
    Args:
        mesh_before_path: Path to original mesh PLY
        mesh_after_path: Path to improved mesh PLY
        offset: Side-by-side offset distance
    """
    try:
        mesh_before = o3d.io.read_triangle_mesh(mesh_before_path)
        mesh_after = o3d.io.read_triangle_mesh(mesh_after_path)
        
        mesh_before.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
        mesh_after.paint_uniform_color([0.3, 0.7, 0.9])   # Blue
        
        mesh_after.translate([offset, 0, 0])
        
        o3d.visualization.draw_geometries(
            [mesh_before, mesh_after],
            window_name="Before (left) vs After (right)"
        )
    except Exception as e:
        print(f"Error visualizing comparison: {e}")


def load_mesh_ply(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from PLY file.
    
    Args:
        mesh_path: Path to PLY file
    
    Returns:
        vertices: (N, 3) vertex array
        faces: (M, 3) face array
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return vertices, faces


def save_mesh_ply(
    mesh_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    binary: bool = False,
) -> bool:
    """
    Save mesh to PLY file.
    
    Args:
        mesh_path: Output path
        vertices: (N, 3) vertex array
        faces: (M, 3) face array
        binary: Use binary format (default: ASCII)
    
    Returns:
        success: True if save succeeded
    """
    try:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces)
        )
        o3d.io.write_triangle_mesh(mesh_path, mesh, write_ascii=not binary)
        return True
    except Exception as e:
        print(f"Error saving mesh: {e}")
        return False


def ensure_output_dir(output_path: str) -> None:
    """Create output directory if it doesn't exist."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
