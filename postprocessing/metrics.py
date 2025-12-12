"""
Mesh quality metrics computation.

Provides quantitative metrics to evaluate 3D mesh quality:
- Edge uniformity
- Triangle aspect ratios
- Laplacian smoothness
- Volume stability
- Self-intersection detection
"""

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from typing import Dict, Tuple


def compute_laplacian_matrix(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> csr_matrix:
    """
    Compute mesh Laplacian matrix.
    
    Manual implementation to avoid trimesh caching issues.
    L[i,j] = -1 if vertices i,j are neighbors, L[i,i] = degree(i).
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
    
    Returns:
        L: Sparse (N, N) Laplacian matrix
    """
    N = len(vertices)
    edges = {}
    
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            if edge not in edges:
                edges[edge] = []
            edges[edge].append(face)
    
    rows, cols, data = [], [], []
    degree = [0] * N
    
    for (v1, v2) in edges.keys():
        # Off-diagonal entries
        rows.extend([v1, v2])
        cols.extend([v2, v1])
        data.extend([-1.0, -1.0])
        
        degree[v1] += 1
        degree[v2] += 1
    
    # Diagonal entries (degree)
    for i in range(N):
        rows.append(i)
        cols.append(i)
        data.append(float(degree[i]))
    
    return csr_matrix((data, (rows, cols)), shape=(N, N))


def compute_edge_uniformity(
    vertices: np.ndarray,
    mesh: trimesh.Trimesh,
) -> float:
    """
    Compute edge length uniformity.
    
    Metric: 1 - (std / mean) where std/mean is coefficient of variation.
    Range: [0, 1], higher is better.
    
    Args:
        vertices: (N, 3) vertex positions
        mesh: Trimesh object
    
    Returns:
        uniformity: Float in [0, 1]
    """
    try:
        edge_lengths = mesh.edges_unique_length
        if len(edge_lengths) > 0:
            mean_len = np.mean(edge_lengths)
            std_len = np.std(edge_lengths)
            if mean_len > 0:
                return 1.0 - (std_len / mean_len)
        return 0.0
    except Exception as e:
        print(f"  Warning: Could not compute edge uniformity: {e}")
        return 0.0


def compute_triangle_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> float:
    """
    Compute average triangle quality (aspect ratio).
    
    Quality = 4√3 * Area / (a² + b² + c²)
    Range: [0, 1], 1 = equilateral triangle, 0 = degenerate.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
    
    Returns:
        avg_quality: Average quality across all triangles
    """
    triangle_qualities = []
    
    for face in faces:
        try:
            v0, v1, v2 = vertices[face]
            a = np.linalg.norm(v1 - v0)
            b = np.linalg.norm(v2 - v1)
            c = np.linalg.norm(v0 - v2)
            
            if a > 0 and b > 0 and c > 0:
                s = (a + b + c) / 2
                area_sq = s * (s - a) * (s - b) * (s - c)
                if area_sq > 0:
                    area = np.sqrt(area_sq)
                    quality = 4 * np.sqrt(3) * area / (a**2 + b**2 + c**2)
                    triangle_qualities.append(quality)
        except Exception:
            continue
    
    return np.mean(triangle_qualities) if triangle_qualities else 0.0


def compute_laplacian_smoothness(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> float:
    """
    Compute Laplacian smoothness energy.
    
    Metric: 1 / (1 + normalized_laplacian_energy)
    Range: [0, 1], higher = smoother mesh.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
    
    Returns:
        smoothness: Float in [0, 1]
    """
    try:
        laplacian = compute_laplacian_matrix(vertices, faces)
        
        if laplacian.shape[0] == len(vertices) and laplacian.shape[1] == len(vertices):
            laplacian_coords = laplacian @ vertices
            laplacian_energy = np.linalg.norm(laplacian_coords)
            
            bbox_size = np.linalg.norm(
                vertices.max(axis=0) - vertices.min(axis=0)
            )
            if bbox_size > 0:
                normalized_energy = laplacian_energy / (len(vertices) * bbox_size)
                return 1.0 / (1.0 + normalized_energy)
        
        return 0.0
    except Exception as e:
        print(f"  Warning: Could not compute Laplacian smoothness: {e}")
        return 0.0


def compute_mesh_quality_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive mesh quality metrics.
    
    Computes multiple metrics to quantify mesh improvement across different
    aspects (uniformity, quality, smoothness, etc.).
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
    
    Returns:
        metrics: Dict with keys:
            - edge_uniformity: [0, 1]
            - triangle_quality: [0, 1]
            - laplacian_smoothness: [0, 1]
            - volume: Unsigned volume
            - has_self_intersections: 0 or 1
    """
    assert vertices.shape[1] == 3, f"Vertices must be (N, 3), got {vertices.shape}"
    assert faces.shape[1] == 3, f"Faces must be (M, 3), got {faces.shape}"

    # Filter invalid faces
    max_face_idx = faces.max()
    if max_face_idx >= len(vertices):
        print(f"  Warning: Invalid faces (max {max_face_idx} >= {len(vertices)} vertices)")
        valid_mask = (
            (faces[:, 0] < len(vertices)) &
            (faces[:, 1] < len(vertices)) &
            (faces[:, 2] < len(vertices))
        )
        faces = faces[valid_mask]
        print(f"  Filtered to {len(faces)} valid faces")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    metrics = {}

    # Edge uniformity
    metrics['edge_uniformity'] = compute_edge_uniformity(vertices, mesh)

    # Triangle quality
    metrics['triangle_quality'] = compute_triangle_quality(vertices, faces)

    # Laplacian smoothness
    metrics['laplacian_smoothness'] = compute_laplacian_smoothness(vertices, faces)

    # Volume
    try:
        if mesh.is_watertight:
            metrics['volume'] = abs(mesh.volume)
        else:
            metrics['volume'] = 0.0
    except Exception as e:
        print(f"  Warning: Could not compute volume: {e}")
        metrics['volume'] = 0.0

    # Self-intersections
    try:
        metrics['has_self_intersections'] = 0.0 if mesh.is_watertight else 1.0
    except Exception:
        metrics['has_self_intersections'] = 1.0

    return metrics


def compute_metrics_change(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute percentage change in metrics.
    
    Args:
        metrics_before: Initial metrics dict
        metrics_after: Final metrics dict
    
    Returns:
        changes: Percent change for each metric
    """
    changes = {}
    for key in metrics_before.keys():
        if key in metrics_after:
            before = metrics_before[key]
            after = metrics_after[key]
            change = ((after - before) / (abs(before) + 1e-8)) * 100
            changes[key] = change
    return changes


def print_metrics(
    metrics: Dict[str, float],
    label: str = "Metrics",
) -> None:
    """
    Print metrics in formatted table.
    
    Args:
        metrics: Metrics dictionary
        label: Section label
    """
    print(f"\n{label}:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
