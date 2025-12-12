"""
Post-processing mesh improvements module.

This module contains functions to enhance 3D face meshes output by MVF-Net
through various post-processing techniques:
- Laplacian smoothing with detail preservation
- Bilateral filtering for edge-preserving smoothing
- Self-intersection removal
- Normal enhancement from image gradients
- Multi-view consistency enforcement
"""

import numpy as np
import trimesh
import cv2
from scipy.sparse import csr_matrix
from typing import Tuple, List, Optional


def laplacian_smoothing_adaptive(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 5,
    lambda_smooth: float = 0.5,
) -> np.ndarray:
    """
    Smooth mesh while preserving high-frequency details.
    
    Reduces MVF-Net's typical over-smoothing artifacts by weighting smoothing
    strength inversely with local curvature (preserves edges, smooths flat areas).
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        iterations: Number of smoothing iterations
        lambda_smooth: Smoothing strength ∈ [0, 1]
    
    Returns:
        smoothed_vertices: (N, 3) improved vertices
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = mesh.vertex_normals
    neighbors = mesh.vertex_neighbors
    
    # Detect high-curvature areas (preserve these)
    curvature = np.zeros(len(vertices))
    for i, nbrs in enumerate(neighbors):
        if len(nbrs) > 0:
            # Normal variation indicates curvature
            normal_diff = np.linalg.norm(normals[nbrs] - normals[i], axis=1)
            curvature[i] = np.mean(normal_diff)
    
    # Normalize curvature to [0, 1]
    curvature = curvature / (curvature.max() + 1e-8)
    
    # Adaptive smoothing: smooth less in high-curvature areas
    smoothed = vertices.copy()
    for iteration in range(iterations):
        new_verts = smoothed.copy()
        for i, nbrs in enumerate(neighbors):
            if len(nbrs) > 0:
                # Weight by inverse curvature (smooth flat areas more)
                weight = lambda_smooth * (1.0 - curvature[i])
                neighbor_mean = np.mean(smoothed[nbrs], axis=0)
                new_verts[i] = (1 - weight) * smoothed[i] + weight * neighbor_mean
        smoothed = new_verts
    
    return smoothed


def bilateral_mesh_filtering(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 3,
    sigma_spatial: float = 0.1,
    sigma_range: float = 0.1,
) -> np.ndarray:
    """
    Apply bilateral filtering to mesh vertices.
    
    Smooths while preserving edges/details (superior to simple Laplacian).
    Key improvement: reduces noise while keeping features sharp.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        iterations: Number of filtering iterations
        sigma_spatial: Spatial weighting parameter
        sigma_range: Range weighting parameter (based on normals)
    
    Returns:
        filtered_vertices: (N, 3) improved vertices
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    neighbors = mesh.vertex_neighbors
    
    filtered_vertices = vertices.copy()
    
    for iteration in range(iterations):
        new_vertices = filtered_vertices.copy()
        
        for i, nbrs in enumerate(neighbors):
            if len(nbrs) == 0:
                continue
            
            v_i = filtered_vertices[i]
            v_neighbors = filtered_vertices[nbrs]
            
            # Spatial weights (based on distance)
            spatial_dists = np.linalg.norm(v_neighbors - v_i, axis=1)
            spatial_weights = np.exp(-(spatial_dists**2) / (2 * sigma_spatial**2))
            
            # Range weights (based on normal similarity)
            normals_i = mesh.vertex_normals[i]
            normals_neighbors = mesh.vertex_normals[nbrs]
            normal_diffs = np.linalg.norm(normals_neighbors - normals_i, axis=1)
            range_weights = np.exp(-(normal_diffs**2) / (2 * sigma_range**2))
            
            # Combined weights
            weights = spatial_weights * range_weights
            weights = weights / (weights.sum() + 1e-8)
            
            # Filtered position
            new_vertices[i] = np.sum(weights[:, np.newaxis] * v_neighbors, axis=0)
        
        filtered_vertices = new_vertices
    
    return filtered_vertices


def _closest_point_on_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute closest points from points to triangles.
    
    Args:
        points: (N, 3) query points
        triangles: (M, 3, 3) triangle vertices
    
    Returns:
        closest_points: (N, M, 3) closest points on triangles
        distances_sq: (N, M) squared distances
    """
    A = triangles[:, 0]  # (M, 3)
    B = triangles[:, 1]
    C = triangles[:, 2]

    AB = B - A  # (M, 3)
    AC = C - A
    AP = points[:, None, :] - A[None, :, :]  # (N, M, 3)

    # Dot products
    d1 = np.sum(AB * AP, axis=2)
    d2 = np.sum(AC * AP, axis=2)

    # Edge lengths
    ABAB = np.sum(AB * AB, axis=1)
    ACAC = np.sum(AC * AC, axis=1)
    ABAC = np.sum(AB * AC, axis=1)

    denom = ABAB * ACAC - ABAC * ABAC
    denom = denom + 1e-12

    v = (ACAC * d1 - ABAC * d2) / denom
    w = (ABAB * d2 - ABAC * d1) / denom

    # Clamp barycentric coordinates to triangle
    v_clamped = np.clip(v, 0, 1)
    w_clamped = np.clip(w, 0, 1 - v_clamped)

    cp = (
        A[None, :, :] +
        v_clamped[:, :, None] * AB[None, :, :] +
        w_clamped[:, :, None] * AC[None, :, :]
    )
    d2_sq = np.sum((cp - points[:, None, :]) ** 2, axis=2)

    return cp, d2_sq


def remove_self_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 3,
    threshold: float = 0.005,
    chunk_size: int = 300,
) -> np.ndarray:
    """
    Detect and resolve self-intersections in mesh.
    
    NumPy-only implementation. Common artifact in MVF-Net output.
    Uses closest-point queries to detect and push away intersecting vertices.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        iterations: Number of correction iterations
        threshold: Distance threshold for considering intersection
        chunk_size: Process triangles in chunks to limit memory
    
    Returns:
        corrected_vertices: (N, 3) vertices with intersections resolved
    """
    vertices = vertices.copy()
    V = len(vertices)
    F = len(faces)

    # Build vertex-to-faces adjacency
    vertex_to_faces = [[] for _ in range(V)]
    for f_idx, (a, b, c) in enumerate(faces):
        vertex_to_faces[a].append(f_idx)
        vertex_to_faces[b].append(f_idx)
        vertex_to_faces[c].append(f_idx)

    for iteration in range(iterations):
        print(f"[Iteration {iteration+1}/{iterations}] Computing normals...")

        # Compute normals
        face_normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]]
        )
        normals = np.zeros_like(vertices)
        for f_idx, (a, b, c) in enumerate(faces):
            n = face_normals[f_idx]
            normals[a] += n
            normals[b] += n
            normals[c] += n

        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

        # Extract triangles
        triangles = vertices[faces]  # (F, 3, 3)

        # Per-vertex tracking
        closest_face = np.zeros(V, dtype=np.int32)
        closest_dist = np.full(V, np.inf)

        # Chunked closest-point computation
        for start in range(0, F, chunk_size):
            end = min(start + chunk_size, F)
            tris_chunk = triangles[start:end]

            cp, d2 = _closest_point_on_triangles(vertices, tris_chunk)
            local_min_idx = np.argmin(d2, axis=1)
            local_min_dist = d2[np.arange(V), local_min_idx]

            mask = local_min_dist < closest_dist
            closest_dist[mask] = local_min_dist[mask]
            closest_face[mask] = start + local_min_idx[mask]

        # Push intersecting vertices along normals
        moved = 0
        for i in range(V):
            if closest_dist[i] < threshold:
                f = closest_face[i]
                if f not in vertex_to_faces[i]:
                    vertices[i] += threshold * normals[i]
                    moved += 1

        print(f"  Moved {moved} vertices")

    return vertices


def enhance_normals_from_image(
    vertices: np.ndarray,
    faces: np.ndarray,
    image_rgb: np.ndarray,
    enhancement_scale: float = 0.002,
) -> np.ndarray:
    """
    Enhance mesh details using input image gradients.
    
    Uses image gradients as proxy for surface detail to recover information
    that MVF-Net may have smoothed out.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        image_rgb: (H, W, 3) input RGB image
        enhancement_scale: Magnitude of enhancement displacement
    
    Returns:
        enhanced_vertices: (N, 3) vertices with enhanced details
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Compute image gradients
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
    
    # Simplified projection (assumes orthographic)
    vertices_2d = vertices[:, :2]
    h, w = image_rgb.shape[:2]
    vertex_gradients = np.zeros(len(vertices))
    
    for i, v2d in enumerate(vertices_2d):
        px = int((v2d[0] + 1) * w / 2)
        py = int((v2d[1] + 1) * h / 2)
        
        if 0 <= px < w and 0 <= py < h:
            vertex_gradients[i] = gradient_magnitude[py, px]
    
    # Enhance vertex positions along normals
    normals = mesh.vertex_normals
    enhanced_vertices = vertices + enhancement_scale * vertex_gradients[:, np.newaxis] * normals
    
    return enhanced_vertices


def enforce_multiview_consistency(
    vertices_list: List[np.ndarray],
    faces: np.ndarray,
    confidence_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine multiple MVF-Net outputs from different views.
    
    Enforces consistency across views and removes self-intersections
    from weighted average.
    
    Args:
        vertices_list: List of (N, 3) vertex arrays from different runs
        faces: (M, 3) face indices (same topology)
        confidence_weights: Optional per-view weights (normalized)
    
    Returns:
        consistent_vertices: (N, 3) improved vertices
    """
    if confidence_weights is None:
        confidence_weights = np.ones(len(vertices_list)) / len(vertices_list)
    
    # Weighted average
    consistent = np.zeros_like(vertices_list[0])
    for verts, weight in zip(vertices_list, confidence_weights):
        consistent += weight * verts
    
    # Remove self-intersections
    mesh = trimesh.Trimesh(vertices=consistent, faces=faces, process=False)
    if mesh.is_watertight:
        consistent = remove_self_intersections(consistent, faces)
    
    return consistent
