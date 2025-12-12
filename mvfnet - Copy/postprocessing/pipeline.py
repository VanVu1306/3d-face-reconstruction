"""
MVF-Net post-processing pipeline.

Main module that orchestrates mesh improvement with comprehensive logging,
metrics tracking, and error handling.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np

from . import improvements
from . import metrics
from . import utils


class MeshImprovementPipeline:
    """
    Complete post-processing pipeline for MVF-Net output.
    
    Orchestrates bilateral filtering, adaptive smoothing, self-intersection
    removal, and quality evaluation.
    """
    
    def __init__(
        self,
        enable_bilateral: bool = True,
        enable_smoothing: bool = True,
        enable_intersection_removal: bool = True,
        bilateral_iterations: int = 5,
        smoothing_iterations: int = 3,
        smoothing_lambda: float = 0.5,
    ):
        """
        Initialize pipeline.
        
        Args:
            enable_bilateral: Enable bilateral filtering
            enable_smoothing: Enable adaptive Laplacian smoothing
            enable_intersection_removal: Enable self-intersection removal
            bilateral_iterations: Bilateral filter iterations
            smoothing_iterations: Smoothing iterations
            smoothing_lambda: Smoothing strength [0, 1]
        """
        self.enable_bilateral = enable_bilateral
        self.enable_smoothing = enable_smoothing
        self.enable_intersection_removal = enable_intersection_removal
        self.bilateral_iterations = bilateral_iterations
        self.smoothing_iterations = smoothing_iterations
        self.smoothing_lambda = smoothing_lambda
    
    def improve(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Run complete improvement pipeline.
        
        Args:
            vertices: (N, 3) MVF-Net output vertices
            faces: (M, 3) face indices
            output_path: Where to save improved mesh (optional)
            verbose: Print progress and metrics
        
        Returns:
            improved_vertices: (N, 3) enhanced vertices
            metrics_before: Quality metrics before improvement
            metrics_after: Quality metrics after improvement
        """
        if verbose:
            print("=" * 70)
            print("MVF-Net Mesh Improvement Pipeline")
            print("=" * 70)
        
        # Compute baseline metrics
        if verbose:
            print("\n[1/4] Computing baseline metrics...")
        metrics_before = metrics.compute_mesh_quality_metrics(vertices, faces)
        if verbose:
            metrics.print_metrics(metrics_before, "Baseline quality")
        
        improved_vertices = vertices.copy()
        
        # Step 1: Bilateral filtering
        if self.enable_bilateral:
            if verbose:
                print("\n[2/4] Applying bilateral filtering...")
            vertices_before = improved_vertices.copy()
            improved_vertices = improvements.bilateral_mesh_filtering(
                improved_vertices,
                faces,
                iterations=self.bilateral_iterations
            )
            if verbose:
                utils.verify_mesh_consistency(
                    vertices_before, faces,
                    improved_vertices, faces,
                    "After Bilateral Filtering"
                )
                print("  ✓ Bilateral filter applied")
        
        # Step 2: Adaptive smoothing
        if self.enable_smoothing:
            if verbose:
                print("\n[3/4] Applying adaptive smoothing...")
            vertices_before = improved_vertices.copy()
            improved_vertices = improvements.laplacian_smoothing_adaptive(
                improved_vertices,
                faces,
                iterations=self.smoothing_iterations,
                lambda_smooth=self.smoothing_lambda
            )
            if verbose:
                utils.verify_mesh_consistency(
                    vertices_before, faces,
                    improved_vertices, faces,
                    "After Adaptive Smoothing"
                )
                print("  ✓ Adaptive smoothing applied")
        
        # Step 3: Self-intersection removal
        if self.enable_intersection_removal:
            if verbose:
                print("\n[4/4] Removing self-intersections...")
            vertices_before = improved_vertices.copy()
            improved_vertices = improvements.remove_self_intersections(
                improved_vertices,
                faces,
                iterations=3
            )
            if verbose:
                is_consistent = utils.verify_mesh_consistency(
                    vertices_before, faces,
                    improved_vertices, faces,
                    "After Self-Intersection Removal"
                )
                if not is_consistent:
                    print("  ⚠️  WARNING: Using pre-removal version")
                    improved_vertices = vertices_before
                else:
                    print("  ✓ Self-intersections removed")
        
        # Compute improved metrics
        if verbose:
            print("\nComputing improved metrics...")
        metrics_after = metrics.compute_mesh_quality_metrics(improved_vertices, faces)
        if verbose:
            metrics.print_metrics(metrics_after, "Improved quality")
        
        # Save mesh if requested
        if output_path:
            utils.ensure_output_dir(output_path)
            utils.save_mesh_ply(output_path, improved_vertices, faces, binary=False)
            if verbose:
                print(f"\n✓ Improved mesh saved: {output_path}")
        
        # Summary
        if verbose:
            self._print_summary(metrics_before, metrics_after)
        
        return improved_vertices, metrics_before, metrics_after
    
    @staticmethod
    def _print_summary(
        metrics_before: Dict,
        metrics_after: Dict,
    ) -> None:
        """Print improvement summary."""
        print("\n" + "=" * 70)
        print("IMPROVEMENT SUMMARY")
        print("=" * 70)
        
        changes = metrics.compute_metrics_change(metrics_before, metrics_after)
        for key, change in changes.items():
            before = metrics_before[key]
            after = metrics_after[key]
            print(f"  {key}:")
            print(f"    Before: {before:.4f}")
            print(f"    After:  {after:.4f}")
            print(f"    Change: {change:+.1f}%")
        
        print("=" * 70)
