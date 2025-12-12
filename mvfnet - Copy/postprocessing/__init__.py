"""
MVF-Net post-processing package.

Post-processing improvements for 3D face mesh reconstruction from MVF-Net.
Provides modular, well-documented functions for mesh enhancement.

Main modules:
    - improvements: Mesh enhancement algorithms
    - metrics: Quality metrics computation
    - utils: Utility functions for I/O and visualization
    - pipeline: Main orchestration pipeline
"""

from .improvements import (
    laplacian_smoothing_adaptive,
    bilateral_mesh_filtering,
    remove_self_intersections,
    enhance_normals_from_image,
    enforce_multiview_consistency,
)

from .metrics import (
    compute_mesh_quality_metrics,
    compute_metrics_change,
    print_metrics,
    compute_laplacian_matrix,
)

from .utils import (
    verify_mesh_consistency,
    visualize_mesh,
    visualize_comparison,
    load_mesh_ply,
    save_mesh_ply,
    ensure_output_dir,
)

from .pipeline import MeshImprovementPipeline

__all__ = [
    # Improvements
    "laplacian_smoothing_adaptive",
    "bilateral_mesh_filtering",
    "remove_self_intersections",
    "enhance_normals_from_image",
    "enforce_multiview_consistency",
    # Metrics
    "compute_mesh_quality_metrics",
    "compute_metrics_change",
    "print_metrics",
    "compute_laplacian_matrix",
    # Utils
    "verify_mesh_consistency",
    "visualize_mesh",
    "visualize_comparison",
    "load_mesh_ply",
    "save_mesh_ply",
    "ensure_output_dir",
    # Pipeline
    "MeshImprovementPipeline",
]

__version__ = "1.0.0"
