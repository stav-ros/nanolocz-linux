"""Core typed contracts and analysis operations for NanoLocz."""

from nanolocz.core.detection import detect_particles, fast_peaks2d
from nanolocz.core.drift import estimate_drift_xcorr, estimate_drift_particles, correct_drift
from nanolocz.core.substacks import (
    extract_particle_substacks,
    extract_drift_corrected_substacks,
    create_gaussian_mask,
    batch_extract_substacks,
)
from nanolocz.core.deskar import (
    directional_deskar,
    remove_scan_lines,
    anisotropic_diffusion,
    process_movie_deskar,
)
from nanolocz.core.classification import (
    classify_particles,
    reduce_dimensions_pca,
    cluster_hdbscan,
    select_n_components,
    ClassificationResult,
    plot_scree,
    plot_clusters_2d,
    plot_cluster_sizes,
)
from nanolocz.core.types import DetectionResult, Frame, Localizations, Meta, ParticleStack

__all__ = [
    "DetectionResult",
    "Frame",
    "Meta",
    "Localizations",
    "ParticleStack",
    "ClassificationResult",
    "detect_particles",
    "fast_peaks2d",
    "estimate_drift_xcorr",
    "estimate_drift_particles",
    "correct_drift",
    "extract_particle_substacks",
    "extract_drift_corrected_substacks",
    "create_gaussian_mask",
    "batch_extract_substacks",
    "directional_deskar",
    "remove_scan_lines",
    "anisotropic_diffusion",
    "process_movie_deskar",
    "classify_particles",
    "reduce_dimensions_pca",
    "cluster_hdbscan",
    "select_n_components",
    "plot_scree",
    "plot_clusters_2d",
    "plot_cluster_sizes",
]
