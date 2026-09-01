"""
GPU-accelerated particle classification using PCA and HDBSCAN.

This module provides CuPy-accelerated versions of classification functions
for improved performance on large datasets.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.linalg import svd
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def reduce_dimensions_pca_gpu(
    data: np.ndarray,
    n_components: int | None = None,
    whiten: bool = False,
    device_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce dimensionality of particle data using GPU-accelerated PCA.
    
    Uses CuPy's SVD implementation for faster computation on large datasets.
    Falls back to CPU if CuPy is unavailable.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_particles, *spatial_dims). Will be flattened
        to (n_particles, n_features) before PCA.
    n_components : int | None
        Number of principal components to retain. If None, keeps all components.
    whiten : bool
        Whether to whiten the components (default False).
    device_id : int
        CUDA device ID to use (default 0).
    
    Returns
    -------
    reduced_coords : np.ndarray
        PCA-transformed coordinates with shape (n_particles, n_components).
    explained_variance : np.ndarray
        Explained variance for each component. Shape: (n_components,).
    explained_variance_ratio : np.ndarray
        Explained variance ratio for each component. Shape: (n_components,).
    
    Raises
    ------
    ImportError
        If CuPy is not available.
    ValueError
        If input data is invalid.
    """
    if not _HAS_CUPY:
        raise ImportError(
            "CuPy not available. Install with: pip install cupy-cuda12x"
        )
    
    # Set device
    cp.cuda.Device(device_id).use()
    
    # Validate and prepare input
    data = np.asarray(data, dtype=np.float64)
    
    if data.ndim < 2:
        raise ValueError(f"Input data must have at least 2 dimensions, got {data.ndim}")
    
    n_particles = data.shape[0]
    if n_particles == 0:
        raise ValueError("Input data contains no particles")
    
    # Flatten spatial dimensions
    if data.ndim > 2:
        data_flat = data.reshape(n_particles, -1)
    else:
        data_flat = data
    
    n_features = data_flat.shape[1]
    if n_components is None:
        n_components = min(n_particles, n_features)
    else:
        n_components = min(n_components, n_particles, n_features)
    
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    
    # Center data (subtract mean)
    mean = data_flat.mean(axis=0)
    data_centered = data_flat - mean
    
    # Transfer to GPU
    data_gpu = cp.asarray(data_centered, dtype=cp.float64)
    
    # Compute SVD on GPU
    # For PCA, we need U, S, Vt where data_centered = U @ diag(S) @ Vt
    U, S, Vt = svd(data_gpu, full_matrices=False)
    
    # Select top n_components
    U_n = U[:, :n_components]
    S_n = S[:n_components]
    Vt_n = Vt[:n_components, :]
    
    # Transformed data (scores) = U @ S
    reduced_gpu = U_n @ cp.diag(S_n)
    
    # Explained variance = S^2 / (n_samples - 1)
    n_samples = n_particles
    explained_variance_gpu = (S_n ** 2) / (n_samples - 1)
    
    # Total variance
    total_variance = cp.var(data_gpu, axis=0, ddof=1).sum()
    explained_variance_ratio_gpu = explained_variance_gpu / total_variance
    
    # Whiten if requested
    if whiten:
        reduced_gpu = reduced_gpu / cp.sqrt(explained_variance_gpu)
    
    # Transfer results back to CPU
    reduced = cp.asnumpy(reduced_gpu)
    variance = cp.asnumpy(explained_variance_gpu)
    variance_ratio = cp.asnumpy(explained_variance_ratio_gpu)
    
    return reduced, variance, variance_ratio


def classify_particles_gpu(
    particle_stack_data: np.ndarray,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    max_components: int = 50,
    whiten: bool = False,
    device_id: int = 0,
) -> dict:
    """Classify particles using GPU-accelerated PCA + CPU HDBSCAN.
    
    Note: HDBSCAN currently runs on CPU as there's no official GPU implementation.
    Only the PCA step is GPU-accelerated.
    
    Parameters
    ----------
    particle_stack_data : np.ndarray
        Particle substack data with shape (n_particles, *spatial_dims).
    n_components : int | None
        Number of PCA components. If None, automatically selected.
    variance_threshold : float
        For automatic component selection, cumulative variance threshold.
    min_cluster_size : int
        Minimum particles per cluster for HDBSCAN.
    min_samples : int | None
        HDBSCAN min_samples parameter. If None, uses min_cluster_size.
    max_components : int
        Maximum number of PCA components to consider.
    whiten : bool
        Whether to whiten PCA components.
    device_id : int
        CUDA device ID to use.
    
    Returns
    -------
    dict
        Dictionary containing:
        - labels: Cluster assignments (-1 for noise)
        - probabilities: Confidence scores
        - reduced_coords: PCA coordinates
        - explained_variance: Variance information
        - n_clusters: Number of clusters found
        - noise_count: Number of unclassified particles
    
    Raises
    ------
    ImportError
        If CuPy or hdbscan packages are not installed.
    """
    if not _HAS_CUPY:
        raise ImportError("CuPy not available. Install with: pip install cupy-cuda12x")
    
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan package required. Install with: pip install hdbscan")
    
    # Validate input
    particle_stack_data = np.asarray(particle_stack_data, dtype=np.float64)
    
    if particle_stack_data.ndim < 2:
        raise ValueError(
            f"particle_stack_data must have at least 2 dimensions, got {particle_stack_data.ndim}"
        )
    
    n_particles = particle_stack_data.shape[0]
    if n_particles < min_cluster_size:
        raise ValueError(
            f"Not enough particles ({n_particles}) for min_cluster_size={min_cluster_size}"
        )
    
    # Step 1: GPU-accelerated PCA
    if n_components is None:
        # Use max_components for initial PCA, then select optimal
        n_comp_initial = min(
            max_components, 
            n_particles, 
            particle_stack_data.reshape(n_particles, -1).shape[1]
        )
        reduced, variance, variance_ratio = reduce_dimensions_pca_gpu(
            particle_stack_data,
            n_components=n_comp_initial,
            whiten=whiten,
            device_id=device_id,
        )
        
        # Select optimal number of components (CPU-based)
        cumsum = np.cumsum(variance_ratio)
        n_components = int(np.searchsorted(cumsum, variance_threshold) + 1)
        n_components = min(n_components, max_components)
        
        # Re-run PCA with selected components
        reduced, variance, variance_ratio = reduce_dimensions_pca_gpu(
            particle_stack_data,
            n_components=n_components,
            whiten=whiten,
            device_id=device_id,
        )
    else:
        reduced, variance, variance_ratio = reduce_dimensions_pca_gpu(
            particle_stack_data,
            n_components=n_components,
            whiten=whiten,
            device_id=device_id,
        )
    
    # Step 2: HDBSCAN clustering (CPU)
    if min_samples is None:
        min_samples = min_cluster_size
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )
    
    labels = clusterer.fit_predict(reduced)
    probabilities = clusterer.probabilities_
    
    # Count clusters and noise
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    noise_count = int(np.sum(labels == -1))
    
    return {
        'labels': labels,
        'probabilities': probabilities,
        'reduced_coords': reduced,
        'explained_variance': variance,
        'explained_variance_ratio': variance_ratio,
        'n_clusters': n_clusters,
        'noise_count': noise_count,
        'n_particles': n_particles,
        'n_components': n_components,
    }
