"""
Particle classification using PCA and HDBSCAN.

This module provides tools for classifying particle substacks by conformational state
using Principal Component Analysis (PCA) for dimensionality reduction followed by
HDBSCAN density-based clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA

try:
    import hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False


@dataclass
class ClassificationResult:
    """Results from particle classification pipeline."""
    
    labels: np.ndarray
    """Cluster assignments for each particle. Shape: (n_particles,). -1 indicates noise."""
    
    probabilities: np.ndarray
    """Confidence scores for cluster assignments. Shape: (n_particles,). Range [0, 1]."""
    
    reduced_coords: np.ndarray
    """PCA-reduced coordinates. Shape: (n_particles, n_components)."""
    
    explained_variance: np.ndarray
    """Explained variance for each component. Shape: (n_components,)."""
    
    explained_variance_ratio: np.ndarray
    """Explained variance ratio for each component. Shape: (n_components,)."""
    
    n_clusters: int
    """Number of clusters found (excluding noise)."""
    
    noise_count: int
    """Number of particles classified as noise (label=-1)."""
    
    n_particles: int = field(init=False)
    """Total number of particles classified."""
    
    n_components: int = field(init=False)
    """Number of PCA components used."""
    
    def __post_init__(self):
        self.n_particles = len(self.labels)
        self.n_components = self.reduced_coords.shape[1]
    
    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """Return indices of particles in specified cluster.
        
        Parameters
        ----------
        cluster_id : int
            Cluster ID to retrieve members for. Use -1 for noise particles.
        
        Returns
        -------
        np.ndarray
            Indices of particles belonging to the cluster.
        """
        return np.where(self.labels == cluster_id)[0]
    
    def get_cluster_sizes(self) -> dict[int, int]:
        """Return dictionary mapping cluster_id to particle count.
        
        Returns
        -------
        dict[int, int]
            Dictionary with cluster IDs as keys and particle counts as values.
            Includes noise cluster (-1) if present.
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_non_noise_indices(self) -> np.ndarray:
        """Return indices of all non-noise particles.
        
        Returns
        -------
        np.ndarray
            Indices where labels != -1.
        """
        return np.where(self.labels != -1)[0]


def reduce_dimensions_pca(
    data: np.ndarray,
    n_components: int | None = None,
    whiten: bool = False,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce dimensionality of particle data using PCA.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_particles, *spatial_dims). Will be flattened
        to (n_particles, n_features) before PCA.
    n_components : int | None
        Number of principal components to retain. If None, keeps all components.
    whiten : bool
        Whether to whiten the components (default False).
    random_state : int | None
        Random seed for reproducibility.
    
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
    ValueError
        If input data is empty or has invalid shape.
    """
    if not _HAS_HDBSCAN:
        raise ImportError("hdbscan package required. Install with: pip install hdbscan")
    
    # Validate input
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
    
    # Handle case where n_features < n_particles
    n_features = data_flat.shape[1]
    if n_components is None:
        n_components = min(n_particles, n_features)
    else:
        n_components = min(n_components, n_particles, n_features)
    
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    
    # Apply PCA
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    reduced = pca.fit_transform(data_flat)
    
    return reduced, pca.explained_variance_, pca.explained_variance_ratio_


def select_n_components(
    variance_ratios: np.ndarray,
    method: Literal["kaiser", "elbow", "cumulative"] = "cumulative",
    cumulative_threshold: float = 0.95,
    max_components: int | None = None,
) -> int:
    """Automatically select optimal number of PCA components.
    
    Parameters
    ----------
    variance_ratios : np.ndarray
        Explained variance ratios from PCA, sorted in descending order.
    method : {"kaiser", "elbow", "cumulative"}
        Selection method:
        - "kaiser": Keep components with eigenvalue > 1 (variance ratio > 1/n_components)
        - "elbow": Detect elbow point in scree plot
        - "cumulative": Keep components until cumulative variance exceeds threshold
    cumulative_threshold : float
        For cumulative method, stop when this fraction of variance is explained.
    max_components : int | None
        Upper bound on number of components to select.
    
    Returns
    -------
    int
        Recommended number of components.
    """
    variance_ratios = np.asarray(variance_ratios, dtype=np.float64)
    
    if len(variance_ratios) == 0:
        raise ValueError("variance_ratios cannot be empty")
    
    n_total = len(variance_ratios)
    
    if method == "kaiser":
        # Kaiser criterion: keep components with variance ratio > 1/n_components
        threshold = 1.0 / n_total
        n_comp = np.sum(variance_ratios > threshold)
    
    elif method == "elbow":
        # Simple elbow detection using maximum curvature
        # Normalize to [0, 1] range
        cumsum = np.cumsum(variance_ratios)
        cumsum_norm = cumsum / cumsum[-1]
        
        # Find point of maximum distance from line connecting first and last points
        line_vec = np.array([n_total - 1, cumsum_norm[-1] - cumsum_norm[0]])
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            n_comp = n_total
        else:
            line_unit = line_vec / line_len
            distances = []
            for i in range(len(cumsum_norm)):
                point_vec = np.array([i, cumsum_norm[i] - cumsum_norm[0]])
                proj = np.dot(point_vec, line_unit)
                proj_point = line_unit * proj
                dist = np.linalg.norm(point_vec - proj_point)
                distances.append(dist)
            
            n_comp = np.argmax(distances) + 1
    
    elif method == "cumulative":
        cumsum = np.cumsum(variance_ratios)
        n_comp = np.searchsorted(cumsum, cumulative_threshold) + 1
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kaiser', 'elbow', or 'cumulative'")
    
    # Apply upper bound
    if max_components is not None:
        n_comp = min(n_comp, max_components)
    
    # Ensure at least 1 component
    n_comp = max(1, n_comp)
    
    return int(n_comp)


def cluster_hdbscan(
    coords: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_method: Literal["eom", "leaf"] = "eom",
    allow_single_cluster: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Cluster particles using HDBSCAN.
    
    Parameters
    ----------
    coords : np.ndarray
        Input coordinates with shape (n_particles, n_dimensions). Typically PCA-reduced data.
    min_cluster_size : int
        Minimum number of particles in a cluster (default 10).
    min_samples : int | None
        Minimum samples parameter for HDBSCAN. If None, uses min_cluster_size.
    metric : str
        Distance metric for HDBSCAN (default "euclidean").
    cluster_selection_method : {"eom", "leaf"}
        Method for selecting clusters (default "eom").
    allow_single_cluster : bool
        Whether to allow a single cluster result (default False).
    
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each particle. Shape: (n_particles,). -1 indicates noise.
    probabilities : np.ndarray
        Cluster membership probabilities. Shape: (n_particles,).
    n_clusters : int
        Number of clusters found (excluding noise).
    noise_count : int
        Number of noise particles (label=-1).
    
    Raises
    ------
    ImportError
        If hdbscan package is not installed.
    ValueError
        If input data is invalid.
    """
    if not _HAS_HDBSCAN:
        raise ImportError("hdbscan package required. Install with: pip install hdbscan")
    
    coords = np.asarray(coords, dtype=np.float64)
    
    if coords.ndim != 2:
        raise ValueError(f"Input coords must be 2D, got shape {coords.shape}")
    
    if coords.shape[0] < min_cluster_size:
        raise ValueError(
            f"Not enough particles ({coords.shape[0]}) for min_cluster_size={min_cluster_size}"
        )
    
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        prediction_data=True,
    )
    
    labels = clusterer.fit_predict(coords)
    
    # Get probabilities (cluster membership strengths)
    probabilities = clusterer.probabilities_
    
    # Count clusters and noise
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    noise_count = np.sum(labels == -1)
    
    return labels, probabilities, n_clusters, noise_count


def classify_particles(
    particle_stack_data: np.ndarray,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    max_components: int = 50,
    whiten: bool = False,
    random_state: int | None = None,
) -> ClassificationResult:
    """Classify particles using PCA + HDBSCAN pipeline.
    
    This is the main entry point for particle classification. It performs:
    1. PCA dimensionality reduction (with automatic component selection if n_components=None)
    2. HDBSCAN clustering in reduced space
    3. Returns structured results with labels, probabilities, and metadata
    
    Parameters
    ----------
    particle_stack_data : np.ndarray
        Particle substack data with shape (n_particles, *spatial_dims).
        Typically from ParticleStack.data.
    n_components : int | None
        Number of PCA components. If None, automatically selected using variance_threshold.
    variance_threshold : float
        For automatic component selection, cumulative variance threshold (default 0.95).
    min_cluster_size : int
        Minimum particles per cluster for HDBSCAN (default 10).
    min_samples : int | None
        HDBSCAN min_samples parameter. If None, uses min_cluster_size.
    max_components : int
        Maximum number of PCA components to consider (default 50).
    whiten : bool
        Whether to whiten PCA components (default False).
    random_state : int | None
        Random seed for reproducibility.
    
    Returns
    -------
    ClassificationResult
        Structured results containing:
        - labels: Cluster assignments (-1 for noise)
        - probabilities: Confidence scores
        - reduced_coords: PCA coordinates
        - explained_variance: Variance information
        - n_clusters: Number of clusters found
        - noise_count: Number of unclassified particles
    
    Raises
    ------
    ImportError
        If required packages (hdbscan) are not installed.
    ValueError
        If input data is invalid or too small.
    
    Examples
    --------
    >>> from nanolocz.core.classification import classify_particles
    >>> result = classify_particles(particle_stack.data, min_cluster_size=15)
    >>> print(f"Found {result.n_clusters} clusters with {result.noise_count} noise particles")
    >>> cluster_0_members = result.get_cluster_members(0)
    """
    if not _HAS_HDBSCAN:
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
    
    # Step 1: PCA dimensionality reduction
    if n_components is None:
        # Use max_components for initial PCA, then select optimal
        n_comp_initial = min(max_components, n_particles, particle_stack_data.reshape(n_particles, -1).shape[1])
        reduced, variance, variance_ratio = reduce_dimensions_pca(
            particle_stack_data,
            n_components=n_comp_initial,
            whiten=whiten,
            random_state=random_state,
        )
        
        # Select optimal number of components
        n_components = select_n_components(
            variance_ratio,
            method="cumulative",
            cumulative_threshold=variance_threshold,
            max_components=max_components,
        )
        
        # Re-run PCA with selected components
        reduced, variance, variance_ratio = reduce_dimensions_pca(
            particle_stack_data,
            n_components=n_components,
            whiten=whiten,
            random_state=random_state,
        )
    else:
        reduced, variance, variance_ratio = reduce_dimensions_pca(
            particle_stack_data,
            n_components=n_components,
            whiten=whiten,
            random_state=random_state,
        )
    
    # Step 2: HDBSCAN clustering
    labels, probabilities, n_clusters, noise_count = cluster_hdbscan(
        reduced,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    
    # Step 3: Return structured results
    return ClassificationResult(
        labels=labels,
        probabilities=probabilities,
        reduced_coords=reduced,
        explained_variance=variance,
        explained_variance_ratio=variance_ratio,
        n_clusters=n_clusters,
        noise_count=noise_count,
    )


def plot_scree(
    explained_variance_ratio: np.ndarray,
    ax=None,
    title: str = "Scree Plot",
    show_cumulative: bool = True,
) -> None:
    """Create a scree plot showing explained variance per component.
    
    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Explained variance ratios from PCA.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Creates new figure if None.
    title : str
        Plot title.
    show_cumulative : bool
        Whether to show cumulative variance curve.
    """
    import matplotlib.pyplot as plt
    
    explained_variance_ratio = np.asarray(explained_variance_ratio)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    n_components = len(explained_variance_ratio)
    x = np.arange(1, n_components + 1)
    
    # Plot individual variance
    ax.bar(x, explained_variance_ratio, alpha=0.7, label="Individual")
    
    if show_cumulative:
        cumsum = np.cumsum(explained_variance_ratio)
        ax.plot(x, cumsum, 'o-', color='red', label="Cumulative")
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label="95% threshold")
    
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_clusters_2d(
    reduced_coords: np.ndarray,
    labels: np.ndarray,
    ax=None,
    title: str = "PCA-HDBSCAN Clustering",
    cmap: str = "tab10",
    show_legend: bool = True,
) -> None:
    """Create 2D scatter plot of clusters in PCA space.
    
    Parameters
    ----------
    reduced_coords : np.ndarray
        PCA coordinates with shape (n_particles, n_components), n_components >= 2.
    labels : np.ndarray
        Cluster labels with shape (n_particles,).
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Creates new figure if None.
    title : str
        Plot title.
    cmap : str
        Colormap for clusters.
    show_legend : bool
        Whether to show legend.
    """
    import matplotlib.pyplot as plt
    
    reduced_coords = np.asarray(reduced_coords)
    labels = np.asarray(labels)
    
    if reduced_coords.shape[1] < 2:
        raise ValueError("reduced_coords must have at least 2 components for 2D plot")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    x = reduced_coords[:, 0]
    y = reduced_coords[:, 1]
    
    # Get unique labels (including noise)
    unique_labels = np.unique(labels)
    
    # Plot each cluster
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # Noise points
            ax.scatter(x[mask], y[mask], c='gray', marker='x', s=50, 
                      label='Noise', alpha=0.5)
        else:
            ax.scatter(x[mask], y[mask], label=f'Cluster {label}', 
                      s=50, alpha=0.7)
    
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)


def plot_cluster_sizes(
    labels: np.ndarray,
    ax=None,
    title: str = "Cluster Size Distribution",
    sort_by_size: bool = True,
) -> None:
    """Create histogram of cluster sizes.
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster labels with shape (n_particles,).
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Creates new figure if None.
    title : str
        Plot title.
    sort_by_size : bool
        Whether to sort clusters by size.
    """
    import matplotlib.pyplot as plt
    
    labels = np.asarray(labels)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    unique, counts = np.unique(labels, return_counts=True)
    
    # Separate noise from clusters
    cluster_mask = unique != -1
    cluster_ids = unique[cluster_mask]
    cluster_counts = counts[cluster_mask]
    noise_count = counts[unique == -1].sum() if -1 in unique else 0
    
    if sort_by_size:
        sort_idx = np.argsort(cluster_counts)[::-1]
        cluster_ids = cluster_ids[sort_idx]
        cluster_counts = cluster_counts[sort_idx]
    
    # Plot clusters
    if len(cluster_ids) > 0:
        ax.bar(range(len(cluster_ids)), cluster_counts, tick_label=cluster_ids, alpha=0.7)
    
    # Add noise bar if present
    if noise_count > 0:
        ax.bar(len(cluster_ids), noise_count, color='gray', alpha=0.5, 
              tick_label=['Noise'] if len(cluster_ids) == 0 else ['Noise'])
    
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Particles")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
