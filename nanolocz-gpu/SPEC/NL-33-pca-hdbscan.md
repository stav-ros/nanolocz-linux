# NL-33 — PCA to HDBSCAN grouping

## Goal
Implement particle classification pipeline using PCA for dimensionality reduction followed by HDBSCAN clustering to group particles by conformational state.

## Motivation
After extracting particle substacks (NL-32), we need to classify particles into structural classes to:
- Identify distinct conformational states of imaged molecules
- Enable in-class averaging for improved SNR (NL-34)
- Detect rare or transient states
- Prepare for dynamics analysis (NL-36)

PCA reduces the high-dimensional substack data to a manageable number of components while preserving variance. HDBSCAN then clusters particles in this reduced space, automatically determining the optimal number of clusters and handling noise/outliers.

## Dependencies
- **NL-16**: Detection and statistics (particle coordinates, masks)
- **NL-32**: Particle substack extraction (input data)
- **NL-03**: Typed contracts (`ParticleStack`, `Localizations`)
- **NL-10**: Zarr schema for storing classification results

## Acceptance Criteria

### 1. PCA dimensionality reduction
Implement `reduce_dimensions_pca()` that:
- Takes `ParticleStack` data from NL-32
- Flattens substacks to feature vectors (particles × features)
- Computes PCA with configurable number of components
- Returns:
  - Reduced coordinates (n_particles × n_components)
  - Explained variance ratios
  - Cumulative variance explained
- Supports incremental PCA for large datasets
- Validates input shape and handles missing data

### 2. Component selection heuristics
Provide automatic component selection via:
- Kaiser criterion (eigenvalues > 1)
- Scree plot elbow detection
- Cumulative variance threshold (default 95%)
- Manual override option

### 3. HDBSCAN clustering
Implement `cluster_hdbscan()` that:
- Takes PCA-reduced coordinates
- Configures HDBSCAN parameters:
  - `min_cluster_size`: Minimum particles per cluster (default: 5-10)
  - `min_samples`: Controls cluster conservatism (default: None → min_cluster_size)
  - `metric`: Distance metric (default: 'euclidean')
  - `cluster_selection_method`: 'eom' or 'leaf' (default: 'eom')
- Returns:
  - Cluster labels for each particle (-1 for noise)
  - Cluster probabilities (confidence scores)
  - Number of clusters found
  - Noise particle count

### 4. Classification pipeline
Implement `classify_particles()` that:
- Combines PCA + HDBSCAN in single workflow
- Accepts `ParticleStack` as input
- Returns `ClassificationResult` with:
  - `labels`: Cluster assignments
  - `probabilities`: Confidence scores
  - `reduced_coords`: PCA coordinates
  - `explained_variance`: PCA variance info
  - `n_clusters`: Number of clusters found
  - `noise_count`: Number of unclassified particles
- Supports parameter tuning and validation

### 5. Visualization utilities
Provide helper functions for:
- Scree plot (explained variance vs components)
- 2D/3D scatter plots of reduced coordinates colored by cluster
- Cluster size distribution histogram
- Representative particle images per cluster

### 6. GPU acceleration
Provide GPU-accelerated PCA via CuPy:
- `reduce_dimensions_pca_gpu()` using CuPy's SVD
- Maintain parity with CPU within tolerance
- Fall back to CPU when CuPy unavailable
- Support for large datasets via batched processing

### 7. Integration
- Export from `nanolocz.core.classification` module
- Type-annotated functions with proper error handling
- Compatible with `ParticleStack` type from NL-03/NL-10/NL-32
- Results storable in Zarr format (NL-10)

## Deliverables

### Specification
- `SPEC/NL-33-pca-hdbscan.md` (this file)

### Implementation
- `nanolocz/core/classification.py` — CPU reference implementations
- `nanolocz/gpu/classification.py` — GPU-accelerated versions

### Tests
- `tests/test_classification_nl33.py` with:
  - TestPCADimensionReduction (8+ tests)
  - TestComponentSelection (4+ tests)
  - TestHDBSCANClustering (6+ tests)
  - TestClassificationPipeline (6+ tests)
  - TestVisualizationHelpers (4+ tests)
  - TestClassificationGPU (6+ tests, skipped if CuPy unavailable)
  - TestClassificationIntegration (5+ tests)

### Documentation
- `SESSIONS/YYYY-MM-DD-NL-33.md` session handoff
- Update `STATUS.md` with completion evidence

## API Design

```python
from nanolocz.core.classification import (
    reduce_dimensions_pca,
    cluster_hdbscan,
    classify_particles,
    select_n_components,
    plot_scree,
    plot_clusters_2d,
    ClassificationResult,
)
from nanolocz.core.types import ParticleStack

# Basic classification pipeline
result = classify_particles(
    particle_stack: ParticleStack,
    n_components: int | None = None,  # Auto-select if None
    variance_threshold: float = 0.95,  # For auto-selection
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    max_components: int = 50,  # Upper bound for auto-selection
)

# Access results
labels = result.labels  # Cluster assignments (-1 for noise)
probs = result.probabilities  # Confidence scores
coords_2d = result.reduced_coords[:, :2]  # First 2 PCs for visualization
n_clusters = result.n_clusters
noise_particles = result.noise_count

# Manual two-step process
reduced = reduce_dimensions_pca(particle_stack.data, n_components=10)
variance = reduced.explained_variance_ratio_
labels, probs = cluster_hdbscan(
    reduced.coordinates,
    min_cluster_size=10,
    min_samples=5,
)

# Automatic component selection
n_comp = select_n_components(
    variance_ratios=variance,
    method='kaiser',  # or 'elbow', 'cumulative'
    cumulative_threshold=0.95,
)

# Visualization
plot_scree(result.explained_variance)
plot_clusters_2d(result.reduced_coords, result.labels, title="PCA-HDBSCAN Clustering")
```

## Data Classes

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ClassificationResult:
    """Results from particle classification pipeline."""
    labels: np.ndarray  # Shape: (n_particles,), -1 for noise
    probabilities: np.ndarray  # Shape: (n_particles,), confidence [0, 1]
    reduced_coords: np.ndarray  # Shape: (n_particles, n_components)
    explained_variance: np.ndarray  # Shape: (n_components,)
    explained_variance_ratio: np.ndarray  # Shape: (n_components,)
    n_clusters: int
    noise_count: int
    n_particles: int
    n_components: int
    
    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """Return indices of particles in specified cluster."""
        ...
    
    def get_cluster_sizes(self) -> dict[int, int]:
        """Return dictionary mapping cluster_id to particle count."""
        ...
```

## Tolerance Policy
- PCA reconstruction: rtol=1e-5, atol=1e-8 (CPU float64)
- Clustering stability: Same input → same output (deterministic with fixed seed)
- GPU parity: rtol=1e-3, atol=1e-5 (GPU float32 mode)
- Variance calculation: rtol=1e-6, atol=1e-10

## Notes
- HDBSCAN requires `hdbscan` package (add to dependencies)
- PCA uses `sklearn.decomposition.PCA` or `IncrementalPCA` for large datasets
- Default min_cluster_size should scale with dataset size (e.g., sqrt(n_particles)/10)
- Noise particles (label=-1) may represent damaged particles, aggregates, or rare states
- Classification quality depends on substack alignment quality from NL-32
- Consider implementing silhouette score or Davies-Bouldin index for cluster validation
- Future extension: hierarchical clustering for multi-resolution analysis

## Test Data Requirements
- Synthetic particle stacks with known conformations
- Real AFM data with expected number of states
- Edge cases: single cluster, all noise, very imbalanced clusters
- Large dataset for performance testing (>1000 particles)
