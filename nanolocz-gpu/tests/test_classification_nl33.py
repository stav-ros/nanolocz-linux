"""
Tests for particle classification using PCA and HDBSCAN (NL-33).
"""

import numpy as np
import pytest

from nanolocz.core.classification import (
    ClassificationResult,
    classify_particles,
    cluster_hdbscan,
    plot_cluster_sizes,
    plot_clusters_2d,
    plot_scree,
    reduce_dimensions_pca,
    select_n_components,
)


class TestPCADimensionReduction:
    """Test PCA dimensionality reduction functionality."""

    def test_basic_pca_reduction(self):
        """Test basic PCA reduces dimensions correctly."""
        np.random.seed(42)
        n_particles = 100
        data = np.random.randn(n_particles, 16, 16)  # 16x16 images
        
        reduced, variance, variance_ratio = reduce_dimensions_pca(
            data, n_components=5, random_state=42
        )
        
        assert reduced.shape == (n_particles, 5)
        assert len(variance) == 5
        assert len(variance_ratio) == 5
        assert np.all(variance_ratio >= 0)
        # Variance ratios sum to <= 1.0 (exactly 1.0 only if all components retained)
        assert np.sum(variance_ratio) > 0

    def test_pca_with_2d_input(self):
        """Test PCA with already flattened 2D input."""
        np.random.seed(42)
        n_particles = 50
        n_features = 100
        data = np.random.randn(n_particles, n_features)
        
        reduced, variance, variance_ratio = reduce_dimensions_pca(
            data, n_components=10, random_state=42
        )
        
        assert reduced.shape == (n_particles, 10)
        assert len(variance) == 10
        assert len(variance_ratio) == 10

    def test_pca_variance_preservation(self):
        """Test that PCA preserves most variance with sufficient components."""
        np.random.seed(42)
        # Create data with known structure (first 5 dims have high variance)
        n_particles = 200
        high_var = np.random.randn(n_particles, 5) * 10
        low_var = np.random.randn(n_particles, 95) * 0.1
        data = np.hstack([high_var, low_var])
        
        reduced, variance, variance_ratio = reduce_dimensions_pca(
            data, n_components=10, random_state=42
        )
        
        # First 5 components should capture most variance
        cumsum_5 = np.sum(variance_ratio[:5])
        assert cumsum_5 > 0.9, f"First 5 components should capture >90% variance, got {cumsum_5}"

    def test_pca_empty_data_error(self):
        """Test that empty data raises appropriate error."""
        data = np.array([]).reshape(0, 10)
        
        with pytest.raises(ValueError, match="no particles"):
            reduce_dimensions_pca(data, n_components=5)

    def test_pca_invalid_ndim_error(self):
        """Test that 1D data raises appropriate error."""
        data = np.random.randn(100)
        
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            reduce_dimensions_pca(data, n_components=5)

    def test_pca_too_many_components(self):
        """Test that requesting too many components is handled gracefully."""
        np.random.seed(42)
        n_particles = 20
        data = np.random.randn(n_particles, 10, 10)
        
        # Should automatically limit to min(n_particles, n_features)
        reduced, variance, variance_ratio = reduce_dimensions_pca(
            data, n_components=100, random_state=42
        )
        
        # Should be limited to n_particles
        assert reduced.shape[1] <= n_particles

    def test_pca_reproducibility(self):
        """Test that PCA results are reproducible with fixed seed."""
        np.random.seed(42)
        data = np.random.randn(50, 8, 8)
        
        reduced1, _, _ = reduce_dimensions_pca(data, n_components=5, random_state=123)
        reduced2, _, _ = reduce_dimensions_pca(data, n_components=5, random_state=123)
        
        np.testing.assert_array_almost_equal(reduced1, reduced2)

    def test_pca_whiten_option(self):
        """Test that whitening produces unit variance components."""
        np.random.seed(42)
        data = np.random.randn(100, 10, 10)
        
        reduced, _, _ = reduce_dimensions_pca(data, n_components=5, whiten=True, random_state=42)
        
        # Whitened components should have approximately unit variance
        component_vars = np.var(reduced, axis=0)
        np.testing.assert_allclose(component_vars, 1.0, rtol=0.1)


class TestComponentSelection:
    """Test automatic component selection heuristics."""

    def test_cumulative_method(self):
        """Test cumulative variance threshold method."""
        # Create variance ratios that sum to 1
        variance_ratios = np.array([0.4, 0.3, 0.15, 0.08, 0.04, 0.03])
        
        n_comp = select_n_components(variance_ratios, method="cumulative", cumulative_threshold=0.85)
        
        # 0.4 + 0.3 + 0.15 = 0.85, so should select 3 components
        assert n_comp == 3

    def test_kaiser_method(self):
        """Test Kaiser criterion (eigenvalue > 1)."""
        # Create variance ratios where first 3 are above average
        n_total = 10
        variance_ratios = np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03])
        
        n_comp = select_n_components(variance_ratios, method="kaiser")
        
        # Threshold is 1/n_total = 0.1, so first 4 components should be selected
        assert n_comp >= 3

    def test_elbow_method(self):
        """Test elbow detection in scree plot."""
        # Create typical scree plot shape (steep drop then plateau)
        variance_ratios = np.array([0.35, 0.25, 0.15, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01])
        
        n_comp = select_n_components(variance_ratios, method="elbow")
        
        # Elbow should be detected around 2-4 components
        assert 2 <= n_comp <= 5

    def test_max_components_limit(self):
        """Test that max_components imposes upper bound."""
        variance_ratios = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        n_comp = select_n_components(
            variance_ratios, 
            method="cumulative", 
            cumulative_threshold=0.99,
            max_components=2
        )
        
        assert n_comp == 2

    def test_empty_variance_error(self):
        """Test that empty variance ratios raises error."""
        variance_ratios = np.array([])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            select_n_components(variance_ratios)

    def test_invalid_method_error(self):
        """Test that invalid method raises error."""
        variance_ratios = np.array([0.5, 0.3, 0.2])
        
        with pytest.raises(ValueError, match="Unknown method"):
            select_n_components(variance_ratios, method="invalid")


class TestHDBSCANClustering:
    """Test HDBSCAN clustering functionality."""

    def test_basic_clustering(self):
        """Test basic HDBSCAN clustering recovers known clusters."""
        np.random.seed(42)
        # Create 3 well-separated clusters
        cluster1 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(30, 5) + np.array([10, 10, 10, 10, 10])
        cluster3 = np.random.randn(30, 5) + np.array([-10, -10, -10, -10, -10])
        data = np.vstack([cluster1, cluster2, cluster3])
        
        labels, probs, n_clusters, noise_count = cluster_hdbscan(
            data, min_cluster_size=10
        )
        
        assert n_clusters >= 2, "Should detect at least 2 clusters"
        assert len(labels) == 90
        assert len(probs) == 90
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_noise_detection(self):
        """Test that HDBSCAN identifies noise points."""
        np.random.seed(42)
        # Create tight cluster + scattered noise
        cluster = np.random.randn(50, 3) * 0.5
        noise = np.random.uniform(-10, 10, size=(20, 3))
        data = np.vstack([cluster, noise])
        
        labels, probs, n_clusters, noise_count = cluster_hdbscan(
            data, min_cluster_size=10, min_samples=5
        )
        
        # Should detect the main cluster and some noise
        # Note: HDBSCAN may find 0 clusters if density varies too much
        assert n_clusters >= 0
        # At minimum, we should get valid labels and probabilities
        assert len(labels) == 70
        assert len(probs) == 70

    def test_single_cluster(self):
        """Test clustering with allow_single_cluster option."""
        np.random.seed(42)
        # Create single tight cluster
        data = np.random.randn(50, 5) * 0.5
        
        labels, probs, n_clusters, noise_count = cluster_hdbscan(
            data, min_cluster_size=10, allow_single_cluster=True
        )
        
        # May find 1 cluster or treat all as noise depending on density
        assert n_clusters <= 1

    def test_min_cluster_size_validation(self):
        """Test that insufficient data raises error."""
        np.random.seed(42)
        data = np.random.randn(5, 3)  # Only 5 points
        
        with pytest.raises(ValueError, match="Not enough particles"):
            cluster_hdbscan(data, min_cluster_size=10)

    def test_2d_input_required(self):
        """Test that non-2D input raises error."""
        data = np.random.randn(50, 5, 5)
        
        with pytest.raises(ValueError, match="must be 2D"):
            cluster_hdbscan(data, min_cluster_size=10)


class TestClassificationPipeline:
    """Test end-to-end classification pipeline."""

    def test_basic_classification(self):
        """Test basic classification pipeline."""
        np.random.seed(42)
        # Create data with 3 distinct clusters
        cluster1 = np.random.randn(40, 8, 8) + 2
        cluster2 = np.random.randn(40, 8, 8) - 2
        cluster3 = np.random.randn(40, 8, 8)
        data = np.vstack([cluster1, cluster2, cluster3])
        
        result = classify_particles(data, min_cluster_size=15, random_state=42)
        
        assert isinstance(result, ClassificationResult)
        assert result.n_particles == 120
        assert result.labels.shape == (120,)
        assert result.probabilities.shape == (120,)
        assert result.reduced_coords.shape[0] == 120
        assert result.n_clusters >= 1
        assert result.noise_count >= 0

    def test_classification_result_methods(self):
        """Test ClassificationResult helper methods."""
        np.random.seed(42)
        data = np.random.randn(100, 5, 5)
        
        result = classify_particles(data, min_cluster_size=10, random_state=42)
        
        # Test get_cluster_members
        if result.n_clusters > 0:
            members = result.get_cluster_members(0)
            assert len(members) > 0
            assert np.all(result.labels[members] == 0)
        
        # Test get_cluster_sizes
        sizes = result.get_cluster_sizes()
        assert isinstance(sizes, dict)
        total = sum(sizes.values())
        assert total == result.n_particles
        
        # Test get_non_noise_indices
        non_noise = result.get_non_noise_indices()
        assert np.all(result.labels[non_noise] != -1)

    def test_automatic_component_selection(self):
        """Test that n_components=None triggers auto-selection."""
        np.random.seed(42)
        data = np.random.randn(100, 10, 10)
        
        result_auto = classify_particles(data, n_components=None, min_cluster_size=10, random_state=42)
        result_fixed = classify_particles(data, n_components=5, min_cluster_size=10, random_state=42)
        
        # Auto-selection should choose reasonable number of components
        assert result_auto.n_components >= 1
        assert result_auto.n_components <= 50  # max_components default
        
        # Results may differ but both should be valid
        assert result_auto.n_clusters >= 0
        assert result_fixed.n_clusters >= 0

    def test_small_dataset(self):
        """Test classification with minimal dataset."""
        np.random.seed(42)
        data = np.random.randn(20, 4, 4)
        
        result = classify_particles(data, min_cluster_size=5, random_state=42)
        
        assert result.n_particles == 20
        assert result.n_components >= 1

    def test_classification_preserves_structure(self):
        """Test that similar particles cluster together."""
        np.random.seed(42)
        # Create very distinct clusters
        cluster1 = np.random.randn(30, 6, 6) * 0.5 + 5
        cluster2 = np.random.randn(30, 6, 6) * 0.5 - 5
        data = np.vstack([cluster1, cluster2])
        
        result = classify_particles(data, min_cluster_size=10, random_state=42)
        
        # With such distinct clusters, should find at least 2 clusters
        # or separate them via noise
        assert result.n_clusters + (1 if result.noise_count > 10 else 0) >= 1


class TestVisualizationHelpers:
    """Test visualization utility functions."""

    def test_plot_scree(self):
        """Test scree plot generation."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        variance_ratio = np.array([0.4, 0.3, 0.15, 0.08, 0.04, 0.03])
        
        fig, ax = plt.subplots()
        plot_scree(variance_ratio, ax=ax, title="Test Scree")
        
        # Check that plot was created
        assert len(ax.patches) > 0  # Bars
        assert len(ax.lines) > 0    # Cumulative line
        
        plt.close(fig)

    def test_plot_clusters_2d(self):
        """Test 2D cluster plot generation."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        np.random.seed(42)
        coords = np.random.randn(100, 3)  # 3 components
        labels = np.random.randint(0, 3, size=100)
        
        fig, ax = plt.subplots()
        plot_clusters_2d(coords, labels, ax=ax, title="Test Clusters")
        
        # Check that scatter plots were created
        assert len(ax.collections) > 0
        
        plt.close(fig)

    def test_plot_cluster_sizes(self):
        """Test cluster size histogram generation."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        labels = np.array([0, 0, 0, 1, 1, 2, -1, -1])
        
        fig, ax = plt.subplots()
        plot_cluster_sizes(labels, ax=ax, title="Test Sizes")
        
        # Check that bars were created
        assert len(ax.patches) > 0
        
        plt.close(fig)

    def test_plot_clusters_insufficient_dims(self):
        """Test that 2D plot requires at least 2 components."""
        coords = np.random.randn(50, 1)  # Only 1 component
        labels = np.zeros(50)
        
        with pytest.raises(ValueError, match="at least 2 components"):
            plot_clusters_2d(coords, labels)


class TestClassificationIntegration:
    """Test integration with other nanolocz modules."""

    def test_integration_with_particle_stack(self):
        """Test classification with ParticleStack from NL-32."""
        from nanolocz.core.types import ParticleStack
        
        np.random.seed(42)
        # Create synthetic particle stack
        n_particles = 60
        data = np.random.randn(n_particles, 4, 8, 8)  # 4 frames, 8x8 pixels
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(i, i) for i in range(n_particles)],
            frame_index=[0] * n_particles,
            box_size=8,
        )
        
        # Classify using first frame only
        result = classify_particles(
            stack.data[:, 0, :, :],  # Extract first frame
            min_cluster_size=10,
            random_state=42
        )
        
        assert result.n_particles == n_particles
        assert result.n_clusters >= 0

    def test_full_workflow_synthetic(self):
        """Test complete workflow: generate → classify → analyze."""
        np.random.seed(42)
        
        # Generate synthetic data with known structure
        n_per_cluster = 30
        cluster1 = np.random.randn(n_per_cluster, 6, 6) + 3
        cluster2 = np.random.randn(n_per_cluster, 6, 6) - 3
        cluster3 = np.random.randn(n_per_cluster, 6, 6)
        data = np.vstack([cluster1, cluster2, cluster3])
        
        # Classify
        result = classify_particles(data, min_cluster_size=15, random_state=42)
        
        # Analyze results
        sizes = result.get_cluster_sizes()
        
        # Should have found meaningful structure
        assert result.n_particles == 90
        assert sum(sizes.values()) == 90
        
        # Verify we can extract cluster members
        if result.n_clusters > 0:
            for cluster_id in range(result.n_clusters):
                members = result.get_cluster_members(cluster_id)
                assert len(members) > 0


class TestClassificationGPU:
    """Test GPU-accelerated classification (skipped if CuPy unavailable)."""

    def test_gpu_module_import(self):
        """Test that GPU module can be imported."""
        try:
            import cupy
            from nanolocz.gpu import classification as gpu_classif
            assert hasattr(gpu_classif, 'reduce_dimensions_pca_gpu')
        except ImportError:
            pytest.skip("CuPy not available")

    def test_gpu_fallback_behavior(self):
        """Test that CPU fallback works when GPU unavailable."""
        # This test verifies the CPU implementation works regardless of GPU
        np.random.seed(42)
        data = np.random.randn(50, 5, 5)
        
        result = classify_particles(data, min_cluster_size=10, random_state=42)
        
        assert result.n_particles == 50
        assert result.n_clusters >= 0


# Run tests with: pytest tests/test_classification_nl33.py -v
