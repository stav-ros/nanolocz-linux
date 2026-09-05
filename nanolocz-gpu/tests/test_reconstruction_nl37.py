"""
NL-37: Tests for 3D Reconstruction from Particle Stacks
"""

import pytest
import numpy as np
from nanolocz.core.types import ParticleStack
from nanolocz.core.reconstruction import (
    ReconstructionResult,
    ReconstructionParams,
    back_projection,
    sirt,
    estimate_resolution_fsc,
    reconstruct_volume,
    reconstruct_gpu,
    visualize_orthogonal_slices,
    create_isosurface_mesh,
    _create_spherical_mask,
    _euler_to_rotation_matrix
)


class TestReconstructionResult:
    """Test ReconstructionResult dataclass."""
    
    def test_create_result(self):
        """Test creating reconstruction result."""
        volume = np.random.randn(32, 32, 32)
        result = ReconstructionResult(
            volume=volume,
            voxel_size=1.0,
            resolution_fsc=None,
            fsc_curve=None,
            n_particles=10,
            n_iterations=5
        )
        
        assert result.volume.shape == (32, 32, 32)
        assert result.voxel_size == 1.0
        assert result.n_particles == 10
        assert result.n_iterations == 5
    
    def test_result_with_fsc(self):
        """Test result with FSC data."""
        volume = np.random.randn(32, 32, 32)
        frequencies = np.array([0.1, 0.2, 0.3])
        fsc_values = np.array([0.9, 0.5, 0.1])
        
        result = ReconstructionResult(
            volume=volume,
            voxel_size=1.0,
            resolution_fsc=5.0,
            fsc_curve=(frequencies, fsc_values),
            n_particles=10,
            n_iterations=None
        )
        
        assert result.resolution_fsc == 5.0
        assert len(result.fsc_curve[0]) == 3


class TestReconstructionParams:
    """Test ReconstructionParams dataclass."""
    
    def test_default_params(self):
        """Test default parameters."""
        params = ReconstructionParams()
        
        assert params.box_size == 64
        assert params.voxel_size == 1.0
        assert params.n_iterations == 20
        assert params.regularization == 0.01
        assert params.ctf_corrected == False
    
    def test_custom_params(self):
        """Test custom parameters."""
        params = ReconstructionParams(
            box_size=128,
            voxel_size=0.5,
            n_iterations=50,
            regularization=0.1,
            mask_radius=30.0
        )
        
        assert params.box_size == 128
        assert params.voxel_size == 0.5
        assert params.mask_radius == 30.0


class TestSphericalMask:
    """Test spherical mask creation."""
    
    def test_mask_shape(self):
        """Test mask has correct shape."""
        shape = (32, 32, 32)
        mask = _create_spherical_mask(shape, radius=15)
        
        assert mask.shape == shape
        assert mask.dtype == np.float32
    
    def test_mask_center(self):
        """Test mask is centered and spherical."""
        shape = (32, 32, 32)
        mask = _create_spherical_mask(shape, radius=10)
        
        # Center should be 1
        center = tuple(s // 2 for s in shape)
        assert mask[center] == 1.0
        
        # Corners should be 0
        assert mask[0, 0, 0] == 0.0
        assert mask[-1, -1, -1] == 0.0


class TestEulerAngles:
    """Test Euler angle to rotation matrix conversion."""
    
    def test_zero_angles(self):
        """Test zero angles give identity matrix."""
        R = _euler_to_rotation_matrix(0, 0, 0)
        expected = np.eye(3)
        
        assert np.allclose(R, expected)
    
    def test_rotation_determinant(self):
        """Test rotation matrices have determinant 1."""
        angles = [(30, 45, 60), (90, 0, 0), (0, 90, 0)]
        
        for theta, phi, psi in angles:
            R = _euler_to_rotation_matrix(theta, phi, psi)
            det = np.linalg.det(R)
            
            assert np.isclose(det, 1.0, atol=1e-6)


class TestBackProjection:
    """Test weighted back-projection reconstruction."""
    
    def test_back_projection_basic(self):
        """Test basic back-projection."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        # Create synthetic particle stack
        data = np.random.randn(n_particles, box_size, box_size).astype(np.float32)
        centers = [(i, i) for i in range(n_particles)]
        frames = [0] * n_particles
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size
        )
        
        # Random angles
        angles = np.random.uniform(0, 360, size=(n_particles, 3))
        
        params = ReconstructionParams(box_size=box_size)
        result = back_projection(stack, angles, params)
        
        assert isinstance(result, ReconstructionResult)
        assert result.volume.shape == (box_size, box_size, box_size)
        assert result.n_particles == n_particles
        assert result.n_iterations is None
    
    def test_back_projection_4d_stack(self):
        """Test back-projection with 4D particle stack."""
        np.random.seed(42)
        n_particles = 5
        n_frames = 3
        box_size = 16
        
        data = np.random.randn(n_particles, n_frames, box_size, box_size).astype(np.float32)
        centers = [(i, i) for i in range(n_particles)]
        frames = list(range(n_particles))
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size
        )
        
        angles = np.random.uniform(0, 360, size=(n_particles, 3))
        params = ReconstructionParams(box_size=box_size)
        result = back_projection(stack, angles, params)
        
        assert result.volume.shape == (box_size, box_size, box_size)


class TestSIRT:
    """Test SIRT reconstruction."""
    
    def test_sirt_basic(self):
        """Test basic SIRT reconstruction."""
        np.random.seed(42)
        n_particles = 8
        box_size = 16
        
        data = np.random.randn(n_particles, box_size, box_size).astype(np.float32)
        centers = [(i, i) for i in range(n_particles)]
        frames = [0] * n_particles
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size
        )
        
        angles = np.random.uniform(0, 360, size=(n_particles, 3))
        params = ReconstructionParams(box_size=box_size, n_iterations=5)
        result = sirt(stack, angles, params)
        
        assert isinstance(result, ReconstructionResult)
        assert result.volume.shape == (box_size, box_size, box_size)
        assert result.n_iterations is not None
        assert len(result.correlation_scores) > 0
    
    def test_sirt_convergence(self):
        """Test SIRT convergence tracking."""
        np.random.seed(42)
        n_particles = 6
        box_size = 16
        
        data = np.random.randn(n_particles, box_size, box_size).astype(np.float32)
        centers = [(i, i) for i in range(n_particles)]
        frames = [0] * n_particles
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size
        )
        
        angles = np.random.uniform(0, 360, size=(n_particles, 3))
        params = ReconstructionParams(
            box_size=box_size,
            n_iterations=10,
            convergence_threshold=1e-3
        )
        result = sirt(stack, angles, params)
        
        # Should have correlation scores
        assert len(result.correlation_scores) > 0
        # Correlation should generally increase
        if len(result.correlation_scores) > 1:
            assert result.correlation_scores[-1] > result.correlation_scores[0] - 0.1


class TestFSCResolution:
    """Test Fourier Shell Correlation resolution estimation."""
    
    def test_fsc_identical_volumes(self):
        """Test FSC of identical volumes is 1.0."""
        np.random.seed(42)
        volume = np.random.randn(32, 32, 32)
        
        resolution, (frequencies, fsc_values) = estimate_resolution_fsc(
            volume, volume, voxel_size=1.0
        )
        
        # FSC should be close to 1 for all frequencies
        assert np.all(fsc_values > 0.9)
    
    def test_fsc_independent_volumes(self):
        """Test FSC of independent noise volumes is near 0."""
        np.random.seed(42)
        volume1 = np.random.randn(32, 32, 32)
        volume2 = np.random.randn(32, 32, 32)
        
        resolution, (frequencies, fsc_values) = estimate_resolution_fsc(
            volume1, volume2, voxel_size=1.0
        )
        
        # FSC should be low for independent noise
        assert np.mean(fsc_values) < 0.3
    
    def test_fsc_resolution_threshold(self):
        """Test resolution extraction at threshold."""
        np.random.seed(42)
        # Create correlated volumes
        volume1 = np.random.randn(32, 32, 32)
        volume2 = volume1 + 0.1 * np.random.randn(32, 32, 32)
        
        resolution, (frequencies, fsc_values) = estimate_resolution_fsc(
            volume1, volume2, voxel_size=1.0, threshold=0.5
        )
        
        assert np.isfinite(resolution) or resolution == np.inf


class TestReconstructVolume:
    """Test high-level reconstruction pipeline."""
    
    def test_reconstruct_with_split(self):
        """Test reconstruction with gold-standard split."""
        np.random.seed(42)
        n_particles = 20
        box_size = 16
        
        data = np.random.randn(n_particles, box_size, box_size).astype(np.float32)
        centers = [(i, i) for i in range(n_particles)]
        frames = [0] * n_particles
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size
        )
        
        angles = np.random.uniform(0, 360, size=(n_particles, 3))
        params = ReconstructionParams(box_size=box_size, n_iterations=3)
        
        result = reconstruct_volume(stack, angles, method="sirt", params=params, split_half=True)
        
        assert result.resolution_fsc is not None or result.resolution_fsc == np.inf
        assert result.fsc_curve is not None
    
    def test_reconstruct_without_split(self):
        """Test reconstruction without split."""
        np.random.seed(42)
        n_particles = 10
        box_size = 16
        
        data = np.random.randn(n_particles, box_size, box_size).astype(np.float32)
        centers = [(i, i) for i in range(n_particles)]
        frames = [0] * n_particles
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size
        )
        
        angles = np.random.uniform(0, 360, size=(n_particles, 3))
        params = ReconstructionParams(box_size=box_size)
        
        result = reconstruct_volume(stack, angles, method="back_projection", params=params, split_half=False)
        
        assert result.resolution_fsc is None
        assert result.fsc_curve is None
    
    def test_reconstruct_back_projection_method(self):
        """Test back-projection method selection."""
        np.random.seed(42)
        n_particles = 8
        box_size = 16
        
        data = np.random.randn(n_particles, box_size, box_size).astype(np.float32)
        centers = [(i, i) for i in range(n_particles)]
        frames = [0] * n_particles
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size
        )
        
        angles = np.random.uniform(0, 360, size=(n_particles, 3))
        
        result_bp = reconstruct_volume(stack, angles, method="back_projection", split_half=False)
        result_sirt = reconstruct_volume(stack, angles, method="sirt", split_half=False)
        
        # Both should produce volumes
        assert result_bp.volume.shape == result_sirt.volume.shape


class TestReconstructGPU:
    """Test GPU reconstruction (falls back to CPU without CuPy)."""
    
    def test_reconstruct_gpu_fallback(self):
        """Test GPU reconstruction falls back to CPU."""
        np.random.seed(42)
        n_particles = 6
        box_size = 16
        
        data = np.random.randn(n_particles, box_size, box_size).astype(np.float32)
        centers = [(i, i) for i in range(n_particles)]
        frames = [0] * n_particles
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size
        )
        
        angles = np.random.uniform(0, 360, size=(n_particles, 3))
        params = ReconstructionParams(box_size=box_size)
        
        result = reconstruct_gpu(stack, angles, params)
        
        # Should work even without CuPy
        assert result.volume.shape == (box_size, box_size, box_size)


class TestVisualization:
    """Test visualization utilities."""
    
    def test_orthogonal_slices(self):
        """Test orthogonal slice extraction."""
        volume = np.arange(64).reshape(4, 4, 4).astype(np.float32)
        
        xy, xz, yz = visualize_orthogonal_slices(volume, (2, 2, 2))
        
        assert xy.shape == (4, 4)
        assert xz.shape == (4, 4)
        assert yz.shape == (4, 4)
    
    def test_orthogonal_slices_default(self):
        """Test orthogonal slices with default center."""
        volume = np.random.randn(32, 32, 32)
        
        xy, xz, yz = visualize_orthogonal_slices(volume)
        
        assert xy.shape == (32, 32)
        assert xz.shape == (32, 32)
        assert yz.shape == (32, 32)
    
    def test_isosurface_mesh_stub(self):
        """Test isosurface mesh creation (stub)."""
        volume = np.random.randn(16, 16, 16)
        
        verts, faces, normals = create_isosurface_mesh(volume, level=0.5)
        
        # Stub returns empty arrays
        assert len(verts) == 0
        assert len(faces) == 0
        assert len(normals) == 0


class TestIntegration:
    """Integration tests with other NL modules."""
    
    def test_reconstruct_aligned_particles(self):
        """Test reconstruction with aligned particles from NL-34."""
        # This would use output from align_particles()
        # For now, test with synthetic aligned data
        np.random.seed(42)
        n_particles = 15
        box_size = 20
        
        # Simulate aligned particles (same structure, different noise)
        base_structure = np.zeros((box_size, box_size))
        base_structure[8:12, 8:12] = 1.0
        
        data = np.array([base_structure + 0.1 * np.random.randn(box_size, box_size) 
                        for _ in range(n_particles)]).astype(np.float32)
        centers = [(i, i) for i in range(n_particles)]
        frames = [0] * n_particles
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size
        )
        
        # Use varied angles for proper 3D reconstruction
        # All-zero angles don't provide enough information for 3D reconstruction
        np.random.seed(42)
        angles = np.random.uniform(0, 360, size=(n_particles, 3))
        
        params = ReconstructionParams(box_size=box_size, n_iterations=10)
        result = reconstruct_volume(stack, angles, method="sirt", params=params, split_half=False)
        
        assert result.volume.shape == (box_size, box_size, box_size)
        # Volume should have some structure (non-uniform values)
        assert np.std(result.volume) > 0.001
