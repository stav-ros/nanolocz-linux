"""
Tests for NL-03 Typed Contracts.

This module validates that all typed contracts, protocols, and data classes
defined in nanolocz.core.types function correctly and maintain type safety.
"""

import pytest
import numpy as np
from typing import runtime_checkable
from nanolocz.core.types import (
    # Data classes
    Meta, Frame, Localizations, ParticleStack,
    DetectionParams, DetectionResult, DetectionMethod,
    LocalizationParams, LocalizedParticle, LocalizationModel,
    TrackParams, ParticleTrack,
    ImageMetadata, FileFormat,
    GPUConfig, DeviceType, ProcessingContext,
    AnalysisPipeline,
    # Protocols
    Detector, Localizer, Tracker, FileReader, FileWriter,
)


class TestMeta:
    """Test Meta dataclass."""
    
    def test_create_meta_minimal(self):
        """Test creating Meta with minimal required fields."""
        meta = Meta(pixel_size=(1.0, 1.0))
        assert meta.pixel_size == (1.0, 1.0)
        assert meta.height_unit == "nm"
        assert meta.channel == "height"
    
    def test_create_meta_full(self):
        """Test creating Meta with all fields."""
        meta = Meta(
            pixel_size=(0.5, 0.5),
            height_unit="um",
            channel="phase",
            scan_size=(10.0, 10.0),
            scan_rate=1.0,
            lines=512,
            samples_per_line=512
        )
        assert meta.pixel_size == (0.5, 0.5)
        assert meta.height_unit == "um"
        assert meta.channel == "phase"
        assert meta.scan_size == (10.0, 10.0)
        assert meta.scan_rate == 1.0
        assert meta.lines == 512
        assert meta.samples_per_line == 512


class TestFrame:
    """Test Frame dataclass."""
    
    def test_create_frame(self):
        """Test creating a Frame with valid data."""
        data = np.random.rand(100, 100)
        meta = Meta(pixel_size=(1.0, 1.0))
        frame = Frame(data=data, meta=meta, frame_index=0, timestamp=0.0)
        
        assert frame.shape == (100, 100)
        assert frame.dtype == np.float64
        assert frame.frame_index == 0
        assert frame.timestamp == 0.0
    
    def test_frame_rejects_3d_data(self):
        """Test that Frame rejects 3D data."""
        data = np.random.rand(10, 100, 100)
        meta = Meta(pixel_size=(1.0, 1.0))
        
        with pytest.raises(ValueError, match="Frame data must be 2D"):
            Frame(data=data, meta=meta)
    
    def test_frame_requires_metadata(self):
        """Test that Frame requires metadata."""
        data = np.random.rand(100, 100)
        
        with pytest.raises(ValueError, match="Frame must have metadata"):
            Frame(data=data, meta=None)
    
    def test_frame_converts_list_to_array(self):
        """Test that Frame converts list input to numpy array."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        meta = Meta(pixel_size=(1.0, 1.0))
        frame = Frame(data=data, meta=meta)
        
        assert isinstance(frame.data, np.ndarray)
        assert frame.shape == (2, 2)


class TestLocalizations:
    """Test Localizations dataclass."""
    
    def test_create_localizations_minimal(self):
        """Test creating Localizations with minimal fields."""
        locs = Localizations(
            xy=[(1.0, 2.0), (3.0, 4.0)],
            frame_index=[0, 1]
        )
        assert len(locs) == 2
        assert locs.xy == [(1.0, 2.0), (3.0, 4.0)]
        assert locs.frame_index == [0, 1]
    
    def test_create_localizations_full(self):
        """Test creating Localizations with all fields."""
        locs = Localizations(
            xy=[(1.0, 2.0), (3.0, 4.0)],
            frame_index=[0, 1],
            intensities=[100.0, 200.0],
            sigmas=[(0.5, 0.6), (0.7, 0.8)]
        )
        assert len(locs) == 2
        assert locs.intensities == [100.0, 200.0]
        assert locs.sigmas == [(0.5, 0.6), (0.7, 0.8)]
    
    def test_localizations_length_mismatch(self):
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError, match="xy and frame_index must have same length"):
            Localizations(
                xy=[(1.0, 2.0)],
                frame_index=[0, 1]
            )
    
    def test_localizations_to_array(self):
        """Test conversion to numpy array."""
        locs = Localizations(
            xy=[(1.0, 2.0), (3.0, 4.0)],
            frame_index=[0, 1]
        )
        arr = locs.to_array()
        
        assert arr.shape == (2, 3)
        assert np.allclose(arr[:, 0], [0, 1])  # frames
        assert np.allclose(arr[:, 1], [1.0, 3.0])  # x
        assert np.allclose(arr[:, 2], [2.0, 4.0])  # y


class TestLocalizedParticle:
    """Test LocalizedParticle dataclass."""
    
    def test_create_particle_minimal(self):
        """Test creating particle with minimal fields."""
        particle = LocalizedParticle(
            x=1.5, y=2.5, intensity=100.0
        )
        assert particle.x == 1.5
        assert particle.y == 2.5
        assert particle.intensity == 100.0
        assert particle.sigma_x is None
        assert particle.sigma_y is None
        assert particle.background == 0.0
        assert np.isnan(particle.chi_squared)
        assert particle.frame == 0
    
    def test_create_particle_full(self):
        """Test creating particle with all fields."""
        particle = LocalizedParticle(
            x=1.5, y=2.5, intensity=100.0,
            sigma_x=0.5, sigma_y=0.6,
            background=10.0, chi_squared=0.01,
            frame=5
        )
        assert particle.sigma_x == 0.5
        assert particle.sigma_y == 0.6
        assert particle.background == 10.0
        assert particle.chi_squared == 0.01
        assert particle.frame == 5
    
    def test_particle_position_property(self):
        """Test position property returns numpy array."""
        particle = LocalizedParticle(x=1.5, y=2.5, intensity=100.0)
        pos = particle.position
        
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (2,)
        assert pos[0] == 1.5
        assert pos[1] == 2.5
    
    def test_particle_sigma_property_with_sigmas(self):
        """Test sigma property when sigmas are available."""
        particle = LocalizedParticle(
            x=1.5, y=2.5, intensity=100.0,
            sigma_x=0.5, sigma_y=0.6
        )
        sigma = particle.sigma
        
        assert isinstance(sigma, np.ndarray)
        assert sigma.shape == (2,)
        assert sigma[0] == 0.5
        assert sigma[1] == 0.6
    
    def test_particle_sigma_property_without_sigmas(self):
        """Test sigma property when sigmas are not available."""
        particle = LocalizedParticle(x=1.5, y=2.5, intensity=100.0)
        assert particle.sigma is None


class TestParticleTrack:
    """Test ParticleTrack dataclass."""
    
    def test_create_track_single_particle(self):
        """Test creating track with single particle."""
        particle = LocalizedParticle(x=1.0, y=1.0, intensity=100.0, frame=0)
        track = ParticleTrack(track_id=0, particles=[particle])
        
        assert track.track_id == 0
        assert len(track.particles) == 1
        assert track.duration == 1
        assert track.mean_displacement == 0.0
        assert len(track.displacements) == 0
    
    def test_create_track_multiple_particles(self):
        """Test creating track with multiple particles."""
        particles = [
            LocalizedParticle(x=1.0, y=1.0, intensity=100.0, frame=0),
            LocalizedParticle(x=2.0, y=2.0, intensity=100.0, frame=1),
            LocalizedParticle(x=3.0, y=3.0, intensity=100.0, frame=2),
        ]
        track = ParticleTrack(track_id=0, particles=particles)
        
        assert track.track_id == 0
        assert len(track.particles) == 3
        assert track.duration == 3
        assert len(track.frames) == 3
        assert len(track.displacements) == 2
        
        # Displacements should be sqrt(2) ≈ 1.414 for diagonal movement
        expected_disp = np.sqrt(2.0)
        assert np.allclose(track.displacements, [expected_disp, expected_disp])
    
    def test_track_mean_displacement(self):
        """Test mean displacement calculation."""
        particles = [
            LocalizedParticle(x=0.0, y=0.0, intensity=100.0, frame=0),
            LocalizedParticle(x=3.0, y=4.0, intensity=100.0, frame=1),  # 5 pixels away
        ]
        track = ParticleTrack(track_id=0, particles=particles)
        
        assert track.mean_displacement == 5.0


class TestGPUConfig:
    """Test GPUConfig dataclass."""
    
    def test_create_config_default(self):
        """Test creating GPUConfig with defaults."""
        config = GPUConfig()
        assert config.device_type == DeviceType.AUTO
        assert config.device_id is None
        assert config.memory_limit is None
        assert config.batch_size == 64
    
    def test_resolve_device_cpu(self):
        """Test device resolution when CPU is specified."""
        config = GPUConfig(device_type=DeviceType.CPU)
        assert config.resolve_device() == DeviceType.CPU
    
    def test_resolve_device_auto_fallback(self):
        """Test that AUTO falls back to CPU when CuPy unavailable."""
        config = GPUConfig(device_type=DeviceType.AUTO)
        device = config.resolve_device()
        # Should be CPU since CuPy is not installed in test environment
        assert device == DeviceType.CPU


class TestProcessingContext:
    """Test ProcessingContext for GPU/CPU management."""
    
    def test_create_context_cpu(self):
        """Test creating context for CPU execution."""
        config = GPUConfig(device_type=DeviceType.CPU)
        ctx = ProcessingContext(config=config)
        
        assert ctx.xp.__name__ == 'numpy'
        assert ctx.device_id is None
    
    def test_context_allocate(self):
        """Test array allocation on current device."""
        config = GPUConfig(device_type=DeviceType.CPU)
        ctx = ProcessingContext(config=config)
        
        arr = ctx.allocate((10, 10), dtype=np.float32)
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 10)
        assert arr.dtype == np.float32
        assert np.all(arr == 0.0)
    
    def test_context_to_device_cpu(self):
        """Test transferring array to CPU device."""
        config = GPUConfig(device_type=DeviceType.CPU)
        ctx = ProcessingContext(config=config)
        
        arr = np.array([1.0, 2.0, 3.0])
        result = ctx.to_device(arr)
        
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, arr)
    
    def test_context_to_cpu(self):
        """Test transferring array to CPU."""
        config = GPUConfig(device_type=DeviceType.CPU)
        ctx = ProcessingContext(config=config)
        
        arr = np.array([1.0, 2.0, 3.0])
        result = ctx.to_cpu(arr)
        
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, arr)


class TestDetectorProtocol:
    """Test Detector protocol compliance."""
    
    def test_protocol_recognition(self):
        """Test that classes implementing Detector are recognized."""
        from nanolocz.core.types import DetectionParams, DetectionResult, Image2D
        
        class MockDetector:
            def __init__(self, params: DetectionParams):
                self.params = params
            
            def detect(self, image: Image2D) -> DetectionResult:
                # Minimal implementation
                return DetectionResult(
                    coordinates=np.array([[1.0, 2.0]]),
                    intensities=np.array([100.0]),
                    scores=np.array([0.9])
                )
        
        detector = MockDetector(DetectionParams(threshold=0.5))
        assert isinstance(detector, Detector)
    
    def test_detect_method_signature(self):
        """Test that detect method has correct signature."""
        from nanolocz.core.types import DetectionParams, DetectionResult, Image2D
        
        class MockDetector:
            def detect(self, image: Image2D) -> DetectionResult:
                pass
        
        # Check that the method exists and is callable
        assert hasattr(MockDetector, 'detect')
        assert callable(getattr(MockDetector, 'detect'))


class TestLocalizerProtocol:
    """Test Localizer protocol compliance."""
    
    def test_protocol_recognition(self):
        """Test that classes implementing Localizer are recognized."""
        from nanolocz.core.types import LocalizationParams, LocalizedParticle, Image2D
        
        class MockLocalizer:
            def __init__(self, params: LocalizationParams = None):
                self.params = params
            
            def localize(self, image_patch: Image2D, 
                        initial_guess: tuple[float, float]) -> LocalizedParticle:
                return LocalizedParticle(
                    x=initial_guess[0],
                    y=initial_guess[1],
                    intensity=100.0
                )
        
        localizer = MockLocalizer(LocalizationParams())
        assert isinstance(localizer, Localizer)


class TestTrackerProtocol:
    """Test Tracker protocol compliance."""
    
    def test_protocol_recognition(self):
        """Test that classes implementing Tracker are recognized."""
        from nanolocz.core.types import TrackParams, ParticleTrack, LocalizedParticle
        
        class MockTracker:
            def __init__(self, params: TrackParams = None):
                self.params = params
            
            def track(self, localizations, n_frames: int):
                return []
        
        tracker = MockTracker(TrackParams())
        assert isinstance(tracker, Tracker)


class TestDetectionResult:
    """Test DetectionResult dataclass."""
    
    def test_create_detection_result(self):
        """Test creating detection result."""
        result = DetectionResult(
            coordinates=np.array([[1.0, 2.0], [3.0, 4.0]]),
            intensities=np.array([100.0, 200.0]),
            scores=np.array([0.9, 0.8])
        )
        
        assert result.coordinates.shape == (2, 2)
        assert len(result.intensities) == 2
        assert len(result.scores) == 2
    
    def test_detection_result_invalid_coordinates_shape(self):
        """Test that invalid coordinates shape raises error."""
        with pytest.raises(ValueError, match="coordinates must be shape"):
            DetectionResult(
                coordinates=np.array([1.0, 2.0]),  # 1D instead of 2D
                intensities=np.array([100.0]),
                scores=np.array([0.9])
            )
    
    def test_detection_result_length_mismatch(self):
        """Test that length mismatches raise errors."""
        coords = np.array([[1.0, 2.0]])
        
        with pytest.raises(ValueError, match="intensities length must match"):
            DetectionResult(
                coordinates=coords,
                intensities=np.array([100.0, 200.0]),
                scores=np.array([0.9])
            )
        
        with pytest.raises(ValueError, match="scores length must match"):
            DetectionResult(
                coordinates=coords,
                intensities=np.array([100.0]),
                scores=np.array([0.9, 0.8])
            )


class TestAnalysisPipeline:
    """Test AnalysisPipeline dataclass."""
    
    def test_create_pipeline_empty(self):
        """Test creating pipeline with empty results."""
        meta = ImageMetadata(width=100, height=100)
        pipeline = AnalysisPipeline(
            detections=[],
            localizations=[],
            tracks=[],
            metadata=meta
        )
        
        assert pipeline.total_particles == 0
        assert pipeline.total_tracks == 0
    
    def test_create_pipeline_with_results(self):
        """Test creating pipeline with actual results."""
        meta = ImageMetadata(width=100, height=100)
        
        # Create some mock data
        det_result = DetectionResult(
            coordinates=np.array([[1.0, 2.0]]),
            intensities=np.array([100.0]),
            scores=np.array([0.9])
        )
        
        particle = LocalizedParticle(x=1.0, y=2.0, intensity=100.0, frame=0)
        
        pipeline = AnalysisPipeline(
            detections=[det_result],
            localizations=[[particle]],
            tracks=[],
            metadata=meta
        )
        
        assert pipeline.total_particles == 1
        assert pipeline.total_tracks == 0
    
    def test_pipeline_to_dataframe_requires_pandas(self):
        """Test that DataFrame conversion requires pandas."""
        meta = ImageMetadata(width=100, height=100)
        pipeline = AnalysisPipeline(
            detections=[],
            localizations=[],
            tracks=[],
            metadata=meta
        )
        
        # Try without pandas installed (should fail gracefully)
        try:
            import pandas
            has_pandas = True
        except ImportError:
            has_pandas = False
        
        if not has_pandas:
            with pytest.raises(ImportError, match="pandas required"):
                pipeline.to_dataframe()


class TestEnumTypes:
    """Test enumeration types."""
    
    def test_detection_method_enum(self):
        """Test DetectionMethod enum values."""
        assert DetectionMethod.FAST_PEAKS.value == "fast_peaks"
        assert DetectionMethod.LAPLACIAN.value == "laplacian"
        assert DetectionMethod.WAVELET.value == "wavelet"
        assert DetectionMethod.CUSTOM.value == "custom"
    
    def test_localization_model_enum(self):
        """Test LocalizationModel enum values."""
        assert LocalizationModel.GAUSSIAN_2D.value == "gaussian_2d"
        assert LocalizationModel.SPHERE.value == "sphere"
        assert LocalizationModel.ELLIPSE.value == "ellipse"
        assert LocalizationModel.INTERPOLATED.value == "interpolated"
    
    def test_device_type_enum(self):
        """Test DeviceType enum values."""
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.GPU.value == "gpu"
        assert DeviceType.AUTO.value == "auto"
    
    def test_file_format_enum(self):
        """Test FileFormat enum values."""
        assert FileFormat.TIFF.value == "tiff"
        assert FileFormat.HDF5.value == "hdf5"
        assert FileFormat.NPZ.value == "npz"
        assert FileFormat.MAT.value == "mat"
        assert FileFormat.UNKNOWN.value == "unknown"


class TestParticleStack:
    """Test ParticleStack dataclass."""
    
    def test_create_stack_3d(self):
        """Test creating 3D particle stack."""
        data = np.random.rand(5, 7, 7)  # 5 particles, 7x7 boxes
        stack = ParticleStack(
            data=data,
            centers_xy=[(1.0, 2.0)] * 5,
            frame_index=[0, 0, 1, 1, 2],
            box_size=7
        )
        
        assert stack.n_particles == 5
        assert len(stack.centers_xy) == 5
        assert len(stack.frame_index) == 5
    
    def test_create_stack_4d(self):
        """Test creating 4D particle stack (particles x frames x h x w)."""
        data = np.random.rand(3, 2, 7, 7)  # 3 particles, 2 frames each
        stack = ParticleStack(
            data=data,
            centers_xy=[(1.0, 2.0)] * 3,
            frame_index=[0, 1, 2],
            box_size=7
        )
        
        assert stack.n_particles == 3
        assert stack.data.ndim == 4
    
    def test_stack_length_mismatch(self):
        """Test that centers/frame_index length mismatch raises error."""
        data = np.random.rand(5, 7, 7)
        
        with pytest.raises(ValueError, match="centers_xy and frame_index must have same length"):
            ParticleStack(
                data=data,
                centers_xy=[(1.0, 2.0)] * 5,
                frame_index=[0, 1, 2],  # Wrong length
                box_size=7
            )
    
    def test_stack_rejects_invalid_ndim(self):
        """Test that invalid array dimensions raise error."""
        data = np.random.rand(10, 10)  # 2D instead of 3D or 4D
        
        with pytest.raises(ValueError, match="ParticleStack data must be 3D or 4D"):
            ParticleStack(
                data=data,
                centers_xy=[(1.0, 2.0)],
                frame_index=[0],
                box_size=7
            )
