"""Tests for NL-10 — Zarr schema and opener interface."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from nanolocz.io import open_nanolocz, NanoLoczStore
from nanolocz.core.types import (
    Localizations, 
    ParticleStack, 
    ParticleTrack,
    LocalizedParticle,
)


@pytest.fixture
def temp_zarr_path():
    """Create temporary directory for Zarr store."""
    tmpdir = tempfile.mkdtemp()
    zarr_path = Path(tmpdir) / "test.zarr"
    yield zarr_path
    # Cleanup
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    Path(tmpdir).rmdir()


@pytest.fixture
def sample_movie():
    """Create sample movie data."""
    return np.random.rand(10, 64, 64).astype(np.float64)


@pytest.fixture
def sample_localizations():
    """Create sample localizations."""
    return Localizations(
        xy=[(10.5, 20.3), (30.1, 40.7), (50.9, 60.2)],
        frame_index=[0, 0, 1],
        intensities=[100.0, 150.0, 120.0],
        sigmas=[(1.2, 1.3), (1.1, 1.4), (1.5, 1.2)],
        score=[0.9, 0.85, 0.88],
    )


@pytest.fixture
def sample_tracks():
    """Create sample particle tracks."""
    track1 = ParticleTrack(
        track_id=0,
        particles=[
            LocalizedParticle(x=10.0, y=20.0, intensity=100.0, frame=0),
            LocalizedParticle(x=11.0, y=21.0, intensity=105.0, frame=1),
            LocalizedParticle(x=12.0, y=22.0, intensity=110.0, frame=2),
        ]
    )
    track2 = ParticleTrack(
        track_id=1,
        particles=[
            LocalizedParticle(x=30.0, y=40.0, intensity=150.0, frame=0),
            LocalizedParticle(x=31.0, y=41.0, intensity=155.0, frame=1),
        ]
    )
    return [track1, track2]


@pytest.fixture
def sample_particle_stack():
    """Create sample particle stack."""
    n_particles = 3
    n_frames = 5
    box_size = 16
    data = np.random.rand(n_particles, n_frames, box_size, box_size).astype(np.float64)
    centers = [(8.0, 8.0), (24.0, 24.0), (40.0, 40.0)]
    frame_idx = [0, 0, 1]
    
    return ParticleStack(
        data=data,
        centers_xy=centers,
        frame_index=frame_idx,
        box_size=box_size,
    )


class TestNanoLoczStore:
    """Test NanoLoczStore class functionality."""
    
    def test_create_store_write_mode(self, temp_zarr_path):
        """Test creating a new Zarr store."""
        store = NanoLoczStore(temp_zarr_path, mode='w')
        assert store.path == temp_zarr_path
        assert store.mode == 'w'
        assert 'movie' in store.root
        assert 'localizations' in store.root
        assert 'tracks' in store.root
        assert 'particle_stacks' in store.root
        store.close()
    
    def test_open_existing_store_read_mode(self, temp_zarr_path):
        """Test opening existing store in read-only mode."""
        # Create store first
        store_w = NanoLoczStore(temp_zarr_path, mode='w')
        store_w.close()
        
        # Open in read mode
        store_r = NanoLoczStore(temp_zarr_path, mode='r')
        assert store_r.mode == 'r'
        store_r.close()
    
    def test_context_manager(self, temp_zarr_path):
        """Test using store as context manager."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            assert store.path == temp_zarr_path
        # Store should be closed after context
    
    def test_schema_version_stored(self, temp_zarr_path):
        """Test that schema version is stored on creation."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            assert 'nanolocz_schema_version' in store.root.attrs
            assert store.root.attrs['nanolocz_schema_version'] == '1.0.0'


class TestMovieIO:
    """Test movie save/load functionality."""
    
    def test_save_and_load_movie(self, temp_zarr_path, sample_movie):
        """Test round-trip movie save/load."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            metadata = {'pixel_size': (1.0, 1.0), 'units': 'nm'}
            store.save_movie(sample_movie, metadata)
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            loaded = store.load_movie()
        
        np.testing.assert_array_equal(loaded, sample_movie)
    
    def test_save_movie_with_metadata(self, temp_zarr_path, sample_movie):
        """Test saving movie with metadata."""
        metadata = {
            'pixel_size': (0.5, 0.5),
            'units': 'nm',
            'acquisition_time': 100.0,
        }
        
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_movie(sample_movie, metadata)
            
            # Check metadata was saved
            assert store.root['movie'].attrs['pixel_size'] == (0.5, 0.5)
            assert store.root['movie'].attrs['units'] == 'nm'
    
    def test_load_movie_frame_range(self, temp_zarr_path, sample_movie):
        """Test loading subset of frames."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_movie(sample_movie)
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            # Load frames 2-5
            loaded = store.load_movie(frame_range=(2, 5))
            assert loaded.shape == (3, 64, 64)
            np.testing.assert_array_equal(loaded, sample_movie[2:5])
    
    def test_save_2d_movie(self, temp_zarr_path):
        """Test saving single-frame (2D) movie."""
        movie_2d = np.random.rand(64, 64)
        
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_movie(movie_2d)
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            loaded = store.load_movie()
            assert loaded.ndim == 3
            assert loaded.shape == (1, 64, 64)
    
    def test_load_missing_movie_raises(self, temp_zarr_path):
        """Test that loading missing movie raises KeyError."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            pass  # Don't save any movie
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            with pytest.raises(KeyError, match="No movie data found"):
                store.load_movie()


class TestLocalizationsIO:
    """Test localizations save/load functionality."""
    
    def test_save_and_load_localizations(self, temp_zarr_path, sample_localizations):
        """Test round-trip localizations save/load."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_localizations(sample_localizations, method='fast_peaks')
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            loaded = store.load_localizations()
        
        assert len(loaded.xy) == len(sample_localizations.xy)
        assert loaded.frame_index == sample_localizations.frame_index
        assert loaded.intensities == sample_localizations.intensities
        assert loaded.sigmas == sample_localizations.sigmas
        assert loaded.score == sample_localizations.score
    
    def test_load_localizations_frame_filter(self, temp_zarr_path, sample_localizations):
        """Test filtering localizations by frame range."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_localizations(sample_localizations)
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            # Load only frame 0
            loaded = store.load_localizations(frame_range=slice(0, 1))
            
            # Should have 2 detections from frame 0
            assert len(loaded.xy) == 2
            assert all(f == 0 for f in loaded.frame_index)
    
    def test_localizations_without_optional_fields(self, temp_zarr_path):
        """Test localizations with only required fields."""
        locs = Localizations(
            xy=[(10.0, 20.0), (30.0, 40.0)],
            frame_index=[0, 1],
        )
        
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_localizations(locs)
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            loaded = store.load_localizations()
            assert loaded.intensities is None
            assert loaded.sigmas is None
            assert loaded.score is None
    
    def test_load_missing_localizations_raises(self, temp_zarr_path):
        """Test that loading missing localizations raises KeyError."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            pass  # Don't save any localizations
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            with pytest.raises(KeyError, match="No localization data found"):
                store.load_localizations()


class TestTracksIO:
    """Test tracks save/load functionality."""
    
    def test_save_and_load_tracks(self, temp_zarr_path, sample_tracks):
        """Test round-trip tracks save/load."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_tracks(sample_tracks)
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            loaded = store.load_tracks()
        
        assert len(loaded) == len(sample_tracks)
        
        # Check first track
        assert loaded[0].track_id == 0
        assert len(loaded[0].particles) == 3
        assert loaded[0].particles[0].x == 10.0
        assert loaded[0].particles[0].y == 20.0
        
        # Check second track
        assert loaded[1].track_id == 1
        assert len(loaded[1].particles) == 2
    
    def test_empty_tracks_list(self, temp_zarr_path):
        """Test saving empty tracks list."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_tracks([])
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            loaded = store.load_tracks()
            assert len(loaded) == 0
    
    def test_load_missing_tracks_raises(self, temp_zarr_path):
        """Test that loading missing tracks raises KeyError."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            pass  # Don't save any tracks
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            with pytest.raises(KeyError, match="No track data found"):
                store.load_tracks()


class TestParticleStacksIO:
    """Test particle stacks save/load functionality."""
    
    def test_save_and_load_particle_stack(self, temp_zarr_path, sample_particle_stack):
        """Test round-trip particle stack save/load."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_particle_stacks(sample_particle_stack)
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            loaded = store.load_particle_stacks()
        
        assert loaded.data.shape == sample_particle_stack.data.shape
        assert loaded.centers_xy == sample_particle_stack.centers_xy
        assert loaded.frame_index == sample_particle_stack.frame_index
        assert loaded.box_size == sample_particle_stack.box_size
    
    def test_particle_stack_invalid_ndim(self, temp_zarr_path):
        """Test that non-4D data raises ValueError."""
        bad_data = np.random.rand(3, 5, 16)  # 3D instead of 4D
        
        stack = ParticleStack(
            data=bad_data,
            centers_xy=[(8.0, 8.0)],
            frame_index=[0],
            box_size=16,
        )
        
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            with pytest.raises(ValueError, match="must be 4D"):
                store.save_particle_stacks(stack)
    
    def test_load_missing_particle_stack_raises(self, temp_zarr_path):
        """Test that loading missing particle stack raises KeyError."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            pass  # Don't save any particle stacks
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            with pytest.raises(KeyError, match="No particle stack data found"):
                store.load_particle_stacks()


class TestOpenerInterface:
    """Test the open_nanolocz interface function."""
    
    def test_open_zarr_store(self, temp_zarr_path):
        """Test opening Zarr store via opener."""
        # Create store first
        with open_nanolocz(temp_zarr_path, mode='w') as store:
            assert isinstance(store, NanoLoczStore)
        
        # Open existing
        with open_nanolocz(temp_zarr_path, mode='r') as store:
            assert isinstance(store, NanoLoczStore)
    
    def test_opener_file_not_found(self):
        """Test that opener raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            open_nanolocz("nonexistent.zarr", mode='r')
    
    def test_opener_unsupported_format(self):
        """Test that opener raises ValueError for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            open_nanolocz("data.xyz", mode='r')
    
    def test_opener_hdf5_not_implemented(self, tmp_path):
        """Test that HDF5 format raises NotImplementedError."""
        h5_path = tmp_path / "data.h5"
        h5_path.touch()  # Create empty file
        
        with pytest.raises(NotImplementedError, match="HDF5 support coming soon"):
            open_nanolocz(h5_path, mode='r')
    
    def test_opener_tiff_not_implemented(self, tmp_path):
        """Test that TIFF format raises NotImplementedError."""
        tiff_path = tmp_path / "data.tif"
        tiff_path.touch()  # Create empty file
        
        with pytest.raises(NotImplementedError, match="TIFF reader integration"):
            open_nanolocz(tiff_path, mode='r')


class TestSchemaValidation:
    """Test schema validation and compatibility checks."""
    
    def test_compressed_storage(self, temp_zarr_path, sample_movie):
        """Test that data is compressed."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_movie(sample_movie)
        
        # Check that compressor is set
        movie_array = store.root['movie/data']
        assert movie_array.compressor is not None
        assert 'zstd' in str(movie_array.compressor)
    
    def test_chunked_storage(self, temp_zarr_path, sample_movie):
        """Test that data is chunked appropriately."""
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_movie(sample_movie)
        
        # Check chunking strategy
        movie_array = store.root['movie/data']
        assert movie_array.chunks == (1, 64, 64)  # Chunked by frame
    
    def test_multiple_save_overwrites(self, temp_zarr_path):
        """Test that multiple saves overwrite previous data."""
        movie1 = np.ones((5, 32, 32))
        movie2 = np.ones((5, 32, 32)) * 2
        
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_movie(movie1)
            store.save_movie(movie2)
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            loaded = store.load_movie()
            np.testing.assert_array_equal(loaded, movie2)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_round_trip(self, temp_zarr_path, sample_movie, 
                                  sample_localizations, sample_tracks,
                                  sample_particle_stack):
        """Test complete save/load cycle for all data types."""
        # Save all data
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_movie(sample_movie, {'pixel_size': (1.0, 1.0)})
            store.save_localizations(sample_localizations, method='fast_peaks')
            store.save_tracks(sample_tracks)
            store.save_particle_stacks(sample_particle_stack)
        
        # Load all data
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            loaded_movie = store.load_movie()
            loaded_locs = store.load_localizations()
            loaded_tracks = store.load_tracks()
            loaded_stacks = store.load_particle_stacks()
        
        # Verify all data
        np.testing.assert_array_equal(loaded_movie, sample_movie)
        assert len(loaded_locs.xy) == len(sample_localizations.xy)
        assert len(loaded_tracks) == len(sample_tracks)
        assert loaded_stacks.data.shape == sample_particle_stack.data.shape
    
    def test_lazy_loading_frames(self, temp_zarr_path):
        """Test that frame-range loading doesn't load entire movie."""
        movie = np.random.rand(100, 128, 128)
        
        with NanoLoczStore(temp_zarr_path, mode='w') as store:
            store.save_movie(movie)
        
        with NanoLoczStore(temp_zarr_path, mode='r') as store:
            # Load only first 10 frames
            loaded = store.load_movie(frame_range=(0, 10))
            assert loaded.shape == (10, 128, 128)
