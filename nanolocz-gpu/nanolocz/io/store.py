"""NanoLoczStore — Zarr-based storage for AFM particle data.

This module implements the storage backend following the schema defined in
SPEC/NL-10-zarr-schema.md. It provides a unified interface for reading and
writing movies, localizations, tracks, and particle stacks.
"""

import json
from pathlib import Path
from typing import Any, Literal
import numpy as np
import zarr

from nanolocz.core.types import (
    Localizations, 
    ParticleStack, 
    ParticleTrack,
    ImageMetadata,
    FileFormat,
)


# Schema version for compatibility tracking
SCHEMA_VERSION = "1.0.0"


class NanoLoczStore:
    """Zarr-based storage for NanoLocz AFM data.
    
    Provides methods to load and save:
    - Raw or processed image movies
    - Particle localizations (coordinates, intensities)
    - Particle tracks (trajectories across frames)
    - Extracted particle substacks
    
    The store follows the Zarr schema documented in SPEC/NL-10-zarr-schema.md.
    
    Parameters
    ----------
    path : str or Path
        Path to Zarr store directory
    mode : {'r', 'r+', 'w', 'w-', 'a'}
        Opening mode passed to zarr.open():
        - 'r': Read-only
        - 'r+': Read-write (must exist)
        - 'w': Read-write (overwrite if exists)
        - 'w-': Read-write (fail if exists)
        - 'a': Read-write (append to existing)
    
    Attributes
    ----------
    root : zarr.Group
        Root Zarr group containing all data
    mode : str
        Current access mode
    
    Examples
    --------
    >>> from nanolocz.io import open_nanolocz
    >>> store = open_nanolocz("experiment.zarr", mode="w")
    >>> store.save_movie(movie_data, metadata)
    >>> loaded_movie = store.load_movie()
    >>> store.close()
    """
    
    def __init__(self, path: str | Path, mode: Literal['r', 'r+', 'w', 'w-', 'a'] = 'r'):
        self.path = Path(path)
        self.mode = mode
        
        # Initialize or open Zarr store
        if mode == 'w' and self.path.exists():
            import shutil
            shutil.rmtree(self.path)
        
        self.root = zarr.open(str(self.path), mode=mode)
        
        # Initialize schema version and structure for new stores
        if mode in ('w', 'w-'):
            self._initialize_store()
        elif mode == 'a' and len(list(self.root.keys())) == 0:
            # Empty store opened in append mode - initialize it
            self._initialize_store()
    
    def _initialize_store(self):
        """Initialize Zarr store structure with schema version."""
        # Store schema version
        self.root.attrs['nanolocz_schema_version'] = SCHEMA_VERSION
        self.root.attrs['nanolocz_version'] = '0.1.0.dev0'
        
        # Create groups (will be populated when data is saved)
        if 'movie' not in self.root:
            self.root.create_group('movie')
        if 'localizations' not in self.root:
            self.root.create_group('localizations')
        if 'tracks' not in self.root:
            self.root.create_group('tracks')
        if 'particle_stacks' not in self.root:
            self.root.create_group('particle_stacks')
    
    def _validate_schema(self):
        """Validate that store follows expected schema.
        
        Raises
        ------
        ValueError
            If schema version is incompatible or structure is invalid
        """
        if '.zgroup' not in list(self.root.array_keys()) + ['.zgroup']:
            raise ValueError(f"Invalid Zarr store: {self.path}")
        
        stored_version = self.root.attrs.get('nanolocz_schema_version', '1.0.0')
        major_version = stored_version.split('.')[0]
        if major_version != SCHEMA_VERSION.split('.')[0]:
            raise ValueError(
                f"Incompatible schema version: stored={stored_version}, "
                f"expected ~{SCHEMA_VERSION}"
            )
    
    def save_movie(self, data: np.ndarray, metadata: dict[str, Any] | None = None):
        """Save image movie data.
        
        Parameters
        ----------
        data : np.ndarray
            Movie data with shape (frames, height, width) or (height, width)
            for single-frame data
        metadata : dict, optional
            Image metadata including pixel_size, units, acquisition parameters
            
        Notes
        -----
        Data is chunked by frame: (1, height, width) for efficient frame-wise access.
        Compression uses Blosc Zstandard (zstd) level 3.
        """
        data = np.asarray(data)
        
        # Ensure 3D shape
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError(f"Movie data must be 2D or 3D, got {data.ndim}D")
        
        # Chunk by frame
        chunks = (1, data.shape[1], data.shape[2])
        
        # Convert metadata tuples to lists for Zarr JSON compatibility
        zarr_metadata = None
        if metadata:
            zarr_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, tuple):
                    zarr_metadata[key] = list(value)
                else:
                    zarr_metadata[key] = value
        
        # Delete existing array if it exists (to allow overwrites)
        if 'movie/data' in self.root:
            del self.root['movie/data']
        
        # Zarr v3 API: use create with compressor parameter
        self.root.create(
            'movie/data',
            data=data,
            chunks=chunks,
            compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3, shuffle=1)],
        )
        
        # Save metadata
        if zarr_metadata:
            self.root['movie'].attrs.update(zarr_metadata)
        
        # Add standard metadata
        self.root['movie'].attrs['shape'] = data.shape
        self.root['movie'].attrs['dtype'] = str(data.dtype)
        self.root['movie'].attrs['units'] = zarr_metadata.get('units', 'nm') if zarr_metadata else 'nm'
    
    def load_movie(self, frame_range: slice | tuple[int, int] | None = None) -> np.ndarray:
        """Load image movie data.
        
        Parameters
        ----------
        frame_range : slice or tuple, optional
            Range of frames to load. Can be:
            - slice(start, stop, step)
            - tuple (start, stop)
            - None for all frames
            
        Returns
        -------
        np.ndarray
            Movie data with shape (frames, height, width) or (height, width)
            
        Raises
        ------
        KeyError
            If movie data does not exist in store
        """
        if 'movie/data' not in self.root:
            raise KeyError("No movie data found in store")
        
        movie = self.root['movie/data']
        
        if frame_range is not None:
            if isinstance(frame_range, tuple):
                frame_range = slice(frame_range[0], frame_range[1])
            return movie[frame_range]
        
        return movie[:]
    
    def save_localizations(self, localizations: Localizations, method: str = 'unknown'):
        """Save particle localizations.
        
        Parameters
        ----------
        localizations : Localizations
            Localizations object containing coordinates and metadata
        method : str, optional
            Detection/localization method name (default: 'unknown')
            
        Notes
        -----
        Localizations are stored as separate arrays for each field:
        - xy: (n_detections, 2) float64
        - frame_index: (n_detections,) int32
        - intensities: (n_detections,) float64 (optional)
        - sigmas: (n_detections, 2) float64 (optional)
        - score: (n_detections,) float64 (optional)
        
        Chunking: 1000 detections per chunk for batch processing efficiency.
        """
        locs = localizations
        
        # Convert to arrays
        xy = np.asarray(locs.xy, dtype=np.float64)
        frame_idx = np.asarray(locs.frame_index, dtype=np.int32)
        
        # Chunk size - must match dimensionality of arrays
        chunks_xy = (1000, 2)  # 2D chunk for xy coordinates
        chunks_1d = (1000,)    # 1D chunk for other fields
        
        # Save required fields with data parameter
        if 'localizations/xy' in self.root:
            del self.root['localizations/xy']
        if 'localizations/frame_index' in self.root:
            del self.root['localizations/frame_index']
        
        self.root.create('localizations/xy', data=xy, chunks=chunks_xy,
                       compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
        self.root.create('localizations/frame_index', data=frame_idx, chunks=chunks_1d, 
                       compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
        
        # Save optional fields
        if locs.intensities is not None:
            if 'localizations/intensities' in self.root:
                del self.root['localizations/intensities']
            self.root.create('localizations/intensities', 
                           data=np.asarray(locs.intensities, dtype=np.float64),
                           chunks=chunks_1d,
                           compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
        
        if locs.sigmas is not None:
            if 'localizations/sigmas' in self.root:
                del self.root['localizations/sigmas']
            self.root.create('localizations/sigmas',
                           data=np.asarray(locs.sigmas, dtype=np.float64),
                           chunks=(1000, 2) if np.asarray(locs.sigmas).ndim == 2 else chunks_1d,
                           compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
        
        if locs.score is not None:
            if 'localizations/score' in self.root:
                del self.root['localizations/score']
            self.root.create('localizations/score',
                           data=np.asarray(locs.score, dtype=np.float64),
                           chunks=chunks_1d,
                           compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
        
        # Metadata
        self.root['localizations'].attrs['n_detections'] = len(xy)
        self.root['localizations'].attrs['method'] = method
    
    def load_localizations(self, frame_range: slice | None = None) -> Localizations:
        """Load particle localizations.
        
        Parameters
        ----------
        frame_range : slice, optional
            Filter localizations by frame range
            
        Returns
        -------
        Localizations
            Loaded localizations object
            
        Raises
        ------
        KeyError
            If localization data does not exist in store
        """
        if 'localizations/xy' not in self.root:
            raise KeyError("No localization data found in store")
        
        # Load required fields
        xy = self.root['localizations/xy'][:]
        frame_idx = self.root['localizations/frame_index'][:]
        
        # Apply frame filter if requested
        if frame_range is not None:
            mask = (frame_idx >= frame_range.start) & (frame_idx < frame_range.stop)
            xy = xy[mask]
            frame_idx = frame_idx[mask]
        
        # Load optional fields
        intensities = None
        if 'localizations/intensities' in self.root:
            intensities = self.root['localizations/intensities'][:]
            if frame_range is not None:
                intensities = intensities[mask]
        
        sigmas = None
        if 'localizations/sigmas' in self.root:
            sigmas = self.root['localizations/sigmas'][:]
            if frame_range is not None:
                sigmas = sigmas[mask]
        
        score = None
        if 'localizations/score' in self.root:
            score = self.root['localizations/score'][:]
            if frame_range is not None:
                score = score[mask]
        
        # Convert to tuples where expected by the dataclass
        xy_tuples = [tuple(row) for row in xy]
        sigmas_tuples = None
        if sigmas is not None:
            sigmas_tuples = [tuple(row) for row in sigmas]
        
        return Localizations(
            xy=xy_tuples,
            frame_index=frame_idx.tolist(),
            intensities=intensities.tolist() if intensities is not None else None,
            sigmas=sigmas_tuples,
            score=score.tolist() if score is not None else None,
        )
    
    def save_tracks(self, tracks: list[ParticleTrack]):
        """Save particle tracks.
        
        Parameters
        ----------
        tracks : list[ParticleTrack]
            List of tracked particle trajectories
            
        Notes
        -----
        Each track is stored in a separate group (track_000, track_001, etc.)
        for independent access and parallel processing.
        
        Track structure:
        - frames: frame indices (int32)
        - x: x coordinates (float64)
        - y: y coordinates (float64)
        """
        # Remove existing tracks
        if 'tracks' in self.root:
            del self.root['tracks']
        tracks_group = self.root.create_group('tracks')
        tracks_group.attrs['tracks_saved'] = True
        tracks_group.attrs['n_tracks'] = len(tracks)
        
        for i, track in enumerate(tracks):
            track_group = self.root.create_group(f'tracks/track_{i:03d}')
            
            # Extract positions
            if track.particles:
                frames = np.array([p.frame for p in track.particles], dtype=np.int32)
                x = np.array([p.x for p in track.particles], dtype=np.float64)
                y = np.array([p.y for p in track.particles], dtype=np.float64)
                
                track_group.create('frames', data=frames,
                                 compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
                track_group.create('x', data=x,
                                 compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
                track_group.create('y', data=y,
                                 compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
                
                track_group.attrs['duration'] = len(track.particles)
                track_group.attrs['track_id'] = track.track_id
        
        self.root['tracks'].attrs['n_tracks'] = len(tracks)
    
    def load_tracks(self) -> list[ParticleTrack]:
        """Load particle tracks.
        
        Returns
        -------
        list[ParticleTrack]
            List of loaded particle tracks
            
        Raises
        ------
        KeyError
            If track data does not exist in store
        """
        if 'tracks' not in self.root:
            raise KeyError("No track data found in store")
        
        track_group = self.root['tracks']
        saved_marker = bool(track_group.attrs.get('tracks_saved', False))
        
        # Find all track subgroups. Existing children are treated as legacy
        # saved data when the marker is absent.
        track_names = sorted([k for k in track_group.keys() if k.startswith('track_')])
        if not saved_marker and not track_names:
            raise KeyError("No track data found in store")
        
        # An explicitly saved empty list is distinct from a new store whose
        # scaffolding contains no track data.
        if saved_marker and track_group.attrs.get('n_tracks') == 0:
            return []
        
        # Check if there are any actual tracks with data
        has_track_data = False
        for track_name in track_names:
            track_grp = track_group[track_name]
            if 'frames' in track_grp:
                has_track_data = True
                break
        
        if not has_track_data:
            # Return empty list for empty tracks (not an error)
            return []
        
        tracks = []
        
        for track_name in track_names:
            track_grp = track_group[track_name]
            
            if 'frames' not in track_grp:
                continue
            
            frames = track_grp['frames'][:]
            x = track_grp['x'][:]
            y = track_grp['y'][:]
            
            # Reconstruct particles
            from nanolocz.core.types import LocalizedParticle
            particles = [
                LocalizedParticle(x=x[i], y=y[i], frame=frames[i], intensity=0.0)
                for i in range(len(frames))
            ]
            
            track_id = track_grp.attrs.get('track_id', int(track_name.split('_')[1]))
            tracks.append(ParticleTrack(track_id=track_id, particles=particles))
        
        return tracks
    
    def save_particle_stacks(self, stack: ParticleStack):
        """Save extracted particle substacks.
        
        Parameters
        ----------
        stack : ParticleStack
            Particle stack containing substacks and metadata
            
        Notes
        -----
        Particle stacks are stored as 4D array:
        (n_particles, n_frames, box_size, box_size)
        
        Chunking: (1, n_frames, box_size, box_size) for per-particle access.
        """
        data = np.asarray(stack.data)
        
        if data.ndim != 4:
            raise ValueError(f"Particle stack data must be 4D, got {data.ndim}D")
        
        # Chunk by particle
        chunks = (1, data.shape[1], data.shape[2], data.shape[3])
        
        # Save data
        if 'particle_stacks/data' in self.root:
            del self.root['particle_stacks/data']
        
        self.root.create('particle_stacks/data', data=data, chunks=chunks,
                       compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
        
        # Save centers
        centers = np.asarray(stack.centers_xy, dtype=np.float64)
        if 'particle_stacks/centers_xy' in self.root:
            del self.root['particle_stacks/centers_xy']
        self.root.create('particle_stacks/centers_xy', data=centers,
                       compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
        
        # Save frame indices
        frame_idx = np.asarray(stack.frame_index, dtype=np.int32)
        if 'particle_stacks/frame_index' in self.root:
            del self.root['particle_stacks/frame_index']
        self.root.create('particle_stacks/frame_index', data=frame_idx,
                       compressors=[zarr.codecs.Blosc(cname='zstd', clevel=3)])
        
        # Metadata
        self.root['particle_stacks'].attrs['shape'] = data.shape
        self.root['particle_stacks'].attrs['box_size'] = stack.box_size
        self.root['particle_stacks'].attrs['n_particles'] = stack.n_particles
    
    def load_particle_stacks(self) -> ParticleStack:
        """Load particle substacks.
        
        Returns
        -------
        ParticleStack
            Loaded particle stack
            
        Raises
        ------
        KeyError
            If particle stack data does not exist in store
        """
        if 'particle_stacks/data' not in self.root:
            raise KeyError("No particle stack data found in store")
        
        data = self.root['particle_stacks/data'][:]
        centers = self.root['particle_stacks/centers_xy'][:]
        frame_idx = self.root['particle_stacks/frame_index'][:]
        
        box_size = self.root['particle_stacks'].attrs.get('box_size', None)
        
        # Convert to tuples where expected by the dataclass
        centers_tuples = [tuple(row) for row in centers]
        
        return ParticleStack(
            data=data,
            centers_xy=centers_tuples,
            frame_index=frame_idx.tolist(),
            box_size=box_size,
        )
    
    def close(self):
        """Close the Zarr store.
        
        This ensures all data is flushed to disk.
        """
        # Zarr automatically flushes on deletion, but we can be explicit
        self.root.store.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure store is closed."""
        self.close()
