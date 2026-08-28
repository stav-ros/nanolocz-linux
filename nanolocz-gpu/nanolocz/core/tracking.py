"""
Deterministic single-particle tracking module for NanoLocz.

Ports MATLAB tracking functionality to Python.
Implements Hungarian algorithm-based tracking with gap-closing.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from nanolocz.core.types import (
    LocalizedParticle,
    ParticleTrack,
    TrackParams,
)


def track_particles(
    localizations: list[list[LocalizedParticle]],
    params: TrackParams | None = None,
) -> list[ParticleTrack]:
    """
    Track particles across frames using deterministic assignment.
    
    Implements Hungarian algorithm for frame-to-frame assignment
    with gap-closing support.
    
    Parameters
    ----------
    localizations : list[list[LocalizedParticle]]
        List of per-frame localizations.
        localizations[frame_idx] = list of LocalizedParticle objects
    params : TrackParams, optional
        Tracking parameters. Uses defaults if None.
        
    Returns
    -------
    tracks : list[ParticleTrack]
        List of particle tracks
        
    Examples
    --------
    >>> from nanolocz.core.types import LocalizedParticle, TrackParams
    >>> # Create sample localizations for 3 frames
    >>> frame0 = [LocalizedParticle(x=10.0, y=10.0, frame=0),
    ...           LocalizedParticle(x=50.0, y=50.0, frame=0)]
    >>> frame1 = [LocalizedParticle(x=11.0, y=10.5, frame=1),
    ...           LocalizedParticle(x=51.0, y=49.5, frame=1)]
    >>> frame2 = [LocalizedParticle(x=12.0, y=11.0, frame=2),
    ...           LocalizedParticle(x=52.0, y=49.0, frame=2)]
    >>> localizations = [frame0, frame1, frame2]
    >>> tracks = track_particles(localizations)
    >>> print(f"Found {len(tracks)} tracks")
    """
    if params is None:
        params = TrackParams()
    
    n_frames = len(localizations)
    if n_frames == 0:
        return []
    
    # Initialize track management
    # track_id -> list of (frame_idx, particle)
    active_tracks: dict[int, list[tuple[int, LocalizedParticle]]] = {}
    next_track_id = 0
    
    # Track which particles have been assigned in current frame
    for frame_idx in range(n_frames):
        frame_locs = localizations[frame_idx]
        n_particles = len(frame_locs)
        
        if frame_idx == 0:
            # First frame: initialize new tracks for all particles
            for particle in frame_locs:
                active_tracks[next_track_id] = [(frame_idx, particle)]
                next_track_id += 1
            continue
        
        # Get previous frame assignments
        prev_frame = frame_idx - 1
        prev_tracks = {
            track_id: trajectory[-1]
            for track_id, trajectory in active_tracks.items()
            if trajectory[-1][0] == prev_frame
        }
        
        if not prev_tracks or n_particles == 0:
            # No previous tracks or no current particles
            # Start new tracks for unassigned particles
            for particle in frame_locs:
                active_tracks[next_track_id] = [(frame_idx, particle)]
                next_track_id += 1
            continue
        
        # Build cost matrix for assignment
        n_prev = len(prev_tracks)
        track_ids = list(prev_tracks.keys())
        
        # Cost matrix: rows = previous tracks, cols = current particles
        # Add extra columns for gap-closing (dummy assignments)
        cost_matrix = np.full((n_prev, n_particles + n_prev), np.inf)
        
        # Fill cost matrix with distances
        for i, (track_id, (_, prev_particle)) in enumerate(prev_tracks.items()):
            for j, curr_particle in enumerate(frame_locs):
                dist = np.sqrt(
                    (curr_particle.x - prev_particle.x)**2 +
                    (curr_particle.y - prev_particle.y)**2
                )
                
                if dist <= params.max_displacement:
                    cost_matrix[i, j] = dist
            
            # Gap-closing costs (allow track to continue without observation)
            if params.gap_closing_max_frames > 0:
                gap_cost = params.gap_closing_max_distance
                if params.penalize_gap_closing:
                    gap_cost *= 2.0  # Penalize gap closing
                cost_matrix[i, n_particles + i] = gap_cost
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Track which current particles are assigned
        assigned_particles = set()
        
        for i, j in zip(row_ind, col_ind):
            track_id = track_ids[i]
            
            if j < n_particles:
                # Valid assignment to current particle
                particle = frame_locs[j]
                assigned_particles.add(j)
                
                # Check if cost is acceptable
                if cost_matrix[i, j] <= params.max_displacement:
                    active_tracks[track_id].append((frame_idx, particle))
                else:
                    # Cost too high, start new track
                    active_tracks[next_track_id] = [(frame_idx, particle)]
                    next_track_id += 1
            else:
                # Gap closing (dummy assignment)
                gap_idx = j - n_particles
                if gap_idx == i:  # Same track index
                    # Allow track to continue without observation
                    pass  # Track continues, no particle added
        
        # Start new tracks for unassigned particles
        for j, particle in enumerate(frame_locs):
            if j not in assigned_particles:
                active_tracks[next_track_id] = [(frame_idx, particle)]
                next_track_id += 1
    
    # Convert active tracks to ParticleTrack objects
    # Apply gap-closing logic
    tracks = []
    for track_id, trajectory in active_tracks.items():
        if len(trajectory) == 0:
            continue
        
        # Sort by frame index
        trajectory.sort(key=lambda x: x[0])
        
        # Extract particles, handling gaps
        particles = [particle for _, particle in trajectory]
        
        # Validate gap-closing constraints
        if _validate_track_gaps(particles, params):
            tracks.append(ParticleTrack(track_id=track_id, particles=particles))
    
    return tracks


def _validate_track_gaps(
    particles: list[LocalizedParticle],
    params: TrackParams,
) -> bool:
    """
    Validate that track gaps satisfy constraints.
    
    Parameters
    ----------
    particles : list[LocalizedParticle]
        Particles in track order
    params : TrackParams
        Tracking parameters
        
    Returns
    -------
    valid : bool
        True if track satisfies gap constraints
    """
    if len(particles) < 2:
        return True
    
    max_gap = params.gap_closing_max_frames
    max_gap_dist = params.gap_closing_max_distance
    
    for i in range(len(particles) - 1):
        p1, p2 = particles[i], particles[i + 1]
        frame_gap = p2.frame - p1.frame
        
        if frame_gap > 1:
            # There's a gap
            if frame_gap - 1 > max_gap:
                return False
            
            # Check distance across gap
            dist = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            if dist > max_gap_dist:
                return False
    
    return True


def compute_track_statistics(tracks: list[ParticleTrack]) -> dict:
    """
    Compute statistics for a set of tracks.
    
    Parameters
    ----------
    tracks : list[ParticleTrack]
        List of particle tracks
        
    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'n_tracks': Total number of tracks
        - 'mean_duration': Mean track duration
        - 'max_duration': Maximum track duration
        - 'mean_displacement': Mean displacement per frame
        - 'total_displacement': Total path length
    """
    if not tracks:
        return {
            'n_tracks': 0,
            'mean_duration': 0.0,
            'max_duration': 0,
            'mean_displacement': 0.0,
            'total_displacement': 0.0,
        }
    
    durations = [t.duration for t in tracks]
    displacements = [t.mean_displacement for t in tracks if t.duration > 1]
    
    total_path_length = sum(
        sum(t.displacements) for t in tracks if len(t.displacements) > 0
    )
    
    return {
        'n_tracks': len(tracks),
        'mean_duration': float(np.mean(durations)),
        'max_duration': int(np.max(durations)),
        'mean_displacement': float(np.mean(displacements)) if displacements else 0.0,
        'total_displacement': float(total_path_length),
    }


def filter_tracks(
    tracks: list[ParticleTrack],
    min_duration: int = 3,
    max_duration: int | None = None,
    min_displacement: float = 0.0,
) -> list[ParticleTrack]:
    """
    Filter tracks based on criteria.
    
    Parameters
    ----------
    tracks : list[ParticleTrack]
        List of particle tracks
    min_duration : int
        Minimum track duration (frames)
    max_duration : int, optional
        Maximum track duration (frames)
    min_displacement : float
        Minimum mean displacement
        
    Returns
    -------
    filtered : list[ParticleTrack]
        Filtered list of tracks
    """
    filtered = []
    
    for track in tracks:
        if track.duration < min_duration:
            continue
        
        if max_duration is not None and track.duration > max_duration:
            continue
        
        if track.mean_displacement < min_displacement:
            continue
        
        filtered.append(track)
    
    return filtered
