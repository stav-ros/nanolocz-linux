"""Tests for NL-17: Deterministic single-particle tracking."""

import numpy as np
import pytest

from nanolocz.core.types import LocalizedParticle, ParticleTrack, TrackParams
from nanolocz.core.tracking import (
    track_particles,
    compute_track_statistics,
    filter_tracks,
)


class TestTrackParticles:
    """Test particle tracking functionality."""

    def test_simple_tracking_no_motion(self):
        """Test tracking with stationary particles."""
        # Create 2 stationary particles across 5 frames
        localizations = []
        for frame_idx in range(5):
            frame_locs = [
                LocalizedParticle(x=10.0, y=10.0, intensity=100.0, frame=frame_idx),
                LocalizedParticle(x=50.0, y=50.0, intensity=100.0, frame=frame_idx),
            ]
            localizations.append(frame_locs)

        tracks = track_particles(localizations)

        assert len(tracks) == 2
        assert all(t.duration == 5 for t in tracks)

    def test_tracking_with_motion(self):
        """Test tracking with moving particles."""
        # Create 2 particles moving linearly
        localizations = []
        for frame_idx in range(10):
            frame_locs = [
                LocalizedParticle(x=10.0 + frame_idx, y=10.0, intensity=100.0, frame=frame_idx),
                LocalizedParticle(x=50.0, y=50.0 + frame_idx * 0.5, intensity=100.0, frame=frame_idx),
            ]
            localizations.append(frame_locs)

        tracks = track_particles(localizations)

        assert len(tracks) == 2
        assert all(t.duration == 10 for t in tracks)

    def test_empty_localizations(self):
        """Test tracking with empty input."""
        tracks = track_particles([])
        assert tracks == []

    def test_single_frame(self):
        """Test tracking with single frame."""
        frame_locs = [
            LocalizedParticle(x=10.0, y=10.0, intensity=100.0, frame=0),
            LocalizedParticle(x=50.0, y=50.0, intensity=100.0, frame=0),
        ]
        tracks = track_particles([frame_locs])

        assert len(tracks) == 2
        assert all(t.duration == 1 for t in tracks)

    def test_gap_closing(self):
        """Test tracking with missing detections (gap closing)."""
        # Frame 0: 2 particles
        # Frame 1: 2 particles (both present)
        # Frame 2: 1 particle (one missing - should gap close)
        localizations = [
            [
                LocalizedParticle(x=10.0, y=10.0, intensity=100.0, frame=0),
                LocalizedParticle(x=50.0, y=50.0, intensity=100.0, frame=0),
            ],
            [
                LocalizedParticle(x=11.0, y=10.0, intensity=100.0, frame=1),
                LocalizedParticle(x=50.0, y=50.5, intensity=100.0, frame=1),
            ],
            [
                LocalizedParticle(x=12.0, y=10.0, intensity=100.0, frame=2),
                # Second particle missing in frame 2, should gap close from frame 1
            ],
        ]

        params = TrackParams(
            max_displacement=5.0,
            gap_closing_max_frames=1,
            gap_closing_max_distance=5.0,
        )
        tracks = track_particles(localizations, params)

        # Should have 2 tracks with gap closing
        assert len(tracks) == 2

    def test_track_splitting(self):
        """Test that tracks split when particles move apart."""
        # Particles start close and move apart beyond max_displacement
        localizations = [
            [LocalizedParticle(x=10.0, y=10.0, intensity=100.0, frame=0)],
            [LocalizedParticle(x=10.0, y=10.0, intensity=100.0, frame=1)],
            [LocalizedParticle(x=10.0, y=10.0, intensity=100.0, frame=2)],
        ]

        params = TrackParams(max_displacement=5.0)
        tracks = track_particles(localizations, params)

        assert len(tracks) == 1
        assert tracks[0].duration == 3

    def test_custom_parameters(self):
        """Test tracking with custom parameters."""
        localizations = [
            [
                LocalizedParticle(x=10.0, y=10.0, intensity=100.0, frame=0),
                LocalizedParticle(x=20.0, y=20.0, intensity=100.0, frame=0),
            ],
            [
                LocalizedParticle(x=10.5, y=10.0, intensity=100.0, frame=1),
                LocalizedParticle(x=20.5, y=20.0, intensity=100.0, frame=1),
            ],
        ]

        params = TrackParams(
            max_displacement=2.0,
            gap_closing_max_frames=0,
            penalize_gap_closing=False,
        )
        tracks = track_particles(localizations, params)

        assert len(tracks) == 2


class TestComputeTrackStatistics:
    """Test track statistics computation."""

    def test_empty_tracks(self):
        """Test statistics with no tracks."""
        stats = compute_track_statistics([])

        assert stats['n_tracks'] == 0
        assert stats['mean_duration'] == 0.0
        assert stats['max_duration'] == 0

    def test_single_track(self):
        """Test statistics with one track."""
        particles = [
            LocalizedParticle(x=i, y=i, intensity=100.0, frame=i)
            for i in range(5)
        ]
        track = ParticleTrack(track_id=0, particles=particles)

        stats = compute_track_statistics([track])

        assert stats['n_tracks'] == 1
        assert stats['mean_duration'] == 5.0
        assert stats['max_duration'] == 5

    def test_multiple_tracks(self):
        """Test statistics with multiple tracks."""
        track1 = ParticleTrack(
            track_id=0,
            particles=[LocalizedParticle(x=i, y=i, intensity=100.0, frame=i) for i in range(5)]
        )
        track2 = ParticleTrack(
            track_id=1,
            particles=[LocalizedParticle(x=i*2, y=i, intensity=100.0, frame=i) for i in range(10)]
        )

        stats = compute_track_statistics([track1, track2])

        assert stats['n_tracks'] == 2
        assert stats['mean_duration'] == 7.5
        assert stats['max_duration'] == 10


class TestFilterTracks:
    """Test track filtering functionality."""

    def test_filter_by_min_duration(self):
        """Test filtering by minimum duration."""
        short_track = ParticleTrack(
            track_id=0,
            particles=[LocalizedParticle(x=i, y=i, intensity=100.0, frame=i) for i in range(2)]
        )
        long_track = ParticleTrack(
            track_id=1,
            particles=[LocalizedParticle(x=i, y=i, intensity=100.0, frame=i) for i in range(5)]
        )

        filtered = filter_tracks([short_track, long_track], min_duration=3)

        assert len(filtered) == 1
        assert filtered[0].track_id == 1

    def test_filter_by_max_duration(self):
        """Test filtering by maximum duration."""
        short_track = ParticleTrack(
            track_id=0,
            particles=[LocalizedParticle(x=i, y=i, intensity=100.0, frame=i) for i in range(3)]
        )
        long_track = ParticleTrack(
            track_id=1,
            particles=[LocalizedParticle(x=i, y=i, intensity=100.0, frame=i) for i in range(10)]
        )

        filtered = filter_tracks(
            [short_track, long_track],
            min_duration=1,
            max_duration=5
        )

        assert len(filtered) == 1
        assert filtered[0].track_id == 0

    def test_filter_by_displacement(self):
        """Test filtering by minimum displacement."""
        stationary = ParticleTrack(
            track_id=0,
            particles=[LocalizedParticle(x=10.0, y=10.0, intensity=100.0, frame=i) for i in range(5)]
        )
        moving = ParticleTrack(
            track_id=1,
            particles=[LocalizedParticle(x=i, y=i, intensity=100.0, frame=i) for i in range(5)]
        )

        filtered = filter_tracks(
            [stationary, moving],
            min_duration=1,
            min_displacement=0.5
        )

        assert len(filtered) == 1
        assert filtered[0].track_id == 1

    def test_combined_filters(self):
        """Test combined filtering criteria."""
        tracks = [
            ParticleTrack(
                track_id=i,
                particles=[LocalizedParticle(x=j, y=j, intensity=100.0, frame=j) for j in range(dur)]
            )
            for i, dur in enumerate([2, 3, 5, 10])
        ]

        filtered = filter_tracks(
            tracks,
            min_duration=3,
            max_duration=8,
        )

        assert len(filtered) == 2
        assert [t.track_id for t in filtered] == [1, 2]


class TestParticleTrackProperties:
    """Test ParticleTrack dataclass properties."""

    def test_duration_property(self):
        """Test track duration calculation."""
        particles = [LocalizedParticle(x=i, y=i, intensity=100.0, frame=i) for i in range(7)]
        track = ParticleTrack(track_id=0, particles=particles)

        assert track.duration == 7

    def test_frames_property(self):
        """Test frames list extraction."""
        particles = [LocalizedParticle(x=i, y=i, intensity=100.0, frame=i*2) for i in range(5)]
        track = ParticleTrack(track_id=0, particles=particles)

        assert track.frames == [0, 2, 4, 6, 8]

    def test_displacements_property(self):
        """Test displacement calculation."""
        # Linear motion: each step is sqrt(2) ≈ 1.414
        particles = [LocalizedParticle(x=i, y=i, intensity=100.0, frame=i) for i in range(5)]
        track = ParticleTrack(track_id=0, particles=particles)

        displacements = track.displacements
        assert len(displacements) == 4
        assert np.allclose(displacements, np.sqrt(2))

    def test_mean_displacement_property(self):
        """Test mean displacement calculation."""
        particles = [LocalizedParticle(x=i, y=i, intensity=100.0, frame=i) for i in range(5)]
        track = ParticleTrack(track_id=0, particles=particles)

        assert np.isclose(track.mean_displacement, np.sqrt(2))

    def test_empty_track_properties(self):
        """Test properties with empty track."""
        track = ParticleTrack(track_id=0, particles=[])

        assert track.duration == 0
        assert track.frames == []
        assert len(track.displacements) == 0
        assert track.mean_displacement == 0.0

    def test_single_particle_track(self):
        """Test properties with single particle."""
        track = ParticleTrack(
            track_id=0,
            particles=[LocalizedParticle(x=10.0, y=10.0, intensity=100.0, frame=0)]
        )

        assert track.duration == 1
        assert track.frames == [0]
        assert len(track.displacements) == 0
        assert track.mean_displacement == 0.0
