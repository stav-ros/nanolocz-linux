"""
NL-36: Dynamics Traces, Transitions, and Dwell Times

Test suite for dynamics analysis functionality.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import List

# Mock Track class for testing
@dataclass
class MockTrack:
    particle_id: int
    times: List[float]
    frame_indices: List[int]

@dataclass  
class MockTrackCollection:
    tracks: List[MockTrack]

# Import after defining mocks to avoid circular imports
import sys
sys.path.insert(0, '/workspace/nanolocz-gpu')

from nanolocz.core.dynamics import (
    DynamicsTrace,
    DynamicsResult,
    extract_particle_traces,
    compute_dwell_times,
    estimate_rate_constants,
    plot_trace_with_states,
    plot_dwell_histogram,
    plot_transition_network,
    plot_state_timeline
)


class TestDynamicsTrace:
    """Test DynamicsTrace dataclass."""
    
    def test_create_trace(self):
        """Test creating a basic trace."""
        trace = DynamicsTrace(
            particle_id=0,
            time_points=np.array([0, 1, 2, 3]),
            features={'height': np.array([10, 12, 11, 13])}
        )
        
        assert trace.particle_id == 0
        assert len(trace.time_points) == 4
        assert 'height' in trace.features
        assert trace.state_sequence is None
    
    def test_trace_with_states(self):
        """Test trace with state sequence."""
        trace = DynamicsTrace(
            particle_id=1,
            time_points=np.array([0, 1, 2]),
            features={'intensity': np.array([5, 6, 5])},
            state_sequence=np.array([0, 1, 0]),
            confidence=np.array([0.9, 0.8, 0.9])
        )
        
        assert np.array_equal(trace.state_sequence, [0, 1, 0])
        assert len(trace.confidence) == 3


class TestDwellTimes:
    """Test dwell time computation."""
    
    def test_single_state(self):
        """Test dwell times for single state sequence."""
        seq = [np.array([0, 0, 0, 0])]
        time_step = 0.1
        
        dwell_times = compute_dwell_times(seq, time_step)
        
        assert 0 in dwell_times
        assert len(dwell_times[0]) == 1
        assert np.isclose(dwell_times[0][0], 0.4)  # 4 frames * 0.1s
    
    def test_two_states(self):
        """Test dwell times for two-state sequence."""
        seq = [np.array([0, 0, 1, 1, 1])]
        time_step = 1.0
        
        dwell_times = compute_dwell_times(seq, time_step)
        
        assert 0 in dwell_times
        assert 1 in dwell_times
        assert len(dwell_times[0]) == 1
        assert len(dwell_times[1]) == 1
        assert dwell_times[0][0] == 2.0  # 2 frames
        assert dwell_times[1][0] == 3.0  # 3 frames
    
    def test_multiple_transitions(self):
        """Test dwell times with multiple state transitions."""
        seq = [np.array([0, 1, 0, 1, 0])]
        time_step = 0.5
        
        dwell_times = compute_dwell_times(seq, time_step)
        
        # Each state appears 3 times (state 0) and 2 times (state 1)
        assert len(dwell_times[0]) == 3  # Three separate visits
        assert len(dwell_times[1]) == 2  # Two separate visits
        assert all(dt == 0.5 for dt in dwell_times[0])  # Single frame each
    
    def test_multiple_sequences(self):
        """Test dwell times across multiple sequences."""
        seqs = [
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            np.array([1, 1, 0, 0])
        ]
        time_step = 1.0
        
        dwell_times = compute_dwell_times(seqs, time_step)
        
        # Count visits: seq1 has 1 visit to 0, 1 visit to 1
        #               seq2 has 1 visit to 0, 1 visit to 1
        #               seq3 has 1 visit to 1, 1 visit to 0
        # Total: 3 visits to state 0, 3 visits to state 1
        assert len(dwell_times[0]) == 3  # Three visits total
        assert len(dwell_times[1]) == 3  # Three visits total
    
    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        seqs = [np.array([]), np.array([0, 0, 1])]
        time_step = 1.0
        
        dwell_times = compute_dwell_times(seqs, time_step)
        
        assert 0 in dwell_times
        assert 1 in dwell_times
    
    def test_no_state_sequence(self):
        """Test handling of None sequences."""
        seqs = [None, np.array([0, 0, 1, 1])]
        time_step = 1.0
        
        dwell_times = compute_dwell_times(seqs, time_step)
        
        assert 0 in dwell_times
        assert 1 in dwell_times


class TestRateConstants:
    """Test rate constant estimation."""
    
    def test_simple_transition(self):
        """Test rate constants for simple 2-state system."""
        trans_matrix = np.array([[0.8, 0.2],
                                  [0.3, 0.7]])
        dwell_times = {0: np.array([2.0, 2.0, 2.0]),
                       1: np.array([1.5, 1.5, 1.5])}
        time_step = 0.1
        
        rates = estimate_rate_constants(trans_matrix, dwell_times, time_step)
        
        assert rates.shape == (2, 2)
        # Diagonal should be zero or negative (self-transitions)
        assert rates[0, 0] == 0  # No self-transition rate
        assert rates[1, 1] == 0
        # Off-diagonal should be positive
        assert rates[0, 1] > 0
        assert rates[1, 0] > 0
    
    def test_zero_dwell_time(self):
        """Test handling of zero dwell times."""
        trans_matrix = np.array([[0.5, 0.5],
                                  [0.5, 0.5]])
        dwell_times = {0: np.array([]),  # Empty
                       1: np.array([1.0, 1.0])}
        time_step = 0.1
        
        rates = estimate_rate_constants(trans_matrix, dwell_times, time_step)
        
        assert rates.shape == (2, 2)
        # State 0 has no dwell times, so rates from it should be 0
        assert rates[0, 1] == 0


class TestTraceExtraction:
    """Test trace extraction from particle stacks."""
    
    def test_extract_basic(self):
        """Test basic trace extraction."""
        # Create mock particle stack
        from nanolocz.core.types import ParticleStack
        
        n_particles = 3
        n_frames = 5
        box_size = 16
        
        data = np.random.randn(n_particles, n_frames, box_size, box_size)
        centers = [(8, 8)] * n_particles
        frame_indices = [0] * n_particles
        
        stack = ParticleStack(data=data, centers_xy=centers, 
                             frame_index=frame_indices, box_size=box_size)
        
        # Create mock tracks
        tracks = MockTrackCollection([
            MockTrack(particle_id=i, 
                     times=list(range(n_frames)),
                     frame_indices=list(range(n_frames)))
            for i in range(n_particles)
        ])
        
        traces = extract_particle_traces(tracks, stack, features=['height', 'intensity'])
        
        assert len(traces) == n_particles
        for trace in traces:
            assert len(trace.time_points) == n_frames
            assert 'height' in trace.features
            assert 'intensity' in trace.features
    
    def test_extract_with_gaps(self):
        """Test trace extraction with missing frames."""
        from nanolocz.core.types import ParticleStack
        
        n_particles = 1
        n_frames = 5
        box_size = 16
        
        data = np.ones((n_particles, n_frames, box_size, box_size))
        centers = [(8, 8)]
        frame_indices = [0]
        
        stack = ParticleStack(data=data, centers_xy=centers,
                             frame_index=frame_indices, box_size=box_size)
        
        # Track with gap (missing frame 2)
        tracks = MockTrackCollection([
            MockTrack(particle_id=0,
                     times=[0, 1, 3, 4],  # Missing time 2
                     frame_indices=[0, 1, 3, 4])
        ])
        
        traces = extract_particle_traces(tracks, stack, features=['intensity'])
        
        assert len(traces) == 1
        assert len(traces[0].time_points) == 4


class TestVisualization:
    """Test visualization functions."""
    
    def test_plot_trace_with_states(self):
        """Test trace plotting with states."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        trace = DynamicsTrace(
            particle_id=0,
            time_points=np.array([0, 1, 2, 3, 4]),
            features={'height': np.array([10, 12, 11, 13, 12])},
            state_sequence=np.array([0, 0, 1, 1, 0])
        )
        
        fig, ax = plt.subplots()
        result_ax = plot_trace_with_states(trace, 'height', ax=ax)
        
        assert result_ax is ax
        plt.close(fig)
    
    def test_plot_dwell_histogram(self):
        """Test dwell time histogram."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        dwell_times = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        
        fig, ax = plt.subplots()
        result_ax = plot_dwell_histogram(dwell_times, ax=ax)
        
        assert result_ax is ax
        plt.close(fig)
    
    def test_plot_transition_network(self):
        """Test transition network visualization."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        trans_matrix = np.array([[0.7, 0.3],
                                  [0.2, 0.8]])
        
        fig, ax = plt.subplots()
        result_ax = plot_transition_network(trans_matrix, ax=ax)
        
        assert result_ax is ax
        plt.close(fig)
    
    def test_plot_state_timeline(self):
        """Test state timeline visualization."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        sequences = [
            np.array([0, 0, 1, 1, 0]),
            np.array([1, 1, 1, 0, 0]),
            np.array([0, 1, 0, 1, 0])
        ]
        
        fig, ax = plt.subplots()
        result_ax = plot_state_timeline(sequences, ax=ax)
        
        assert result_ax is ax
        plt.close(fig)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_dynamics(self):
        """Test complete dynamics analysis workflow."""
        from nanolocz.core.types import ParticleStack
        
        # Generate synthetic 2-state trajectory
        np.random.seed(42)
        n_particles = 5
        n_frames = 50
        box_size = 16
        
        # Create data with two distinct intensity levels
        data = np.zeros((n_particles, n_frames, box_size, box_size))
        for p in range(n_particles):
            for f in range(n_frames):
                # Alternate between two states
                state = (f // 10) % 2
                base_intensity = 10 if state == 0 else 20
                data[p, f] = base_intensity + np.random.randn(box_size, box_size) * 0.5
        
        centers = [(8, 8)] * n_particles
        frame_indices = [0] * n_particles
        
        stack = ParticleStack(data=data, centers_xy=centers,
                             frame_index=frame_indices, box_size=box_size)
        
        # Create tracks
        tracks = MockTrackCollection([
            MockTrack(particle_id=i,
                     times=list(range(n_frames)),
                     frame_indices=list(range(n_frames)))
            for i in range(n_particles)
        ])
        
        # Extract traces
        traces = extract_particle_traces(tracks, stack, features=['intensity'])
        
        assert len(traces) == n_particles
        
        # Try HMM fitting (skip if hmmlearn not available)
        try:
            result = fit_hmm(traces, 'intensity', n_states=2)
            
            assert result.n_states == 2
            assert result.transition_matrix.shape == (2, 2)
            assert len(result.dwell_times) > 0
            assert result.rate_constants.shape == (2, 2)
        except ImportError:
            pytest.skip("hmmlearn not available")


# Import fit_hmm for integration test
try:
    from nanolocz.core.dynamics import fit_hmm
except ImportError:
    fit_hmm = None
