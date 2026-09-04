"""
NL-36: Dynamics Traces, Transitions, and Dwell Times

Core implementation for extracting and analyzing time-series dynamics from particle trajectories.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

# Try to import hmmlearn for HMM
try:
    from hmmlearn import hmm
    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False

# Try to import ruptures for change-point detection
try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except ImportError:
    _HAS_RUPTURES = False

from .types import ParticleTrack, ParticleStack


@dataclass
class DynamicsTrace:
    """Time-series data for a single particle trajectory."""
    particle_id: int
    time_points: np.ndarray  # shape: (n_frames,)
    features: dict[str, np.ndarray]  # feature_name -> (n_frames,)
    state_sequence: Optional[np.ndarray] = None  # shape: (n_frames,)
    confidence: Optional[np.ndarray] = None  # state assignment confidence


@dataclass
class DynamicsResult:
    """Complete dynamics analysis results."""
    traces: list[DynamicsTrace]
    n_states: int
    transition_matrix: np.ndarray  # shape: (n_states, n_states)
    dwell_times: dict[int, np.ndarray]  # state_id -> array of dwell times
    rate_constants: np.ndarray  # shape: (n_states, n_states)
    change_points: list[np.ndarray] = field(default_factory=list)  # per-trace change point indices
    bic_scores: Optional[np.ndarray] = None  # model selection scores
    emission_means: Optional[np.ndarray] = None  # Gaussian emission means
    emission_covars: Optional[np.ndarray] = None  # Gaussian emission covariances


def extract_particle_traces(
    tracks: List[ParticleTrack],
    particle_stacks: ParticleStack,
    features: list[str] = None,
    interpolate_gaps: bool = True,
    max_gap_size: int = 2
) -> list[DynamicsTrace]:
    """
    Extract time-series features from tracked particles.
    
    Parameters
    ----------
    tracks : TrackCollection
        Output from NL-17 tracking
    particle_stacks : ParticleStack
        Aligned particle substacks from NL-32/NL-34
    features : list[str]
        Features to extract: 'height', 'volume', 'area', 'intensity'
    interpolate_gaps : bool
        Whether to interpolate missing frames in tracks
    max_gap_size : int
        Maximum gap size to interpolate (larger gaps remain as NaN)
    
    Returns
    -------
    list[DynamicsTrace]
        One trace per particle with requested features
    """
    if features is None:
        features = ['height', 'intensity']
    
    traces = []
    
    # Get particle data
    data = particle_stacks.data  # (n_particles, n_frames, H, W) or (n_particles, H, W)
    centers = particle_stacks.centers_xy
    frame_indices = particle_stacks.frame_index
    
    n_particles = particle_stacks.n_particles
    
    for track_id, track in enumerate(tracks.tracks):
        if track_id >= n_particles:
            break
            
        # Get time points for this track
        time_points = np.array(track.times)
        n_frames = len(time_points)
        
        if n_frames == 0:
            continue
        
        # Extract features for each time point
        extracted_features = {}
        
        for feature_name in features:
            feature_values = np.full(n_frames, np.nan)
            
            for i, (frame_idx, time) in enumerate(zip(track.frame_indices, track.times)):
                if frame_idx < 0 or frame_idx >= data.shape[1] if data.ndim == 4 else 1:
                    continue
                
                # Extract feature based on type
                if data.ndim == 4:
                    particle_frame = data[track_id, frame_idx]
                else:
                    particle_frame = data[track_id]
                
                if feature_name == 'height':
                    feature_values[i] = np.max(particle_frame) - np.min(particle_frame)
                elif feature_name == 'intensity':
                    feature_values[i] = np.mean(particle_frame)
                elif feature_name == 'volume':
                    feature_values[i] = np.sum(particle_frame)
                elif feature_name == 'area':
                    threshold = np.mean(particle_frame) + 2 * np.std(particle_frame)
                    feature_values[i] = np.sum(particle_frame > threshold)
                else:
                    # Custom feature
                    feature_values[i] = np.mean(particle_frame)
            
            # Interpolate gaps if requested
            if interpolate_gaps and max_gap_size > 0:
                valid_mask = ~np.isnan(feature_values)
                if np.any(valid_mask):
                    x_valid = np.where(valid_mask)[0]
                    y_valid = feature_values[valid_mask]
                    
                    # Check gap sizes
                    gaps = np.diff(x_valid)
                    large_gaps = np.where(gaps > max_gap_size)[0]
                    
                    if len(large_gaps) == 0:
                        # Interpolate all gaps
                        interpolator = interp1d(x_valid, y_valid, kind='linear', 
                                               fill_value='extrapolate', bounds_error=False)
                        feature_values = interpolator(np.arange(n_frames))
                    else:
                        # Only interpolate small gaps
                        for start_idx in range(len(x_valid) - 1):
                            gap_size = x_valid[start_idx + 1] - x_valid[start_idx]
                            if gap_size <= max_gap_size:
                                # Interpolate this gap
                                for j in range(x_valid[start_idx] + 1, x_valid[start_idx + 1]):
                                    t = (j - x_valid[start_idx]) / gap_size
                                    feature_values[j] = y_valid[start_idx] * (1 - t) + y_valid[start_idx + 1] * t
            
            extracted_features[feature_name] = feature_values
        
        trace = DynamicsTrace(
            particle_id=track_id,
            time_points=time_points,
            features=extracted_features
        )
        traces.append(trace)
    
    return traces


def fit_hmm(
    traces: list[DynamicsTrace],
    feature_name: str,
    n_states: int = None,
    max_states: int = 10,
    criterion: str = 'bic',
    random_state: int = 42
) -> DynamicsResult:
    """
    Fit Hidden Markov Model to extract latent states.
    
    Parameters
    ----------
    traces : list[DynamicsTrace]
        Input traces with features
    feature_name : str
        Which feature to use for HMM fitting
    n_states : int, optional
        Fixed number of states (if None, auto-select via BIC/AIC)
    max_states : int
        Maximum states to try for model selection
    criterion : str
        Model selection criterion: 'bic' or 'aic'
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    DynamicsResult
        Fitted HMM with state sequences, transition matrix, etc.
    """
    if not _HAS_HMMLEARN:
        raise ImportError("hmmlearn is required for HMM fitting. Install with: pip install hmmlearn")
    
    np.random.seed(random_state)
    
    # Prepare data: concatenate all traces
    observations = []
    trace_lengths = []
    
    for trace in traces:
        if feature_name not in trace.features:
            continue
        feat_data = trace.features[feature_name]
        # Remove NaN values
        valid_mask = ~np.isnan(feat_data)
        if np.any(valid_mask):
            observations.append(feat_data[valid_mask].reshape(-1, 1))
            trace_lengths.append(np.sum(valid_mask))
    
    if len(observations) == 0:
        raise ValueError("No valid observations found")
    
    X = np.vstack(observations)
    
    # Model selection if n_states not specified
    if n_states is None:
        bic_scores = []
        models = []
        
        for n in range(2, min(max_states + 1, len(X) // 10)):
            model = hmm.GaussianHMM(n_components=n, covariance_type='full', 
                                   n_iter=100, random_state=random_state)
            try:
                model.fit(X)
                if criterion == 'bic':
                    score = model.bic(X)
                else:  # aic
                    score = model.aic(X)
                bic_scores.append(score)
                models.append(model)
            except Exception:
                bic_scores.append(np.inf)
                models.append(None)
        
        # Select best model (lowest BIC/AIC)
        best_idx = np.argmin(bic_scores)
        n_states = best_idx + 2
        model = models[best_idx]
        bic_scores = np.array(bic_scores)
    else:
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='full',
                               n_iter=100, random_state=random_state)
        model.fit(X)
        bic_scores = None
    
    # Decode state sequences for each trace
    n_states = model.n_components
    transition_matrix = model.transmat_
    emission_means = model.means_.flatten()
    emission_covars = model.covars_.flatten() if model.covars_.ndim > 1 else model.covars_
    
    # Assign states to each trace
    state_sequences = []
    obs_idx = 0
    
    for trace in traces:
        if feature_name not in trace.features:
            state_sequences.append(None)
            continue
            
        feat_data = trace.features[feature_name]
        valid_mask = ~np.isnan(feat_data)
        n_valid = np.sum(valid_mask)
        
        if n_valid == 0:
            state_sequences.append(None)
            continue
        
        # Get observations for this trace
        X_trace = feat_data[valid_mask].reshape(-1, 1)
        
        # Predict states
        states_trace = model.predict(X_trace)
        
        # Create full state sequence with NaN for missing frames
        full_states = np.full(len(feat_data), -1, dtype=int)
        full_states[valid_mask] = states_trace
        
        trace.state_sequence = full_states
        state_sequences.append(full_states)
    
    # Compute dwell times
    time_step = np.median([np.diff(t.time_points).mean() for t in traces if len(t.time_points) > 1])
    dwell_times = compute_dwell_times(state_sequences, time_step)
    
    # Estimate rate constants
    rate_constants = estimate_rate_constants(transition_matrix, dwell_times, time_step)
    
    return DynamicsResult(
        traces=traces,
        n_states=n_states,
        transition_matrix=transition_matrix,
        dwell_times=dwell_times,
        rate_constants=rate_constants,
        bic_scores=bic_scores,
        emission_means=emission_means,
        emission_covars=emission_covars
    )


def detect_change_points(
    traces: list[DynamicsTrace],
    feature_name: str,
    method: str = 'pelt',
    penalty: float = None,
    min_segment_length: int = 5,
    cost_function: str = 'l2'
) -> list[np.ndarray]:
    """
    Detect abrupt transitions using change-point detection.
    
    Parameters
    ----------
    traces : list[DynamicsTrace]
        Input traces
    feature_name : str
        Feature to analyze
    method : str
        Algorithm: 'pelt', 'binary_seg', 'window'
    penalty : float, optional
        Penalty parameter (auto-estimated if None)
    min_segment_length : int
        Minimum segment length between change points
    cost_function : str
        Cost function: 'l2', 'l1', 'ar', 'kernel'
    
    Returns
    -------
    list[np.ndarray]
        Change point indices for each trace
    """
    if not _HAS_RUPTURES:
        raise ImportError("ruptures is required for change-point detection. Install with: pip install ruptures")
    
    all_change_points = []
    
    for trace in traces:
        if feature_name not in trace.features:
            all_change_points.append(np.array([]))
            continue
        
        signal = trace.features[feature_name]
        
        # Remove NaN values
        valid_mask = ~np.isnan(signal)
        if np.sum(valid_mask) < min_segment_length * 2:
            all_change_points.append(np.array([]))
            continue
        
        signal_clean = signal[valid_mask]
        
        # Auto-estimate penalty if not provided
        if penalty is None:
            # Use heuristic based on signal variance
            penalty = np.var(signal_clean) * np.log(len(signal_clean))
        
        # Select algorithm
        if method == 'pelt':
            algo = rpt.Pelt(model=cost_function, min_length=min_segment_length).fit(signal_clean)
            change_points = algo.predict(pen=penalty)
        elif method == 'binary_seg':
            algo = rpt.Binseg(model=cost_function, min_length=min_segment_length).fit(signal_clean)
            change_points = algo.predict(n_bkps=5)  # Default to 5 change points
        elif method == 'window':
            algo = rpt.Window(model=cost_function, min_length=min_segment_length).fit(signal_clean)
            change_points = algo.predict(n_bkps=5)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to original indices
        if len(change_points) > 0:
            # Map back to original time indices
            valid_indices = np.where(valid_mask)[0]
            if len(change_points) < len(valid_indices):
                mapped_cps = valid_indices[change_points[:-1]]  # Exclude last point (end of signal)
            else:
                mapped_cps = change_points[:-1]
        else:
            mapped_cps = []
        
        all_change_points.append(np.array(mapped_cps))
    
    return all_change_points


def compute_dwell_times(
    state_sequences: list[np.ndarray],
    time_step: float
) -> dict[int, np.ndarray]:
    """
    Extract dwell times per state from state sequences.
    
    Parameters
    ----------
    state_sequences : list[np.ndarray]
        State assignments over time for each trace
    time_step : float
        Time between frames in seconds
    
    Returns
    -------
    dict[int, np.ndarray]
        Dwell times (in seconds) for each state
    """
    dwell_times = {}
    
    for seq in state_sequences:
        if seq is None or len(seq) == 0:
            continue
        
        # Find contiguous segments of each state
        unique_states = np.unique(seq[seq >= 0])  # Exclude -1 (no state)
        
        for state in unique_states:
            if state not in dwell_times:
                dwell_times[state] = []
            
            # Find runs of this state
            mask = (seq == state).astype(int)
            diff = np.diff(mask)
            
            # Starts are where diff goes from 0 to 1
            starts = np.where(diff == 1)[0] + 1
            # Ends are where diff goes from 1 to 0
            ends = np.where(diff == -1)[0] + 1
            
            # Handle edge cases
            if mask[0] == 1:
                starts = np.concatenate([[0], starts])
            if mask[-1] == 1:
                ends = np.concatenate([ends, [len(mask)]])
            
            # Compute dwell times
            durations = (ends - starts) * time_step
            dwell_times[state].extend(durations.tolist())
    
    # Convert to arrays
    dwell_times = {k: np.array(v) for k, v in dwell_times.items()}
    
    return dwell_times


def estimate_rate_constants(
    transition_matrix: np.ndarray,
    dwell_times: dict[int, np.ndarray],
    time_step: float
) -> np.ndarray:
    """
    Estimate kinetic rate constants from transitions and dwell times.
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        State transition probabilities
    dwell_times : dict[int, np.ndarray]
        Dwell times per state
    time_step : float
        Time between observations
    
    Returns
    -------
    np.ndarray
        Rate constant matrix (units: 1/s)
    """
    n_states = len(transition_matrix)
    rate_matrix = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        if i in dwell_times and len(dwell_times[i]) > 0:
            # Mean dwell time for state i
            mean_dwell = np.mean(dwell_times[i])
            
            if mean_dwell > 0:
                # Total exit rate from state i
                total_exit_rate = 1.0 / mean_dwell
                
                # Distribute to individual transitions
                for j in range(n_states):
                    if i != j:
                        rate_matrix[i, j] = total_exit_rate * transition_matrix[i, j]
    
    return rate_matrix


def plot_trace_with_states(trace: DynamicsTrace, feature_name: str, ax=None):
    """Plot trace with inferred state colors."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    if feature_name not in trace.features:
        raise ValueError(f"Feature '{feature_name}' not found in trace")
    
    y = trace.features[feature_name]
    t = trace.time_points
    
    ax.plot(t, y, 'k-', linewidth=1, label='Signal')
    
    if trace.state_sequence is not None:
        # Color-code by state
        unique_states = np.unique(trace.state_sequence[trace.state_sequence >= 0])
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
        
        for state, color in zip(unique_states, colors):
            mask = trace.state_sequence == state
            if np.any(mask):
                ax.fill_between(t[mask], y[mask], alpha=0.3, color=color, 
                              label=f'State {state}')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(feature_name)
    ax.legend()
    
    return ax


def plot_dwell_histogram(dwell_times: np.ndarray, ax=None):
    """Plot dwell time histogram with exponential fit."""
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    if len(dwell_times) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return ax
    
    # Histogram
    counts, bins, _ = ax.hist(dwell_times, bins=30, alpha=0.7, density=True, 
                             label='Data', edgecolor='black')
    
    # Fit exponential
    def exp_func(x, tau):
        return (1/tau) * np.exp(-x/tau)
    
    try:
        popt, _ = curve_fit(exp_func, bins[:-1], counts, p0=[np.mean(dwell_times)])
        tau_fit = popt[0]
        x_fit = np.linspace(0, max(dwell_times), 100)
        ax.plot(x_fit, exp_func(x_fit, tau_fit), 'r-', linewidth=2, 
               label=f'Exp fit (τ={tau_fit:.2f}s)')
    except Exception:
        pass
    
    ax.set_xlabel('Dwell time (s)')
    ax.set_ylabel('Probability density')
    ax.legend()
    
    return ax


def plot_transition_network(transition_matrix: np.ndarray, ax=None):
    """Visualize transition probabilities as network diagram."""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    n_states = transition_matrix.shape[0]
    
    # Create graph
    G = nx.DiGraph()
    
    for i in range(n_states):
        G.add_node(i)
        for j in range(n_states):
            if i != j and transition_matrix[i, j] > 0.01:  # Threshold for visualization
                G.add_edge(i, j, weight=transition_matrix[i, j])
    
    # Layout
    pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Draw edges with width proportional to weight
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]
    if len(weights) > 0:
        nx.draw_networkx_edges(G, pos, width=np.array(weights) * 5, 
                              arrowstyle='->', arrowsize=20, ax=ax)
    
    ax.set_title('Transition Network')
    ax.axis('off')
    
    return ax


def plot_state_timeline(state_sequences: list[np.ndarray], ax=None):
    """Gantt-style plot of state sequences across particles."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_sequences = [(i, seq) for i, seq in enumerate(state_sequences) 
                      if seq is not None and len(seq) > 0]
    
    if len(valid_sequences) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return ax
    
    # Limit number of traces for visualization
    max_traces = min(50, len(valid_sequences))
    valid_sequences = valid_sequences[:max_traces]
    
    all_states = set()
    for _, seq in valid_sequences:
        all_states.update(seq[seq >= 0])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_states)))
    state_to_color = {s: colors[i] for i, s in enumerate(sorted(all_states))}
    
    for idx, (particle_id, seq) in enumerate(valid_sequences):
        # Find runs of each state
        for state in all_states:
            mask = (seq == state).astype(int)
            if not np.any(mask):
                continue
            
            diff = np.diff(mask)
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            
            if mask[0] == 1:
                starts = np.concatenate([[0], starts])
            if mask[-1] == 1:
                ends = np.concatenate([ends, [len(mask)]])
            
            for start, end in zip(starts, ends):
                ax.barh(idx, end - start, left=start, height=0.8, 
                       color=state_to_color.get(state, 'gray'), alpha=0.7)
    
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Particle index')
    ax.set_title('State Timeline')
    
    return ax
