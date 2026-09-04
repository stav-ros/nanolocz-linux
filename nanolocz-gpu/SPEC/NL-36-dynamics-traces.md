# NL-36: Dynamics Traces, Transitions, and Dwell Times

**Priority:** High  
**Phase:** Phase 3 — LAFM+ Core Pipeline  
**Status:** in_progress  
**Created:** 2026-09-04  
**Dependencies:** NL-17 (Tracking), NL-33 (PCA/HDBSCAN Classification)

---

## Goals

Extract time-series dynamics from classified particle tracks to enable quantitative analysis of:
1. **State transitions** — Detect conformational changes or binding events
2. **Dwell times** — Measure residence times in each state
3. **Transition kinetics** — Estimate rate constants between states
4. **Hidden Markov Models** — Infer latent states from noisy observations
5. **Change-point detection** — Identify abrupt transitions in trajectories

---

## Dependencies

| Card | Status | Usage |
|------|--------|-------|
| NL-17 | done | Single-particle tracking provides trajectories with temporal ordering |
| NL-33 | done | PCA/HDBSCAN classification assigns discrete states to particles |
| NL-34 | done | Aligned class averages improve signal quality for trace extraction |
| NL-32 | done | Particle substacks provide time-resolved intensity data |

---

## Acceptance Criteria

### AC-1: Trace Extraction
- [ ] Extract intensity/feature traces from tracked particles over time
- [ ] Support multiple features: height, volume, area, PCA scores
- [ ] Handle missing frames (gaps in tracks) with interpolation or masking
- [ ] Batch extraction for all tracks in a dataset

### AC-2: Hidden Markov Model (HMM) Analysis
- [ ] Implement Gaussian HMM for state inference
- [ ] Automatic selection of number of states (BIC/AIC criteria)
- [ ] Estimate transition probability matrix
- [ ] Decode most likely state sequence (Viterbi algorithm)
- [ ] GPU acceleration for HMM fitting (optional, via hmmlearn or custom CuPy)

### AC-3: Change-Point Detection
- [ ] Implement PELT (Pruned Exact Linear Time) algorithm
- [ ] Implement binary segmentation for comparison
- [ ] Support multiple cost functions: L2, L1, kernel-based
- [ ] Detect change-points in multi-dimensional traces

### AC-4: Dwell Time Analysis
- [ ] Extract dwell times per state from HMM/state assignments
- [ ] Fit exponential distributions to dwell times
- [ ] Estimate mean lifetimes and rate constants
- [ ] Statistical tests for multi-exponential behavior

### AC-5: Transition Kinetics
- [ ] Build transition count matrices
- [ ] Compute transition rates (probability / time step)
- [ ] Identify preferred transition pathways
- [ ] Visualize transition networks

### AC-6: Visualization Tools
- [ ] Plot traces with inferred states overlaid
- [ ] Histogram dwell times with fitted distributions
- [ ] Network diagram of transition probabilities
- [ ] State sequence timeline (Gantt-style plot)

---

## API Design

### Trace Extraction

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

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
    change_points: list[np.ndarray]  # per-trace change point indices
    bic_scores: Optional[np.ndarray] = None  # model selection scores
```

### Main Functions

```python
# nanolocz/core/dynamics.py

def extract_particle_traces(
    tracks: TrackCollection,
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
        Features to extract: 'height', 'volume', 'area', 'pca_scores'
    interpolate_gaps : bool
        Whether to interpolate missing frames in tracks
    max_gap_size : int
        Maximum gap size to interpolate (larger gaps remain as NaN)
    
    Returns
    -------
    list[DynamicsTrace]
        One trace per particle with requested features
    """
    ...

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
    ...

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
    ...

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
    ...

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
    ...

def plot_trace_with_states(trace: DynamicsTrace, feature_name: str, ax=None):
    """Plot trace with inferred state colors."""
    ...

def plot_dwell_histogram(dwell_times: np.ndarray, ax=None):
    """Plot dwell time histogram with exponential fit."""
    ...

def plot_transition_network(transition_matrix: np.ndarray, ax=None):
    """Visualize transition probabilities as network diagram."""
    ...

def plot_state_timeline(state_sequences: list[np.ndarray], ax=None):
    """Gantt-style plot of state sequences across particles."""
    ...
```

### GPU Acceleration

```python
# nanolocz/gpu/dynamics.py

def fit_hmm_gpu(
    data: cp.ndarray,
    n_states: int,
    max_iter: int = 100,
    tol: float = 1e-6
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    GPU-accelerated HMM fitting using Baum-Welch algorithm.
    
    Parameters
    ----------
    data : cp.ndarray
        Observation sequences, shape: (n_sequences, seq_length, n_features)
    n_states : int
        Number of hidden states
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    
    Returns
    -------
    transition_matrix : cp.ndarray
        State transition probabilities
    emission_means : cp.ndarray
        Gaussian emission means
    emission_covars : cp.ndarray
        Gaussian emission covariances
    """
    ...
```

---

## Test Plan

### Unit Tests
1. **Trace extraction**
   - Single particle with complete track
   - Particle with gaps (interpolation vs masking)
   - Multi-feature extraction
   - Empty track handling

2. **HMM fitting**
   - Known 2-state system recovery
   - Model selection (BIC/AIC correctness)
   - Viterbi decoding accuracy
   - Transition matrix estimation

3. **Change-point detection**
   - Step function recovery
   - Multiple change points
   - Noise robustness
   - Method comparison (PELT vs binary seg)

4. **Dwell time analysis**
   - Single exponential distribution
   - Multi-exponential mixture
   - Mean lifetime accuracy
   - Rate constant estimation

5. **Visualization**
   - Trace plotting with states
   - Dwell histogram with fit
   - Network diagram layout
   - Timeline rendering

### Integration Tests
1. End-to-end: tracks → traces → HMM → dwell times → rates
2. Combined with NL-33 classification states
3. Batch processing multiple trajectories
4. GPU parity tests (when CuPy available)

---

## Implementation Notes

### HMM Implementation Options
1. **hmmlearn library** — Mature, scikit-learn compatible API
2. **pomegranate** — Modern, supports Bayesian HMMs
3. **Custom CuPy implementation** — For GPU acceleration

Recommendation: Start with `hmmlearn` for CPU, add CuPy backend for GPU.

### Change-Point Detection
- **ruptures library** — Comprehensive implementation of PELT, binary segmentation
- Add as optional dependency in `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  dynamics = ["hmmlearn>=0.3", "ruptures>=1.1"]
  ```

### Handling Missing Data
- Tracks from NL-17 may have gaps (missing detections)
- Options:
  1. Linear interpolation for small gaps (< max_gap_size)
  2. Mask NaN values during HMM fitting
  3. Use HMM variants that handle missing observations

---

## Deliverables

- [x] SPEC/NL-36-dynamics-traces.md (this file)
- [ ] `nanolocz/core/dynamics.py` — Core implementation
- [ ] `nanolocz/gpu/dynamics.py` — GPU acceleration
- [ ] `tests/test_dynamics_nl36.py` — Test suite
- [ ] SESSIONS/2026-09-04-NL-36.md — Session handoff
- [ ] Update STATUS.md — Mark NL-36 complete

---

## Success Metrics

- **Test coverage**: >90% for core algorithms
- **Performance**: Process 1000 traces (100 frames each) in <10s on CPU
- **Accuracy**: Recover known states in synthetic data with >95% accuracy
- **GPU speedup**: 5-10x faster for HMM fitting with >100 sequences

---

## Future Extensions

- **NL-36b**: Bayesian HMM with Dirichlet priors
- **NL-36c**: Hierarchical HMM for multi-level dynamics
- **NL-36d**: Deep learning-based state inference (variational autoencoders)
- **NL-36e**: Real-time dynamics analysis for streaming AFM data
