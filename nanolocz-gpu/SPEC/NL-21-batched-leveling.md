# NL-21 — Batched levelling kernel

## Goal
Provide GPU-accelerated batched line and plane leveling for AFM movie sequences.

## Motivation
Processing AFM movies (hundreds to thousands of frames) requires efficient per-frame leveling operations. The CPU implementation iterates over frames in Python, creating a performance bottleneck. GPU acceleration enables:
- Parallel processing of all frames simultaneously
- Consistent precision policy across CPU/GPU backends
- Integration with the NL-20 backend system

## Dependencies
- **NL-20**: NumPy/CuPy backend switch and precision policy
- **NL-14**: Line/plane leveling algorithms (CPU reference)

## Acceptance Criteria

### 1. Batched line leveling
Implement `batch_line_leveling()` that:
- Takes a 3D movie array `(frames, rows, cols)` or list of `Frame` objects
- Applies line leveling to all frames in a single batched operation
- Uses `BackendContext` from NL-20 for array operations
- Supports optional 2D boolean mask broadcast across frames
- Preserves first line as reference level (matching `line_leveling()` semantics)
- Returns leveled movie and per-frame offsets

### 2. Batched plane leveling
Implement `batch_plane_leveling()` that:
- Takes a 3D movie array or list of `Frame` objects
- Applies plane leveling (polynomial surface fit) to all frames
- Uses masked least-squares fitting for robust estimation
- Returns leveled movie and per-frame plane coefficients

### 3. Backend integration
- Both functions accept optional `BackendContext` parameter
- Respect precision mode (REFERENCE=MIXED=HIGH)
- Use appropriate dtype based on backend configuration
- Fall back to CPU when GPU unavailable

### 4. Precision policy
- CPU REFERENCE mode: float64 throughout
- GPU MIXED mode: float32 computation, float64 output
- Results match CPU reference within tolerance policy

### 5. Error handling
- Validate input shapes (must be 3D for movies)
- Validate mask compatibility
- Raise `ValueError` for invalid inputs

## Implementation Notes

### Architecture
```
nanolocz/core/leveling.py (CPU reference + batch wrapper)
nanolocz/gpu/leveling.py (GPU kernels - future)
```

For NL-21, the batched operations are implemented in `nanolocz/core/leveling.py` using the NL-20 `BackendContext`. This provides:
- Single code path for CPU and GPU (via NumPy/CuPy abstraction)
- Easy migration to dedicated GPU kernels later
- Immediate performance benefits on large movies

### API
```python
def batch_line_leveling(
    movie: np.ndarray | list[Frame],
    mask: np.ndarray | None = None,
    context: BackendContext | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply line leveling to all frames in a movie.
    
    Args:
        movie: 3D array (frames, rows, cols) or list of Frame objects
        mask: Optional 2D boolean mask (rows, cols), broadcast to all frames
        context: Optional BackendContext for GPU execution
        
    Returns:
        Tuple of (leveled_movie, per_frame_offsets)
    """
```

## Test Strategy

### Unit tests (`tests/test_leveling_nl21.py`)
- Batch line leveling matches per-frame reference
- Mask handling and broadcasting
- Precision policy enforcement
- Input validation (shape, mask errors)

### Integration tests
- Large movie processing (>100 frames)
- Memory efficiency verification
- Round-trip with detection pipeline

## Evidence
- Implementation: `nanolocz/core/leveling.py` (batch_line_leveling function)
- Tests: `tests/test_leveling_nl21.py` (4 passing tests)
- Session doc: `SESSIONS/2026-08-29-NL-21.md`
