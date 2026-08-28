# NL-10 specification — Zarr schema and opener interface

Date: 2026-08-28
Card: NL-10
State: in_progress
Dependencies: NL-02 ✓, NL-03 ✓

## Objective

Define a Zarr-based storage schema for AFM particle data and implement a unified
opener interface that provides consistent access to raw movies, processed 
localizations, and particle stacks across all file formats.

## Motivation

AFM experiments generate multi-dimensional data:
- Time series of height images (movies)
- Detected particle coordinates per frame
- Extracted particle substacks for analysis
- Tracking results linking particles across frames

Zarr provides:
- Chunked, compressed N-dimensional arrays
- Cloud-native storage (S3, GCS, local filesystem)
- Parallel read/write access
- Metadata alongside data in hierarchical structure
- Interoperability with scientific Python ecosystem

## Acceptance criteria

1. **Zarr schema documented** in `SPEC/zarr-schema.md`:
   - Group structure for movies, localizations, tracks, and particle stacks
   - Array shapes, dtypes, and chunking recommendations
   - Metadata conventions following CF-like conventions for microscopy
   - Provenance tracking for processing steps

2. **Opener interface implemented** in `nanolocz/io/opener.py`:
   - `open_nanolocz(path: str, mode: str = "r") -> NanoLoczStore`
   - Support for `.zarr`, `.h5`, `.tiff` input formats
   - Automatic schema validation on open
   - Lazy loading of large arrays

3. **NanoLoczStore class** with methods:
   - `load_movie(frame_range=None) -> np.ndarray`
   - `load_localizations(frame_range=None) -> Localizations`
   - `load_tracks() -> list[ParticleTrack]`
   - `load_particle_stacks() -> ParticleStack`
   - `save_movie(data, metadata)`
   - `save_localizations(localizations)`
   - `save_tracks(tracks)`
   - `close()`

4. **Tests** in `tests/test_io_nl10.py`:
   - Schema validation tests
   - Round-trip save/load tests
   - Multi-format opener tests
   - Lazy loading verification
   - Edge cases (empty data, missing groups)

5. **Documentation**:
   - Usage examples in docstrings
   - Migration guide from MATLAB structures
   - Performance tuning recommendations

## Deliverables

- [ ] `SPEC/zarr-schema.md` — Complete schema specification
- [ ] `nanolocz/io/opener.py` — Opener interface implementation
- [ ] `nanolocz/io/store.py` — NanoLoczStore class
- [ ] `nanolocz/io/__init__.py` — Public API exports
- [ ] `tests/test_io_nl10.py` — Comprehensive test suite
- [ ] Update `STATUS.md` with progress
- [ ] Session handoff document in `SESSIONS/`

## Technical notes

### Proposed Zarr structure

```
experiment.zarr/
├── .zgroup              # Zarr group metadata
├── .zattrs              # Experiment-level metadata
├── movie/               # Raw or processed image stack
│   ├── .zarray          # Shape: (frames, height, width)
│   └── .zattrs          # Pixel size, units, acquisition params
├── localizations/       # Detected coordinates
│   ├── xy/.zarray       # Shape: (n_detections, 2), float64
│   ├── frame_index/.zarray  # Shape: (n_detections,), int32
│   ├── intensities/.zarray  # Shape: (n_detections,), float64
│   └── .zattrs          # Detection method, thresholds
├── tracks/              # Linked trajectories
│   ├── track_000/
│   │   ├── frames/.zarray   # Frame indices
│   │   ├── x/.zarray        # X coordinates
│   │   ├── y/.zarray        # Y coordinates
│   │   └── .zattrs          # Track metadata
│   ├── track_001/
│   └── .zattrs          # Tracking parameters
└── particle_stacks/     # Extracted substacks
    ├── data/.zarray     # Shape: (n_particles, n_frames, box_size, box_size)
    ├── centers_xy/.zarray   # Center positions
    └── .zattrs          # Box size, extraction method
```

### Chunking strategy

- Movie: `(1, height, width)` for frame-wise access
- Localizations: `(1000,)` for batch processing
- Tracks: Per-track storage for independent access
- Particle stacks: `(1, n_frames, box_size, box_size)` per particle

### Compression

- Default: Blosc Zstandard (zstd) level 3
- Lossless for integer data
- Consider lossy compression for large movies (Blosc shuffle + zstd)

### Version compatibility

- Schema version stored in `.zattrs` at root
- Backward compatibility plan for future schema changes
- Migration utilities for old format data

## Out of scope (future cards)

- `.gwy`, `.spm`, `.jpk`, `.ibw`, `.asd` format readers (NL-11, NL-12, NL-13)
- GPU-accelerated I/O (NL-20+)
- Napari integration (NL-37, NL-41)
- Cloud storage backends (deferred)

## Next steps

1. Draft Zarr schema specification
2. Implement basic opener interface
3. Create NanoLoczStore class
4. Write comprehensive tests
5. Document usage patterns
