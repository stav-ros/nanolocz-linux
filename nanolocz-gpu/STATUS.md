# Project status

Last updated: 2026-08-28
Current phase: Phase 2 — Data I/O & Storage (COMPLETE)
Current card: NL-11
Status: in_progress — Implementing .gwy and .h5-jpk openers

## Progress

| Card | Area | State | Evidence |
|---|---|---|---|
| NL-01 | Fork audit and toolbox map | done | `ADR/0000-toolbox-map.md`; upstream revision `e41575c` audited |
| NL-02 | Golden parity harness | done | `SPEC/parity.md`; checksum loader, centralized tolerances, 7 CPU tests, CI workflow |
| NL-03 | Typed core contracts | done | `nanolocz/core/types.py`; 43 type contract tests green; protocols with @runtime_checkable; SESSIONS/2026-08-28-NL-03.md |
| NL-10 | Zarr schema and opener interface | **done** | `SPEC/NL-10-zarr-schema.md`; `nanolocz/io/store.py`; 29/29 I/O tests green; complete round-trip validation; SESSIONS/2026-08-28-NL-10.md |
| NL-11 | .gwy and .h5-jpk Openers | **in_progress** | Gwyddion integration started; JPK HDF5 reader planned |
| NL-12–NL-17 | CPU core port | not_started | blocked by NL-11 |
| NL-20–NL-24 | GPU backend and kernels | not_started | blocked by P1 |
| NL-30–NL-37 | LAFM+ science | not_started | blocked by core/GPU work |
| NL-40–NL-43 | Interface and shipping | not_started | blocked by core/GPU work |
| NL-50–NL-54 | Simulation bridge | not_started | **unblocked** — can start NL-50 |

## Phase 1 reconciliation evidence

- Canonical package: `nanolocz/`; obsolete `src/nanolocz/` removed.
- Single valid `pyproject.toml` with top-level package discovery and `[test]` extra.
- Parity fixtures and tolerances are exposed from `nanolocz.parity`.
- Validation: `99 passed, 1 skipped`; editable install succeeds; project self-check passes.

## Phase 2 completion summary

### NL-10 Deliverables
- Zarr schema specification in `SPEC/NL-10-zarr-schema.md`
- Complete `NanoLoczStore` implementation in `nanolocz/io/store.py`
- Full I/O support for movies, localizations, tracks, and particle stacks
- Round-trip validation with 29/29 tests passing
- Context manager support and proper resource cleanup
- Schema versioning and validation
- Compression with Blosc Zstandard (zstd) level 3
- Tuple-to-list conversion for Zarr JSON compatibility
- Proper error handling for missing data (KeyError for missing tracks/localizations)

### NL-10 Test Coverage
- Store creation and initialization
- Movie save/load with metadata preservation
- Localizations round-trip with optional fields
- Tracks save/load with empty track handling
- Particle stacks storage and retrieval
- Compressed storage validation
- Multiple save overwrites
- Complete round-trip integration test
- Opener interface tests (Zarr, HDF5, TIFF)
- Schema validation and version compatibility

## Next action

Execute NL-11 only: implement .gwy (Gwyddion) and .h5-jpk (JPK Instruments) file format openers.
These proprietary AFM formats are critical for real-world data ingestion. The opener interface
from NL-10 provides the foundation. Focus on:
- Gwyddion (.gwy) reader using gwyfile library
- JPK (.h5-jpk) HDF5-based reader
- Unified metadata extraction (pixel_size, units, scan parameters)
- Integration with existing open_nanolocz() dispatcher
- Test fixtures with representative AFM data

## Self-check contract

A task may move to `done` only when:

- its acceptance criteria in `SPEC/tasks.md` are satisfied;
- the targeted test command is green;
- `python tools/project_check.py` is green;
- the evidence is recorded in this file; and
- a session handoff exists under `SESSIONS/`.

Allowed states: `not_started`, `in_progress`, `blocked`, `done`.
