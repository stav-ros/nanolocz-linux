# Task cards

Each card is one agent-sized unit. Do not start a card until dependencies are done.
The status ledger lives in `../STATUS.md`.

## P0 — foundation

- **NL-01** Audit the upstream MATLAB core and map every toolbox call. Depends: none. Acceptance: every call has file and line; GPL headers and NOTICE drafted.
- **NL-02** Build the golden parity harness. Depends: NL-01. Acceptance: CPU-only pytest, checksum validation, tolerance policy.
- **NL-03** Define typed core contracts. Depends: NL-01. Acceptance: `Frame`, `Meta`, `Localizations`, and `ParticleStack`; strict type checks.

## P1 — CPU core port

- **NL-10** Zarr schema and opener interface. Depends: NL-02, NL-03.
- **NL-11** `.gwy` and `.h5-jpk` openers. Depends: NL-10.
- **NL-12** `.spm`, `.jpk`, and `.ibw` openers. Depends: NL-10.
- **NL-13** `.asd` opener including trace-only files. Depends: NL-12.
- **NL-14** Line, plane, and weighted multi-plane levelling. Depends: NL-10.
- **NL-15** Filters, masks, scar removal, and profiles. Depends: NL-10.
- **NL-16** Detection, statistics, and masks. Depends: NL-15.
- **NL-17** Deterministic single-particle tracking. Depends: NL-16.

## P2 — GPU

- **NL-20** NumPy/CuPy backend switch and precision policy. Depends: NL-14.
- **NL-21** Batched levelling kernel. Depends: NL-20.
- **NL-22** Detection and statistics kernels. Depends: NL-20.
- **NL-23** LAFM splat kernel and FRC. Depends: NL-20.
- **NL-24** Simulation AFM kernels. Depends: NL-20.

## P3 — LAFM+

- **NL-30** Per-frame drift estimation. Depends: NL-22.
- **NL-31** Directional deskar filter. Depends: NL-15.
- **NL-32** Particle substack extraction. Depends: NL-16, NL-30.
- **NL-33** PCA to HDBSCAN grouping. Depends: NL-32.
- **NL-34** In-class alignment and averaging. Depends: NL-33.
- **NL-35** Tip estimation and regularized deconvolution. Depends: NL-34, NL-24.
- **NL-36** Dynamics traces, transitions, and dwell times. Depends: NL-33, NL-17.
- **NL-37** Napari replay hooks. Depends: NL-36, NL-41.

## P2.6 — simulation bridge

- **NL-50** Minimal PDB-to-AFM workflow. Depends: NL-03. Acceptance: import ATOM/HETATM coordinates, convert Angstroms to nanometres, render a coarse conical-tip AFM image, estimate a usable tip radius, and perform rough tip/translation fitting. This is intentionally approximate and CPU-first.
- **NL-51** CPU hard-collision height field and simulation parity improvements. Depends: NL-50.
- **NL-52** CUDA synthesis kernel. Depends: NL-51, NL-20.
- **NL-53** Masked NCC scorer and rough fitting improvements. Depends: NL-52.
- **NL-54** Optional BioAFMviewer validation and napari overlay. Depends: NL-53, NL-41.

## P4 — interface and ship

- **NL-40** Headless CLI and batch runner. Depends: NL-16, NL-23. **DONE** — 18/18 tests passing, CLI fully functional with 5 subcommands, JSON config support, batch processing queue. See `SPEC/NL-40-cli-batch-runner.md`, `SESSIONS/2026-09-05-NL-40.md`.
- **NL-41** Napari plugin v1. Depends: NL-40. Split into NL-41a (minimum usable plugin) and NL-41b (full analysis workflow). **DONE** — NL-41a complete with 13/13 tests, integrated API calls, shared PipelineConfig. See `SPEC/NL-41-napari-plugin.md`, `SESSIONS/2026-09-05-NL-41.md`.
- **NL-42** Docker and conda packaging. Depends: NL-40. **DONE** — Dockerfile, Dockerfile.gpu, .dockerignore, conda.recipe/, installation documentation. See `SPEC/NL-42-docker-conda-packaging.md`, `docs/installation/`.
- **NL-43** Benchmark report and v1.0-gpu release. Depends: all. **CURRENT CARD** — Benchmark suite created, RELEASE_CHECKLIST.md ready, example gallery structure in place. Final release pending user validation and artifact upload. See `SPEC/NL-43-benchmark-release.md`.
