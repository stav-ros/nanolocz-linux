# NL-02 — golden parity harness contract

## Purpose

Provide one CPU-only test surface for comparing future Python implementations with
MATLAB/Octave reference outputs. This card creates infrastructure only; it must not
implement NanoLocz analysis algorithms.

## Fixture layout

Each module owns a directory under `golden/`:

```text
golden/<module>/<case>.npy
golden/<module>/<case>.npy.sha256
golden/<module>/ENV.md
```

The checksum sidecar contains a SHA-256 digest for the exact `.npy` bytes, optionally
followed by whitespace and the filename. Pickle-enabled NumPy arrays are rejected.
Missing sidecars, malformed digests, or mismatches fail loudly.

## Tolerance policy

- CPU/reference and MATLAB float64 paths: `rtol=1e-5`, `atol=1e-8`.
- Future GPU float32 paths: `rtol=1e-3`, `atol=1e-5`.
- Integer, boolean, shape, and metadata values: exact equality.
- NaNs compare equal only at corresponding positions.

The policy is centralized in `nanolocz.parity.tolerance`; tests must not introduce
ad-hoc tolerances without an ADR or an explicit fixture-level reason.

## Required commands

```bash
python -m pytest -q
python tools/project_check.py
```

CI must run without a GPU and without MATLAB. Golden fixture capture is a separate
workflow and is intentionally not part of NL-02.
