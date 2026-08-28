# Phase 1 reconciliation handoff

Date: 2026-08-28
Scope: reconcile the merged Python port foundation before NL-03

## Completed

- Selected `nanolocz-gpu/nanolocz/` as the canonical package layout.
- Migrated typed contracts and NL-02 parity utilities from `src/nanolocz/` into the canonical package.
- Removed the obsolete `src/` package tree and duplicate `nanolocz/core/parity.py` implementation.
- Replaced the conflicting `pyproject.toml` with one valid configuration.
- Removed the undeclared `nanolocz.cli:main` entry point until the CLI task is implemented.
- Updated `AGENT.md`, the ADR hierarchy, and status/progress documentation.
- Extended `tools/project_check.py` to validate TOML, package discovery, required parity files, and obsolete-layout removal.
- Fixed TIFF loading so normal reads do not require tifffile's optional Zarr adapter.
- Added project-structure tests for the canonical package and packaging metadata.

## Validation

From `nanolocz-gpu/`:

```text
python -m pytest -q
27 passed, 1 skipped (CuPy unavailable)

python tools/project_check.py
PASS project scaffold is self-consistent; current card NL-03
PASS canonical package and pyproject.toml are aligned

python -m pip install --no-deps -e ".[test]"
Successfully built and installed nanolocz-0.1.0.dev0
```

## Current state

Phase 1 foundation reconciliation is complete. `STATUS.md` keeps NL-03 as the next task card. The broad detection and I/O modules remain preliminary and are not claimed as MATLAB-parity-complete.

## Next step

Start NL-03 only: define strict typed core contracts and serialization tests. Do not expand detection, file-format support, GPU kernels, or CLI scope during that card.
