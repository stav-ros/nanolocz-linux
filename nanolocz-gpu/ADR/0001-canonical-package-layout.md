# ADR 001: Canonical Package Layout

## Status
Accepted

## Context
After consolidating the NanoLocz Python port into the `nanolocz-gpu` directory, we discovered two potential package layouts:

1. **Top-level layout**: `nanolocz-gpu/nanolocz/` (current implementation)
   - Contains: detection, file readers, GPU utilities, LAFM/simulation placeholders
   - Currently used by pyproject.toml with `where = ["."]`

2. **src layout**: `nanolocz-gpu/src/nanolocz/` (from original scaffold)
   - Contains: typed core contracts, parity fixtures, tolerance utilities
   - Would require `where = ["src"]` in pyproject.toml

The merged main branch had both layouts present, creating ambiguity about which is canonical and risking that build tools might omit critical code.

## Decision
We adopt the **top-level layout** (`nanolocz-gpu/nanolocz/`) as the canonical package structure for the following reasons:

### Advantages of Top-Level Layout
1. **Simplicity**: Shorter import paths and easier navigation for new contributors
2. **Current momentum**: Existing implementation already uses this layout
3. **Visibility**: Package code is immediately visible at repository root
4. **Common in scientific Python**: Many scientific packages use this layout (e.g., numpy, scipy)

### Mitigation of src Layout Benefits
The src layout's primary benefit is preventing accidental imports of local code during testing. We mitigate this by:
- Using proper test isolation with pytest
- Running tests in clean environments
- Using `pytest --import-mode=importlib` when needed

## Implementation

### Package Structure
```
nanolocz-gpu/
├── nanolocz/              # Canonical package root
│   ├── __init__.py
│   ├── core/              # Core algorithms (detection, localization)
│   ├── gpu/               # GPU acceleration (CuPy wrappers)
│   ├── formats/           # File I/O (TIFF, HDF5, AFM formats)
│   ├── lafm/              # LAFM integration (placeholder)
│   └── simafm/            # SimAFM integration (placeholder)
├── tests/                 # Test suite
├── benchmarks/            # Performance benchmarks
├── examples/              # Usage examples
├── docs/                  # Documentation source
└── tools/                 # Development tools
```

### pyproject.toml Configuration
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["nanolocz*"]
```

### Parity Infrastructure Integration
The NL-02 parity utilities from the src layout scaffold will be migrated into the top-level layout under:
- `nanolocz/parity/__init__.py` - Public parity testing API
- `nanolocz/parity/fixtures.py` - SHA-256 verified MATLAB parity fixtures
- `nanolocz/parity/tolerance.py` - Centralized numerical tolerance policy

## Consequences

### Positive
- ✅ Clear, single source of truth for package location
- ✅ Simplified directory structure
- ✅ No need to refactor existing implementation
- ✅ `.gitignore` now properly protects Python artifacts
- ✅ Build configuration matches actual layout

### Negative
- ⚠️ Slightly higher risk of accidental imports during development (mitigated by testing practices)
- ⚠️ Need to migrate parity infrastructure from src/ concept to top-level

### Neutral
- Package distribution behavior remains identical
- No impact on end users

## Compliance
This ADR ensures compliance with the agent workflow by:
1. Establishing canonical layout before NL-03 (typed contracts)
2. Preserving NL-02 parity utilities in the canonical package
3. Updating packaging configuration correctly
4. Restoring Python artifact exclusions in .gitignore
5. Documenting the decision for future reference

## References
- Original issue: Package layout reconciliation task
- Related: NL-02 parity infrastructure, NL-03 typed contracts
- Python packaging guide: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/

---
**Date**: 2025-01-XX  
**Decided by**: NanoLocz Python Development Team  
**Reviewers**: Required before NL-03 implementation begins
