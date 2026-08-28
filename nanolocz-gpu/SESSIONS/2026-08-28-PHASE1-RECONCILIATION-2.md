# Phase 1 reconciliation handoff (second pass — regression fix)

Date: 2026-08-28
Scope: repair a regression on `main` that reintroduced the exact duplication
the first Phase 1 reconciliation (`2026-08-28-PHASE1-RECONCILIATION.md`)
had removed, plus unrelated repo hygiene issues discovered during audit.

## Root cause

After the first reconciliation merged to `main` (PR #6), a second,
uncoordinated agent branch (`gpu-accelerated-nanolocz-fork-ef8b6`) merged
PRs #7 and #8 straight to `main` without running `pytest` or
`tools/project_check.py`. Those merges:

- reintroduced a duplicate `nanolocz/core/parity.py` implementation
  alongside the canonical `nanolocz/parity/` package;
- reintroduced a bloated `nanolocz/core/types.py` (NL-10+ scope: enums,
  `DetectionParams`, `ParticleTrack`, GPU config, pandas export) in place
  of the minimal NL-03-scoped contracts (`Frame`, `Meta`, `Localizations`,
  `ParticleStack`);
- duplicated `[project]` and `[tool.pytest.ini_options]` tables in
  `pyproject.toml`, making it invalid TOML;
- as a side effect of an earlier PR in the same chain (#3/#4), committed
  ~11,200 `node_modules/` files directly into git history and replaced the
  working root `.gitignore` with a broken 3-line file containing literal
  markdown code-fence characters.

Result: `pip install -e ".[test]"` and `pytest` both failed outright on
`main` (`tomllib.TOMLDecodeError: Cannot declare ('project',) twice`).

## Completed in this pass

- Restored the canonical, minimal `nanolocz/core/types.py`, `nanolocz/__init__.py`,
  `nanolocz/core/__init__.py`, `tests/test_parity.py`, and `tools/project_check.py`
  from the first reconciliation.
- Removed the duplicate `nanolocz/core/parity.py` again.
- Replaced `pyproject.toml` with the single valid configuration (top-level
  package discovery, `test` extra, no duplicate tables).
- Restored a real root `.gitignore` (Node + Python) and removed the
  ~11,200 tracked `node_modules/` files from git tracking.
- Removed the two empty, unregistered gitlinks `NanoLocz/` and
  `nano-locz-original/` (no `.gitmodules`, pointed at the same upstream
  commit, contributed nothing).
- Pruned unused root `package.json` dependencies that are not referenced
  anywhere in `src/` or elsewhere in the repo: `@dnd-kit/*`,
  `@supabase/supabase-js`, `canvas-confetti`, `date-fns`, `framer-motion`,
  `lucide-react`, `react-router-dom`, `recharts`, `uuid` (and their
  matching `@types/*` dev deps).
- Marked `PROGRESS_TRACKER.md` deprecated in favor of `STATUS.md`, since
  having two trackers with different taxonomies was a contributing factor
  to the drift that let this regression land unnoticed.

## Validation

From `nanolocz-gpu/`:

```text
python -m pip install --no-deps -e ".[test]"
Successfully installed nanolocz-0.1.0.dev0

python -m pytest -q
27 passed, 1 skipped (CuPy unavailable)

python tools/project_check.py
PASS project scaffold is self-consistent; current card NL-03
PASS 11 required memory/contract files present
PASS 3 tracked task status rows use valid states
PASS canonical package and pyproject.toml are aligned
```

From repo root:

```text
npm install && npm run build
✓ built in ~2s, no errors

npm run typecheck
tsc --noEmit — no errors
```

## Process fix recommended (not yet enforced)

The actual root cause was two agent branches merging to `main` without a
required, green CI gate. `.github/workflows/python.yml` already runs
`pytest` and `tools/project_check.py` on every push/PR touching
`nanolocz-gpu/**` — but nothing currently blocks a merge if that workflow
is red. Enable required status checks / branch protection on `main` so a
failing CI run cannot be merged, regardless of which agent or branch is
proposing the change.

## Next step

Once this fix lands on `main`, resume exactly where `STATUS.md` says:
**NL-03** — typed core contracts, strict type checks, serialization tests
only. Do not expand into NL-10+ scope (file openers, detection statistics,
GPU kernels) until NL-03 is marked done with evidence recorded in
`STATUS.md`.
