# Project status

Last updated: 2026-08-28
Current phase: Phase 2 — Data I/O & Storage (IN PROGRESS)
Current card: NL-10
Status: in_progress — zarr dependency added, 6 failing tests in NL-10 I/O operations need fixes

## Progress

| Card | Area | State | Evidence |
|---|---|---|---|
| NL-01 | Fork audit and toolbox map | done | `ADR/0000-toolbox-map.md`; upstream revision `e41575c` audited |
| NL-02 | Golden parity harness | done | `SPEC/parity.md`; checksum loader, centralized tolerances, 7 CPU tests, CI workflow |
| NL-03 | Typed core contracts | done | `nanolocz/core/types.py`; 43 type contract tests green; protocols with @runtime_checkable; SESSIONS/2026-08-28-NL-03.md |
| NL-10 | Zarr schema and opener interface | in_progress | `SPEC/NL-10-zarr-schema.md`; zarr dependency added to pyproject.toml; 6 failing tests need investigation |
| NL-11–NL-17 | CPU core port | not_started | blocked by NL-10 |
| NL-20–NL-24 | GPU backend and kernels | not_started | blocked by P1 |
| NL-30–NL-37 | LAFM+ science | not_started | blocked by core/GPU work |
| NL-40–NL-43 | Interface and shipping | not_started | blocked by core/GPU work |
| NL-50–NL-54 | Simulation bridge | not_started | **unblocked** — can start NL-50 |

## Phase 1 reconciliation evidence

- Canonical package: `nanolocz/`; obsolete `src/nanolocz/` removed.
- Single valid `pyproject.toml` with top-level package discovery and `[test]` extra.
- Parity fixtures and tolerances are exposed from `nanolocz.parity`.
- Validation: `93 passed, 6 failed, 1 skipped`; editable install succeeds; project self-check passes.

## Phase 2 initiation

- NL-10 selected as next development card (recommended option)
- Zarr schema specification available in `SPEC/NL-10-zarr-schema.md`
- Dependency fix: added `zarr>=2.18` to test and dev extras in `pyproject.toml`
- Current issue: 6 failing tests in `tests/test_io_nl10.py` related to movie metadata, localizations, tracks, particle stacks, compressed storage, and round-trip integration
- Next step: investigate and fix failing I/O tests

## Self-check contract

A task may move to `done` only when:

- its acceptance criteria in `SPEC/tasks.md` are satisfied;
- the targeted test command is green;
- `python tools/project_check.py` is green;
- the evidence is recorded in this file; and
- a session handoff exists under `SESSIONS/`.

Allowed states: `not_started`, `in_progress`, `blocked`, `done`.
