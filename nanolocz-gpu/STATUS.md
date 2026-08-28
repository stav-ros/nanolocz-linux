# Project status

Last updated: 2026-08-28
Current phase: P0 — foundation
Current card: NL-01
Status: ready

## Progress

| Card | Area | State | Evidence |
|---|---|---|---|
| NL-01 | Fork audit and toolbox map | in_progress | `SPEC/audit.md` is the pending contract |
| NL-02 | Golden parity harness | not_started | blocked by NL-01 |
| NL-03 | Typed core contracts | not_started | independent of NL-02 |
| NL-10–NL-17 | CPU core port | not_started | blocked by foundation |
| NL-20–NL-24 | GPU backend and kernels | not_started | blocked by P1 |
| NL-30–NL-37 | LAFM+ science | not_started | blocked by core/GPU work |
| NL-40–NL-43 | Interface and shipping | not_started | blocked by core/GPU work |
| NL-50–NL-54 | Simulation bridge | not_started | can begin after NL-03 |

## Next action

Execute NL-01 only: inspect the upstream/externalized MATLAB core and record every
MATLAB toolbox call in `ADR/0000-toolbox-map.md`. Do not port library logic yet.

## Self-check contract

A task may move to `done` only when:

- its acceptance criteria in `SPEC/tasks.md` are satisfied;
- the targeted test command is green;
- `python tools/project_check.py` is green;
- the evidence is recorded in this file; and
- a session handoff exists under `SESSIONS/`.

Allowed states: `not_started`, `in_progress`, `blocked`, `done`.
