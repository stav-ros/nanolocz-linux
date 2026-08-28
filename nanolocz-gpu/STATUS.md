# Project status

Last updated: 2026-08-28
Current phase: Phase 1 — foundation reconciliation complete
Current card: NL-03
Status: ready for next card

## Progress

| Card | Area | State | Evidence |
|---|---|---|---|
| NL-01 | Fork audit and toolbox map | done | `ADR/0000-toolbox-map.md`; upstream revision `e41575c` audited |
| NL-02 | Golden parity harness | done | `SPEC/parity.md`; checksum loader, centralized tolerances, 7 CPU tests, CI workflow |
| NL-03 | Typed core contracts | not_started | independent of NL-02 |
| NL-10–NL-17 | CPU core port | not_started | blocked by foundation |
| NL-20–NL-24 | GPU backend and kernels | not_started | blocked by P1 |
| NL-30–NL-37 | LAFM+ science | not_started | blocked by core/GPU work |
| NL-40–NL-43 | Interface and shipping | not_started | blocked by core/GPU work |
| NL-50–NL-54 | Simulation bridge | not_started | can begin after NL-03 |

## Next action

Execute NL-03 only: implement the typed core contracts, strict type checks, and
serialization tests. Do not implement numerical analysis or file openers yet. NL-03
is the next foundation card now that the parity harness is green.

## Self-check contract

A task may move to `done` only when:

- its acceptance criteria in `SPEC/tasks.md` are satisfied;
- the targeted test command is green;
- `python tools/project_check.py` is green;
- the evidence is recorded in this file; and
- a session handoff exists under `SESSIONS/`.

Allowed states: `not_started`, `in_progress`, `blocked`, `done`.
