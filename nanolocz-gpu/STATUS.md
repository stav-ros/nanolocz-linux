# Project status

Last updated: 2026-08-28
Current phase: P0 — foundation
Current card: NL-02
Status: ready for next card

## Progress

| Card | Area | State | Evidence |
|---|---|---|---|
| NL-01 | Fork audit and toolbox map | done | `ADR/0000-toolbox-map.md`; upstream revision `e41575c` audited |
| NL-02 | Golden parity harness | not_started | unblocked by NL-01; next recommended card |
| NL-03 | Typed core contracts | not_started | independent of NL-02 |
| NL-10–NL-17 | CPU core port | not_started | blocked by foundation |
| NL-20–NL-24 | GPU backend and kernels | not_started | blocked by P1 |
| NL-30–NL-37 | LAFM+ science | not_started | blocked by core/GPU work |
| NL-40–NL-43 | Interface and shipping | not_started | blocked by core/GPU work |
| NL-50–NL-54 | Simulation bridge | not_started | can begin after NL-03 |

## Next action

Execute NL-02 only: build the CPU-only golden parity harness, fixture checksum
validation, and tolerance policy. Do not port analysis logic yet. NL-03 remains an
independent follow-up after the harness contract is established.

## Self-check contract

A task may move to `done` only when:

- its acceptance criteria in `SPEC/tasks.md` are satisfied;
- the targeted test command is green;
- `python tools/project_check.py` is green;
- the evidence is recorded in this file; and
- a session handoff exists under `SESSIONS/`.

Allowed states: `not_started`, `in_progress`, `blocked`, `done`.
