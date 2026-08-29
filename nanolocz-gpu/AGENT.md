# AGENT.md — nanolocz-gpu working agreement

Read this file before changing anything in this port.

## Mission
Build a Linux-first, MATLAB-free Python/CUDA port of NanoLocz while preserving
scientific behavior, a CPU reference path, and reproducible handoffs.

The current BioAFM-inspired scope is intentionally small: import PDB atom
coordinates, render a coarse AFM height image with a conical tip, estimate a
usable tip radius from an AFM image, and perform rough tip/translation fitting.
Do not expand this into full molecular viewing, flexible fitting, electrostatics,
or BioAFMviewer file compatibility without a new approved task card.

## Source of truth
- `STATUS.md` is the current project state and next-card pointer.
- `SPEC/` defines contracts. Do not silently change a spec during implementation.
- `ADR/` records architecture decisions. Draft changes for human approval.
- `tests/` and `golden/` define correctness; visual plausibility is not enough.
- `SESSIONS/` records what the next agent needs to resume safely.

## Non-negotiable rules
1. Work on exactly one task card from `SPEC/tasks.md` per session.
2. Read the selected card, its referenced spec, and current `STATUS.md` first.
3. Write or update the failing test before implementation code.
4. Public APIs use typed dataclasses from `nanolocz/core/types.py` (the canonical package).
5. Keep NumPy/float64 as the reference path; GPU support is optional until P1 is green.
6. Never edit `SPEC/` or `ADR/` to make a failing implementation pass without approval.
7. Run `python tools/project_check.py` before handoff. A green check is required.
8. Update `STATUS.md` and write a session note after each card.
9. One focused commit per completed card. Stop when scope creeps or the card is green.
10. Treat `nanolocz/simafm/` as a deliberately approximate visual workflow;
    preserve explicit limitations in tests and documentation.

## Session start
```bash
cd nanolocz-gpu
cat AGENT.md STATUS.md
python tools/project_check.py
```

## Session finish
```bash
python -m pytest -q
python tools/project_check.py
# update STATUS.md and create SESSIONS/YYYY-MM-DD-<card>.md
git status --short
```
