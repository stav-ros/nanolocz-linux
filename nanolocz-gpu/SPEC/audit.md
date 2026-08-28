# NL-01 — fork audit contract

## Goal
Create an evidence-backed inventory of the externalized NanoLocz MATLAB core before
writing Python. This prevents hidden toolbox dependencies from appearing late.

## Required output
`ADR/0000-toolbox-map.md` must list each toolbox function with:

- source file and line number;
- MATLAB toolbox or built-in classification;
- proposed Python replacement;
- whether behavior needs a golden fixture; and
- unresolved questions or licensing concerns.

Also record GPL-3.0 and attribution obligations in `NOTICE.md`.

## Stop condition
Do not implement port logic in this card. If the upstream source is unavailable,
record the exact blocker and the command or URL needed to continue.
