"""Independent repository self-check for agent sessions."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "AGENT.md",
    "STATUS.md",
    "SPEC/tasks.md",
    "SPEC/audit.md",
    "ADR/README.md",
    "golden/README.md",
    "SESSIONS/README.md",
    "pyproject.toml",
    "src/nanolocz/core/types.py",
]
STATES = {"not_started", "in_progress", "blocked", "done"}


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        print("FAIL missing required files:")
        print("\n".join(f"- {path}" for path in missing))
        return 1

    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    current = re.search(r"^Current card: (NL-\d+)$", status, re.MULTILINE)
    if not current:
        print("FAIL STATUS.md must declare Current card: NL-XX")
        return 1

    rows = re.findall(r"^\| (NL-\d+) \|.*?\| (\w+) \|", status, re.MULTILINE)
    invalid = [(card, state) for card, state in rows if state not in STATES]
    if invalid:
        print(f"FAIL invalid task states: {invalid}")
        return 1

    if "Allowed states:" not in status:
        print("FAIL STATUS.md must document allowed states")
        return 1

    print(f"PASS project scaffold is self-consistent; current card {current.group(1)}")
    print(f"PASS {len(REQUIRED)} required memory/contract files present")
    print(f"PASS {len(rows)} tracked task status rows use valid states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
