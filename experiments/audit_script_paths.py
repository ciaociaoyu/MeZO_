#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(?:\"([^\"]*)\"|'([^']*)')\s*$")
CALL_RE = re.compile(r"\b(source|bash|python)\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s#;]+))")
VAR_RE = re.compile(r"\$(\w+)|\$\{([^}]+)\}")
CD_LINE_RE = re.compile(r"^\s*cd\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s&;]+))\s*$")
CD_PREFIX_RE = re.compile(r"^\s*cd\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s&;]+))\s*&&\s*(.*)$")


@dataclass
class Problem:
    script: Path
    line_no: int
    command: str
    target: str
    resolved: str
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit shell scripts under experiments/ for broken source/bash/python file paths."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/scratch/jy03364/MeZO_/experiments"),
        help="Path to the experiments root. Defaults to the current shared workspace path.",
    )
    return parser.parse_args()


def expand_vars(template: str, mapping: dict[str, str]) -> str:
    previous = None
    current = template
    for _ in range(20):
        if current == previous:
            break
        previous = current
        current = VAR_RE.sub(lambda m: mapping.get(m.group(1) or m.group(2), m.group(0)), current)
    return current


def iter_problems(root: Path) -> list[Problem]:
    problems: list[Problem] = []
    for script in sorted(root.rglob("*.sh")):
        variables: dict[str, str] = {}
        cwd = script.parent
        for line_no, line in enumerate(script.read_text(errors="replace").splitlines(), start=1):
            assign_match = ASSIGN_RE.match(line)
            if assign_match:
                key = assign_match.group(1)
                raw_value = assign_match.group(2) if assign_match.group(2) is not None else assign_match.group(3)
                variables[key] = expand_vars(raw_value, variables)

            cd_match = CD_LINE_RE.match(line)
            if cd_match:
                raw_cd = next(group for group in cd_match.groups() if group is not None)
                expanded_cd = expand_vars(raw_cd, variables)
                cwd = Path(expanded_cd) if expanded_cd.startswith("/") else (cwd / expanded_cd).resolve()
                continue

            analysis_line = line
            base_dir = cwd
            cd_prefix_match = CD_PREFIX_RE.match(line)
            if cd_prefix_match:
                raw_cd = next(group for group in cd_prefix_match.groups()[:3] if group is not None)
                expanded_cd = expand_vars(raw_cd, variables)
                base_dir = Path(expanded_cd) if expanded_cd.startswith("/") else (cwd / expanded_cd).resolve()
                analysis_line = cd_prefix_match.group(4)

            for call_match in CALL_RE.finditer(analysis_line):
                command = call_match.group(1)
                target = next(group for group in call_match.groups()[1:] if group is not None)
                if target.startswith("-"):
                    continue
                expanded = expand_vars(target, variables)
                if command == "python" and not ("/" in expanded or expanded.endswith(".py")):
                    continue
                if expanded.startswith("$"):
                    problems.append(Problem(script, line_no, command, target, expanded, "unresolved"))
                    continue
                resolved = Path(expanded) if expanded.startswith("/") else (base_dir / expanded).resolve()
                if not resolved.exists():
                    problems.append(Problem(script, line_no, command, target, str(resolved), "missing"))
    return problems


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    problems = iter_problems(root)
    if problems:
        print(f"[audit] found {len(problems)} broken script path reference(s) under {root}")
        for problem in problems:
            print(
                f"{problem.script}:{problem.line_no}: {problem.command} {problem.target} -> "
                f"{problem.resolved} ({problem.kind})"
            )
        return 1
    print(f"[audit] all script path references under {root} resolved successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
