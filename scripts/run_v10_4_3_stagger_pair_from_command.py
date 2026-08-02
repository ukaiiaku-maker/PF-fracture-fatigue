#!/usr/bin/env python3
"""Run a controlled n_stagger comparison from a previously audited command.sh.

The template command preserves the exact material, kernel, geometry, seed, and
loading options of an existing case.  This harness changes only:

* ``--n-stagger``;
* ``--out``;
* optionally ``--steps``, ``--print-every``, and snapshot count.

It is intended for the v10.4.3 constitutive-time qualification before any
production campaign is restarted.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


_REQUIRED_MODEL = "arrhenius_fracture.sharp_front_v10_4_2_plastic_flow_audited"


def _replace_flag(text: str, flag: str, value: str, *, required: bool = True) -> str:
    pattern = re.compile(
        rf"({re.escape(flag)}(?:=|\s+))(?:\"[^\"]*\"|'[^']*'|[^\s\\]+)"
    )
    replaced, count = pattern.subn(lambda match: match.group(1) + value, text, count=1)
    if required and count != 1:
        raise RuntimeError(f"expected exactly one {flag} in template; found {count}")
    if not required and count == 0:
        return text.rstrip() + f" \\\n  {flag} {value}\n"
    return replaced


def _steps_csv(case_root: Path) -> Path:
    matches = sorted(case_root.glob("steps_*K.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one steps_*K.csv in {case_root}; found {matches}"
        )
    return matches[0]


def _numeric_row(path: Path, index: int = -1) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty step history: {path}")
    row = rows[index]
    result: dict[str, float] = {"row_count": float(len(rows))}
    for key, value in row.items():
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return value / reference


def _summarize(cases: dict[int, Path]) -> dict[str, Any]:
    rows = {n: _numeric_row(_steps_csv(root)) for n, root in cases.items()}
    reference_n = min(rows)
    reference = rows[reference_n]
    fields = [
        "Uapp_m",
        "Uapp",
        "Ftop_N",
        "Ftop",
        "KJ_MPa_sqrt_m",
        "KJ",
        "J_effective_J_per_m2",
        "J_positive_J_per_m2",
        "W_p_J_per_m",
        "W_p",
        "U_el_J_per_m",
        "U_el",
        "rho_mean_m2",
        "rho_mean",
        "rho_max_m2",
        "rho_max",
        "B",
        "a_tip_m",
        "a_tip",
        "n_fire",
    ]
    comparisons: dict[str, Any] = {}
    for n, row in rows.items():
        if n == reference_n:
            continue
        ratios = {
            field: _ratio(row.get(field), reference.get(field))
            for field in fields
            if field in row or field in reference
        }
        comparisons[f"n_stagger_{n}_over_{reference_n}"] = ratios
    return {
        "schema": "v10.4.3_n_stagger_pair_summary_v1",
        "reference_n_stagger": reference_n,
        "cases": {str(n): str(path.resolve()) for n, path in cases.items()},
        "final_rows": {str(n): row for n, row in rows.items()},
        "ratios": comparisons,
        "interpretation": (
            "Corrected runs need not be identical at n_stagger=1 and 4 because "
            "they are different fixed-point convergence depths, but cumulative "
            "plastic state/work must not scale approximately as n_stagger. "
            "Use n_stagger=2 versus 4 as the tighter convergence comparison."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-command", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--staggers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--save-snapshots", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if len(set(args.staggers)) != len(args.staggers) or any(n < 1 for n in args.staggers):
        raise SystemExit("--staggers must contain unique positive integers")
    if args.outroot.exists():
        raise SystemExit(f"refusing to overwrite existing output root: {args.outroot}")

    template = args.template_command.read_text()
    if _REQUIRED_MODEL not in template:
        raise SystemExit(
            "template must use the audited public model entry " + _REQUIRED_MODEL
        )
    for flag in ("--n-stagger", "--out", "--steps"):
        if flag not in template:
            raise SystemExit(f"template is missing required flag {flag}")

    args.outroot.mkdir(parents=True)
    cases: dict[int, Path] = {}
    for n_stagger in args.staggers:
        case_root = args.outroot / f"n_stagger_{n_stagger}"
        command = template
        command = _replace_flag(command, "--n-stagger", str(n_stagger))
        command = _replace_flag(command, "--out", shlex.quote(str(case_root.resolve())))
        command = _replace_flag(command, "--steps", str(args.steps))
        command = _replace_flag(
            command, "--print-every", str(args.print_every), required=False
        )
        command = _replace_flag(
            command, "--save-snapshots", str(args.save_snapshots), required=False
        )

        command_path = args.outroot / f"command_n_stagger_{n_stagger}.sh"
        command_path.write_text("set -euo pipefail\n" + command.lstrip())
        command_path.chmod(0o755)
        cases[n_stagger] = case_root
        print(f"Prepared n_stagger={n_stagger}: {command_path}")

        if args.dry_run:
            continue
        log_path = args.outroot / f"n_stagger_{n_stagger}.log"
        with log_path.open("w") as log:
            completed = subprocess.run(
                ["bash", str(command_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-80:])
            raise SystemExit(
                f"n_stagger={n_stagger} failed with exit {completed.returncode}\n{tail}"
            )
        _steps_csv(case_root)
        print(f"Completed n_stagger={n_stagger}: {case_root}")

    if not args.dry_run:
        summary = _summarize(cases)
        summary_path = args.outroot / "n_stagger_comparison.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
