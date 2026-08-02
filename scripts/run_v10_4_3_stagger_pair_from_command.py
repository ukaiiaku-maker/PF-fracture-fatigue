#!/usr/bin/env python3
"""Run a controlled stagger fixed-point comparison from an audited command.sh.

The template command preserves the exact material, kernel, geometry, seed, and
loading options of an existing case.  This harness changes only:

* ``--n-stagger`` (the maximum fixed-point iteration count);
* ``--out``;
* optionally ``--steps``, ``--print-every``, and snapshot count;
* the explicit fixed-point relaxation and convergence tolerances; and
* the rejected-trial timestep subdivision controls.

It is intended for the v10.4.3 constitutive-time/fixed-point qualification
before any production campaign is restarted.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shlex
import subprocess
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


def _summarize(cases: dict[int, Path], controls: dict[str, float | int]) -> dict[str, Any]:
    rows = {n: _numeric_row(_steps_csv(root)) for n, root in cases.items()}
    reference_n = min(rows)
    reference = rows[reference_n]
    fields = [
        "step",
        "Uapp_m",
        "dt_cur_s",
        "adaptive_frac",
        "Ftop_N",
        "J_effective_direct_J_per_m2",
        "J_signed_direct_J_per_m2",
        "KJ_Pa_sqrtm",
        "sigma_tip_Pa",
        "sigma_back_Pa",
        "sigma_cleave_eff_Pa",
        "W_bulk_plastic_cumulative_J_per_m",
        "U_elastic_J_per_m",
        "W_ext_cumulative_J_per_m",
        "W_fracture_residual_cumulative_J_per_m",
        "W_emit_J_per_m",
        "W_tip_emit_cumulative_J_per_m",
        "N_em",
        "lambda_e",
        "lambda_c",
        "B",
        "a_tip_m",
        "crack_extension_m",
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
        "schema": "v10.4.3_adaptive_converged_stagger_pair_summary_v3",
        "reference_n_stagger": reference_n,
        "fixed_point_controls": controls,
        "cases": {str(n): str(path.resolve()) for n, path in cases.items()},
        "final_rows": {str(n): row for n, row in rows.items()},
        "ratios": comparisons,
        "interpretation": (
            "Every successful accepted step passed the strict relaxed fixed-point "
            "gate. Unconverged trials were rolled back and retried with dt and dU "
            "reduced together at fixed loading rate. Runs with different maximum "
            "iteration ceilings should agree if both ceilings are sufficiently high; "
            "a hard failure is diagnostic and is not a completed case."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-command", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--staggers", type=int, nargs="+", default=[40, 80])
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--save-snapshots", type=int, default=0)
    parser.add_argument("--stagger-relaxation", type=float, default=0.25)
    parser.add_argument("--stagger-rtol", type=float, default=1.0e-6)
    parser.add_argument("--stagger-ep-atol", type=float, default=1.0e-12)
    parser.add_argument("--stagger-rho-atol-m2", type=float, default=1.0e3)
    parser.add_argument("--stagger-dt-shrink", type=float, default=0.25)
    parser.add_argument("--stagger-min-dt-fraction", type=float, default=1.0e-8)
    parser.add_argument("--stagger-max-dt-retries", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if len(set(args.staggers)) != len(args.staggers) or any(n < 1 for n in args.staggers):
        raise SystemExit("--staggers must contain unique positive integers")
    if not (0.0 < args.stagger_relaxation <= 1.0):
        raise SystemExit("--stagger-relaxation must satisfy 0 < alpha <= 1")
    if min(args.stagger_rtol, args.stagger_ep_atol, args.stagger_rho_atol_m2) < 0.0:
        raise SystemExit("fixed-point tolerances must be non-negative")
    if not (0.0 < args.stagger_dt_shrink < 1.0):
        raise SystemExit("--stagger-dt-shrink must satisfy 0 < factor < 1")
    if not (0.0 < args.stagger_min_dt_fraction <= 1.0):
        raise SystemExit("--stagger-min-dt-fraction must satisfy 0 < fraction <= 1")
    if args.stagger_max_dt_retries < 0:
        raise SystemExit("--stagger-max-dt-retries must be non-negative")
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
        controls = [
            ("--stagger-relaxation", args.stagger_relaxation),
            ("--stagger-rtol", args.stagger_rtol),
            ("--stagger-ep-atol", args.stagger_ep_atol),
            ("--stagger-rho-atol-m2", args.stagger_rho_atol_m2),
            ("--stagger-dt-shrink", args.stagger_dt_shrink),
            ("--stagger-min-dt-fraction", args.stagger_min_dt_fraction),
            ("--stagger-max-dt-retries", args.stagger_max_dt_retries),
        ]
        for flag, value in controls:
            rendered = str(value) if isinstance(value, int) else f"{value:.17g}"
            command = _replace_flag(command, flag, rendered, required=False)

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
            tail = "\n".join(log_path.read_text().splitlines()[-140:])
            raise SystemExit(
                f"n_stagger={n_stagger} failed with exit {completed.returncode}\n{tail}"
            )
        _steps_csv(case_root)
        print(f"Completed n_stagger={n_stagger}: {case_root}")

    if not args.dry_run:
        control_payload: dict[str, float | int] = {
            "stagger_relaxation": args.stagger_relaxation,
            "stagger_rtol": args.stagger_rtol,
            "stagger_ep_atol": args.stagger_ep_atol,
            "stagger_rho_atol_m2": args.stagger_rho_atol_m2,
            "stagger_dt_shrink": args.stagger_dt_shrink,
            "stagger_min_dt_fraction": args.stagger_min_dt_fraction,
            "stagger_max_dt_retries": args.stagger_max_dt_retries,
        }
        summary = _summarize(cases, control_payload)
        summary_path = args.outroot / "n_stagger_comparison.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
