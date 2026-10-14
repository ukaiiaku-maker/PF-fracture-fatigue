#!/usr/bin/env python3
"""Build a no-scale-factor three-level FEM pre-event convergence receipt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

FILES = (
    "steps_0900K.csv",
    "anisotropic_calibrated_tip_first_passage_summary.json",
    "compact_corridor_mesh_v91852.json",
    "generic_current_pf_companion_contract_v1.json",
    "run_args.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observe(root: Path) -> dict:
    root = root.resolve()
    paths = [root / name for name in FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    with paths[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one pre-event step in {paths[0]}")
    step = rows[0]
    summary = json.loads(paths[1].read_text())
    mesh = json.loads(paths[2].read_text())
    contract = json.loads(paths[3].read_text())
    run_args = json.loads(paths[4].read_text())
    if not contract["run_completed_without_exception"]:
        raise ValueError(f"incomplete native run: {root}")

    C11 = float(run_args["crystal_C11"])
    C12 = float(run_args["crystal_C12"])
    C44 = float(run_args["crystal_C44"])
    anisotropy = 2.0 * C44 / (C11 - C12)
    if not math.isclose(anisotropy, 1.0, rel_tol=1.0e-12):
        raise ValueError("common-J reconstruction is currently restricted to Zener A=1")
    young = (C11 - C12) * (C11 + 2.0 * C12) / (C11 + C12)
    poisson = C12 / (C11 + C12)
    Eprime = young / (1.0 - poisson * poisson)
    K = float(step["KJ_Pa_sqrtm"])
    common_J = K * K / Eprime
    native_J = float(summary["J_first_J_per_m2"])
    return {
        "directory": root.name,
        "node_count": int(mesh["node_count"]),
        "triangle_count": int(mesh["triangle_count"]),
        "initial_hbar_tip_m": float(mesh["initial_hbar_tip_m"]),
        "maximum_sampled_hbar_tip_m": float(mesh["maximum_sampled_hbar_tip_m"]),
        "remote_half_opening_m": float(step["Uapp_m"]),
        "top_reaction_N_per_unit_thickness": float(step["Ftop_N"]),
        "sigma_tip_Pa": float(step["sigma_tip_Pa"]),
        "native_J_J_per_m2": native_J,
        "native_KJ_Pa_sqrt_m": K,
        "common_isotropic_plane_strain_J_J_per_m2": common_J,
        "common_to_native_J_relative_difference": abs(common_J - native_J) / native_J,
        "material_option": contract["material"]["option_key"],
        "sha256": {path.name: _sha(path) for path in paths},
    }


def _relative_change(a: float, b: float) -> float:
    return abs(b - a) / max(abs(b), 1.0e-300)


def qualify(roots: list[Path], tolerance: float) -> dict:
    levels = sorted((observe(root) for root in roots),
                    key=lambda row: row["initial_hbar_tip_m"], reverse=True)
    if len(levels) != 3:
        raise ValueError("exactly three related levels are required")
    metrics = (
        "top_reaction_N_per_unit_thickness",
        "sigma_tip_Pa",
        "native_J_J_per_m2",
        "native_KJ_Pa_sqrt_m",
    )
    changes = {
        metric: {
            "coarse_to_medium": _relative_change(levels[0][metric], levels[1][metric]),
            "medium_to_fine": _relative_change(levels[1][metric], levels[2][metric]),
        }
        for metric in metrics
    }
    max_fine = max(item["medium_to_fine"] for item in changes.values())
    passed = max_fine <= tolerance
    return {
        "schema": "generic-fem-three-level-pre-event-convergence/1",
        "status": "PASS" if passed else "FAIL_NOT_CONVERGED",
        "no_fitted_scale_factor": True,
        "relative_tolerance": tolerance,
        "levels": levels,
        "relative_changes": changes,
        "maximum_medium_to_fine_relative_change": max_fine,
        "interpretation": (
            "The generic adapter executes three genuine physical tip-resolution "
            "levels. Failure is retained as a numerical finding and does not "
            "qualify R08 convergence or cross-backend agreement."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    roots = args.levels
    if args.compact_root is not None:
        compact = args.compact_root
        compact.mkdir(parents=True, exist_ok=True)
        copied = []
        for index, source in enumerate(roots, 1):
            destination = compact / f"level_{index}"
            destination.mkdir(exist_ok=True)
            for name in FILES:
                shutil.copy2(source / name, destination / name)
            copied.append(destination)
        roots = copied
    payload = qualify(roots, args.relative_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
