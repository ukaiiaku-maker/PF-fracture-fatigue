#!/usr/bin/env python3
"""Diagnose exact signed-slip ribbon support without solving FEM systems.

The diagnostic reproduces the station selection, exact triangle/ribbon overlap,
and stiffness masking used by the v10.2.14 load-invariance evaluator.  It reports
the exact state/system/bin/station whose terminal neighborhood is unsupported.
No snapshot fields or production mechanics are modified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.physical_fem_snapshot_v10212 import load_snapshot
from arrhenius_fracture.physical_fem_station_responses_v10212 import (
    _active_ribbon_geometry,
    _station_indices,
    _unit,
)
from arrhenius_fracture.slip_ribbon_overlap_v10214 import (
    overlap_weighted_slip_ribbon_increment,
)
from arrhenius_fracture.unit_slip_perturbation_v10212 import (
    DEFAULT_MINIMUM_RESIDUAL_STIFFNESS_FRACTION,
    DEFAULT_STIFFNESS_KAPPA,
    SlipRibbonPerturbation,
    element_residual_stiffness_fraction,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ribbon-width-m", type=float)
    parser.add_argument("--minimum-station-spacing-m", type=float)
    parser.add_argument(
        "--minimum-residual-stiffness-fraction",
        type=float,
        default=DEFAULT_MINIMUM_RESIDUAL_STIFFNESS_FRACTION,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = load_snapshot(args.snapshot.expanduser().resolve())
    meta = data["metadata"]
    mesh = data["mesh"]
    damage = np.asarray(data["d"], dtype=float)
    width = (
        max(2.0 * float(mesh.hbar_tip), 10.0 * float(data["mat"].b))
        if args.ribbon_width_m is None
        else float(args.ribbon_width_m)
    )
    spacing = (
        max(2.0 * width, 2.0 * float(mesh.hbar_tip))
        if args.minimum_station_spacing_m is None
        else float(args.minimum_station_spacing_m)
    )
    if not math.isfinite(width) or width <= 0.0:
        raise SystemExit("ribbon width must be positive and finite")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise SystemExit("station spacing must be positive and finite")

    tip = np.asarray(meta.crack_tip_xy_m, dtype=float)
    forward = _unit(meta.crack_direction)
    directions = [_unit(row) for row in meta.channel_directions]
    normals = [_unit(row) for row in meta.channel_normals]
    active_grid = tuple(float(value) for value in meta.active_x_m)
    indices = _station_indices(active_grid, spacing)
    residual = element_residual_stiffness_fraction(
        mesh,
        damage,
        stiffness_kappa=DEFAULT_STIFFNESS_KAPPA,
    )
    killed = residual < float(args.minimum_residual_stiffness_fraction)

    rows = []
    failures = []
    for system, (slip, normal) in enumerate(zip(directions, normals)):
        for bin_index in indices:
            x_m = float(active_grid[bin_index])
            row = {
                "state_id": str(meta.state_id),
                "system": int(system),
                "bin": int(bin_index),
                "x_m": x_m,
                "x_um": 1.0e6 * x_m,
            }
            try:
                start, end, placement = _active_ribbon_geometry(
                    system=system,
                    x_m=x_m,
                    width_m=width,
                    mesh=mesh,
                    damage=damage,
                    tip=tip,
                    forward=forward,
                    slip_direction=slip,
                    minimum_residual_stiffness_fraction=float(
                        args.minimum_residual_stiffness_fraction
                    ),
                    stiffness_kappa=DEFAULT_STIFFNESS_KAPPA,
                )
                perturbation = SlipRibbonPerturbation(
                    system=system,
                    region="active",
                    bin_index=bin_index,
                    start_xy_m=start,
                    end_xy_m=end,
                    slip_direction=slip,
                    plane_normal=normal,
                    width_m=width,
                    burgers_m=float(data["mat"].b),
                    signed_line_content=0.25,
                )
                _, overlap_audit, support = overlap_weighted_slip_ribbon_increment(
                    mesh, perturbation
                )
                overlap = np.asarray(support.overlap_area_e_m2, dtype=float)
                terminal = np.asarray(
                    support.terminal_overlap_area_e_m2, dtype=float
                )
                geometric_area = float(np.sum(overlap))
                supported_area = float(np.sum(overlap[~killed]))
                terminal_geometric = float(np.sum(terminal))
                terminal_supported = float(np.sum(terminal[~killed]))
                row.update(
                    {
                        "start_xy_m": placement["start_xy_m"],
                        "end_xy_m": placement["end_xy_m"],
                        "ribbon_width_m": width,
                        "terminal_window_m": float(support.terminal_window_m),
                        "selected_elements": int(
                            overlap_audit["selected_elements"]
                        ),
                        "geometric_overlap_area_m2": geometric_area,
                        "supported_overlap_area_m2": supported_area,
                        "terminal_geometric_overlap_area_m2": terminal_geometric,
                        "terminal_supported_overlap_area_m2": terminal_supported,
                        "terminal_supported_fraction": terminal_supported
                        / max(terminal_geometric, 1.0e-30),
                        "passed": terminal_supported > 1.0e-30,
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "passed": False,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                )
            rows.append(row)
            if not row["passed"]:
                failures.append(row)

    payload = {
        "schema": "v10.2.27_signed_ribbon_support_diagnostic_v1",
        "snapshot": str(args.snapshot.expanduser().resolve()),
        "state_id": str(meta.state_id),
        "temperature_K": float(meta.temperature_K),
        "crystal_theta_deg": float(
            meta.engine_config.get("anisotropic_config", {}).get(
                "crystal_theta_deg", float("nan")
            )
        ),
        "mesh_hbar_tip_m": float(mesh.hbar_tip),
        "ribbon_width_m": width,
        "minimum_station_spacing_m": spacing,
        "minimum_residual_stiffness_fraction": float(
            args.minimum_residual_stiffness_fraction
        ),
        "active_grid_points": len(active_grid),
        "selected_station_count_per_system": len(indices),
        "selected_station_indices": indices,
        "evaluated_placements": len(rows),
        "failed_placements": len(failures),
        "pass": not failures,
        "failures": failures,
        "placements": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
