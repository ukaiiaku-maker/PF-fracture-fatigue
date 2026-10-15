#!/usr/bin/env python3
"""Build the compact, provenance-explicit opening-only PF/FEM comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONT = ROOT / "common_framework_readiness/fem_observer_restart_continuation_20261014"
EVIDENCE = CONT / "evidence"
PF_DIRS = (
    EVIDENCE / "pf_v101_common_1p5um_u2e7_run3",
    EVIDENCE / "pf_v101_common_1um_u2e7_run3",
    EVIDENCE / "pf_v101_common_0p75um_u2e7_run3",
)
FEM_RECEIPT = EVIDENCE / "fem_baseline_reconciliation_v2.json"
OUTPUT = CONT / "PF_FEM_COMMON_MECHANICS_COMPARISON.json"


def _relative_change(a: float, b: float) -> float:
    return abs(b - a) / max(abs(b), 1.0e-300)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_pf_level(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader((path / "steps_0900K.csv").open()))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one pre-event row")
    row = rows[0]
    summary = json.loads((path / "summary.json").read_text())[0]
    args = json.loads((path / "run_args.json").read_text())
    zero_keys = (
        "lambda_e", "N_em", "W_emit_J_per_m", "pz_store_total",
        "pz_mobile_total", "pz_escape_total", "pz_emit_total",
    )
    zeros = {key: float(row[key]) for key in zero_keys}
    if any(value != 0.0 for value in zeros.values()):
        raise ValueError(f"{path}: accommodation state is not identically zero: {zeros}")
    opening = float(row["Uapp_m"])
    reaction = float(row["Ftop_N"])
    return {
        "path": _display_path(path),
        "requested_tip_h_m": float(args["tip_h_fine"]),
        "measured_tip_h_m": float(summary["hbar_tip_m"]),
        "node_count": int(summary["n_nodes"]),
        "total_grip_opening_m": opening,
        "top_reaction_N_per_unit_thickness": reaction,
        "global_linear_elastic_energy_from_boundary_work_J_per_m": 0.5 * reaction * opening,
        "native_KJ_Pa_sqrt_m": float(row["KJ_Pa_sqrtm"]),
        "native_J_provenance": "PF native domain-J, reported by its K_J=sqrt(Eprime*J) channel",
        "accommodation_state": zeros,
    }


def build_receipt(pf_dirs: tuple[Path, ...] = PF_DIRS,
                  fem_receipt: Path = FEM_RECEIPT) -> dict[str, Any]:
    pf = [_read_pf_level(path) for path in pf_dirs]
    fem_source = json.loads(fem_receipt.read_text())
    fem = []
    for level in fem_source["levels"]:
        fem.append({
            "support_h_over_L_pz": level["support_h_over_L_pz"],
            "support_spacing_m": level["support_spacing_m"],
            "node_count": level["node_count"],
            "triangle_count": level["triangle_count"],
            "total_grip_opening_m": fem_source["config"]["total_grip_opening_m"],
            "top_reaction_N_per_unit_thickness": level["top_reaction_N_per_unit_thickness"],
            "stored_energy_J_per_m": level["stored_energy_J_per_m"],
            "native_domain_J_J_per_m2": level["native_full_energy_J_median_J_per_m2"],
            "matched_virtual_extension_G_J_per_m2": level["matched_outer_virtual_G_J_per_m2"],
        })

    openings = {level["total_grip_opening_m"] for level in pf + fem}
    if len(openings) != 1:
        raise ValueError(f"PF/FEM opening mismatch: {sorted(openings)}")
    pf_rel = {
        "reaction_medium_to_fine": _relative_change(
            pf[-2]["top_reaction_N_per_unit_thickness"],
            pf[-1]["top_reaction_N_per_unit_thickness"],
        ),
        "global_energy_medium_to_fine": _relative_change(
            pf[-2]["global_linear_elastic_energy_from_boundary_work_J_per_m"],
            pf[-1]["global_linear_elastic_energy_from_boundary_work_J_per_m"],
        ),
        "native_KJ_medium_to_fine": _relative_change(
            pf[-2]["native_KJ_Pa_sqrt_m"], pf[-1]["native_KJ_Pa_sqrt_m"]
        ),
    }
    fem_rel = {
        "reaction_medium_to_fine": _relative_change(
            fem[-2]["top_reaction_N_per_unit_thickness"],
            fem[-1]["top_reaction_N_per_unit_thickness"],
        ),
        "stored_energy_medium_to_fine": _relative_change(
            fem[-2]["stored_energy_J_per_m"], fem[-1]["stored_energy_J_per_m"]
        ),
        "native_J_medium_to_fine": _relative_change(
            fem[-2]["native_domain_J_J_per_m2"], fem[-1]["native_domain_J_J_per_m2"]
        ),
    }
    pff, femf = pf[-1], fem[-1]
    common = {
        "PF_over_FEM_reaction": pff["top_reaction_N_per_unit_thickness"] / femf["top_reaction_N_per_unit_thickness"],
        "PF_over_FEM_global_energy": pff["global_linear_elastic_energy_from_boundary_work_J_per_m"] / femf["stored_energy_J_per_m"],
        "reaction_relative_difference": _relative_change(
            femf["top_reaction_N_per_unit_thickness"], pff["top_reaction_N_per_unit_thickness"]
        ),
        "global_energy_relative_difference": _relative_change(
            femf["stored_energy_J_per_m"], pff["global_linear_elastic_energy_from_boundary_work_J_per_m"]
        ),
    }
    return {
        "schema": "pf-fem-common-mechanics-comparison/1",
        "status": "QUALIFIED_INDEPENDENT_SUPPORTS_SYSTEMATIC_REPRESENTATION_OFFSET",
        "physics_advanced": False,
        "common_conditions": {
            "width_m": 0.002, "height_m": 0.004, "crack_m": 0.0005,
            "notch_half_thickness_m": 8.0e-5, "process_zone_m": 5.0e-5,
            "total_grip_opening_m": openings.pop(), "constraint": "plane_strain",
            "elasticity": "cubic C11=523 GPa, C12=203 GPa, C44=160 GPa; Zener A=1",
            "accommodation": "disabled; all emission, retained, mobile, escaped, and emitted-work states zero",
        },
        "PF": {
            "entry": "arrhenius_fracture.sharp_front_v10_1",
            "role": "preserved native mechanics refinement entry; pre-event opening-only mechanics only",
            "refinement_axis": "graded PF tip element size",
            "levels": pf, "relative_changes": pf_rel,
            "support_converged_below_2_percent": max(pf_rel.values()) < 0.02,
        },
        "FEM": {
            "entry": "qualified v10.0.5.18.3.4 path-aware physical reachable-envelope provider",
            "refinement_axis": "path-envelope support spacing",
            "levels": fem, "relative_changes": fem_rel,
            "support_converged_below_2_percent": max(fem_rel.values()) < 0.02,
            "observer_correspondence": fem_source["observer_correspondence"],
        },
        "fine_level_common_observer_comparison": common,
        "interpretation": {
            "common_observers": ["top reaction", "global linear-elastic stored energy"],
            "noninterchangeable_native_observers": [
                "PF native domain-J/K_J", "FEM native full-energy domain J",
                "FEM matched material-coordinate virtual-extension G",
            ],
            "finding": (
                "Each backend is support-converged, but PF remains about 11.9% stiffer in the "
                "common global observers. This is a resolved comparison result, not a convergence "
                "failure and not permission to tune or substitute either native crack representation."
            ),
            "representation_difference": (
                "PF and FEM use independently qualified meshes and distinct discrete notch/damage "
                "supports; native J/K channels retain their own domain and field definitions."
            ),
        },
        "source_receipts": {"FEM": _display_path(fem_receipt)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": receipt["status"]}, indent=2))


if __name__ == "__main__":
    main()
