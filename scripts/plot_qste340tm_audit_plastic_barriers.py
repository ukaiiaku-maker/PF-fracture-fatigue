#!/usr/bin/env python3
"""Plot audit-qualified QSTE340TM emission, Peierls, and Taylor barriers."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_qste340tm_stateful_exp_floor import (
    BURGERS_M, PROCESS_ZONE_M, RHO0_M2, TEMPERATURE_K, exp_floor_barrier,
)
from scripts.qste340tm_faithful_reduction import RESOLVED_STRESS_FRACTION
from scripts.refit_qste340tm_restored_native import build_records
import scripts.refit_qste340tm_resolved_sloping_barrier as slope
import scripts.refit_qste340tm_signed_bulk_1d as signed

BASE = ROOT / "literature_data/fatigue_recovered_review/model_conditional/emission_owned_blunting"
OUT = BASE / "native_signed_bulk_refit_20261028"
CASES = (
    ("Signed bulk", BASE / "resolved_increment_1d/signed_bulk_joint_refit_20261028/fit.json",
     "#0072B2", "-", True),
    ("No bulk, best refit", BASE / "resolved_increment_1d/no_bulk_matched_refit_20261028/fit.json",
     "#D55E00", "--", False),
    ("No bulk, repeat refit", BASE / "resolved_increment_1d/no_bulk_equal_effort_control_20261028/fit.json",
     "#009E73", ":", False),
)


def reconstruct(label, path, color, linestyle, has_signed_bulk, records):
    fit = json.loads(path.read_text())
    vector = np.asarray(fit["parameters_vector"], dtype=float)
    phases = int(fit.get("final_phases", fit.get("settings", {}).get("final_phases", 48)))
    step = float(fit.get("advection_step_m", 1.0e-6 * fit.get("settings", {}).get("advection_step_um", 0.25)))
    state = (signed.trajectory(vector, records, phases=phases, advection_step_m=step)
             if has_signed_bulk else slope.state_trajectory(vector, records, phases, step))
    p = slope.unpack(vector)
    effective = np.maximum(state["stress"] - state["backstress_Pa"][:, None], 0.0)
    projection = math.sqrt(PROCESS_ZONE_M / (PROCESS_ZONE_M + 0.5 * p["blunting_length_m"]))
    retained = np.maximum(state["retained"] + projection * state["wake_retained"], 0.0)
    forest = RHO0_M2 + retained / PROCESS_ZONE_M**2
    return {
        "label": label, "path": str(path), "color": color, "linestyle": linestyle,
        "parameters": p, "effective_stress_Pa": [float(effective.min()), float(effective.max())],
        "forest_density_m-2": [float(forest.min()), float(np.median(forest)), float(forest.max())],
    }


def main() -> int:
    records = build_records("index_rank", premeasurement=True)
    curves = [reconstruct(*case, records) for case in CASES]
    maximum = 1.05 * max(item["effective_stress_Pa"][1] for item in curves)
    stress = np.linspace(0.0, maximum, 1400)
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), sharex=True)

    for item in curves:
        p = item["parameters"]
        emission = exp_floor_barrier("audit_emit", p["emission_G0_298_eV"],
                                     p["plastic_sigc_298_Pa"], 1.0e11)
        transport = exp_floor_barrier("audit_transport", p["transport_G0_298_eV"],
                                      p["plastic_sigc_298_Pa"], 1.0e12)
        emit_eV = emission.deltaG_eV(stress, TEMPERATURE_K)
        peierls_eV = transport.deltaG_eV(RESOLVED_STRESS_FRACTION * stress, TEMPERATURE_K)
        densities = np.asarray(item["forest_density_m-2"])
        spacing = 1.0 / (2.0 * np.sqrt(densities))
        taylor_argument = (RESOLVED_STRESS_FRACTION * stress[None, :] *
                           spacing[:, None] / BURGERS_M)
        taylor_eV = transport.deltaG_eV(taylor_argument, TEMPERATURE_K)
        low = np.minimum(taylor_eV[0], taylor_eV[2])
        high = np.maximum(taylor_eV[0], taylor_eV[2])

        common = dict(color=item["color"], linestyle=item["linestyle"], lw=2.3)
        axes[0].plot(stress / 1.0e9, emit_eV, label=item["label"], **common)
        axes[1].plot(stress / 1.0e9, peierls_eV, **common)
        axes[2].plot(stress / 1.0e9, taylor_eV[1], **common)
        axes[2].fill_between(stress / 1.0e9, low, high, color=item["color"], alpha=0.10)
        item["barrier_eV_ranges"] = {
            "emission": [float(emit_eV.min()), float(emit_eV.max())],
            "peierls_runtime_mapping": [float(peierls_eV.min()), float(peierls_eV.max())],
            "taylor_runtime_envelope": [float(low.min()), float(high.max())],
        }

    axes[0].set_title("Dislocation nucleation\nEXP-floor")
    axes[1].set_title("Peierls transport\n" + r"argument $=\sigma_{eff}/\sqrt{3}$")
    axes[2].set_title("Taylor release\nforest-density envelope")
    axes[0].set_ylabel(r"activation barrier $\Delta G^*$ (eV)")
    for axis in axes:
        axis.set_xlabel("effective tip stress (GPa)")
        axis.set_ylim(bottom=0.0)
        axis.grid(False)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Audit-qualified QSTE340TM plastic activation barriers at 298.15 K")
    figure.tight_layout()
    png = OUT / "audit_plastic_barrier_comparison.png"
    svg = OUT / "audit_plastic_barrier_comparison.svg"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt.close(figure)

    receipt = {
        "schema": "qste340tm-audit-plastic-barrier-comparison/1",
        "temperature_K": TEMPERATURE_K,
        "effective_tip_stress_grid_Pa": [0.0, float(maximum)],
        "resolved_stress_fraction": RESOLVED_STRESS_FRACTION,
        "burgers_m": BURGERS_M,
        "process_zone_m": PROCESS_ZONE_M,
        "background_forest_density_m-2": RHO0_M2,
        "barrier_contract": {
            "emission": "EXP_floor(emission_G0, sigma_eff)",
            "peierls": "EXP_floor(transport_G0, sigma_eff/sqrt(3))",
            "taylor": "EXP_floor(transport_G0, sigma_eff/sqrt(3) * spacing(rho)/b)",
            "note": "Peierls and Taylor share the fitted transport energy surface in this 1-D reduction; attempt-frequency and downstream rate operators differ.",
        },
        "curves": [{k: v for k, v in item.items() if k not in {"color", "linestyle"}}
                   for item in curves],
        "outputs": [str(png), str(svg)],
    }
    (OUT / "audit_plastic_barrier_comparison.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
