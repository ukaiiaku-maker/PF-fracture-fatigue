#!/usr/bin/env python3
"""Plot audit-qualified QSTE340TM opening barriers on their sampled stress support."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis
from scripts.refit_qste340tm_restored_native import build_records
import scripts.refit_qste340tm_resolved_sloping_barrier as slope
import scripts.refit_qste340tm_signed_bulk_1d as signed

BASE = ROOT / "literature_data/fatigue_recovered_review/model_conditional/emission_owned_blunting"
OUT = BASE / "native_signed_bulk_refit_20261028"
TEMPERATURE_K = 298.15

CASES = (
    ("Signed bulk", BASE / "resolved_increment_1d/signed_bulk_joint_refit_20261028/fit.json",
     "#0072B2", "-", True),
    ("No bulk, best refit", BASE / "resolved_increment_1d/no_bulk_matched_refit_20261028/fit.json",
     "#D55E00", "--", False),
    ("No bulk, repeat refit", BASE / "resolved_increment_1d/no_bulk_equal_effort_control_20261028/fit.json",
     "#009E73", ":", False),
)


def main() -> int:
    records = build_records("index_rank", premeasurement=True)
    evaluated = []
    for label, path, color, linestyle, has_signed_bulk in CASES:
        fit = json.loads(path.read_text())
        vector = np.asarray(fit["parameters_vector"], dtype=float)
        phases = int(fit.get("final_phases", fit.get("settings", {}).get("final_phases", 48)))
        step = float(fit.get("advection_step_m", 1.0e-6 * fit.get("settings", {}).get("advection_step_um", 0.25)))
        if has_signed_bulk:
            state = signed.trajectory(vector, records, phases=phases, advection_step_m=step)
        else:
            state = slope.state_trajectory(vector, records, phases, step)
        p = slope.unpack(vector)
        opening_stress = np.maximum(
            state["stress"] - p["cleavage_backstress_fraction"] * state["backstress_Pa"][:, None],
            0.0,
        )
        evaluated.append({
            "label": label, "path": str(path), "color": color, "linestyle": linestyle,
            "parameters": p, "sampled_stress_Pa": [float(opening_stress.min()),
                                                     float(opening_stress.max())],
        })

    maximum = 1.05 * max(item["sampled_stress_Pa"][1] for item in evaluated)
    stress = np.linspace(0.0, maximum, 1400)
    figure, axis = plt.subplots(figsize=(9.2, 6.0))
    for item in evaluated:
        p = item["parameters"]
        basis = SmoothBroadBasis(p["opening_sigma_b_Pa"], p["opening_exponent"],
                                 p["opening_smoothing_ratio"])
        barrier_kbt = p["opening_floor_over_kBT"] + p["opening_amplitude_over_kBT"] * basis.value(stress)
        barrier_eV = barrier_kbt * KB_EV_PER_K * TEMPERATURE_K
        item["barrier_eV_over_grid"] = [float(barrier_eV.min()), float(barrier_eV.max())]
        axis.plot(stress / 1.0e9, barrier_eV, color=item["color"],
                  linestyle=item["linestyle"], lw=2.5, label=item["label"])
        lo, hi = item["sampled_stress_Pa"]
        axis.plot([lo / 1.0e9, hi / 1.0e9], [0.018, 0.018], color=item["color"],
                  linestyle=item["linestyle"], lw=3.0, solid_capstyle="butt")

    axis.text(0.01, 0.025, "colored segments: sampled effective-stress support",
              transform=axis.transAxes, fontsize=9, color="0.35")
    axis.set(xlabel="effective opening stress (GPa)",
             ylabel=r"constitutive $G^*_{open}$ (eV)",
             title="Audit-qualified QSTE340TM sloping-floor opening barriers at 298.15 K")
    axis.set_ylim(bottom=0.0)
    axis.grid(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    png = OUT / "audit_opening_barrier_comparison.png"
    svg = OUT / "audit_opening_barrier_comparison.svg"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt.close(figure)

    receipt = {
        "schema": "qste340tm-audit-opening-barrier-comparison/1",
        "temperature_K": TEMPERATURE_K,
        "definition": "G_open = kBT [floor + amplitude * SmoothBroadBasis(effective opening stress)]",
        "state_shift_included": False,
        "state_shift_note": "Retained-population embrittlement is history-dependent and is not a unique stress-only barrier.",
        "stress_grid_Pa": [0.0, float(maximum)],
        "curves": [{k: v for k, v in item.items() if k not in {"color", "linestyle"}}
                   for item in evaluated],
        "outputs": [str(png), str(svg)],
    }
    (OUT / "audit_opening_barrier_comparison.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
