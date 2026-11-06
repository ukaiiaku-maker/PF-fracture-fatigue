"""Best per-temperature joint fits (families A and B) against the measured curves."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .per_temperature_joint_fit import OUTPUT, measured_extension_limit_u, objective, targets

COLORS = {"A": "C0", "B": "C3", "A_emission_active": "C2", "A_signed_pileup": "C1"}


def best(output, family, C, cases):
    """Best point under the current objective (stored objectives may be stale)."""
    path = Path(output) / f"model_{family}" / f"T{C}_evaluations.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        r["objective"] = objective(cases, r["results"])[0]
    return min(rows, key=lambda r: r["objective"]) if rows else None


def main(C=21, output=OUTPUT):
    """Plot the best point of every family/variant run present under output."""
    output = Path(output)
    cases = targets()[C]
    fits = {f: best(output, f, C, cases) for f in COLORS}
    # Without a measured load trace, show the model's own (unscored) load and stiffness.
    any_linked = any("load_u_um" in case for case in cases)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for case in cases:
        linked = "load_u_um" in case
        axes[0].plot(case["J_da_um"], case["J"], "o" if linked else "s", mfc="none",
                     color="k" if linked else "0.55", ms=4, label=f"measured {case['curve_id']}")
        if linked:
            axes[1].plot(case["load_u_um"], case["load_K"], ".", color="k", ms=2, label="measured")
            axes[2].plot(case["stiff_u_um"], case["stiff"], ".", color="k", ms=2, label="measured")
    for family, row in fits.items():
        if row is None:
            continue
        for case, result in zip(cases, row["results"]):
            linked = "load_u_um" in case
            style = "-" if linked else "--"
            label = f"{family} ({row['objective']:.1f}) {result['curve_id']} [{result['status']}]"
            axes[0].plot(result.get("J_da_um", []), result.get("J", []), style,
                         color=COLORS[family], label=label)
            if linked or not any_linked:
                u_lim = measured_extension_limit_u(case, result) if linked else np.inf
                if np.isfinite(u_lim):
                    for ax in axes[1:]:
                        ax.axvline(u_lim, color="0.6", lw=.8, ls=":")
                # Tip and bulk state are read from the last crack event; none without one.
                upl = result.get("bulk_upl_um")
                state = ("no crack event" if upl is None else
                         f"bulk u_pl {upl:.2f} µm, tip emissions {result.get('tip_activations') or 0:.0f}")
                axes[1].plot(result["u_um"], result["K_nominal"], style, color=COLORS[family],
                             label=f"{family}{'' if linked else ' (unscored)'}, {state}")
                axes[2].plot(result["u_um"], result["stiffness"], style, color=COLORS[family],
                             label=family)
    axes[0].set(xlabel="Crack extension (µm)", ylabel="J (N/m)", xlim=(0, 1.05))
    axes[1].set(xlabel="Imposed displacement (µm)", ylabel="Nominal K (MPa√m)")
    axes[2].set(xlabel="Imposed displacement (µm)", ylabel="Relative stiffness")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{C} °C per-temperature joint fit")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"T{C}_best_A_vs_B.{suffix}", dpi=170)
    return fits


if __name__ == "__main__":
    for family, row in main(int(sys.argv[1]) if len(sys.argv) > 1 else 21).items():
        if row:
            print(family, round(row["objective"], 3), row["stage"], row["x"])
