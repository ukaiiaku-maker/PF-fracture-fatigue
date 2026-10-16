#!/usr/bin/env python3
"""Decision figures for the R10 mechanism-balanced information block."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "integration_evidence/r10_global_mechanism_search/balanced_information_batch"
FIG = DATA / "figures"


def read(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def num(row, key):
    value = row.get(key, "")
    return None if value in ("", None) else float(value)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fatigue = read("fatigue_outcomes.csv")
    fracture = read("fracture_outcomes.csv")
    colors = {"OPENING_DIRECTED": "#2468a2", "MIXED_DIRECTED": "#d17c0f",
              "PLASTICITY_DIRECTED": "#8b3a8f"}

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for lane in colors:
        rows = [r for r in fatigue if r["lane"] == lane and r["stream"] == "PRIMARY"]
        developed = sorted([r for r in rows if num(r, "developed_20_60_rate_m_per_cycle")],
                           key=lambda r: num(r, "DeltaK_MPa_sqrt_m"))
        if developed:
            ax.plot([num(r, "DeltaK_MPa_sqrt_m") for r in developed],
                    [num(r, "developed_20_60_rate_m_per_cycle") for r in developed],
                    "o-", color=colors[lane], label=lane.replace("_DIRECTED", " full"))
            ax.plot([num(r, "DeltaK_MPa_sqrt_m") for r in developed],
                    [num(r, "A0_physical_rate_m_per_cycle") for r in developed],
                    "--", color=colors[lane], alpha=.65, label=lane.replace("_DIRECTED", " A0"))
        censored = [r for r in rows if num(r, "zero_event_95pct_rate_upper_m_per_cycle")]
        if censored:
            ax.scatter([num(r, "DeltaK_MPa_sqrt_m") for r in censored],
                       [num(r, "zero_event_95pct_rate_upper_m_per_cycle") for r in censored],
                       marker="v", color=colors[lane], label=lane.replace("_DIRECTED", " 95% upper"))
            for r in censored:
                ax.plot([num(r, "DeltaK_MPa_sqrt_m")]*2,
                        [num(r, "zero_event_95pct_rate_upper_m_per_cycle")/8,
                         num(r, "zero_event_95pct_rate_upper_m_per_cycle")],
                        color=colors[lane], lw=.8)
    ax.set(xscale="log", yscale="log", xlabel=r"$\Delta K$ (MPa $\sqrt{m}$)",
           ylabel=r"developed $da/dN$ (m/cycle)")
    ax.legend(fontsize=7, ncol=2, frameon=False); ax.grid(True, which="both", alpha=.2)
    fig.tight_layout(); fig.savefig(FIG / "developed_rates_and_censors.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for lane in colors:
        rows = sorted([r for r in fracture if r["lane"] == lane], key=lambda r: num(r, "temperature_K"))
        ax.plot([num(r, "temperature_K") for r in rows],
                [num(r, "K_init_MPa_sqrt_m") for r in rows], "o-",
                color=colors[lane], label=lane.replace("_DIRECTED", ""))
    ax.set(xlabel="Temperature (K)", ylabel=r"native $K_{init}$ (MPa $\sqrt{m}$)")
    ax.legend(frameon=False); ax.grid(True, alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "same_row_fracture_reachability.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for lane in ("OPENING_DIRECTED", "MIXED_DIRECTED"):
        rows = sorted([r for r in fatigue if r["lane"] == lane and r["stream"] == "PRIMARY"
                       and num(r, "C_rate_full_over_A0")], key=lambda r: num(r, "DeltaK_MPa_sqrt_m"))
        ax.plot([num(r, "DeltaK_MPa_sqrt_m") for r in rows],
                [num(r, "C_rate_full_over_A0") for r in rows], "o-",
                color=colors[lane], label=lane.replace("_DIRECTED", ""))
    ax.axhline(1.0, color="black", lw=.8); ax.axhspan(1/1.5, 1.5, color="gray", alpha=.12)
    ax.set(yscale="log", xlabel=r"$\Delta K$ (MPa $\sqrt{m}$)", ylabel=r"$C_{rate}=v_{full}/v_{A0}$")
    ax.legend(frameon=False); ax.grid(True, which="both", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "mechanism_rate_effect.png", dpi=180); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
