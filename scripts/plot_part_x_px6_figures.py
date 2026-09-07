"""PX6 required figures. Reads ONLY tracked artifacts/crack_rebonding_
part_x_v1/*.csv -- never a gitignored runs/ directory.

Usage:
    <pinned interpreter> scripts/plot_part_x_px6_figures.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

PROTOCOL_LABEL = {
    "D1": "D1 (R=-0.95)", "D2": "D2 (COMPETING_REVERSIBLE)", "D3": "D3 (f=316Hz)",
    "D5": "D5 (PASSIVATION_LIMITED)", "D6_conditional_persistent": "D6 (persistent)",
}
PROTOCOL_COLOR = {
    "D1": "tab:blue", "D2": "tab:orange", "D3": "tab:green", "D5": "tab:red", "D6_conditional_persistent": "tab:purple",
}


def plot_S_h_developed_vs_Kmax(out_dir: Path) -> str:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_rate_table.csv")))
    by_protocol: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if r["seed"] != "1720":
            continue
        by_protocol.setdefault(r["protocol"], []).append((float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, float(r["S_h_developed"])))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for protocol, pts in by_protocol.items():
        pts.sort()
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax.plot(x, y, marker="o", color=PROTOCOL_COLOR[protocol], label=PROTOCOL_LABEL[protocol])
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$K_{max}$ (MPa $\sqrt{m}$)")
    ax.set_ylabel(r"$S_h$ developed $= \log_{10}(\dot{a}_{finite}/\dot{a}_{zero})$")
    ax.set_title("PX4: developed-regime rebonding slowdown vs Kmax, seed=1720")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "px6_S_h_developed_vs_Kmax_all_protocols.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def plot_px5_static_shield_comparison(out_dir: Path) -> str:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_analysis.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), sharey=True)
    for ax, protocol in zip(axes, ("D2", "D5")):
        pts = sorted((float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, r) for r in rows if r["protocol"] == protocol)
        x = [p[0] for p in pts]
        y_dyn = [float(p[1]["S_h_dynamic"]) for p in pts]
        y_ceil = [float(p[1]["ceiling_static_S_h"]) for p in pts]
        y_orbit = [float(p[1]["orbit_matched_static_S_h"]) for p in pts]
        ax.plot(x, y_dyn, marker="o", color="black", linewidth=2.5, label="dynamic rebonding (PX4)")
        ax.plot(x, y_ceil, marker="s", linestyle="--", color="tab:red", label="ceiling static shield")
        ax.plot(x, y_orbit, marker="^", linestyle="--", color="tab:blue", label="periodic-orbit-matched static shield")
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel(r"$K_{max}$ (MPa $\sqrt{m}$)")
        ax.set_title(PROTOCOL_LABEL[protocol])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$S_h$ developed")
    axes[0].legend(fontsize=8)
    fig.suptitle("PX5: dynamic rebonding vs prescribed static-shield controls")
    fig.tight_layout()
    path = out_dir / "px6_px5_static_shield_vs_dynamic.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def plot_seed_robustness(out_dir: Path) -> str:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_slope_table.csv")))
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    m_min = min(min(float(r["seed_1720_slope_m"]), float(r["seed_1001723_slope_m"])) for r in rows)
    m_max = max(max(float(r["seed_1720_slope_m"]), float(r["seed_1001723_slope_m"])) for r in rows)
    pad = 0.05 * (m_max - m_min)
    ax.plot([m_min - pad, m_max + pad], [m_min - pad, m_max + pad], color="grey", linestyle=":", label="y = x")
    for r in rows:
        ax.scatter(float(r["seed_1720_slope_m"]), float(r["seed_1001723_slope_m"]), s=60, label=r["base_protocol"])
    ax.set_xlabel("log-log Kmax slope m, seed=1720")
    ax.set_ylabel("log-log Kmax slope m, seed=1001723")
    ax.set_title("PX4.1: seed-to-seed slope robustness (all 5 axes REBONDING_SEED_ROBUST)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "px6_seed_robustness_slope_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def main() -> None:
    out_dir = ARTIFACTS_DIR
    names = [
        plot_S_h_developed_vs_Kmax(out_dir),
        plot_px5_static_shield_comparison(out_dir),
        plot_seed_robustness(out_dir),
    ]
    for name in names:
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
