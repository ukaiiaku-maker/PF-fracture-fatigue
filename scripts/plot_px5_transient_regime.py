"""Figure for the PX5 transient-regime analysis: rolling (5-event window)
finite/zero and static/zero waiting-time ratio vs event index, for D2 and
D5 at Kmax=12 MPa*sqrt(m) (the largest-effect condition) and Kmax=18
MPa*sqrt(m) (a representative mid-grid point). Reads only tracked
artifacts/crack_rebonding_part_x_v1/px5_transient_regime_analysis.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"


def main() -> None:
    d = json.loads((ARTIFACTS_DIR / "px5_transient_regime_analysis.json").read_text())
    by_key = {(p["protocol"], p["Kmax_MPa_sqrt_m"]): p for p in d["points"]}

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    panels = [("D2", 12.0), ("D2", 18.0), ("D5", 12.0), ("D5", 18.0)]
    for ax, (protocol, Kmax) in zip(axes.flat, panels):
        p = by_key[(protocol, Kmax)]
        rolling = p["rolling_5event_ratio"]
        x = list(range(len(rolling["dynamic_over_zero"])))
        ax.plot(x, rolling["dynamic_over_zero"], marker="o", color="black", label="dynamic/zero")
        ax.plot(x, rolling["ceiling_over_zero"], marker="s", linestyle="--", color="tab:red", label="ceiling/zero")
        ax.plot(x, rolling["orbit_matched_over_zero"], marker="^", linestyle="--", color="tab:blue", label="orbit-matched/zero")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        ax.set_title(f"{protocol}, Kmax={Kmax} MPa$\\sqrt{{m}}$", fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("event index")
    for ax in axes[:, 0]:
        ax.set_ylabel("rolling (5-event) waiting-time ratio")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("PX5 transient-regime analysis: approach to the developed-regime waiting-time ratio")
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig_px5_transient_regime_approach.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
