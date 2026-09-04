"""Section C required figures for the developed-response confirmation
campaign. Reads ONLY the tracked developed_confirmation_decision.json
(built by scripts/analyze_developed_confirmation.py) -- never a
gitignored runs/ directory.

Usage:
    <pinned interpreter> scripts/plot_developed_confirmation.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_developed_confirmation"
KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)
SEED_COLORS = {1720: "tab:blue", 1001723: "tab:orange"}
SEED_MARKERS = {1720: "o", 1001723: "s"}


def _seed_analyses(decision: dict) -> dict[int, dict]:
    out = {}
    for seed in (1720, 1001723):
        a = decision.get(f"seed_{seed}_analysis")
        if a is not None:
            out[seed] = a
    return out


def plot_developed_da_dN_vs_Kmax(seed_analyses: dict[int, dict], out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = [K / 1.0e6 for K in KMAX_GRID_Pa_sqrt_m]
    for seed, a in seed_analyses.items():
        zero_y, finite_y = [], []
        for K in KMAX_GRID_Pa_sqrt_m:
            pair = a["per_pair"][f"K{int(round(K/1e6))}MPa_seed{seed}"]
            zero_y.append(pair["gate_zero"]["developed_interval"]["da_dN"])
            finite_y.append(pair["gate_finite"]["developed_interval"]["da_dN"])
        ax.plot(x, zero_y, marker=SEED_MARKERS[seed], linestyle="--",
                 color=SEED_COLORS[seed], label=f"zero cohesion, seed {seed}")
        ax.plot(x, finite_y, marker=SEED_MARKERS[seed], linestyle="-",
                 color=SEED_COLORS[seed], label=f"finite cohesion, seed {seed}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$K_{max}$ (MPa $\sqrt{m}$)")
    ax.set_ylabel("developed da/dN (m/cycle)")
    ax.set_title("Developed-window da/dN vs Kmax")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "developed_da_dN_vs_Kmax_zero_and_finite.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def plot_S_h_vs_Kmax_all_windows(seed_analyses: dict[int, dict], out_dir: Path) -> str:
    windows = ["developed", "all_event", "final_half", "final_six"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    x = [math.log10(K) for K in KMAX_GRID_Pa_sqrt_m]
    for ax, window in zip(axes.flat, windows):
        for seed, a in seed_analyses.items():
            fit = a["fits_by_window"][window]
            if fit is None:
                continue
            y = [fit["S_h_by_Kmax"][str(int(K))] for K in KMAX_GRID_Pa_sqrt_m]
            ax.plot(x, y, marker=SEED_MARKERS[seed], color=SEED_COLORS[seed],
                     label=f"seed {seed} (delta_m={fit['delta_m_least_squares']:.3f})")
        ax.set_title(window)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}(K_{max})$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$S_h$ (decade)")
    fig.suptitle("Rate-shift S_h(K) across analysis windows")
    fig.tight_layout()
    path = out_dir / "S_h_vs_Kmax_all_windows.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def plot_delta_m_and_secants(seed_analyses: dict[int, dict], out_dir: Path) -> str:
    windows = ["developed", "all_event", "final_half", "final_six"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.35
    xpos = range(len(windows))
    for i, seed in enumerate(seed_analyses):
        a = seed_analyses[seed]
        dm = [a["fits_by_window"][w]["delta_m_least_squares"] if a["fits_by_window"][w] else float("nan") for w in windows]
        offset = (i - 0.5) * width
        ax1.bar([p + offset for p in xpos], dm, width=width, color=SEED_COLORS[seed], label=f"seed {seed}")
    ax1.axhline(0.25, color="red", linestyle=":", label="slope_gate=0.25")
    ax1.set_xticks(list(xpos)); ax1.set_xticklabels(windows, rotation=20)
    ax1.set_ylabel("delta_m (3-point least squares)")
    ax1.set_title("delta_m by window")
    ax1.legend(fontsize=7); ax1.grid(axis="y", alpha=0.3)

    for seed, a in seed_analyses.items():
        fit = a["fits_by_window"]["developed"]
        if fit is None:
            continue
        ax2.plot([1, 2], [fit["secant_15_to_18"], fit["secant_18_to_21"]],
                  marker="o", color=SEED_COLORS[seed], label=f"seed {seed}")
    ax2.set_xticks([1, 2]); ax2.set_xticklabels(["15->18", "18->21"])
    ax2.set_ylabel("local secant slope (developed window)")
    ax2.set_title("Adjacent secants (developed window)")
    ax2.legend(fontsize=7); ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "local_delta_m_and_secants.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def plot_eventwise_and_rolling_ratios(seed_analyses: dict[int, dict], out_dir: Path) -> str:
    fig, axes = plt.subplots(1, len(seed_analyses) or 1, figsize=(6.5 * max(len(seed_analyses), 1), 5), squeeze=False)
    for col, (seed, a) in enumerate(seed_analyses.items()):
        ax = axes[0][col]
        for K in KMAX_GRID_Pa_sqrt_m:
            pair = a["per_pair"][f"K{int(round(K/1e6))}MPa_seed{seed}"]
            rows = pair["eventwise_and_rolling_waiting_time_ratios"]["eventwise"]
            xs = [r["event_index"] for r in rows]
            ys = [r["log10_ratio_signed"] for r in rows]
            ax.plot(xs, ys, marker=".", alpha=0.6, label=f"K={int(round(K/1e6))} eventwise")
            rolling = pair["eventwise_and_rolling_waiting_time_ratios"]["rolling_4event"]
            rxs = [sum(r["window_event_indices"]) / len(r["window_event_indices"]) for r in rolling]
            rys = [r["log10_ratio_signed"] for r in rolling]
            ax.plot(rxs, rys, linewidth=2, label=f"K={int(round(K/1e6))} 4-event rolling")
        ax.axhline(0.0, color="k", linewidth=0.5)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("event index")
        ax.set_ylabel("log10(waiting_time_finite / waiting_time_zero)")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.suptitle("Eventwise and 4-event rolling finite/zero waiting-time ratios")
    fig.tight_layout()
    path = out_dir / "eventwise_and_rolling_waiting_time_ratios.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def plot_pB_contact_action_vs_Kmax(seed_analyses: dict[int, dict], out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = [K / 1.0e6 for K in KMAX_GRID_Pa_sqrt_m]
    for seed, a in seed_analyses.items():
        pB, contact_t, aw_uncond, aw_cond = [], [], [], []
        for K in KMAX_GRID_Pa_sqrt_m:
            pair = a["per_pair"][f"K{int(round(K/1e6))}MPa_seed{seed}"]
            fa = pair["finite_exposure_and_action"]
            pB.append(fa["pre_event_max_pB_mean"])
            contact_t.append(fa["total_negative_K_contact_time_s"])
            aw_uncond.append(fa["action_weighted_K_rebond_unconditional_mean_Pa_sqrt_m"])
            aw_cond.append(fa["action_weighted_K_rebond_conditional_mean_nonzero_Pa_sqrt_m"])
        axes[0].plot(x, pB, marker=SEED_MARKERS[seed], color=SEED_COLORS[seed], label=f"seed {seed}")
        axes[1].plot(x, contact_t, marker=SEED_MARKERS[seed], color=SEED_COLORS[seed], label=f"seed {seed}")
        axes[2].plot(x, aw_uncond, marker=SEED_MARKERS[seed], color=SEED_COLORS[seed],
                      linestyle="--", label=f"seed {seed} unconditional")
        axes[2].plot(x, aw_cond, marker=SEED_MARKERS[seed], color=SEED_COLORS[seed],
                      linestyle="-", label=f"seed {seed} conditional (nonzero)")
    axes[0].set_title("pre-event max p_B (mean)"); axes[0].set_xlabel(r"$K_{max}$ (MPa $\sqrt{m}$)")
    axes[1].set_title("integrated negative-K contact time (s)"); axes[1].set_xlabel(r"$K_{max}$ (MPa $\sqrt{m}$)")
    axes[2].set_title("action-weighted K_rebond (Pa sqrt(m))"); axes[2].set_xlabel(r"$K_{max}$ (MPa $\sqrt{m}$)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=6)
    fig.tight_layout()
    path = out_dir / "pB_contact_time_action_weighted_Krebond_vs_Kmax.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def main() -> int:
    decision = json.loads((DEV_ARTIFACTS / "developed_confirmation_decision.json").read_text())
    seed_analyses = _seed_analyses(decision)
    if not seed_analyses:
        raise SystemExit("no seed analyses present in developed_confirmation_decision.json")

    outputs = [
        plot_developed_da_dN_vs_Kmax(seed_analyses, DEV_ARTIFACTS),
        plot_S_h_vs_Kmax_all_windows(seed_analyses, DEV_ARTIFACTS),
        plot_delta_m_and_secants(seed_analyses, DEV_ARTIFACTS),
        plot_eventwise_and_rolling_ratios(seed_analyses, DEV_ARTIFACTS),
        plot_pB_contact_action_vs_Kmax(seed_analyses, DEV_ARTIFACTS),
    ]
    for name in outputs:
        print(f"wrote {DEV_ARTIFACTS / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
