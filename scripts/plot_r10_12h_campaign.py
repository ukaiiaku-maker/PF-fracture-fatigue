#!/usr/bin/env python3
"""Create compact, reproducible figures from accepted R10 campaign artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_effective(path: Path):
    result = json.loads(path.read_text())
    progress_path = path.with_name("progress.json")
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if float(progress.get("accepted_cycles", 0.)) > float(result.get("accepted_cycles", 0.)):
            result = {**result, **progress}
    return result


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fatigue_plot(analysis: Path, out: Path):
    rows = read_csv(analysis / "event_waiting_and_windows.csv")
    rows = [row for row in rows if row.get("developed_rate_m_per_cycle")]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    groups = {}
    for row in rows:
        label = (f"{row['candidate_id']} | T={float(row['temperature_K']):g} K, "
                 f"R={float(row['R']):g}, f={float(row['frequency_Hz']):g} Hz")
        groups.setdefault(label, []).append(row)
    for label, group in groups.items():
        group.sort(key=lambda row: float(row["DeltaK_MPa_sqrt_m"]))
        ax.plot([float(row["DeltaK_MPa_sqrt_m"]) for row in group],
                [float(row["developed_rate_m_per_cycle"]) for row in group],
                marker="o", linewidth=1.2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta K$ (MPa $\sqrt{\mathrm{m}}$)")
    ax.set_ylabel(r"developed $da/dN$ (m cycle$^{-1}$)")
    if groups:
        ax.legend(fontsize=6, frameon=False)
    save(fig, out / "developed_fatigue_rates.png")


def growth_history_plot(campaign: Path, out: Path):
    patterns = (
        "hcf_round_3_17780_developed/*/result.json",
        "hcf_round_5_28340_developed/*/result.json",
        "hcf_seed_replicates_17780/*/result.json",
    )
    paths = []
    for pattern in patterns:
        paths.extend(campaign.glob(pattern))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for path in sorted(paths):
        result = load_effective(path)
        events = result.get("events", [])
        if not events:
            continue
        ax.step([max(float(row["cycle"]), 1e-12) for row in events],
                [1e6 * float(row["extension_m"]) for row in events], where="post",
                linewidth=1., label=f"{result['candidate_id']} f={float(result['reference_factor']):g}")
    ax.set_xscale("log")
    ax.set_xlabel("accepted cycles, N")
    ax.set_ylabel(r"crack extension, $\Delta a$ ($\mu$m)")
    if paths:
        ax.legend(fontsize=6, frameon=False, ncol=2)
    save(fig, out / "crack_extension_histories.png")


def thermal_plot(campaign: Path, out: Path):
    roots = (
        campaign / "thermal_ramps_17780",
        campaign / "thermal_ramps_22871",
        campaign / "thermal_ramps_active_plastic_descendants_K400",
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    censor_labeled = False
    for root in roots:
        if not root.is_dir():
            continue
        for candidate_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            resolved, censored = [], []
            for path in sorted(candidate_dir.glob("T_*K.json")):
                row = json.loads(path.read_text())
                temperature = float(row["temperature_K"])
                value = row.get("first_committed_separation_K_MPa_sqrt_m")
                if value is not None:
                    resolved.append((temperature, float(value)))
                elif row.get("status") == "KMAX_CENSOR":
                    censored.append((temperature, float(row["Kmax_MPa_sqrt_m"])))
            label = candidate_dir.name.replace("R10F_", "")
            if resolved:
                ax.plot(*zip(*resolved), marker="o", linewidth=1.1, label=label)
            if censored:
                ax.scatter(*zip(*censored), marker="^", facecolors="none", edgecolors="black", s=32,
                           label="Kmax censor" if not censor_labeled else None)
                censor_labeled = True
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"committed separation load (MPa $\sqrt{\mathrm{m}}$)")
    ax.legend(fontsize=5.5, frameon=False, ncol=2)
    save(fig, out / "thermal_fracture_responses.png")


def mechanism_plot(analysis: Path, out: Path):
    rows = read_csv(analysis / "matched_22871_event_comparison.csv")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if rows:
        ax.plot([int(row["event_index"]) for row in rows],
                [100 * float(row["relative_difference"]) for row in rows], marker="o")
    ax.set_xlabel("matched event index")
    ax.set_ylabel("full vs frozen event-time difference (%)")
    save(fig, out / "matched_plastic_feedback_event_times.png")


def state_plot(campaign: Path, out: Path):
    paths = {
        "evolving full state": campaign / "hcf_round_1_block_2/R10F_V2_22871__MS/T_300__R_0.1__Hz_1000/f_1.0100/result.json",
        "frozen opening feedback": campaign / "hcf_mechanism_controls_22871/frozen/result.json",
    }
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    for label, path in paths.items():
        if not path.is_file():
            continue
        result = load_effective(path)
        history = result.get("history", [])
        cycles = [max(float(row["accepted_cycles"]), 1.) for row in history]
        axes[0].plot(cycles, [1e6 * float(row.get("tip_radius_m", 0.)) for row in history],
                     marker="o", markersize=2.5, label=label)
        axes[1].plot(cycles, [float(row.get("mobile_count", 0.)) for row in history],
                     marker="o", markersize=2.5, label=label)
    axes[0].set_ylabel(r"tip radius ($\mu$m)")
    axes[1].set_ylabel("mobile population")
    axes[1].set_xlabel("accepted cycles, N")
    for ax in axes:
        ax.set_xscale("log")
        ax.legend(fontsize=7, frameon=False)
    save(fig, out / "matched_state_histories.png")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fatigue_plot(args.analysis, args.out)
    growth_history_plot(args.campaign, args.out)
    thermal_plot(args.campaign, args.out)
    mechanism_plot(args.analysis, args.out)
    state_plot(args.campaign, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
