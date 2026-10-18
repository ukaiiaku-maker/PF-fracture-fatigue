#!/usr/bin/env python3
"""Create compact, reproducible figures from accepted R10 campaign artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

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
        reference = row.get("plastic_feedback_reference") or "NATIVE_FULL_STATE"
        label = (f"{row['candidate_id']} | T={float(row['temperature_K']):g} K, "
                 f"R={float(row['R']):g}, f={float(row['frequency_Hz']):g} Hz | "
                 f"{reference}")
        groups.setdefault(label, []).append(row)
    for label, group in groups.items():
        by_amplitude = {}
        for row in group:
            by_amplitude.setdefault(float(row["DeltaK_MPa_sqrt_m"]), []).append(
                float(row["developed_rate_m_per_cycle"]))
        x, y, low, high = [], [], [], []
        for amplitude, rates in sorted(by_amplitude.items()):
            center = median(rates)
            x.append(amplitude); y.append(center)
            low.append(center - min(rates)); high.append(max(rates) - center)
        ax.errorbar(x, y, yerr=[low, high], marker="o", linewidth=1.2,
                    capsize=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta K$ (MPa $\sqrt{\mathrm{m}}$)")
    ax.set_ylabel(r"developed $da/dN$ (m cycle$^{-1}$)")
    if groups:
        ax.legend(fontsize=6, frameon=False)
    save(fig, out / "developed_fatigue_rates.png")


def growth_history_plot(campaign: Path, out: Path):
    representative_factors = {
        "R10F_V2_17780__MS": {1.0375, 1.0525, 1.075},
        "R10F_V2_28475__MS": {1.0025, 1.01, 1.02},
        "R10F_V2_22871__MS": {1.01, 1.02, 1.04},
        "R10F_V2_01796__MS": {1.01, 1.03},
        "R10F_V2_00116__MS": {1.00001, 1.005, 1.01},
        "R10F_V2_11566__MS": {1.06, 1.1},
    }
    patterns = (
        "hcf_round_3_17780_developed/*/result.json",
        "hcf_round_5_28340_developed/*/result.json",
        "hcf_seed_replicates_17780/*/result.json",
        "hcf_round_17_smooth_mixed_28475/**/result.json",
        "hcf_round_18_smooth_mixed_28475_dense/*/result.json",
        "hcf_seed_replicates_28475/*/result.json",
        "hcf_round_24_fast_recurrent_17444/*/result.json",
        "hcf_round_28_intermediate_recurrent_01796/*/result.json",
        "hcf_round_30_22871_amplitude_ladder/*/result.json",
        "hcf_round_31_high_barrier_00116/*/result.json",
        "hcf_round_32_state_active_11566/*/result.json",
        "hcf_round_1_block_2/R10F_V2_22871__MS/**/result.json",
    )
    paths = []
    for pattern in patterns:
        paths.extend(campaign.glob(pattern))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    plotted = set()
    for path in sorted(paths):
        result = load_effective(path)
        events = result.get("events", [])
        if not events:
            continue
        candidate = result["candidate_id"]
        factor = float(result["reference_factor"])
        if (candidate not in representative_factors or
                not any(math.isclose(factor, value, rel_tol=0., abs_tol=1e-10)
                        for value in representative_factors[candidate])):
            continue
        key = (candidate, round(factor, 8))
        if key in plotted:
            continue
        plotted.add(key)
        ax.step([max(float(row["cycle"]), 1e-12) for row in events],
                [1e6 * float(row["extension_m"]) for row in events], where="post",
                linewidth=1.1, label=f"{candidate} f={factor:g}")
    ax.set_xscale("log")
    ax.set_xlabel("accepted cycles, N")
    ax.set_ylabel(r"crack extension, $\Delta a$ ($\mu$m)")
    if plotted:
        ax.legend(fontsize=6.5, frameon=False, loc="upper left",
                  bbox_to_anchor=(1.01, 1.))
    save(fig, out / "crack_extension_histories.png")


def long_exposure_history_plot(campaign: Path, out: Path):
    paths = (
        campaign / ("hcf_round_1_block_2/R10F_V2_22871__MS/"
                    "T_300__R_0.1__Hz_1000/f_1.0100/result.json"),
        campaign / "hcf_round_32_state_active_11566/f_1.06/result.json",
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    plotted = False
    for path in paths:
        if not path.is_file():
            continue
        result = load_effective(path)
        events = result.get("events", [])
        if not events:
            continue
        cycles = [max(float(row["cycle"]), 1e-12) for row in events]
        extensions = [1e6 * float(row["extension_m"]) for row in events]
        accepted = float(result.get("accepted_cycles", cycles[-1]))
        if accepted > cycles[-1]:
            cycles.append(accepted)
            extensions.append(extensions[-1])
        label = (f"{result['candidate_id']} f="
                 f"{float(result['reference_factor']):g}")
        ax.step(cycles, extensions, where="post", linewidth=1.3, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("accepted cycles, N")
    ax.set_ylabel(r"crack extension, $\Delta a$ ($\mu$m)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, out / "long_exposure_hcf_and_arrest.png")


def local_slope_plot(analysis: Path, out: Path):
    """Plot every adjacent developed-window secant without regression averaging."""
    rows = read_csv(analysis / "local_developed_slopes.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    groups = {}
    for row in rows:
        key = (row["candidate_id"], row.get("temperature_K"), row.get("R"),
               row.get("frequency_Hz"), row.get("plastic_feedback_reference"))
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        group.sort(key=lambda row: float(row["DeltaK_low_MPa_sqrt_m"]))
        x = [math.sqrt(float(row["DeltaK_low_MPa_sqrt_m"]) *
                       float(row["DeltaK_high_MPa_sqrt_m"])) for row in group]
        y = [float(row["local_log_slope"]) for row in group]
        label = (f"{key[0]} | T={float(key[1]):g} K, R={float(key[2]):g}, "
                 f"f={float(key[3]):g} Hz | {key[4]}")
        ax.plot(x, y, marker="o", linewidth=1.1, label=label)
    ax.axhspan(3.4, 4.6, color="tab:green", alpha=.15, label="strict target")
    ax.set_yscale("symlog", linthresh=5.)
    ax.set_xlabel(r"geometric-mean $\Delta K$ (MPa $\sqrt{\mathrm{m}}$)")
    ax.set_ylabel("adjacent local log slope")
    ax.legend(fontsize=5.5, frameon=False)
    save(fig, out / "adjacent_local_slopes.png")


def thermal_plot(campaign: Path, out: Path):
    roots = (
        campaign / "thermal_ramps_17780",
        campaign / "thermal_ramps_22871",
        campaign / "thermal_ramps_28340",
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


def thermal_mechanism_plot(campaign: Path, out: Path):
    """Show which thermal response survives a frozen-plastic reference."""
    controls = []
    for path in sorted(campaign.glob("mechanism_controls_*/mechanism_control_summary.csv")):
        rows = read_csv(path)
        if rows:
            controls.append((rows[0]["candidate_id"], rows))
    if not controls:
        return
    fig, axes = plt.subplots(len(controls), 1, figsize=(6.4, 3.2 * len(controls)),
                             sharex=True, squeeze=False)
    for ax, (candidate, rows) in zip(axes[:, 0], controls):
        rows.sort(key=lambda row: float(row["temperature_K"]))
        for prefix, marker, label in (
                ("evolving", "o", "evolving full state"),
                ("frozen", "s", "frozen initial plastic state")):
            resolved = [(float(row["temperature_K"]),
                         float(row[f"{prefix}_commit_K_MPa_sqrt_m"]))
                        for row in rows
                        if row.get(f"{prefix}_commit_K_MPa_sqrt_m") and
                        row.get(f"{prefix}_unsupported", "").lower() != "true"]
            if resolved:
                ax.plot(*zip(*resolved), marker=marker, label=label)
            censored = [(float(row["temperature_K"]),
                         float(row[f"{prefix}_terminal_K_MPa_sqrt_m"]))
                        for row in rows
                        if row.get(f"{prefix}_terminal_K_MPa_sqrt_m") and
                        row.get(f"{prefix}_status") == "KMAX_CENSOR"]
            if censored:
                ax.scatter(*zip(*censored), marker="^", facecolors="none",
                           edgecolors="black", s=38,
                           label=f"{prefix} Kmax censor")
            unsupported = [(float(row["temperature_K"]),
                            float(row.get(f"{prefix}_commit_K_MPa_sqrt_m") or
                                  row.get(f"{prefix}_terminal_K_MPa_sqrt_m")))
                           for row in rows
                           if (row.get(f"{prefix}_commit_K_MPa_sqrt_m") or
                               row.get(f"{prefix}_terminal_K_MPa_sqrt_m")) and
                           row.get(f"{prefix}_unsupported", "").lower() == "true"]
            if unsupported:
                ax.scatter(*zip(*unsupported), marker="x", color="black", s=38,
                           label=f"{prefix} unsupported")
        ax.set_ylabel(r"commit load (MPa $\sqrt{\mathrm{m}}$)")
        ax.set_title(candidate)
        ax.legend(fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel("temperature (K)")
    save(fig, out / "thermal_plastic_state_controls.png")


def mechanism_plot(analysis: Path, out: Path):
    general = analysis / "matched_plastic_feedback_event_comparison.csv"
    rows = read_csv(general if general.is_file() else
                    analysis / "matched_22871_event_comparison.csv")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    groups = {}
    for row in rows:
        groups.setdefault(row.get("candidate_id", "R10F_V2_22871__MS"), []).append(row)
    for candidate, group in groups.items():
        group.sort(key=lambda row: int(row["event_index"]))
        ax.plot([int(row["event_index"]) for row in group],
                [100 * float(row["relative_difference"]) for row in group],
                marker="o", markersize=3, label=candidate)
    ax.set_xlabel("matched event index")
    ax.set_ylabel("full vs frozen event-time difference (%)")
    if groups:
        ax.legend(fontsize=7, frameon=False)
    save(fig, out / "matched_plastic_feedback_event_times.png")


def state_plot(campaign: Path, out: Path):
    paths = {
        "evolving full state": campaign / "hcf_round_1_block_2/R10F_V2_22871__MS/T_300__R_0.1__Hz_1000/f_1.0100/result.json",
        "frozen opening feedback": campaign / "hcf_mechanism_controls_22871/frozen_phase16_same_seed/result.json",
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


def athermal_transition_plot(campaign: Path, out: Path):
    """Keep runouts, arrested increments, and cascades visually distinct."""
    wanted = {"R10F_V2_06081__AB", "R10F_V2_21280__AB"}
    points = {}
    for pattern in ("hcf_round_9_athermal_new/*/*/*/result.json",
                    "hcf_round_12_athermal_brackets_*/*/*/*/result.json",
                    "hcf_round_13_athermal_renewal_*/*/*/*/result.json"):
        for path in campaign.glob(pattern):
            result = load_effective(path)
            if result.get("candidate_id") not in wanted:
                continue
            key = (result["candidate_id"], float(result["reference_factor"]))
            if key not in points or float(result["accepted_cycles"]) > float(
                    points[key]["accepted_cycles"]):
                points[key] = result
    if not points:
        return
    styles = {
        "runout": ("o", "none"),
        "arrested one-cell increment": ("^", "tab:orange"),
        "single-excursion cascade": ("X", "tab:red"),
    }
    colors = {"R10F_V2_06081__AB": "tab:blue", "R10F_V2_21280__AB": "tab:green"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    labels = set()
    for (candidate, factor), result in sorted(points.items()):
        events = len(result.get("events", []))
        if events == 0:
            category = "runout"
        elif events == 1:
            category = "arrested one-cell increment"
        else:
            category = "single-excursion cascade"
        marker, fill = styles[category]
        label = f"{candidate} — {category}"
        edge = colors[candidate]
        ax.scatter([factor], [max(float(result["accepted_cycles"]), 1e-6)],
                   marker=marker, s=55, facecolors=fill if fill != "none" else "none",
                   edgecolors=edge, linewidths=1.2,
                   label=label if label not in labels else None)
        labels.add(label)
    ax.set_yscale("log")
    ax.set_xlabel("reference-load factor")
    ax.set_ylabel("accepted cycles before disposition")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    save(fig, out / "athermal_bilinear_transition_map.png")


def active_plastic_transition_plot(campaign: Path, out: Path):
    """Contrast plasticity-arrested bursts with matched frozen cascades."""
    paths = []
    for pattern in (
            "hcf_round_23_active_plastic_low_area/*/result.json",
            "hcf_round_25_active_plastic_bracket/*/result.json",
            "hcf_mechanism_controls_active_07172/*/result.json"):
        paths.extend(campaign.glob(pattern))
    records = []
    for path in paths:
        result = load_effective(path)
        records.append({
            "factor": float(result["reference_factor"]),
            "events": len(result.get("events", [])),
            "cycles": max(float(result.get("accepted_cycles", 0.)), 1e-12),
            "reference": result.get("plastic_feedback_reference", "EVOLVING_FULL_STATE"),
            "disposition": result.get("disposition", ""),
            "unsupported": bool(result.get("unsupported_validities")) or
                           result.get("disposition") == "UNRESOLVED_OPENING_PHYSICS",
        })
    if not records:
        return
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    styles = {
        "EVOLVING_FULL_STATE": ("o", "tab:blue", "evolving full state"),
        "FROZEN_INITIAL_PLASTIC_STATE": ("s", "tab:orange", "frozen initial plastic state"),
    }
    for reference, (marker, color, label) in styles.items():
        group = sorted((row for row in records if row["reference"] == reference and
                        not row["unsupported"]),
                       key=lambda row: row["factor"])
        if not group:
            continue
        axes[0].plot([row["factor"] for row in group],
                     [row["events"] for row in group], marker=marker,
                     color=color, linewidth=1.1, label=label)
        axes[1].plot([row["factor"] for row in group],
                     [row["cycles"] for row in group], marker=marker,
                     color=color, linewidth=1.1, label=label)
    unresolved = [row for row in records if row["unsupported"]]
    if unresolved:
        axes[0].scatter([row["factor"] for row in unresolved],
                        [row["events"] for row in unresolved], marker="x",
                        color="black", s=55, label="unsupported opening transit")
        axes[1].scatter([row["factor"] for row in unresolved],
                        [row["cycles"] for row in unresolved], marker="x",
                        color="black", s=55, label="unsupported opening transit")
    axes[0].set_ylabel("physical advances")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("accepted cycles")
    axes[1].set_xlabel("reference-load factor")
    for ax in axes:
        ax.legend(fontsize=7, frameon=False)
    save(fig, out / "active_plastic_burst_arrest_transition.png")


def state_active_11566_plot(campaign: Path, out: Path):
    """Show the newly acquired plasticity-created arrest/reactivation bracket."""
    paths = list(campaign.glob("hcf_round_32_state_active_11566/*/result.json"))
    paths.extend(campaign.glob("hcf_mechanism_controls_11566/*/result.json"))
    if not paths:
        return
    records = []
    for path in paths:
        result = load_effective(path)
        records.append((
            float(result["reference_factor"]), len(result.get("events", [])),
            max(float(result.get("accepted_cycles", 0.)), 1e-12),
            result.get("plastic_feedback_reference", "EVOLVING_FULL_STATE")))
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    styles = {
        "EVOLVING_FULL_STATE": ("o", "tab:blue", "evolving full state"),
        "FROZEN_INITIAL_PLASTIC_STATE": ("s", "tab:orange", "frozen initial plastic state"),
    }
    for reference, (marker, color, label) in styles.items():
        rows = sorted(row for row in records if row[3] == reference)
        if not rows:
            continue
        axes[0].plot([row[0] for row in rows], [row[1] for row in rows],
                     marker=marker, color=color, label=label)
        axes[1].plot([row[0] for row in rows], [row[2] for row in rows],
                     marker=marker, color=color, label=label)
    axes[0].set_ylabel("physical advances")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("accepted cycles")
    axes[1].set_xlabel("reference-load factor")
    for ax in axes:
        ax.legend(fontsize=7, frameon=False)
    save(fig, out / "state_active_11566_transition.png")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fatigue_plot(args.analysis, args.out)
    local_slope_plot(args.analysis, args.out)
    growth_history_plot(args.campaign, args.out)
    long_exposure_history_plot(args.campaign, args.out)
    thermal_plot(args.campaign, args.out)
    thermal_mechanism_plot(args.campaign, args.out)
    mechanism_plot(args.analysis, args.out)
    state_plot(args.campaign, args.out)
    athermal_transition_plot(args.campaign, args.out)
    active_plastic_transition_plot(args.campaign, args.out)
    state_active_11566_plot(args.campaign, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
