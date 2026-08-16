#!/usr/bin/env python3
"""Analyze prospective hybrid HCF/LCF fatigue transfers and joint response."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--loads", type=Path, required=True)
    parser.add_argument("--accelerated-root", type=Path, required=True)
    parser.add_argument("--explicit-root", type=Path, required=True)
    parser.add_argument("--fracture-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def classify_status(status: str, extension_um: float) -> tuple[str, str]:
    token = str(status).lower()
    if "target" in token and extension_um >= 99.0:
        return "developed_target_reached", "filled"
    if "maximum_cycle" in token or "cycle_censor" in token or "hazard_censor" in token:
        return "cycle_or_hazard_censor", "downward_triangle"
    if "fail" in token or "reject" in token or "partial" in token:
        return "partial_or_numerical_unresolved", "open_square"
    return "partial_or_numerical_unresolved", "open_square"


def developed_from_events(events: list[dict], development_m: float = 20e-6) -> float:
    if len(events) < 2:
        return float("nan")
    chosen = [event for event in events if float(event.get("cumulative_extension_m", 0.0)) >= development_m]
    if len(chosen) < 2:
        return float("nan")
    first, last = chosen[0], chosen[-1]
    da = float(last["cumulative_extension_m"]) - float(first["cumulative_extension_m"])
    dN = float(last["cycles"]) - float(first["cycles"])
    return da / dN if dN > 0.0 else float("nan")


def load_accelerated(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rates, event_rows = [], []
    for path in sorted(root.rglob("fatigue_result.json")):
        payload = json.loads(path.read_text())
        events = list(payload.get("events", []))
        extension_um = float(payload.get("final_extension_m", 0.0)) * 1e6
        status_class, marker = classify_status(payload.get("status", ""), extension_um)
        value = payload.get("developed_da_dN_m_per_cycle")
        rate = float(value) if value is not None else developed_from_events(events)
        if status_class != "developed_target_reached":
            rate = float("nan")
        loading = payload.get("loading", {})
        row = {
            "candidate_id": payload["candidate_id"],
            "integration_mode": "accelerated",
            "seed": int(payload["seed"]),
            "normalized_f": float(payload["fraction"]),
            "deltaK_MPa_sqrt_m": float(loading.get("deltaK_MPa_sqrt_m", payload["reference_deltaK_MPa_sqrt_m"] * payload["fraction"])),
            "status": payload["status"],
            "status_class": status_class,
            "plot_marker_semantics": marker,
            "target_reached": status_class == "developed_target_reached",
            "cycles": float(payload["final_cycles"]),
            "projected_extension_um": extension_um,
            "path_extension_um": extension_um,
            "event_count": len(events),
            "developed_da_dN_m_per_cycle": rate,
            "wall_seconds": float(payload.get("wall_seconds", np.nan)),
            "result_path": str(path.resolve()),
        }
        rates.append(row)
        previous_cycles = 0.0
        previous_extension = 0.0
        for event in events:
            cycles = float(event["cycles"])
            extension = float(event["cumulative_extension_m"])
            dN = cycles - previous_cycles
            da = extension - previous_extension
            event_rows.append(
                {
                    **{key: row[key] for key in ("candidate_id", "integration_mode", "seed", "normalized_f", "deltaK_MPa_sqrt_m")},
                    "event_index": int(event["event_index"]),
                    "cumulative_cycles": cycles,
                    "interval_cycles": dN,
                    "committed_advance_m": da,
                    "event_da_dN_m_per_cycle": da / dN if dN > 0.0 else np.nan,
                    "threshold_action": event.get("threshold_action"),
                    "physical_hazard_action": event.get("physical_hazard_action"),
                    "event_length_factor": event.get("event_length_factor"),
                    "backstress_Pa": event.get("backstress_Pa"),
                    "shielding_MPa_sqrt_m": event.get("shielding_MPa_sqrt_m"),
                }
            )
            previous_cycles = cycles
            previous_extension = extension
    return pd.DataFrame(rates), pd.DataFrame(event_rows)


def load_explicit(root: Path, loads: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rates, events = [], []
    for path in sorted(root.rglob("developed_fatigue_growth_summary.json")):
        payload = json.loads(path.read_text())
        provenance = payload.get("provenance", {})
        candidate_id = str(provenance.get("parameter_option") or path.parts[-3])
        deltaK = float(provenance["deltaK_MPa_sqrt_m"])
        seed = int(provenance["hazard_seed"])
        match = loads[loads.candidate_id.eq(candidate_id)].copy()
        if len(match):
            index = (match.deltaK_MPa_sqrt_m - deltaK).abs().idxmin()
            fraction = float(match.loc[index, "normalized_f"])
        else:
            fraction = float("nan")
        extension_um = float(payload.get("final_projected_extension_um", 0.0))
        status_class, marker = classify_status(payload.get("status", ""), extension_um)
        developed = payload.get("developed_interval", {}).get("da_dN")
        rate = float(developed) if developed is not None and status_class == "developed_target_reached" else float("nan")
        row = {
            "candidate_id": candidate_id,
            "integration_mode": "explicit",
            "seed": seed,
            "normalized_f": fraction,
            "deltaK_MPa_sqrt_m": deltaK,
            "status": payload.get("status"),
            "status_class": status_class,
            "plot_marker_semantics": marker,
            "target_reached": status_class == "developed_target_reached",
            "cycles": float(payload.get("cycles_consumed", np.nan)),
            "projected_extension_um": extension_um,
            "path_extension_um": float(payload.get("final_path_extension_um", np.nan)),
            "event_count": int(payload.get("event_count", 0)),
            "developed_da_dN_m_per_cycle": rate,
            "wall_seconds": float(payload.get("wall_seconds", np.nan)),
            "late_to_early_rate_ratio": payload.get("late_to_early_rate_ratio"),
            "stable_growth_provisional": payload.get("stable_growth_provisional"),
            "result_path": str(path.resolve()),
        }
        rates.append(row)
        for event in payload.get("event_measurements", []):
            events.append(
                {
                    **{key: row[key] for key in ("candidate_id", "integration_mode", "seed", "normalized_f", "deltaK_MPa_sqrt_m")},
                    "event_index": event.get("event_index"),
                    "cumulative_cycles": event.get("cycles_post"),
                    "interval_cycles": event.get("cycles_between_events"),
                    "committed_advance_m": event.get("projected_advance_m"),
                    "path_advance_m": event.get("path_advance_m"),
                    "event_da_dN_m_per_cycle": event.get("da_dN_m_per_cycle"),
                    "event_ds_dN_m_per_cycle": event.get("ds_dN_m_per_cycle"),
                    "tortuosity": event.get("tortuosity"),
                    "threshold_action": event.get("threshold_action"),
                    "physical_hazard_action": event.get("physical_hazard_action"),
                    "event_length_factor": event.get("event_length_factor"),
                    "backstress_Pa": event.get("sigma_back_Pa"),
                    "shielding_MPa_sqrt_m": event.get("K_shield_Pa_sqrt_m"),
                }
            )
    return pd.DataFrame(rates), pd.DataFrame(events)


def morphology_table(rates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    preferred = rates[
        ((rates.integration_mode.eq("explicit")) & rates.normalized_f.ge(1.0))
        | ((rates.integration_mode.eq("accelerated")) & rates.normalized_f.lt(1.0))
    ]
    for candidate_id, group in preferred.groupby("candidate_id"):
        finite = group[group.target_reached & group.developed_da_dN_m_per_cycle.notna()].sort_values("normalized_f")
        x = finite.normalized_f.to_numpy(float)
        y = np.log10(finite.developed_da_dN_m_per_cycle.to_numpy(float))
        slope = np.gradient(y, x) if len(finite) >= 2 else np.asarray([np.nan])
        knee_index = int(np.argmax(np.abs(np.diff(slope)))) + 1 if len(finite) >= 4 else None
        rows.append(
            {
                "candidate_id": candidate_id,
                "finite_developed_points": len(finite),
                "cycle_or_hazard_censors": int(group.status_class.eq("cycle_or_hazard_censor").sum()),
                "partial_or_numerical_unresolved": int(group.status_class.eq("partial_or_numerical_unresolved").sum()),
                "dynamic_rate_range_decades": float(np.ptp(y)) if len(y) >= 2 else np.nan,
                "knee_normalized_f": float(x[knee_index]) if knee_index is not None else np.nan,
                "knee_deltaK_MPa_sqrt_m": float(finite.iloc[knee_index].deltaK_MPa_sqrt_m) if knee_index is not None else np.nan,
                "low_branch_log_slope_per_f": float(np.mean(slope[: max(len(slope) // 2, 1)])) if len(slope) else np.nan,
                "upper_branch_log_slope_per_f": float(np.mean(slope[len(slope) // 2 :])) if len(slope) else np.nan,
                "maximum_developed_da_dN_m_per_cycle": float(finite.developed_da_dN_m_per_cycle.max()) if len(finite) else np.nan,
                "minimum_developed_da_dN_m_per_cycle": float(finite.developed_da_dN_m_per_cycle.min()) if len(finite) else np.nan,
                "morphology_known": len(finite) >= 4,
            }
        )
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, out: Path, stem: str, data: pd.DataFrame) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    data.to_csv(out / f"{stem}_plot_data.csv", index=False)


def plot_rates(ax, rates: pd.DataFrame, x: str) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, max(rates.candidate_id.nunique(), 1)))
    for color, (candidate_id, group) in zip(colors, rates.groupby("candidate_id")):
        resolved = group[group.status_class.eq("developed_target_reached")].sort_values(x)
        ax.plot(resolved[x], resolved.developed_da_dN_m_per_cycle, "o-", color=color, label=candidate_id, alpha=0.8)
        censor = group[group.status_class.eq("cycle_or_hazard_censor")]
        if len(censor):
            floor = resolved.developed_da_dN_m_per_cycle.min() / 5 if len(resolved) else 1e-18
            ax.scatter(censor[x], np.full(len(censor), floor), marker="v", facecolors="none", edgecolors=color)
        partial = group[group.status_class.eq("partial_or_numerical_unresolved")]
        if len(partial):
            floor = resolved.developed_da_dN_m_per_cycle.min() / 2 if len(resolved) else 2e-18
            ax.scatter(partial[x], np.full(len(partial), floor), marker="s", facecolors="none", edgecolors=color)
    ax.set_yscale("log")
    ax.legend(fontsize=6, ncol=2)


def figures(out: Path, rates: pd.DataFrame, comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.5)); plot_rates(ax, rates, "deltaK_MPa_sqrt_m")
    ax.set(xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)", ylabel=r"developed $da/dN$ (m/cycle)", title="Prospective joint fatigue response")
    savefig(fig, out, "prospective_joint_da_dN_vs_deltaK", rates)
    fig, ax = plt.subplots(figsize=(8.2, 5.5)); plot_rates(ax, rates, "normalized_f")
    ax.set(xlabel="normalized f", ylabel=r"developed $da/dN$ (m/cycle)", title="Prospective response on normalized loading")
    savefig(fig, out, "prospective_joint_da_dN_vs_f", rates)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(comparison.DBTT_magnitude_MPa_sqrt_m, comparison.dynamic_rate_range_decades, c=comparison.F1, cmap="viridis", edgecolor="black")
    ax1.set(xlabel="Fracture DBTT magnitude", ylabel="Fatigue rate range (decades)", title="Fracture and fatigue amplitude")
    ax2.scatter(comparison.peak_prominence_MPa_sqrt_m, comparison.knee_normalized_f, c=comparison.F3, cmap="coolwarm", edgecolor="black")
    ax2.set(xlabel="Fracture Peak-T prominence", ylabel="Fatigue knee f", title="Response-shape transfer")
    savefig(fig, out, "prospective_joint_fracture_and_fatigue_panels", comparison)

    q = rates[rates.normalized_f.ge(0.8)]
    fig, ax = plt.subplots(figsize=(8.2, 5.5)); plot_rates(ax, q, "normalized_f")
    ax.set(xlabel="normalized f", ylabel=r"developed $da/dN$ (m/cycle)", title="Knee-to-LCF comparison (explicit above transition)")
    savefig(fig, out, "prospective_joint_knee_LCF_comparison", q)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    loads = pd.read_csv(args.loads)
    accelerated, accelerated_events = load_accelerated(args.accelerated_root)
    explicit, explicit_events = load_explicit(args.explicit_root, loads)
    rates = pd.concat([accelerated, explicit], ignore_index=True, sort=False)
    events = pd.concat([accelerated_events, explicit_events], ignore_index=True, sort=False)
    if rates.empty:
        raise RuntimeError("no prospective fatigue results found")
    morphology = morphology_table(rates)
    fracture = pd.read_csv(args.fracture_summary)
    selection = pd.read_csv(args.selection)[["candidate_id", "selection_category", "selection_reason"]]
    comparison = fracture.merge(morphology, on="candidate_id", how="inner").merge(selection, on="candidate_id", how="inner")
    rates.to_csv(args.out / "prospective_joint_fatigue_rates.csv", index=False)
    morphology.to_csv(args.out / "prospective_joint_fatigue_morphology.csv", index=False)
    events.to_csv(args.out / "prospective_joint_fatigue_event_statistics.csv", index=False)
    comparison.to_csv(args.out / "prospective_joint_candidate_comparison.csv", index=False)
    figures(args.out, rates, comparison)
    manifest = {
        "schema": "v914_prospective_joint_fatigue_analysis_v1",
        "candidate_count": int(rates.candidate_id.nunique()),
        "rate_case_count": len(rates),
        "event_count": len(events),
        "developed_target_rates": int(rates.status_class.eq("developed_target_reached").sum()),
        "cycle_or_hazard_censors": int(rates.status_class.eq("cycle_or_hazard_censor").sum()),
        "partial_or_numerical_unresolved": int(rates.status_class.eq("partial_or_numerical_unresolved").sum()),
        "artificial_rates_for_censors": False,
        "physics_changed": False,
    }
    (args.out / "prospective_joint_fatigue_analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"V914_PROSPECTIVE_JOINT_ANALYSIS_COMPLETE candidates={manifest['candidate_count']} rates={len(rates)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
