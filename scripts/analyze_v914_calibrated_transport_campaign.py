#!/usr/bin/env python3
"""Summarize calibrated v7 REV/IRREV event-to-event campaign outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "v914_calibrated_transport_campaign_analysis_v1"
ACCEPTED_TERMINAL_STATUSES = {"growth_target_reached", "cycle_horizon_reached"}


def _cycle_at_extension(events: list[dict[str, Any]], extension_m: float) -> float | None:
    if not events or float(events[-1]["cumulative_extension_m"]) < extension_m:
        return None
    left_cycle = 0.0
    left_extension = 0.0
    for event in events:
        right_cycle = float(event["cumulative_cycles"])
        right_extension = float(event["cumulative_extension_m"])
        if right_extension >= extension_m:
            span = right_extension - left_extension
            if span <= 0.0:
                return right_cycle
            fraction = (extension_m - left_extension) / span
            return left_cycle + fraction * (right_cycle - left_cycle)
        left_cycle = right_cycle
        left_extension = right_extension
    return None


def summarize_result(result_path: Path, developed_start_m: float = 20.0e-6) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    contract_path = result_path.with_name("run_contract.json")
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
    events = sorted(result.get("events", []), key=lambda row: float(row["cumulative_cycles"]))
    final_event_cycle = float(events[-1]["cumulative_cycles"]) if events else math.nan
    extension = float(result.get("cumulative_extension_m", 0.0))
    total_rate = (
        extension / final_event_cycle
        if events and final_event_cycle > 0.0
        else math.nan
    )
    start_cycle = _cycle_at_extension(events, developed_start_m)
    developed_rate = math.nan
    developed_span = 0.0
    if start_cycle is not None and events and final_event_cycle > start_cycle:
        developed_span = max(extension - developed_start_m, 0.0)
        developed_rate = developed_span / (final_event_cycle - start_cycle)

    final_reversibility = result.get("final_reversibility", {})
    final_gross = result.get("final_gross_activity", {})
    last_state = events[-1] if events else (
        result.get("anchor_history", [{}])[-1] if result.get("anchor_history") else {}
    )
    total_hazard = sum(float(row.get("physical_hazard_action", 0.0)) for row in events)
    total_hazard += float(result.get("hazard_clock_action", 0.0))
    frequency = float(contract.get("frequency_Hz", result.get("identity_contract", {}).get("loading", {}).get("frequency_Hz", math.nan)))
    delta_k = float(contract.get("deltaK_MPa_sqrt_m", result.get("identity_contract", {}).get("loading", {}).get("deltaK_MPa_sqrt_m", math.nan)))
    R = float(contract.get("R", result.get("identity_contract", {}).get("loading", {}).get("R", math.nan)))
    status = result.get("status")
    if status not in ACCEPTED_TERMINAL_STATUSES:
        disposition = "incomplete_or_rejected"
    elif status == "cycle_horizon_reached":
        disposition = "physical_cycle_censor"
    else:
        disposition = "completed_growth_target"
    return {
        "result_path": str(result_path.resolve()),
        "configuration_label": contract.get("configuration_label"),
        "return_control": contract.get("return_control"),
        "state_model_id": contract.get("state_model_id"),
        "candidate": contract.get("candidate", result.get("identity_contract", {}).get("candidate_id")),
        "deltaK_MPa_sqrt_m": delta_k,
        "Kmax_MPa_sqrt_m": delta_k / max(1.0 - R, 1.0e-300),
        "R": R,
        "frequency_Hz": frequency,
        "temperature_K": float(contract.get("temperature_K", math.nan)),
        "seed": int(contract.get("seed", result.get("identity_contract", {}).get("seed", -1))),
        "status": status,
        "campaign_disposition": disposition,
        "censored": status == "cycle_horizon_reached",
        "accepted_terminal_result": status in ACCEPTED_TERMINAL_STATUSES,
        "physical_cycles_integer_boundary": int(result.get("completed_physical_cycles", 0)),
        "final_event_cycle": final_event_cycle,
        "event_count": len(events),
        "extension_m": extension,
        "total_da_dN_m_per_cycle": total_rate,
        "developed_start_m": developed_start_m,
        "developed_span_m": developed_span,
        "developed_da_dN_m_per_cycle": developed_rate,
        "developed_da_dt_m_per_s": developed_rate * frequency,
        "cumulative_cleavage_action": total_hazard,
        "tip_radius_m": float(last_state.get("tip_radius_m", math.nan)),
        "net_source_blunting_count": float(final_reversibility.get("reversible_net_source_slip_count", math.nan)),
        "returned_source_slip_count": float(final_reversibility.get("reversible_returned_source_slip_count", math.nan)),
        "physical_return_fraction": float(final_reversibility.get("reversible_physical_return_fraction_of_emitted", math.nan)),
        "retained_density_sum_m2": float(
            last_state.get("retained_total_m2", math.nan)
        ),
        "retained_line_content": float(
            last_state.get("retained_line_content", math.nan)
        ),
        "Kshield_MPa_sqrt_m": float(last_state.get("shielding_MPa_sqrt_m", math.nan)),
        "total_cycle_map_evaluations": int(result.get("total_cycle_map_evaluations", 0)),
        "accepted_block_count": int(result.get("accepted_block_count", 0)),
        "maximum_accepted_stride": int(result.get("maximum_accepted_stride", 0)),
        "dmd_map_refresh_count": int(result.get("dmd_map_refresh_count", 0)),
        "dmd_map_reuse_count": int(result.get("dmd_map_reuse_count", 0)),
        "dmd_map_invalidation_count": int(result.get("dmd_map_invalidation_count", 0)),
        "gross_source_slip_line_content": float(final_gross.get("gross_source_slip_line_content", math.nan)),
        "gross_returned_slip_line_content": float(final_gross.get("gross_returned_slip_line_content", math.nan)),
    }


def fit_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        rate = float(row["developed_da_dN_m_per_cycle"])
        if (
            row["campaign_disposition"] != "completed_growth_target"
            or not math.isfinite(rate)
            or rate <= 0.0
        ):
            continue
        key = (
            row["configuration_label"], row["return_control"],
            row["R"], row["frequency_Hz"], row["seed"],
        )
        groups.setdefault(key, []).append(row)
    fits = []
    for key, values in groups.items():
        values.sort(key=lambda row: float(row["deltaK_MPa_sqrt_m"]))
        if len(values) < 3:
            continue
        x = np.log10([float(row["deltaK_MPa_sqrt_m"]) for row in values])
        y = np.log10([float(row["developed_da_dN_m_per_cycle"]) for row in values])
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        fits.append({
            "configuration_label": key[0], "return_control": key[1],
            "R": key[2], "frequency_Hz": key[3], "seed": key[4],
            "point_count": len(values), "deltaK_min": float(10 ** x[0]),
            "deltaK_max": float(10 ** x[-1]), "paris_m": float(slope),
            "log10_intercept": float(intercept),
            "R_squared": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
            "fit_is_output_only": True,
        })
    return fits


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot computed rates without treating censored/incomplete points as fits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["configuration_label"], row["return_control"],
            row["R"], row["frequency_Hz"], row["seed"],
        )
        groups.setdefault(key, []).append(row)
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        accepted = [
            row for row in values
            if row["campaign_disposition"] == "completed_growth_target"
            and math.isfinite(float(row["developed_da_dN_m_per_cycle"]))
            and float(row["developed_da_dN_m_per_cycle"]) > 0.0
        ]
        accepted.sort(key=lambda row: float(row["deltaK_MPa_sqrt_m"]))
        label = f"{key[0]}/{key[1]}, R={key[2]:g}, f={key[3]:g} Hz"
        if accepted:
            delta_k = [float(row["deltaK_MPa_sqrt_m"]) for row in accepted]
            da_dn = [float(row["developed_da_dN_m_per_cycle"]) for row in accepted]
            axes[0].loglog(delta_k, da_dn, "o-", label=label)
            axes[1].loglog(
                delta_k,
                [float(row["developed_da_dt_m_per_s"]) for row in accepted],
                "o-", label=label,
            )
        censored = [
            row for row in values
            if row["campaign_disposition"] == "physical_cycle_censor"
            and math.isfinite(float(row["total_da_dN_m_per_cycle"]))
            and float(row["total_da_dN_m_per_cycle"]) > 0.0
        ]
        if censored:
            x = [float(row["deltaK_MPa_sqrt_m"]) for row in censored]
            y = [float(row["total_da_dN_m_per_cycle"]) for row in censored]
            axes[0].loglog(x, y, "v", markerfacecolor="none")
            axes[1].loglog(
                x,
                [rate * float(row["frequency_Hz"]) for rate, row in zip(y, censored)],
                "v", markerfacecolor="none",
            )
    axes[0].set(xlabel=r"$\Delta K$ (MPa $\sqrt{m}$)", ylabel=r"developed $da/dN$ (m/cycle)")
    axes[1].set(xlabel=r"$\Delta K$ (MPa $\sqrt{m}$)", ylabel=r"developed $da/dt$ (m/s)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    if groups:
        axes[0].legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--developed-start-um", type=float, default=20.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted({path for root in args.roots for path in root.rglob("result.json")})
    rows = [summarize_result(path, args.developed_start_um * 1.0e-6) for path in paths]
    fits = fit_groups(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "campaign_summary.csv", rows)
    _write_csv(args.out / "paris_fits.csv", fits)
    write_plots(rows, args.out / "paris_response.png")
    (args.out / "campaign_analysis.json").write_text(json.dumps({
        "schema": SCHEMA,
        "developed_start_um": args.developed_start_um,
        "rows": rows,
        "fits": fits,
    }, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(f"summarized {len(rows)} complete result files; {len(fits)} output fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
