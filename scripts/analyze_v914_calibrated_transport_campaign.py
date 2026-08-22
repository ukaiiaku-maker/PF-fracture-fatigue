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


SCHEMA = "v914_calibrated_transport_campaign_analysis_v2"
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
    status = result.get("status")
    completed_cycles = float(result.get("completed_physical_cycles", 0))
    observation_cycles = (
        completed_cycles
        if status == "cycle_horizon_reached" and completed_cycles > 0.0
        else final_event_cycle
    )
    total_rate = (
        extension / observation_cycles
        if events and observation_cycles > 0.0
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
    restart_count = int(result.get("restart_count", 0))
    if restart_count > 0:
        disposition = "unsupported_restart_result"
    elif status not in ACCEPTED_TERMINAL_STATUSES:
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
        "Kmin_MPa_sqrt_m": R * delta_k / max(1.0 - R, 1.0e-300),
        "R": R,
        "frequency_Hz": frequency,
        "temperature_K": float(contract.get("temperature_K", math.nan)),
        "seed": int(contract.get("seed", result.get("identity_contract", {}).get("seed", -1))),
        "status": status,
        "campaign_disposition": disposition,
        "censored": status == "cycle_horizon_reached",
        "accepted_terminal_result": status in ACCEPTED_TERMINAL_STATUSES and restart_count == 0,
        "restart_count": restart_count,
        "fresh_uninterrupted": restart_count == 0,
        "physical_cycles_integer_boundary": int(result.get("completed_physical_cycles", 0)),
        "observation_cycles_for_total_rate": observation_cycles,
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
        "physical_returned_mobile_per_m": float(final_reversibility.get("reversible_physical_returned_mobile_per_m", math.nan)),
        "reverse_mobile_exposure_m2_s": float(final_reversibility.get("reversible_reverse_mobile_exposure_m2_s", math.nan)),
        "reverse_plastic_displacement_m": math.nan,
        "reverse_plastic_displacement_available": False,
        "mobile_density_sum_m2": float(last_state.get("mobile_total_m2", math.nan)),
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


def event_rate_rows(result_path: Path, developed_start_m: float = 20.0e-6) -> list[dict[str, Any]]:
    """Return event-resolved projected/path rates from one audited result."""
    summary = summarize_result(result_path, developed_start_m)
    if summary["campaign_disposition"] == "unsupported_restart_result":
        return []
    result = json.loads(result_path.read_text())
    events = sorted(result.get("events", []), key=lambda row: float(row["cumulative_cycles"]))
    rows = []
    previous_cycles = 0.0
    previous_extension = 0.0
    previous_path = 0.0
    for event in events:
        cycles = float(event["cumulative_cycles"])
        extension = float(event["cumulative_extension_m"])
        projected_increment = extension - previous_extension
        path_increment = float(event.get(
            "committed_path_advance_m", event.get("committed_advance_m", projected_increment)
        ))
        path = previous_path + path_increment
        interval = cycles - previous_cycles
        rows.append({
            "result_path": summary["result_path"],
            "configuration_label": summary["configuration_label"],
            "return_control": summary["return_control"],
            "deltaK_MPa_sqrt_m": summary["deltaK_MPa_sqrt_m"],
            "Kmax_MPa_sqrt_m": summary["Kmax_MPa_sqrt_m"],
            "R": summary["R"], "frequency_Hz": summary["frequency_Hz"],
            "seed": summary["seed"], "event_index": int(event.get("event_index", len(rows))),
            "cumulative_cycles": cycles, "interval_cycles": interval,
            "cumulative_extension_m": extension, "cumulative_path_m": path,
            "projected_increment_m": projected_increment, "path_increment_m": path_increment,
            "event_da_dN_m_per_cycle": projected_increment / interval if interval > 0.0 else math.nan,
            "event_ds_dN_m_per_cycle": path_increment / interval if interval > 0.0 else math.nan,
            "tortuosity": path_increment / projected_increment if projected_increment > 0.0 else math.nan,
            "developed_event": previous_extension >= developed_start_m,
            "physical_hazard_action": float(event.get("physical_hazard_action", math.nan)),
            "threshold_action": float(event.get("threshold_action", math.nan)),
        })
        previous_cycles, previous_extension, previous_path = cycles, extension, path
    return rows


def moving_window_rows(event_rows: list[dict[str, Any]], window_m: float = 20.0e-6) -> list[dict[str, Any]]:
    """Compute trailing event-aligned windows spanning at least ``window_m``."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        groups.setdefault(str(row["result_path"]), []).append(row)
    output = []
    for values in groups.values():
        values.sort(key=lambda row: float(row["cumulative_cycles"]))
        for right_index, right in enumerate(values):
            left_index = right_index - 1
            while left_index >= 0 and (
                float(right["cumulative_extension_m"])
                - float(values[left_index]["cumulative_extension_m"]) < window_m
            ):
                left_index -= 1
            if left_index < 0:
                continue
            left = values[left_index]
            cycles = float(right["cumulative_cycles"]) - float(left["cumulative_cycles"])
            da = float(right["cumulative_extension_m"]) - float(left["cumulative_extension_m"])
            ds = float(right["cumulative_path_m"]) - float(left["cumulative_path_m"])
            if cycles <= 0.0 or da <= 0.0:
                continue
            output.append({
                **{key: right[key] for key in (
                    "result_path", "configuration_label", "return_control",
                    "deltaK_MPa_sqrt_m", "Kmax_MPa_sqrt_m", "R", "frequency_Hz", "seed",
                )},
                "left_event_index": left["event_index"], "right_event_index": right["event_index"],
                "window_cycles": cycles, "window_extension_m": da, "window_path_m": ds,
                "window_da_dN_m_per_cycle": da / cycles,
                "window_ds_dN_m_per_cycle": ds / cycles,
                "window_tortuosity": ds / da,
            })
    return output


def local_slope_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        rate = float(row["developed_da_dN_m_per_cycle"])
        if row["campaign_disposition"] == "completed_growth_target" and math.isfinite(rate) and rate > 0.0:
            key = (row["configuration_label"], row["return_control"], row["R"], row["frequency_Hz"], row["seed"])
            groups.setdefault(key, []).append(row)
    output = []
    for key, values in groups.items():
        values.sort(key=lambda row: float(row["deltaK_MPa_sqrt_m"]))
        for left, right in zip(values, values[1:]):
            k0, k1 = float(left["deltaK_MPa_sqrt_m"]), float(right["deltaK_MPa_sqrt_m"])
            r0, r1 = float(left["developed_da_dN_m_per_cycle"]), float(right["developed_da_dN_m_per_cycle"])
            output.append({
                "configuration_label": key[0], "return_control": key[1], "R": key[2],
                "frequency_Hz": key[3], "seed": key[4], "deltaK_left": k0, "deltaK_right": k1,
                "deltaK_geometric_midpoint": math.sqrt(k0 * k1),
                "local_paris_m": math.log(r1 / r0) / math.log(k1 / k0),
                "rate_left_m_per_cycle": r0, "rate_right_m_per_cycle": r1,
            })
    return output


def trend_rows(rows: list[dict[str, Any]], variable: str) -> list[dict[str, Any]]:
    """Return adjacent R ratios or frequency power exponents at matched loads."""
    if variable not in {"R", "frequency_Hz"}:
        raise ValueError(variable)
    fixed = "frequency_Hz" if variable == "R" else "R"
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        rate = float(row["developed_da_dN_m_per_cycle"])
        if row["campaign_disposition"] != "completed_growth_target" or not math.isfinite(rate) or rate <= 0.0:
            continue
        load = float(row["Kmax_MPa_sqrt_m"] if variable == "R" else row["deltaK_MPa_sqrt_m"])
        # R curves were launched from rounded DeltaK values chosen on a common
        # Kmax grid.  Milliscale Kmax grouping preserves that intended match
        # without merging distinct adaptive points. Frequency cases use the
        # exact same DeltaK and retain the tighter rounding.
        load_digits = 3 if variable == "R" else 6
        key = (
            row["return_control"], row["seed"], float(row[fixed]),
            round(load, load_digits),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in groups.items():
        values.sort(key=lambda row: float(row[variable]))
        for left, right in zip(values, values[1:]):
            x0, x1 = float(left[variable]), float(right[variable])
            rate0 = float(left["developed_da_dN_m_per_cycle"])
            rate1 = float(right["developed_da_dN_m_per_cycle"])
            record = {
                "return_control": key[0], "seed": key[1], fixed: key[2],
                "matched_load_MPa_sqrt_m": key[3], f"{variable}_left": x0,
                f"{variable}_right": x1, "rate_left_m_per_cycle": rate0,
                "rate_right_m_per_cycle": rate1, "rate_ratio_right_over_left": rate1 / rate0,
            }
            if variable == "frequency_Hz" and x0 > 0.0 and x1 > 0.0:
                record["frequency_exponent_per_cycle"] = math.log(rate1 / rate0) / math.log(x1 / x0)
            output.append(record)
    return output


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
        selected = None
        for length in range(len(values), 2, -1):
            for start in range(len(values) - length + 1):
                candidate = values[start:start + length]
                x = np.log10([float(row["deltaK_MPa_sqrt_m"]) for row in candidate])
                y = np.log10([float(row["developed_da_dN_m_per_cycle"]) for row in candidate])
                if not np.all(np.diff(y) > 0.0):
                    continue
                slope, intercept = np.polyfit(x, y, 1)
                fitted = slope * x + intercept
                ss_res = float(np.sum((y - fitted) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
                if r_squared >= 0.98:
                    selected = (candidate, x, slope, intercept, r_squared)
                    break
            if selected is not None:
                break
        if selected is None:
            continue
        candidate, x, slope, intercept, r_squared = selected
        fits.append({
            "configuration_label": key[0], "return_control": key[1],
            "R": key[2], "frequency_Hz": key[3], "seed": key[4],
            "point_count": len(candidate), "deltaK_min": float(10 ** x[0]),
            "deltaK_max": float(10 ** x[-1]), "paris_m": float(slope),
            "log10_intercept": float(intercept),
            "R_squared": r_squared,
            "selection_rule": "longest_contiguous_monotone_segment_with_R_squared_at_least_0.98",
            "fit_is_output_only": True,
        })
    return fits


def knee_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identify the largest adjacent log-rate transition without fitting a law."""
    slopes = local_slope_rows(rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in slopes:
        key = (row["configuration_label"], row["return_control"], row["R"], row["frequency_Hz"], row["seed"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in groups.items():
        transition = max(values, key=lambda row: abs(math.log10(row["rate_right_m_per_cycle"] / row["rate_left_m_per_cycle"])))
        log_jump = math.log10(transition["rate_right_m_per_cycle"] / transition["rate_left_m_per_cycle"])
        output.append({
            "configuration_label": key[0], "return_control": key[1], "R": key[2],
            "frequency_Hz": key[3], "seed": key[4],
            "deltaK_left": transition["deltaK_left"], "deltaK_right": transition["deltaK_right"],
            "deltaK_knee_geometric_midpoint": transition["deltaK_geometric_midpoint"],
            "rate_left_m_per_cycle": transition["rate_left_m_per_cycle"],
            "rate_right_m_per_cycle": transition["rate_right_m_per_cycle"],
            "log10_rate_jump": log_jump, "local_slope": transition["local_paris_m"],
            "classification": "sharp_kinetic_knee" if abs(transition["local_paris_m"]) >= 50.0 else "graded_transition",
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
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


def write_trend_plots(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    accepted = [
        row for row in rows
        if row["campaign_disposition"] == "completed_growth_target"
        and math.isfinite(float(row["developed_da_dN_m_per_cycle"]))
        and float(row["developed_da_dN_m_per_cycle"]) > 0.0
    ]
    r_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    f_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in accepted:
        r_groups.setdefault((row["return_control"], row["frequency_Hz"], row["seed"]), []).append(row)
        f_groups.setdefault((row["return_control"], row["R"], row["seed"]), []).append(row)
    for key, values in sorted(r_groups.items(), key=lambda item: str(item[0])):
        values.sort(key=lambda row: (float(row["R"]), float(row["Kmax_MPa_sqrt_m"])))
        axes[0].semilogy(
            [float(row["R"]) for row in values],
            [float(row["developed_da_dN_m_per_cycle"]) for row in values],
            "o-", label=f"{key[0]}, {key[1]:g} Hz",
        )
    for key, values in sorted(f_groups.items(), key=lambda item: str(item[0])):
        values.sort(key=lambda row: (float(row["frequency_Hz"]), float(row["deltaK_MPa_sqrt_m"])))
        axes[1].loglog(
            [float(row["frequency_Hz"]) for row in values],
            [float(row["developed_da_dN_m_per_cycle"]) for row in values],
            "o-", label=f"{key[0]}, R={key[1]:g}",
        )
    axes[0].set(xlabel="load ratio R", ylabel=r"developed $da/dN$ (m/cycle)")
    axes[1].set(xlabel="frequency (Hz)", ylabel=r"developed $da/dN$ (m/cycle)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_required_figures(
    rows: list[dict[str, Any]], local_slopes: list[dict[str, Any]],
    knees: list[dict[str, Any]], out: Path,
) -> None:
    """Write the six campaign figures named by the scientific contract."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [row for row in rows if row["campaign_disposition"] != "unsupported_restart_result"]
    completed = [
        row for row in usable
        if row["campaign_disposition"] == "completed_growth_target"
        and math.isfinite(float(row["developed_da_dN_m_per_cycle"]))
        and float(row["developed_da_dN_m_per_cycle"]) > 0.0
    ]
    censored = [row for row in usable if row["campaign_disposition"] == "physical_cycle_censor"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    positive = sorted((row for row in completed if math.isclose(float(row["R"]), 0.1)), key=lambda row: float(row["deltaK_MPa_sqrt_m"]))
    if positive:
        axes[0].loglog([row["deltaK_MPa_sqrt_m"] for row in positive], [row["developed_da_dN_m_per_cycle"] for row in positive], "o-")
    positive_censors = [row for row in censored if math.isclose(float(row["R"]), 0.1)]
    if positive_censors:
        axes[0].loglog([row["deltaK_MPa_sqrt_m"] for row in positive_censors], [row["total_da_dN_m_per_cycle"] for row in positive_censors], "v", markerfacecolor="none", label="1e12-cycle censor")
    positive_slopes = sorted((row for row in local_slopes if math.isclose(float(row["R"]), 0.1)), key=lambda row: float(row["deltaK_geometric_midpoint"]))
    if positive_slopes:
        axes[1].plot([row["deltaK_geometric_midpoint"] for row in positive_slopes], [row["local_paris_m"] for row in positive_slopes], "o-")
    axes[0].set(xlabel=r"$\Delta K$ (MPa $\sqrt{m}$)", ylabel=r"developed $da/dN$ (m/cycle)")
    axes[1].set(xlabel=r"$\Delta K$ midpoint (MPa $\sqrt{m}$)", ylabel="adjacent local slope")
    for axis in axes: axis.grid(True, which="both", alpha=0.25)
    fig.savefig(out / "figure_1_R0p1_paris_and_local_slope.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for R in sorted({float(row["R"]) for row in usable}):
        values = sorted((row for row in completed if math.isclose(float(row["R"]), R)), key=lambda row: float(row["deltaK_MPa_sqrt_m"]))
        if values:
            axes[0].loglog([row["deltaK_MPa_sqrt_m"] for row in values], [row["developed_da_dN_m_per_cycle"] for row in values], "o-", label=f"R={R:g}")
            axes[1].semilogy([row["Kmax_MPa_sqrt_m"] for row in values], [row["developed_da_dN_m_per_cycle"] for row in values], "o-", label=f"R={R:g}")
        marks = [row for row in censored if math.isclose(float(row["R"]), R)]
        if marks:
            axes[0].loglog([row["deltaK_MPa_sqrt_m"] for row in marks], [row["total_da_dN_m_per_cycle"] for row in marks], "v", markerfacecolor="none")
            axes[1].semilogy([row["Kmax_MPa_sqrt_m"] for row in marks], [row["total_da_dN_m_per_cycle"] for row in marks], "v", markerfacecolor="none")
    axes[0].set(xlabel=r"$\Delta K$ (MPa $\sqrt{m}$)", ylabel=r"developed $da/dN$ (m/cycle)")
    axes[1].set(xlabel=r"$K_{max}$ (MPa $\sqrt{m}$)", ylabel=r"developed $da/dN$ (m/cycle)")
    for axis in axes: axis.grid(True, which="both", alpha=0.25)
    if axes[0].lines: axes[0].legend()
    fig.savefig(out / "figure_2_R_dependence_paris_curves.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    if knees:
        ordered = sorted(knees, key=lambda row: float(row["R"]))
        R_values = [float(row["R"]) for row in ordered]
        delta_k = [float(row["deltaK_knee_geometric_midpoint"]) for row in ordered]
        axes[0].plot(R_values, delta_k, "o-")
        axes[1].plot(R_values, [value / (1.0 - R) for value, R in zip(delta_k, R_values)], "o-")
    axes[0].set(xlabel="R", ylabel=r"$\Delta K_{knee}$ (MPa $\sqrt{m}$)")
    axes[1].set(xlabel="R", ylabel=r"$K_{max,knee}$ (MPa $\sqrt{m}$)")
    for axis in axes: axis.grid(True, alpha=0.25)
    fig.savefig(out / "figure_3_knee_shift_with_R.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2), constrained_layout=True)
    for R in sorted({float(row["R"]) for row in usable}):
        values = sorted((row for row in usable if math.isclose(float(row["R"]), R)), key=lambda row: float(row["Kmax_MPa_sqrt_m"]))
        if not values: continue
        x = [row["Kmax_MPa_sqrt_m"] for row in values]
        axes[0].plot(x, [row["physical_return_fraction"] for row in values], "o-", label=f"R={R:g}")
        axes[1].plot(x, [row["tip_radius_m"] for row in values], "o-")
        axes[2].plot(x, [row["Kshield_MPa_sqrt_m"] for row in values], "o-")
        axes[3].plot(x, [row["net_source_blunting_count"] for row in values], "o-")
    axes[0].set(ylabel="physical return fraction"); axes[1].set(ylabel="tip radius (m)"); axes[2].set(ylabel=r"$K_{shield}$ (MPa $\sqrt{m}$)"); axes[3].set(ylabel="net source blunting count")
    for axis in axes: axis.set_xlabel(r"$K_{max}$ (MPa $\sqrt{m}$)"); axis.grid(True, alpha=0.25)
    if axes[0].lines: axes[0].legend()
    fig.savefig(out / "figure_4_return_blunting_shielding.png", dpi=180); plt.close(fig)

    for field, name, ylabel in (
        ("developed_da_dN_m_per_cycle", "figure_5_frequency_da_dN.png", r"developed $da/dN$ (m/cycle)"),
        ("developed_da_dt_m_per_s", "figure_6_frequency_da_dt.png", r"developed $da/dt$ (m/s)"),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
        groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for row in completed: groups.setdefault((float(row["R"]), round(float(row["deltaK_MPa_sqrt_m"]), 6)), []).append(row)
        for key, values in sorted(groups.items()):
            values.sort(key=lambda row: float(row["frequency_Hz"]))
            if len(values) > 1: ax.loglog([row["frequency_Hz"] for row in values], [row[field] for row in values], "o-", label=f"R={key[0]:g}, dK={key[1]:g}")
        ax.set(xlabel="frequency (Hz)", ylabel=ylabel); ax.grid(True, which="both", alpha=0.25)
        if ax.lines: ax.legend(fontsize=7)
        fig.savefig(out / name, dpi=180); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--developed-start-um", type=float, default=20.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted({path for root in args.roots for path in root.rglob("result.json")})
    rows = [summarize_result(path, args.developed_start_um * 1.0e-6) for path in paths]
    fits = fit_groups(rows)
    events = [
        row for path in paths
        for row in event_rate_rows(path, args.developed_start_um * 1.0e-6)
    ]
    windows = moving_window_rows(events)
    local_slopes = local_slope_rows(rows)
    knees = knee_rows(rows)
    r_trends = trend_rows(rows, "R")
    frequency_trends = trend_rows(rows, "frequency_Hz")
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "campaign_summary.csv", rows)
    _write_csv(args.out / "paris_points.csv", rows)
    _write_csv(args.out / "paris_fits.csv", fits)
    _write_csv(args.out / "event_rates.csv", events)
    _write_csv(args.out / "paris_event_rates.csv", events)
    _write_csv(args.out / "moving_window_rates.csv", windows)
    _write_csv(args.out / "local_paris_slopes.csv", local_slopes)
    _write_csv(args.out / "local_slopes.csv", local_slopes)
    _write_csv(args.out / "kinetic_knees.csv", knees)
    _write_csv(args.out / "R_trends.csv", r_trends)
    _write_csv(args.out / "R_dependence.csv", r_trends)
    _write_csv(args.out / "frequency_trends.csv", frequency_trends)
    _write_csv(args.out / "frequency_dependence.csv", frequency_trends)
    write_plots(rows, args.out / "paris_response.png")
    write_trend_plots(rows, args.out / "R_frequency_trends.png")
    write_required_figures(rows, local_slopes, knees, args.out)
    analysis_payload = json.dumps({
        "schema": SCHEMA,
        "developed_start_um": args.developed_start_um,
        "rows": rows,
        "fits": fits,
        "event_rates": events,
        "moving_window_rates": windows,
        "local_slopes": local_slopes,
        "kinetic_knees": knees,
        "R_trends": r_trends,
        "frequency_trends": frequency_trends,
    }, indent=2, sort_keys=True, allow_nan=True) + "\n"
    report = (
        "# Calibrated transport fatigue campaign\n\n"
        f"- Audited result files: {len(rows)}\n"
        f"- Fresh completed growth trajectories: {sum(row['campaign_disposition'] == 'completed_growth_target' for row in rows)}\n"
        f"- Physical cycle censors: {sum(row['campaign_disposition'] == 'physical_cycle_censor' for row in rows)}\n"
        f"- Excluded unsupported restart results: {sum(row['campaign_disposition'] == 'unsupported_restart_result' for row in rows)}\n"
        f"- Event-rate rows: {len(events)}\n"
        f"- Moving-window rows: {len(windows)}\n"
        f"- Global output-only Paris fits: {len(fits)}\n"
        f"- Adjacent local Paris slopes: {len(local_slopes)}\n"
        f"- Kinetic-knee summaries: {len(knees)}\n"
        f"- Matched R comparisons: {len(r_trends)}\n"
        f"- Matched frequency comparisons: {len(frequency_trends)}\n"
    )
    for name in ("campaign_analysis.json", "final_summary.json"):
        (args.out / name).write_text(analysis_payload)
    for name in ("CAMPAIGN_REPORT.md", "final_report.md"):
        (args.out / name).write_text(report)
    print(f"summarized {len(rows)} result files; {len(fits)} output fits; {len(events)} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
