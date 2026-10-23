#!/usr/bin/env python3
"""Assemble the executed QSTE340TM active-calibration evidence and figures."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from arrhenius_fracture.paris_slope_r9 import _NativePhaseController
from arrhenius_fracture.persistent_site_poincare_v10230 import one_cycle_map
from arrhenius_fracture.qste340tm_restored_binding import (
    OpeningCoordinates,
    PlasticCoordinates,
    TEMPERATURE_K,
    build_engine,
    install_opening,
    load_parent_binding,
    material_binding,
    opening_surface,
)
from scripts.complete_qste340tm_active_calibration import (
    CANDIDATES,
    CYCLE_SCHEDULE_POLICY,
    NATIVE_CYCLE_TOLERANCE,
    OUT,
    SCHEMA,
    _checkpoint_dir,
    _restore_fixed_window_checkpoint,
    bind_exact_observation_states,
    prepare_observation_schedule,
)
from scripts.refit_qste340tm_restored_native import (
    ALT_REGISTRY,
    DATA,
    EVENT_LENGTH_M,
    FREQUENCY_HZ,
    PARENT_CANDIDATE,
    PARENT_REGISTRY,
    R_VALUES,
    _stress_trace,
    atomic_json,
    build_records,
    fixed_kmax,
    metrics,
    predict,
)

FIG = OUT / "figures"
TABLE = OUT / "tables"
HISTORICAL = (
    DATA / "model_conditional/stateful_exp_floor_capacity_policy/restored_native_refit/"
    "candidate_cache/sustained_active_f8f63fdfca0e.json"
)
PRIMARY = OUT / "completed_candidates/sustained_active_23b19aa6590c.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def calibration_exclusion_reason(data: dict[str, Any],
                                 records: list[dict[str, Any]]) -> str | None:
    """Gate primary selection on the corrected, complete observation-state path.

    Legacy sparse fits may remain useful labelled comparisons.  Their smaller
    error cannot make them eligible for the corrected calibration frontier.
    This gate checks source/time ownership and the states used for predictions;
    it does not certify native load/phase convergence or parameter optimality.
    """
    if data.get("status") != "complete":
        return "not_complete"
    if data.get("schema") != SCHEMA or data.get("calibration_eligible") is not True:
        return "legacy_or_diagnostic_result_not_calibration_eligible"
    payload = data.get("identity", {}).get("payload", {})
    if (payload.get("schema") != SCHEMA
            or payload.get("calibration_eligible") is not True
            or payload.get("observation_state_binding") != "exact_R_source_order_and_verified_native_cycle"
            or payload.get("observation_cycle_schedule_policy") != CYCLE_SCHEDULE_POLICY):
        return "incompatible_observation_state_identity"
    if payload.get("state_only_acceleration", False) or data.get("state_only_acceleration", False):
        return "state_only_observer_not_long_path_qualified"
    try:
        if int(payload.get("support_points_per_R", 1)) > 0:
            return "sparse_support_identity"
        if int(payload.get("phase_count", 0)) <= 0:
            return "invalid_phase_count"
        if payload.get("alignment") != "index_rank":
            return "analyzer_requires_index_rank_observation_alignment"
        if not math.isclose(float(payload.get("frequency_Hz", float("nan"))),
                            FREQUENCY_HZ, rel_tol=0.0, abs_tol=1.0e-12):
            return "analyzer_frequency_mismatch"
        scheduled = prepare_observation_schedule(records)
        if data.get("observation_count") != len(scheduled):
            return "observation_count_mismatch"
        supports = data.get("support_states", [])
        states = data.get("states", [])
        if data.get("genuine_evolved_support_count") != len(scheduled):
            return "genuine_support_count_mismatch"
        exact = bind_exact_observation_states(scheduled, supports)
        bound_states = bind_exact_observation_states(scheduled, states)
        source_keys = [(float(r["R"]), int(r["source_order"])) for r in scheduled]
        if [(float(s["R"]), int(s["source_order"])) for s in states] != source_keys:
            return "prediction_state_order_mismatch"
        for observation, support, state in zip(scheduled, exact, bound_states):
            if state["support_state_id"] != support["support_state_id"]:
                return "prediction_state_support_identity_mismatch"
            for field in ("tip_radius_m", "K_shield_Pa_sqrt_m"):
                value = float(state[field])
                if not math.isfinite(value) or value != float(support[field]):
                    return "prediction_state_does_not_match_evolved_support"
            if float(state["tip_radius_m"]) <= 0.0:
                return "nonpositive_native_radius"
            # Check the stored observation ownership, not values overwritten by
            # bind_exact_observation_states while building an ordered copy.
        for observation, state in zip(scheduled, states):
            if (state.get("observation_source_order") != observation["source_order"]
                    or state.get("observation_id") != observation["series_id"]):
                return "stored_observation_identity_mismatch"
            for field, expected in (("observation_cycles", observation["cycles"]),
                                    ("mapped_cycles", observation["mapped_cycles"])):
                if not math.isclose(float(state[field]), float(expected), rel_tol=0.0,
                                    abs_tol=NATIVE_CYCLE_TOLERANCE):
                    return "stored_observation_time_mismatch"
        for mode in ("absolute", "shape"):
            fit = data["fits"][mode]
            for field in ("prediction", "frozen_prediction"):
                values = np.asarray(fit[field], dtype=float)
                if values.shape != (len(scheduled),) or not np.all(np.isfinite(values) & (values > 0.0)):
                    return "invalid_prediction_vector"
            for metric in ("overall_rmse_log10_rate", "shape_rmse_log10"):
                if not math.isfinite(float(fit["metrics"][metric])):
                    return "nonfinite_fit_metric"
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        return f"invalid_observation_state_record: {error}"
    return None


def candidate_records(records: list[dict[str, Any]] | None = None,
                      *, excluded: list[dict[str, Any]] | None = None,
                      ) -> list[tuple[Path, dict[str, Any]]]:
    records = (prepare_observation_schedule(build_records("index_rank", premeasurement=True))
               if records is None else prepare_observation_schedule(records))
    paths = [HISTORICAL] if HISTORICAL.is_file() else []
    paths.extend(sorted((OUT / "completed_candidates").glob("*.json")))
    seen: set[str] = set()
    output = []
    for path in paths:
        data = load(path)
        identity = data.get("identity_sha256", data.get("identity", {}).get("sha256", str(path)))
        if identity in seen:
            continue
        seen.add(identity)
        reason = calibration_exclusion_reason(data, records)
        if reason is not None:
            if excluded is not None:
                excluded.append({
                    "path": str(path.resolve()), "sha256": digest(path),
                    "candidate_id": data.get("candidate_id"), "identity": identity,
                    "schema": data.get("schema"), "status": data.get("status"),
                    "role": "excluded_comparator_only", "calibration_eligible": False,
                    "exclusion_reason": reason,
                    "support_count": len(data.get("support_states", data.get("support_rows", []))),
                    "absolute_metrics": data.get("fits", {}).get("absolute", {}).get("metrics"),
                    "shape_metrics": data.get("fits", {}).get("shape", {}).get("metrics"),
                })
            continue
        output.append((path, data))
    return output


def constant_controls(records: list[dict[str, Any]]) -> dict[str, Any]:
    observed_log = np.log10([row["observed_dadn_m_per_cycle"] for row in records])
    r = np.asarray([row["R"] for row in records])
    pooled = np.full(len(records), 10.0 ** float(np.mean(observed_log)))
    block_mean = float(np.mean([
        np.mean(observed_log[r == r_value]) for r_value in R_VALUES
    ]))
    balanced = np.full(len(records), 10.0 ** block_mean)
    return {
        "pooled_optimal_common_constant": {
            "rate_m_per_cycle": float(pooled[0]), "metrics": metrics(records, pooled),
        },
        "equal_history_optimal_common_constant": {
            "rate_m_per_cycle": float(balanced[0]), "metrics": metrics(records, balanced),
        },
    }


def acceleration_profile(candidate: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    failures: dict[str, int] = {}
    accepted_cycles: dict[str, float] = {}
    drift_errors: list[float] = []
    hazard_errors: list[float] = []
    for process_row in candidate.get("process_rows", []):
        for mode in process_row.get("process", {}).get("modes", []):
            name = str(mode.get("mode", "unknown"))
            counts[name] = counts.get(name, 0)+1
            accepted_cycles[name] = accepted_cycles.get(name, 0.0)+float(mode.get("cycles", 0.0))
            failure = mode.get("failure_reason") or mode.get("projective_failure")
            if failure:
                failures[str(failure)] = failures.get(str(failure), 0)+1
            if "drift_error" in mode:
                drift_errors.append(float(mode["drift_error"]))
            if "hazard_error" in mode:
                hazard_errors.append(float(mode["hazard_error"]))
    return {
        "mode_counts": counts,
        "accepted_cycles_by_mode": accepted_cycles,
        "failure_counts": failures,
        "maximum_reported_drift_error": max(drift_errors, default=0.0),
        "maximum_reported_hazard_error": max(hazard_errors, default=0.0),
    }


def support_representation_comparison(historical: dict[str, Any],
                                      primary: dict[str, Any]) -> dict[str, Any]:
    fields = ("mobile_count", "retained_count", "accumulated_slip_count",
              "returned_slip_count", "tip_radius_m", "K_shield_Pa_sqrt_m")
    rows = []
    for r_value in R_VALUES:
        old = [row for row in historical["support_rows"] if row["R"] == r_value][-1]
        new = [row for row in primary["support_states"] if row["R"] == r_value][-1]
        row: dict[str, Any] = {"R": r_value}
        for field in fields:
            row[f"historical_{field}"] = old[field]
            row[f"chronological_{field}"] = new[field]
            scale = max(abs(float(old[field])), abs(float(new[field])), 1.0e-30)
            row[f"scaled_difference_{field}"] = (float(new[field])-float(old[field]))/scale
        rows.append(row)
    return {
        "scope": (
            "combined history-forcing/support-density difference; not a pure support-convergence estimate"
        ),
        "historical_supports_per_R": 2,
        "chronological_supports_per_R": 8,
        "rows": rows,
    }


def state_only_longitudinal_comparison(primary: dict[str, Any]) -> dict[str, Any]:
    path = OUT/"restarts/sustained_active_b97ec2d16143/outer_progress.json"
    if not path.is_file():
        return {"available": False, "long_projective_propagation_qualified": False}
    diagnostic = load(path)
    fields = ("mobile_count", "retained_count", "accumulated_slip_count",
              "returned_slip_count", "tip_radius_m", "K_shield_Pa_sqrt_m",
              "gross_source_activity")
    rows = []
    maxima = {field: 0.0 for field in fields}
    primary_supports = primary["support_states"]
    for diagnostic_row in diagnostic["completed_observation_rows"]:
        matches = [row for row in primary_supports
                   if row["R"] == diagnostic_row["R"]
                   and row["cycles"] == diagnostic_row["cycles"]]
        if not matches:
            continue
        production_row = matches[0]
        row: dict[str, Any] = {"R": diagnostic_row["R"], "cycles": diagnostic_row["cycles"]}
        for field in fields:
            production = float(production_row[field])
            accelerated = float(diagnostic_row[field])
            scale = max(abs(production), abs(accelerated), 1.0e-30)
            difference = (accelerated-production)/scale
            maxima[field] = max(maxima[field], abs(difference))
            row[f"production_{field}"] = production
            row[f"state_only_{field}"] = accelerated
            row[f"scaled_difference_{field}"] = difference
        rows.append(row)
    return {
        "available": True,
        "scope": "same sustained-active chronological history through available common supports",
        "comparison_rows": rows,
        "maximum_absolute_scaled_difference_by_field": maxima,
        "long_projective_propagation_qualified": False,
        "decision": (
            "state-only observer remains diagnostic-only; production opening observer is required "
            "because DMD hazard validation changes the accepted projected state path"
        ),
    }


def restart_inventory() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((OUT/"restarts").glob("*/outer_progress.json")):
        progress = load(path)
        payload = progress["identity"]["payload"]
        command = (
            "python scripts/complete_qste340tm_active_calibration.py "
            f"--candidate {payload['candidate_id']} "
            f"--support-points {payload.get('support_points_per_R', 0)} "
            f"--load-substeps {payload['load_substeps']} "
            f"--phase-count {payload['phase_count']}"
        )
        if payload.get("state_only_acceleration"):
            command += " --state-only-opening-diagnostic"
        if payload.get("state_opening_seed"):
            command += (
                " --state-opening-seed-result results/completed_candidates/"
                "sustained_active_23b19aa6590c.json --state-opening-seed-mode absolute"
            )
        current_resume_supported = (
            payload.get("schema") == SCHEMA
            and payload.get("observation_cycle_schedule_policy") == CYCLE_SCHEDULE_POLICY
            and payload.get("observation_state_binding") == "exact_R_source_order_and_verified_native_cycle"
        )
        rows.append({
            "restart_root": str(path.parent.resolve()),
            "identity_sha256": progress["identity"]["sha256"],
            "candidate_id": progress["identity"]["payload"]["candidate_id"],
            "status": progress["status"],
            "completed_support_count": len(progress["completed_observation_rows"]),
            "current_r_index": progress["current_r_index"],
            "current_observation_position": progress["current_observation_position"],
            "current_load_substep": progress["current_load_substep"],
            "active_observation": progress.get("active_observation"),
            "state_only_acceleration": progress["identity"]["payload"].get(
                "state_only_acceleration", False
            ),
            "state_opening_seed": progress["identity"]["payload"].get("state_opening_seed"),
            "resume_supported_by_current_runner": current_resume_supported,
            "resume_command": command if current_resume_supported else None,
        })
    return rows


def prediction_rows(records: list[dict[str, Any]], primary: dict[str, Any],
                    shape_frontier: dict[str, Any]) -> list[dict[str, Any]]:
    absolute = np.asarray(primary["fits"]["slope_priority"]["prediction"])
    frozen = np.asarray(primary["fits"]["slope_priority"]["frozen_prediction"])
    shape = np.asarray(shape_frontier["fits"]["slope_priority_weak_offset"]["prediction"])
    rows = []
    for record, state, pa, pf, ps in zip(records, primary["states"], absolute, frozen, shape):
        rows.append({
            "series_id": record["series_id"],
            "observation_source_order": record["source_order"],
            "R": record["R"],
            "cycles": record["cycles"],
            "mapped_cycles": record["mapped_cycles"],
            "cycle_rounding_delta": record["cycle_rounding_delta"],
            "crack_length_m": record["crack_length_m"],
            "DeltaK_MPa_sqrt_m": record["DeltaK_MPa_sqrt_m"],
            "Kmax_MPa_sqrt_m": record["DeltaK_MPa_sqrt_m"]/(1.0-record["R"]),
            "observed_dadn_m_per_cycle": record["observed_dadn_m_per_cycle"],
            "active_slope_priority_prediction_m_per_cycle": float(pa),
            "same_parameter_frozen_prediction_m_per_cycle": float(pf),
            "active_weak_offset_prediction_m_per_cycle": float(ps),
            "slope_priority_residual_log10": float(math.log10(pa/record["observed_dadn_m_per_cycle"])),
            "support_state_id": state["support_state_id"],
            "support_source_order": state["source_order"],
            "support_cycles": state["cycles"],
            "support_is_evolved_observation": state["support_is_evolved"],
        })
    return rows


def surface_rows(parent: Any, fit: dict[str, Any], stress_matrix: np.ndarray,
                 fit_label: str) -> list[dict[str, Any]]:
    coordinates = OpeningCoordinates(**fit["coordinates"])
    surface = opening_surface(parent.manifest.cleavage.surface, coordinates)
    sampled = np.unique(np.asarray(stress_matrix, dtype=float).reshape(-1))
    # All actual quadrature stresses are retained; no synthetic stress range is added.
    values = surface._basis_values(sampled)
    derivatives = surface._basis_derivatives(sampled)
    reference = surface.reference_G_eV(sampled)
    entropy = surface.entropy_kB(sampled, 300.0)
    entropy_term = -(TEMPERATURE_K-300.0)*KB_EV_PER_K*entropy
    thermal = (TEMPERATURE_K-300.0)-TEMPERATURE_K*math.log(TEMPERATURE_K/300.0)
    cp_amp = surface.heat_capacity_amplitudes_kB or (0.0,)*len(surface.bases)
    cp = np.zeros_like(sampled)+surface.heat_capacity_infinity_kB
    for amplitude, value in zip(cp_amp, values):
        cp += amplitude*value
    heat_capacity_term = KB_EV_PER_K*cp*thermal
    total = surface.G_eV(sampled, TEMPERATURE_K)
    derivative = surface.dG_dsigma_eV_per_Pa(sampled, TEMPERATURE_K)
    raw = surface.raw_rate_s(sampled, TEMPERATURE_K)
    completed = gammainc(surface.hits, np.minimum(raw*surface.tau_s, 1.0e12))/surface.tau_s
    rows = []
    for index, stress in enumerate(sampled):
        rows.append({
            "fit_label": fit_label,
            "stress_Pa": float(stress),
            "reference_floor_eV": float(surface.reference_floor_eV),
            "reference_broad_contribution_eV": float(surface.reference_amplitudes_eV[0]*values[0][index]),
            "reference_guard_contribution_eV": float(surface.reference_amplitudes_eV[1]*values[1][index]),
            "reference_G_eV": float(reference[index]),
            "entropy_thermal_contribution_eV": float(entropy_term[index]),
            "heat_capacity_contribution_eV": float(heat_capacity_term[index]),
            "Gstar_eV": float(total[index]),
            "dG_dsigma_eV_per_Pa": float(derivative[index]),
            "activation_volume_m3": float(surface.activation_volume_m3(stress, TEMPERATURE_K)),
            "raw_elementary_rate_per_s": float(raw[index]),
            "cooperative_completed_rate_per_s": float(completed[index]),
            "broad_basis_derivative_per_Pa": float(derivatives[0][index]),
            "guard_basis_derivative_per_Pa": float(derivatives[1][index]),
        })
    return rows


def exact_final_support_checks(parent: Any, primary: dict[str, Any],
                               records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plastic = PlasticCoordinates(**primary["plastic_coordinates"])
    rows = []
    restart = Path(primary["restart_root"])
    for mode in ("slope_priority", "slope_priority_weak_offset"):
        opening = OpeningCoordinates(**primary["fits"][mode]["coordinates"])
        for r_value in R_VALUES:
            engine = build_engine(material_binding(parent, plastic), kernel_family_path=R4_KERNEL_FAMILY)
            _restore_fixed_window_checkpoint(engine, _checkpoint_dir(restart, r_value))
            install_opening(engine, opening_surface(parent.manifest.cleavage.surface, opening))
            record_indices = [i for i, row in enumerate(records) if row["R"] == r_value]
            index = record_indices[-1]
            record = records[index]
            waveform = FatigueWaveform(
                Kmax=record["DeltaK_MPa_sqrt_m"]/(1.0-r_value)*1.0e6,
                R=r_value, frequency_Hz=FREQUENCY_HZ,
            )
            exact = one_cycle_map(
                engine, _NativePhaseController(int(primary["identity"]["payload"]["phase_count"])),
                waveform, TEMPERATURE_K,
            )
            exact_prediction = EVENT_LENGTH_M*exact.hazard_action_per_cycle
            approximate = float(primary["fits"][mode]["prediction"][index])
            rows.append({
                "candidate_id": primary["candidate_id"],
                "mode": mode, "R": r_value, "series_id": record["series_id"],
                "source_order": record["source_order"],
                "exact_production_prediction_m_per_cycle": exact_prediction,
                "phase_frozen_prediction_m_per_cycle": approximate,
                "log10_exact_over_phase_frozen": math.log10(exact_prediction/approximate),
                "active_state_evolves_during_exact_cycle": bool(
                    np.any(np.asarray(exact.state_end.vector) != np.asarray(exact.state_start.vector))
                ),
                "geometry_preserved": exact.geometry_preserved,
                "stochastic_state_preserved": exact.stochastic_state_preserved,
            })
    return rows


def figures(records: list[dict[str, Any]], primary: dict[str, Any],
            shape_frontier: dict[str, Any],
            prediction_table: list[dict[str, Any]],
            absolute_surface_table: list[dict[str, Any]],
            shape_surface_table: list[dict[str, Any]],
            matched: list[dict[str, Any]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    r = np.asarray([row["R"] for row in records])
    dk = np.asarray([row["DeltaK_MPa_sqrt_m"] for row in records])
    obs = np.asarray([row["observed_dadn_m_per_cycle"] for row in records])
    active = np.asarray(primary["fits"]["slope_priority"]["prediction"])
    frozen = np.asarray(primary["fits"]["slope_priority"]["frozen_prediction"])
    shape = np.asarray(shape_frontier["fits"]["slope_priority_weak_offset"]["prediction"])
    colors = {0.1: "#0072B2", 0.3: "#E69F00", 0.5: "#009E73", 0.7: "#CC79A7"}
    ymin = min(obs.min(), active.min(), shape.min())*.65
    ymax = max(obs.max(), active.max(), shape.max())*1.55
    xmin, xmax = dk.min()*.96, dk.max()*1.04
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=True)
    for axis, rv in zip(axes.flat, R_VALUES):
        mask = r == rv
        order = np.argsort(dk[mask])
        axis.scatter(dk[mask], obs[mask], s=11, color="#444444", alpha=.48, label="measured")
        axis.plot(dk[mask][order], active[mask][order], color=colors[rv], lw=2.0, label="slope-priority fit")
        axis.plot(dk[mask][order], shape[mask][order], color="#D55E00", lw=1.4, ls="--", label="weak-offset refinement")
        axis.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(ymin, ymax),
                 xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)")
        axis.text(.04, .93, f"R = {rv:.1f}", transform=axis.transAxes, va="top")
    axes[0, 0].set_ylabel("da/dN (m/cycle)")
    axes[1, 0].set_ylabel("da/dN (m/cycle)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(FIG/"active_fit_common_DeltaK_axes.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    frozen_ymin = min(obs.min(), active.min(), frozen.min())*.65
    frozen_ymax = max(obs.max(), active.max(), frozen.max())*1.55
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=True)
    for axis, rv in zip(axes.flat, R_VALUES):
        mask = r == rv
        order = np.argsort(dk[mask])
        axis.scatter(dk[mask], obs[mask], s=11, color="#444444", alpha=.48, label="measured")
        axis.plot(dk[mask][order], active[mask][order], color=colors[rv], lw=2.0,
                  label="slope-priority fit")
        axis.plot(dk[mask][order], frozen[mask][order], color="#56B4E9", lw=1.7, ls=":",
                  label="same parameters, frozen state")
        axis.set(xscale="log", yscale="log", xlim=(xmin, xmax),
                 ylim=(frozen_ymin, frozen_ymax), xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)")
        axis.text(.04, .93, f"R = {rv:.1f}", transform=axis.transAxes, va="top")
    axes[0, 0].set_ylabel("da/dN (m/cycle)")
    axes[1, 0].set_ylabel("da/dN (m/cycle)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(FIG/"same_parameter_active_frozen_common_axes.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True, sharey=True)
    residual = np.log10(active/obs)
    rlim = max(abs(residual.min()), abs(residual.max()))*1.08
    for axis, rv in zip(axes.flat, R_VALUES):
        mask = r == rv
        axis.plot(dk[mask], residual[mask], color=colors[rv], lw=1.4)
        axis.axhline(0.0, color="#666666", lw=.8)
        axis.set(xscale="log", xlim=(xmin, xmax), ylim=(-rlim, rlim),
                 xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)")
        axis.text(.04, .93, f"R = {rv:.1f}", transform=axis.transAxes, va="top")
    axes[0, 0].set_ylabel(r"log$_{10}$(prediction/data)")
    axes[1, 0].set_ylabel(r"log$_{10}$(prediction/data)")
    fig.tight_layout()
    fig.savefig(FIG/"active_absolute_residuals.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=False, sharey=True)
    supports = primary["support_states"]
    for axis, rv in zip(axes.flat, R_VALUES):
        block = [row for row in supports if row["R"] == rv]
        x = np.asarray([row["cycles"] for row in block])
        axis.plot(x, [row["mobile_count"] for row in block], marker="o", label="mobile")
        axis.plot(x, [row["accumulated_slip_count"] for row in block], marker="s", label="accumulated slip")
        axis.plot(x, [row["retained_count"] for row in block], marker="^", label="retained")
        axis.plot(x, [row["returned_slip_count"] for row in block], marker="v", label="returned")
        axis.set(yscale="symlog", xlabel="accepted cycles")
        axis.text(.04, .93, f"R = {rv:.1f}", transform=axis.transAxes, va="top")
    axes[0, 0].set_ylabel("native count")
    axes[1, 0].set_ylabel("native count")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(FIG/"active_state_histories.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    stress = np.asarray([row["stress_Pa"] for row in absolute_surface_table])/1.0e9
    g = np.asarray([row["Gstar_eV"] for row in absolute_surface_table])
    floor = np.asarray([row["reference_floor_eV"] for row in absolute_surface_table])
    broad = np.asarray([row["reference_broad_contribution_eV"] for row in absolute_surface_table])
    guard = np.asarray([row["reference_guard_contribution_eV"] for row in absolute_surface_table])
    order = np.argsort(stress)
    fig, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.plot(stress[order], g[order], color="#0072B2", lw=2.1, label=r"$G^*$ absolute fit")
    shape_stress = np.asarray([row["stress_Pa"] for row in shape_surface_table])/1.0e9
    shape_g = np.asarray([row["Gstar_eV"] for row in shape_surface_table])
    shape_order = np.argsort(shape_stress)
    axis.plot(shape_stress[shape_order], shape_g[shape_order], color="#D55E00", lw=1.8,
              ls="--", label=r"$G^*$ shape frontier")
    axis.plot(stress[order], floor[order], color="#555555", ls=":", label="absolute floor")
    axis.plot(stress[order], broad[order], color="#E69F00", ls="--", label="absolute broad contribution")
    axis.plot(stress[order], guard[order], color="#009E73", ls="-.", label="absolute guard contribution")
    axis.set(xlabel="production opening stress (GPa)", ylabel=r"barrier or contribution (eV)")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG/"active_Gstar_and_contributions.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    targets = sorted({row["target_Kmax_MPa_sqrt_m"] for row in matched})
    target = 48.0 if 48.0 in targets else targets[len(targets)//2]
    block = [row for row in matched if row["target_Kmax_MPa_sqrt_m"] == target]
    rv = np.asarray([row["R"] for row in block])
    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.scatter(rv, [row["observed_dadn_m_per_cycle"] for row in block], s=52,
                 color="#333333", label="nearest measured point", zorder=3)
    axis.plot(rv, [row["active_slope_priority_prediction"] for row in block], marker="o",
              color="#0072B2", label="active absolute fit")
    axis.plot(rv, [row["same_parameter_frozen_prediction"] for row in block], marker="s",
              color="#56B4E9", ls=":", label="same parameters, frozen state")
    axis.set(yscale="log", xlabel="load ratio R", ylabel="da/dN (m/cycle)")
    axis.text(.03, .95, rf"nearest available points to $K_{{max}}={target:.0f}$ MPa$\sqrt{{m}}$",
              transform=axis.transAxes, va="top")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/"matched_peak_R_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    records = prepare_observation_schedule(build_records("index_rank", premeasurement=True))
    excluded_comparators = []
    candidates = candidate_records(records, excluded=excluded_comparators)
    if not candidates:
        raise RuntimeError(
            "no completed schema-2 calibration has exact evolved states for every observation; "
            "legacy/sparse results are comparator-only. Complete the repaired runner with "
            "--support-points 0 before selecting a primary fit."
        )
    sustained_control = load(PRIMARY) if PRIMARY.is_file() else None
    historical = load(HISTORICAL) if HISTORICAL.is_file() else None
    parent = load_parent_binding(ALT_REGISTRY, PARENT_REGISTRY, candidate_id=PARENT_CANDIDATE)
    barrier_admissible = [item for item in candidates if
                          item[1]["fits"]["slope_priority"]["barrier_qualification"]["passed"]]
    if not barrier_admissible:
        raise RuntimeError("no completed schema-2 candidate has a physically admissible slope-priority barrier")
    absolute_frontier_path, primary = min(
        barrier_admissible,
        key=lambda item: item[1]["fits"]["slope_priority"]["slope_priority_evaluation"]["primary_score"],
    )
    shape_frontier_path, shape_frontier = min(
        barrier_admissible,
        key=lambda item: item[1]["fits"]["slope_priority_weak_offset"]["slope_priority_evaluation"]["primary_score"],
    )
    candidate_summary = []
    for path, candidate in candidates:
        states = candidate.get("support_states", candidate.get("support_rows", []))
        candidate_summary.append({
            "path": str(path.resolve()), "sha256": digest(path),
            "candidate_id": candidate["candidate_id"],
            "identity": candidate.get("identity_sha256", candidate.get("identity", {}).get("sha256")),
            "plastic_coordinates": candidate["plastic_coordinates"],
            "support_count": len(states),
            "calibration_eligible": True,
            "phase_count": candidate["identity"]["payload"]["phase_count"],
            "load_substeps": candidate["identity"]["payload"]["load_substeps"],
            "observation_cycle_schedule_policy": CYCLE_SCHEDULE_POLICY,
            "history_mode": candidate.get("history_mode", "historical_endpoint_support"),
            "state_opening_seed": candidate.get("state_opening_seed"),
            "fitted_absolute_opening": candidate["fits"]["absolute"]["coordinates"],
            "absolute_metrics": candidate["fits"]["absolute"]["metrics"],
            "shape_metrics": candidate["fits"]["shape"]["metrics"],
            "slope_priority_evaluation": candidate["fits"]["slope_priority"]["slope_priority_evaluation"],
            "slope_priority_barrier": candidate["fits"]["slope_priority"]["barrier_qualification"],
            "maximum_mobile_count": max(row["mobile_count"] for row in states),
            "maximum_retained_count": max(row["retained_count"] for row in states),
            "maximum_accumulated_slip_count": max(row["accumulated_slip_count"] for row in states),
            "maximum_tip_radius_m": max(row["tip_radius_m"] for row in states),
        })
    prediction_table = prediction_rows(records, primary, shape_frontier)
    states = primary["states"]
    stress = _stress_trace(records, states, int(primary["identity"]["payload"]["phase_count"]))
    absolute_surface_table = surface_rows(
        parent, primary["fits"]["slope_priority"], stress, "primary_slope_priority",
    )
    shape_stress = _stress_trace(
        records, shape_frontier["states"], int(shape_frontier["identity"]["payload"]["phase_count"]),
    )
    shape_surface_table = surface_rows(
        parent, shape_frontier["fits"]["slope_priority_weak_offset"], shape_stress,
        "weak_offset_refinement",
    )
    surface_table = absolute_surface_table+shape_surface_table
    matched = fixed_kmax(records, {
        "active_slope_priority_prediction": np.asarray(primary["fits"]["slope_priority"]["prediction"]),
        "same_parameter_frozen_prediction": np.asarray(primary["fits"]["slope_priority"]["frozen_prediction"]),
    })
    exact_checks = exact_final_support_checks(parent, primary, records)
    if shape_frontier.get("restart_root") != primary.get("restart_root"):
        exact_checks.extend(exact_final_support_checks(parent, shape_frontier, records))
    representation = (
        support_representation_comparison(historical, sustained_control)
        if historical is not None and sustained_control is not None
        else {"available": False, "rows": [], "scope": "optional legacy comparison inputs unavailable"}
    )
    representation["role"] = "legacy_comparator_only_not_repaired_history_convergence"
    state_only_comparison = (
        state_only_longitudinal_comparison(sustained_control)
        if sustained_control is not None
        else {"available": False, "long_projective_propagation_qualified": False}
    )
    state_only_comparison["role"] = "legacy_observer_diagnostic_only"
    restarts = restart_inventory()
    write_csv(TABLE/"chronological_predictions.csv", prediction_table)
    write_csv(TABLE/"genuine_support_state_history.csv", primary["support_states"])
    write_csv(TABLE/"opening_surface_actual_sampled_stresses.csv", surface_table)
    write_csv(TABLE/"matched_peak_comparison.csv", matched)
    write_csv(TABLE/"exact_final_support_opening_checks.csv", exact_checks)
    if representation["rows"]:
        write_csv(TABLE/"historical_vs_chronological_final_states.csv", representation["rows"])
    if state_only_comparison.get("comparison_rows"):
        write_csv(TABLE/"production_vs_state_only_common_supports.csv",
                  state_only_comparison["comparison_rows"])
    atomic_json(OUT/"restart_inventory.json", restarts)
    controls = constant_controls(records)
    kernel = load(Path(R4_KERNEL_FAMILY))
    measured_extensions = {
        str(r_value): (
            max(row["crack_length_m"] for row in records if row["R"] == r_value)
            - min(row["crack_length_m"] for row in records if row["R"] == r_value)
        ) for r_value in R_VALUES
    }
    kernel_limit = max(kernel["cumulative_crack_path_extension_levels_m"])
    summary = {
        "schema": "qste340tm-active-calibration-assessment/2",
        "source_csv_sha256": digest(DATA/"normalized/qste340tm_constant_amplitude.csv"),
        "temperature_K": TEMPERATURE_K,
        "frequency_Hz": FREQUENCY_HZ,
        "frequency_status": "provisional assumed value; active sensitivity remains unresolved",
        "fitting_weights": {
            "absolute_objective": (
                "each observation in R block carries 1/sqrt(n_R), so each of the four histories "
                "contributes equal total squared weight"
            ),
            "shape_objective": (
                "concatenates 0.5 times the block-balanced absolute residual with the "
                "block-balanced within-history centered residual"
            ),
            "headline_overall_rmse": "unweighted pooled RMSE over all 760 observations",
            "matched_Kmax_and_slopes": "diagnostic summaries of the same observations; not added as independent likelihood data",
        },
        "primary_identity": primary["identity"],
        "slope_priority_frontier_path": str(absolute_frontier_path.resolve()),
        "primary_plastic_coordinates": primary["plastic_coordinates"],
        "primary_opening_coordinates": {
            mode: primary["fits"][mode]["coordinates"] for mode in ("slope_priority", "slope_priority_weak_offset")
        },
        "primary_metrics": {
            mode: primary["fits"][mode]["metrics"] for mode in ("slope_priority", "slope_priority_weak_offset")
        },
        "shape_frontier": {
            "path": str(shape_frontier_path.resolve()),
            "candidate_id": shape_frontier["candidate_id"],
            "identity": shape_frontier.get(
                "identity_sha256", shape_frontier.get("identity", {}).get("sha256")
            ),
            "plastic_coordinates": shape_frontier["plastic_coordinates"],
            "opening_coordinates": shape_frontier["fits"]["slope_priority_weak_offset"]["coordinates"],
            "metrics": shape_frontier["fits"]["slope_priority_weak_offset"]["metrics"],
        },
        "same_parameter_frozen_metrics": {
            mode: primary["fits"][mode]["frozen_metrics"] for mode in ("slope_priority", "slope_priority_weak_offset")
        },
        "sustained_chronological_control": {
            "available": True,
            "role": "legacy_comparator_only_not_calibration_frontier",
            "path": str(PRIMARY.resolve()),
            "identity": sustained_control["identity"],
            "plastic_coordinates": sustained_control["plastic_coordinates"],
            "absolute_metrics": sustained_control["fits"]["absolute"]["metrics"],
            "same_parameter_frozen_absolute_metrics": sustained_control["fits"]["absolute"]["frozen_metrics"],
        } if sustained_control is not None else {
            "available": False, "role": "optional_legacy_comparator_only",
        },
        "constant_controls": controls,
        "native_translation_domain": {
            "kernel_path": str(Path(R4_KERNEL_FAMILY).resolve()),
            "kernel_sha256": digest(Path(R4_KERNEL_FAMILY)),
            "qualified_extension_levels_m": kernel["cumulative_crack_path_extension_levels_m"],
            "qualified_maximum_extension_m": kernel_limit,
            "measured_extension_by_R_m": measured_extensions,
            "measured_to_qualified_domain_ratio_by_R": {
                key: value/kernel_limit for key, value in measured_extensions.items()
            },
            "decision": (
                "no full native translation executed: every measured history exceeds the "
                "qualified signed-kernel domain; extrapolation or population reset was rejected"
            ),
        },
        "candidate_summary": candidate_summary,
        "excluded_comparators": excluded_comparators,
        "candidate_selection_policy": (
            "schema-2 calibration_eligible production histories with exact source/time/state "
            "binding for all observations; legacy and diagnostic metrics never select the primary"
        ),
        "native_acceleration_profile": acceleration_profile(primary),
        "historical_vs_chronological_representation": representation,
        "state_only_longitudinal_qualification": state_only_comparison,
        "restart_inventory": restarts,
        "exact_final_support_opening_checks": exact_checks,
        "claim_assessment": (
            "Conditional fixed-local-window calibration with verified observation-state ownership. "
            "Scientific interpretation requires comparison of the measured residuals, load/phase "
            "refinement, continuous plastic-parameter optimization, and the stated model/data "
            "qualifications. No physical conclusion is assigned automatically from legacy metrics."
        ),
        "unresolved": [
            "native signed-kernel translation/remapping beyond 1.175 mm for the measured 6-13 mm extension",
            "active-state frequency sensitivity around the provisional 10 Hz assumption",
            "load-quadrature and phase refinement at the same material/opening parameters",
            "continuous outer plastic-parameter optimization beyond the three fixed candidates",
            "sensitivity to nearest-whole-cycle scheduling of mapped observations",
            "R=0.5 rate/history identifier mismatch",
        ],
    }
    atomic_json(OUT/"active_calibration_assessment.json", summary)
    figures(
        records, primary, shape_frontier, prediction_table,
        absolute_surface_table, shape_surface_table, matched,
    )
    print(json.dumps({
        "assessment": str(OUT/"active_calibration_assessment.json"),
        "candidate_count": len(candidate_summary),
        "slope_priority_score": primary["fits"]["slope_priority"]["slope_priority_evaluation"]["primary_score"],
        "pooled_constant_rmse": controls["pooled_optimal_common_constant"]["metrics"]["overall_rmse_log10_rate"],
        "figures": len(list(FIG.glob("*.png"))),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
