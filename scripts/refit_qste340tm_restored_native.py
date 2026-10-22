#!/usr/bin/env python3
"""Fit QSTE340TM with the restored persistent/reversible production runtime.

The expensive outer coordinates are evaluated with the installed R2/R9 state
machine and its accepted high-cycle accelerator.  Opening coordinates are then
profiled on that prescribed measured-history replay.  The profiling is exact
with respect to the replay because measured crack extension, rather than the
opening prediction, owns the geometry schedule; final saved states are checked
again through the production one-cycle opening observer with feedback active.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammainc
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from arrhenius_fracture.paris_slope_r9 import _NativePhaseController
from arrhenius_fracture.persistent_site_high_cycle_engine_v10230 import (
    integrate_state_coupled_waveform,
    invalidate_high_cycle_cache,
)
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (
    capture_ledgers,
    ledger_delta,
)
from arrhenius_fracture.persistent_site_poincare_v10230 import one_cycle_map
from arrhenius_fracture.qste340tm_restored_binding import (
    MODEL_ID,
    OpeningCoordinates,
    PlasticCoordinates,
    TEMPERATURE_K,
    build_engine,
    install_opening,
    load_parent_binding,
    material_binding,
    opening_surface,
    plastic_vector,
)
from arrhenius_fracture.reduced_persistent_reference_r2 import runtime_receipt


DATA = ROOT / "literature_data/fatigue_recovered_review"
OUT = DATA / "model_conditional/stateful_exp_floor_capacity_policy/restored_native_refit"
ALT_REGISTRY = ROOT / "integration_evidence/r10_alternative_gstar_joint_search/alternative_gstar_registry.json"
PARENT_REGISTRY = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/"
    "v3-fatigue-informed-joint-barrier-search/analysis_outputs/"
    "v3_fatigue_informed_joint_barrier_search/v3_complete_candidate_registry.json"
)
PARENT_CANDIDATE = "R10_ALT_GSTAR_02506"
FREQUENCY_HZ = 10.0
EVENT_LENGTH_M = 5.0e-6
R_VALUES = (0.1, 0.3, 0.5, 0.7)
SEED = 20261022


class CandidateWallLimit(BaseException):
    """Escape internal broad Exception handlers without killing the campaign."""
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    def encode(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        return repr(item)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=encode) + "\n")
    temporary.replace(path)


def build_records(alignment: str, *, premeasurement: bool = True) -> list[dict[str, Any]]:
    rates = read_csv(DATA / "normalized/qste340tm_constant_amplitude.csv")
    histories = read_csv(DATA / "normalized/qste340tm_crack_histories.csv")
    records: list[dict[str, Any]] = []
    for r_value in R_VALUES:
        block = sorted(
            (row for row in rates if float(row["R"]) == r_value),
            key=lambda row: int(row["source_order"]),
        )
        history = sorted(
            (row for row in histories
             if float(row["R"]) == r_value and row["protocol"] == "CONSTANT_AMPLITUDE"),
            key=lambda row: int(row["source_order"]),
        )
        h_cycles = np.asarray([float(row["cycles"]) for row in history])
        h_crack = np.asarray([float(row["reported_crack_length_m"]) for row in history])
        delta_k = np.asarray([float(row["reported_DeltaK_MPa_sqrt_m"]) for row in block])
        if alignment == "index_rank":
            coordinate = np.linspace(0.0, 1.0, len(block))
            history_coordinate = np.linspace(0.0, 1.0, len(history))
            cycles = np.interp(coordinate, history_coordinate, h_cycles)
            crack = np.interp(coordinate, history_coordinate, h_crack)
            exact_index = len(block) == len(history)
        elif alignment == "deltaK_crack_length":
            coordinate = (delta_k-delta_k[0]) / max(delta_k[-1]-delta_k[0], 1.0e-30)
            history_coordinate = (h_crack-h_crack[0]) / max(h_crack[-1]-h_crack[0], 1.0e-30)
            cycles = np.interp(coordinate, history_coordinate, h_cycles)
            crack = np.interp(coordinate, history_coordinate, h_crack)
            exact_index = False
        else:
            raise ValueError(f"unknown alignment {alignment!r}")
        if not premeasurement:
            cycles = cycles - cycles[0]
        for index, (row, cycle, crack_length) in enumerate(zip(block, cycles, crack)):
            records.append({
                "series_id": row["series_id"],
                "source_order": int(row["source_order"]),
                "block_index": index,
                "R": r_value,
                "DeltaK_MPa_sqrt_m": float(row["reported_DeltaK_MPa_sqrt_m"]),
                "observed_dadn_m_per_cycle": float(row["dadn_m_per_cycle"]),
                "cycles": float(cycle),
                "crack_length_m": float(crack_length),
                "history_alignment": alignment,
                "history_index_exact": bool(exact_index),
                "premeasurement_exposure_included": bool(premeasurement),
            })
    if len(records) != 760:
        raise RuntimeError(f"expected 760 QSTE rows, found {len(records)}")
    return records


def support_indices(records: list[dict[str, Any]], points_per_R: int) -> dict[float, list[int]]:
    output: dict[float, list[int]] = {}
    for r_value in R_VALUES:
        indices = [i for i, row in enumerate(records) if row["R"] == r_value]
        positions = np.unique(np.linspace(0, len(indices)-1, int(points_per_R)).round().astype(int))
        output[r_value] = [indices[int(position)] for position in positions]
    return output


def _state_row(engine: Any, row: dict[str, Any], process: dict[str, Any],
               prescribed_advance_m: float) -> dict[str, Any]:
    mpz = engine.mpz
    diagnostics = mpz.diagnostics(engine.G, engine.nu, engine.b, engine.f.r0)
    Kmax = row["DeltaK_MPa_sqrt_m"] / (1.0-row["R"]) * 1.0e6
    sigma = engine.sigma_tip(Kmax)
    rho = mpz.local_forest_density_m2(False)
    profile = mpz.local_stress_profile_Pa(sigma)
    rates = mpz._transport_rates(profile, rho, TEMPERATURE_K, engine.b)
    returned = float(np.sum(getattr(mpz, "returned_slip_positive", 0.0))
                     + np.sum(getattr(mpz, "returned_slip_negative", 0.0)))
    accumulated = float(np.sum(getattr(mpz, "accumulated_slip_positive", 0.0))
                        + np.sum(getattr(mpz, "accumulated_slip_negative", 0.0)))
    return {
        "R": row["R"], "source_order": row["source_order"],
        "block_index": row["block_index"], "cycles": row["cycles"],
        "crack_length_m": row["crack_length_m"],
        "prescribed_advance_m": prescribed_advance_m,
        "mobile_count": float(mpz.mobile_count),
        "retained_count": float(mpz.retained_count),
        "accumulated_slip_count": accumulated,
        "returned_slip_count": returned,
        "net_slip_count": float(mpz.local_slip_count()),
        "gross_source_activity": float(getattr(mpz, "cumulative_gross_source_activity", 0.0)),
        "gross_return_activity": float(getattr(mpz, "cumulative_gross_return_activity", 0.0)),
        "source_activity_mean": float(np.mean(getattr(mpz, "tip_source_activity", np.ones(1)))),
        "tip_radius_m": float(engine.r_eff()),
        "K_shield_Pa_sqrt_m": float(engine.K_shield()),
        "opening_stress_at_Kmax_Pa": float(sigma),
        "mean_forest_density_m2": float(np.mean(rho)),
        "maximum_forest_density_m2": float(np.max(rho)),
        "mean_peierls_rate_s": float(np.mean(rates["peierls"])),
        "mean_taylor_elementary_rate_s": float(np.mean(rates["taylor_single"])),
        "mean_taylor_release_rate_s": float(np.mean(rates["taylor"])),
        "mean_taylor_density_order": float(np.mean(rates["m"])),
        "mean_encounter_rate_s": float(np.mean(rates["encounter"])),
        "persistent_site_multiplicity_per_system": float(diagnostics["persistent_site_multiplicity_per_system"]),
        "interval_process": process,
    }


def _stress_trace(records: list[dict[str, Any]], states: list[dict[str, Any]],
                  phase_count: int) -> np.ndarray:
    phase = (np.arange(int(phase_count), dtype=float) + 0.5) * 2.0*math.pi/int(phase_count)
    output = np.empty((len(records), int(phase_count)))
    for index, (row, state) in enumerate(zip(records, states)):
        fraction = .5*(1.0+row["R"]) + .5*(1.0-row["R"])*np.cos(phase)
        Kmax = row["DeltaK_MPa_sqrt_m"]/(1.0-row["R"])*1.0e6
        K_eff = np.maximum(Kmax*fraction-state["K_shield_Pa_sqrt_m"], 0.0)
        output[index] = np.minimum(
            K_eff / math.sqrt(2.0*math.pi*max(state["tip_radius_m"], 1.0e-30)),
            30.0e9,
        )
    return output


def run_native_history(
    parent: Any,
    plastic: PlasticCoordinates,
    records: list[dict[str, Any]],
    *,
    points_per_R: int,
    phase_count: int,
    keep_engines: bool = False,
    apply_measured_moving_frame: bool = False,
) -> dict[str, Any]:
    selected = support_indices(records, points_per_R)
    state_rows: list[dict[str, Any] | None] = [None] * len(records)
    saved_engines: dict[int, Any] = {}
    support_rows: list[dict[str, Any]] = []
    started = time.time()
    base_binding = material_binding(parent, plastic)
    receipt = None
    for r_value in R_VALUES:
        engine = build_engine(base_binding, kernel_family_path=R4_KERNEL_FAMILY)
        engine.hazard_threshold_action = 1.0e300
        engine.hazard_action_current = 0.0
        engine.B = 0.0
        if receipt is None:
            receipt = runtime_receipt(engine, stage="qste_restored_native_history_initial")
        prior_cycles = 0.0
        prior_crack = records[selected[r_value][0]]["crack_length_m"]
        previous_state: dict[str, Any] | None = None
        support_for_r = selected[r_value]
        for support_number, global_index in enumerate(support_for_r):
            row = records[global_index]
            measured_advance = max(float(row["crack_length_m"])-float(prior_crack), 0.0)
            advance = measured_advance if apply_measured_moving_frame else 0.0
            advance_receipt = engine.mpz.advance(advance) if advance > 0.0 else {
                "status": "not_applied_fixed_local_window",
                "measured_interval_advance_m": measured_advance,
                "reason": (
                    "reported DeltaK already owns specimen crack geometry; the accepted "
                    "signed kernel is qualified only through 0.001175 m cumulative local "
                    "extension and cannot replay the 0.00636-0.01307 m histories"
                ),
            }
            if advance > 0.0:
                invalidate_high_cycle_cache(engine, "prescribed_measured_crack_advance")
            exposure = max(float(row["cycles"])-prior_cycles, 0.0)
            before = capture_ledgers(engine)
            if exposure > 0.0:
                waveform = FatigueWaveform(
                    Kmax=float(row["DeltaK_MPa_sqrt_m"])/(1.0-r_value)*1.0e6,
                    R=r_value,
                    frequency_Hz=FREQUENCY_HZ,
                )
                result = integrate_state_coupled_waveform(
                    engine,
                    _NativePhaseController(phase_count),
                    waveform,
                    TEMPERATURE_K,
                    exposure,
                )
                if result.get("fired"):
                    raise RuntimeError("suppressed-threshold measured-history replay fired")
                if result.get("coupled_hazard_work_budget_exhausted"):
                    raise RuntimeError("native high-cycle work budget exhausted")
            else:
                result = {"coupled_hazard_cycles_consumed": 0.0, "coupled_hazard_modes": []}
            process = {
                "requested_cycles": exposure,
                "accepted_cycles": float(result.get("coupled_hazard_cycles_consumed", 0.0)),
                "accepted_seconds": float(result.get("coupled_hazard_cycles_consumed", 0.0))/FREQUENCY_HZ,
                "modes": result.get("coupled_hazard_modes", []),
                "ledger_delta": ledger_delta(before, capture_ledgers(engine)),
                "moving_frame": advance_receipt,
            }
            current_state = _state_row(engine, row, process, advance)
            support_rows.append(current_state)
            if keep_engines:
                saved_engines[global_index] = copy.deepcopy(engine)
            lo = global_index
            hi = (support_for_r[support_number+1] if support_number+1 < len(support_for_r)
                  else max(i for i, value in enumerate(records) if value["R"] == r_value)+1)
            for index in range(lo, hi):
                state_rows[index] = dict(current_state)
            previous_state = current_state
            prior_cycles = float(row["cycles"])
            prior_crack = float(row["crack_length_m"])
        if previous_state is None:
            raise RuntimeError("empty R block")
    if any(row is None for row in state_rows):
        raise RuntimeError("native history did not cover every observation")
    states = [dict(row) for row in state_rows if row is not None]
    return {
        "states": states,
        "support_rows": support_rows,
        "support_indices": {str(key): value for key, value in selected.items()},
        "stress": _stress_trace(records, states, phase_count),
        "runtime_receipt": receipt,
        "elapsed_s": time.time()-started,
        "saved_engines": saved_engines,
        "moving_frame_policy": (
            "measured_crack_history" if apply_measured_moving_frame
            else "fixed_local_window_reported_DeltaK_owns_specimen_geometry"
        ),
    }


OPENING_LOWER = np.asarray([0.15, 0.0, math.log(0.1e9), math.log(0.15), 0.0])
OPENING_UPPER = np.asarray([0.90, 3.0, math.log(30.0e9), math.log(5.0), 3.0])


def unpack_opening(vector: np.ndarray) -> OpeningCoordinates:
    return OpeningCoordinates(
        floor_eV=float(vector[0]),
        broad_amplitude_eV=float(vector[1]),
        broad_sigma_Pa=float(math.exp(vector[2])),
        broad_exponent=float(math.exp(vector[3])),
        guard_amplitude_eV=float(vector[4]),
    )


def predict(base_surface: Any, vector: np.ndarray, stress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    surface = opening_surface(base_surface, unpack_opening(vector))
    raw = surface.raw_rate_s(stress, TEMPERATURE_K)
    completed = gammainc(surface.hits, np.minimum(raw*surface.tau_s, 1.0e12))/surface.tau_s
    rate = EVENT_LENGTH_M*np.mean(completed, axis=1)/FREQUENCY_HZ
    barrier = surface.G_eV(stress, TEMPERATURE_K)
    return np.maximum(rate, 1.0e-300), barrier


def block_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_values = np.asarray([row["R"] for row in records])
    observed = np.log10([row["observed_dadn_m_per_cycle"] for row in records])
    weights = np.empty(len(records))
    for r_value in R_VALUES:
        weights[r_values == r_value] = 1.0/math.sqrt(np.sum(r_values == r_value))
    return r_values, observed, weights


def fit_opening(mode: str, base_surface: Any, records: list[dict[str, Any]],
                stress: np.ndarray) -> dict[str, Any]:
    r_values, observed, weights = block_arrays(records)
    starts = [
        np.array([0.456, 1.0e-6, math.log(.2e9), math.log(1.75), 0.0]),
        np.array([0.447, .031, math.log(9.8e9), math.log(2.2), 0.0]),
        np.array([.35, .8, math.log(2.0e9), math.log(.6), .8]),
        np.array([.50, 1.5, math.log(8.0e9), math.log(1.5), .2]),
        np.array([.65, 2.0, math.log(15.0e9), math.log(3.0), 1.5]),
    ]

    def residual(vector: np.ndarray) -> np.ndarray:
        prediction, barrier = predict(base_surface, vector, stress)
        error = np.log10(prediction)-observed
        centered = np.empty_like(error)
        for r_value in R_VALUES:
            mask = r_values == r_value
            centered[mask] = error[mask]-np.mean(error[mask])
        primary = error*weights if mode == "absolute" else np.r_[.5*error*weights, centered*weights]
        invalid = max(0.0, 1.0e-5-float(np.min(barrier)))*1.0e5
        return np.r_[primary, invalid]

    solutions = []
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, OPENING_LOWER, OPENING_UPPER),
            bounds=(OPENING_LOWER, OPENING_UPPER),
            max_nfev=240,
            x_scale="jac",
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        solutions.append((float(residual(result.x)@residual(result.x)), result))
    objective, solution = min(solutions, key=lambda item: item[0])
    prediction, barrier = predict(base_surface, solution.x, stress)
    return {
        "mode": mode,
        "objective": objective,
        "coordinates_vector": solution.x.tolist(),
        "coordinates": asdict(unpack_opening(solution.x)),
        "prediction": prediction.tolist(),
        "minimum_barrier_eV": float(np.min(barrier)),
        "maximum_barrier_eV": float(np.max(barrier)),
        "solver_success": bool(solution.success),
        "solver_message": str(solution.message),
        "solver_nfev": int(solution.nfev),
    }


def metrics(records: list[dict[str, Any]], prediction: np.ndarray) -> dict[str, Any]:
    observed = np.asarray([row["observed_dadn_m_per_cycle"] for row in records])
    r_values = np.asarray([row["R"] for row in records])
    error = np.log10(prediction)-np.log10(observed)
    block = {}
    centered_all = []
    for r_value in R_VALUES:
        mask = r_values == r_value
        x = np.log([row["DeltaK_MPa_sqrt_m"] for row in records if row["R"] == r_value])
        yo = np.log(observed[mask]); yp = np.log(prediction[mask])
        unique_x, inverse = np.unique(x, return_inverse=True)
        unique_yo = np.asarray([np.mean(yo[inverse == i]) for i in range(len(unique_x))])
        unique_yp = np.asarray([np.mean(yp[inverse == i]) for i in range(len(unique_x))])
        local_observed = np.gradient(unique_yo, unique_x)
        local_predicted = np.gradient(unique_yp, unique_x)
        centered = error[mask]-np.mean(error[mask]); centered_all.extend(centered.tolist())
        block[str(r_value)] = {
            "points": int(np.sum(mask)),
            "rmse_log10_rate": float(np.sqrt(np.mean(error[mask]**2))),
            "bias_log10_rate": float(np.mean(error[mask])),
            "observed_broad_loglog_slope": float(np.polyfit(x, yo, 1)[0]),
            "predicted_broad_loglog_slope": float(np.polyfit(x, yp, 1)[0]),
            "observed_local_slope_q10_q50_q90": np.quantile(local_observed, [.1,.5,.9]).tolist(),
            "predicted_local_slope_q10_q50_q90": np.quantile(local_predicted, [.1,.5,.9]).tolist(),
        }
    return {
        "overall_rmse_log10_rate": float(np.sqrt(np.mean(error**2))),
        "overall_bias_log10_rate": float(np.mean(error)),
        "shape_rmse_log10": float(np.sqrt(np.mean(np.asarray(centered_all)**2))),
        "block_metrics": block,
    }


def frozen_stress(records: list[dict[str, Any]], phase_count: int) -> np.ndarray:
    states = [{"tip_radius_m": 1.0e-6, "K_shield_Pa_sqrt_m": 0.0} for _ in records]
    return _stress_trace(records, states, phase_count)


def candidate_vectors(count: int) -> list[tuple[str, PlasticCoordinates]]:
    seeds = [
        ("prior_absolute", PlasticCoordinates(2.5240573, .9914368e9, 1.7843938, 1.7843938)),
        ("prior_shape", PlasticCoordinates(2.51045, .973153e9, 1.99524, 1.99524)),
        ("sustained_active", PlasticCoordinates(1.2, 1.5e9, 1.5, 1.5)),
        ("transport_dominated", PlasticCoordinates(.8, 1.0e9, .7, 2.5)),
        ("trapping_dominated", PlasticCoordinates(.8, 1.2e9, 2.8, .8)),
        ("strong_escape", PlasticCoordinates(1.5, 2.0e9, .45, .45)),
        ("weak_state", PlasticCoordinates(2.8, 3.5e9, 3.5, 3.5)),
        ("transition_high", PlasticCoordinates(1.65, 2.5e9, .52, .40)),
        ("transition_low", PlasticCoordinates(1.48, 2.5e9, .52, .40)),
    ]
    needed = max(int(count)-len(seeds), 0)
    if needed:
        power = int(math.ceil(math.log2(needed)))
        points = qmc.Sobol(4, scramble=True, seed=SEED).random_base2(power)[:needed]
        for index, u in enumerate(points):
            seeds.append((f"sobol_{index:03d}", PlasticCoordinates(
                emission_G0_298_eV=.5+2.5*u[0],
                plastic_sigc_298_Pa=math.exp(math.log(.4e9)+u[1]*(math.log(6e9)-math.log(.4e9))),
                peierls_G0_298_eV=.3+3.7*u[2],
                taylor_G0_298_eV=.3+3.7*u[3],
            )))
    return seeds[:int(count)]


def evaluate_candidate(candidate_id: str, plastic: PlasticCoordinates, parent: Any,
                       records: list[dict[str, Any]], points: int, phases: int,
                       *, force: bool = False, max_wall_s: int = 600) -> dict[str, Any]:
    directory = OUT / "candidate_cache"
    directory.mkdir(parents=True, exist_ok=True)
    identity = canonical({"plastic": asdict(plastic), "points": points, "phases": phases,
                          "alignment": records[0]["history_alignment"],
                          "premeasurement": records[0]["premeasurement_exposure_included"],
                          "candidate_wall_limit_s": int(max_wall_s)})
    path = directory / f"{candidate_id}_{identity[:12]}.json"
    if path.is_file() and not force:
        return json.loads(path.read_text())
    print(f"[{candidate_id}] native history start", flush=True)
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        def wall_limit(_signum: int, _frame: Any) -> None:
            raise CandidateWallLimit(
                f"native candidate exceeded declared {int(max_wall_s)} s wall limit"
            )
        old_handler = signal.signal(signal.SIGALRM, wall_limit)
        signal.alarm(max(int(max_wall_s), 1))
        trajectory = run_native_history(
            parent, plastic, records, points_per_R=points, phase_count=phases,
        )
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        base_surface = parent.manifest.cleavage.surface
        fits = {mode: fit_opening(mode, base_surface, records, trajectory["stress"])
                for mode in ("absolute", "shape")}
        for fit in fits.values():
            fit["metrics"] = metrics(records, np.asarray(fit["prediction"]))
        result = {
            "schema": "qste340tm-restored-native-candidate/1",
            "candidate_id": candidate_id,
            "identity_sha256": identity,
            "evaluator_identity": MODEL_ID,
            "plastic_coordinates": asdict(plastic),
            "points_per_R": points,
            "phase_count": phases,
            "elapsed_s": trajectory["elapsed_s"],
            "fits": fits,
            "states": trajectory["states"],
            "support_rows": trajectory["support_rows"],
            "support_indices": trajectory["support_indices"],
            "runtime_receipt": trajectory["runtime_receipt"],
            "status": "complete",
        }
    except (Exception, CandidateWallLimit) as error:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        result = {
            "schema": "qste340tm-restored-native-candidate/1",
            "candidate_id": candidate_id,
            "identity_sha256": identity,
            "evaluator_identity": MODEL_ID,
            "plastic_coordinates": asdict(plastic),
            "points_per_R": points,
            "phase_count": phases,
            "status": "rejected",
            "rejection_type": type(error).__name__,
            "rejection_reason": str(error),
        }
    atomic_json(path, result)
    print(f"[{candidate_id}] {result['status']}", flush=True)
    return result


def local_vectors(best: list[dict[str, Any]], count: int) -> list[tuple[str, PlasticCoordinates]]:
    rng = np.random.default_rng(SEED+1)
    output = []
    centers = [entry["plastic_coordinates"] for entry in best]
    for index in range(int(count)):
        center = centers[index % len(centers)]
        factors = np.exp(rng.normal(0.0, [.10,.12,.12,.12]))
        output.append((f"local_{index:03d}", PlasticCoordinates(
            emission_G0_298_eV=float(np.clip(center["emission_G0_298_eV"]*factors[0], .5, 3.0)),
            plastic_sigc_298_Pa=float(np.clip(center["plastic_sigc_298_Pa"]*factors[1], .4e9, 6e9)),
            peierls_G0_298_eV=float(np.clip(center["peierls_G0_298_eV"]*factors[2], .3, 4.0)),
            taylor_G0_298_eV=float(np.clip(center["taylor_G0_298_eV"]*factors[3], .3, 4.0)),
        )))
    return output


def pareto(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in results:
        if candidate.get("status") != "complete":
            continue
        for mode, fit in candidate["fits"].items():
            rows.append({
                "candidate_id": candidate["candidate_id"], "mode": mode,
                "plastic_coordinates": candidate["plastic_coordinates"],
                "opening_coordinates": fit["coordinates"],
                "absolute_rmse": fit["metrics"]["overall_rmse_log10_rate"],
                "shape_rmse": fit["metrics"]["shape_rmse_log10"],
            })
    front = []
    for row in rows:
        if not any(
            other["absolute_rmse"] <= row["absolute_rmse"]
            and other["shape_rmse"] <= row["shape_rmse"]
            and (other["absolute_rmse"] < row["absolute_rmse"]
                 or other["shape_rmse"] < row["shape_rmse"])
            for other in rows
        ):
            front.append(row)
    return sorted(front, key=lambda row: row["absolute_rmse"])


def exact_qualification(parent: Any, plastic: PlasticCoordinates,
                        opening: OpeningCoordinates, records: list[dict[str, Any]],
                        points: int, phases: int) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = run_native_history(
        parent, plastic, records, points_per_R=points, phase_count=phases,
        keep_engines=True,
    )
    surface = opening_surface(parent.manifest.cleavage.surface, opening)
    rows = []
    selected = trajectory["support_indices"]
    for r_value in R_VALUES:
        indices = selected[str(r_value)]
        probes = [indices[0], indices[len(indices)//2], indices[-1]]
        for index in probes:
            engine = copy.deepcopy(trajectory["saved_engines"][index])
            install_opening(engine, surface)
            row = records[index]
            waveform = FatigueWaveform(
                Kmax=row["DeltaK_MPa_sqrt_m"]/(1.0-row["R"])*1.0e6,
                R=row["R"], frequency_Hz=FREQUENCY_HZ,
            )
            resolution = {}
            for count in sorted({max(phases//2, 8), phases, phases*2}):
                result = one_cycle_map(engine, _NativePhaseController(count), waveform, TEMPERATURE_K)
                resolution[str(count)] = {
                    "prediction_m_per_cycle": EVENT_LENGTH_M*result.hazard_action_per_cycle,
                    "hazard_action_per_cycle": result.hazard_action_per_cycle,
                    "plastic_totals": result.plastic_totals,
                    "ledger_delta": result.ledger_delta_per_cycle,
                    "state_start": result.state_start.diagnostics,
                    "state_end": result.state_end.diagnostics,
                }
            approximate = predict(
                parent.manifest.cleavage.surface,
                np.asarray([
                    opening.floor_eV, opening.broad_amplitude_eV,
                    math.log(opening.broad_sigma_Pa), math.log(opening.broad_exponent),
                    opening.guard_amplitude_eV,
                ]),
                trajectory["stress"][index:index+1],
            )[0][0]
            rows.append({
                "R": row["R"], "source_order": row["source_order"],
                "DeltaK_MPa_sqrt_m": row["DeltaK_MPa_sqrt_m"],
                "approximate_prediction_m_per_cycle": float(approximate),
                "phase_resolution": resolution,
            })
    serializable = {key: value for key, value in trajectory.items()
                    if key not in {"saved_engines", "stress"}}
    return {"rows": rows, "phase_count": phases,
            "comparison": "phase-frozen production drive versus exact production Poincare observer"}, serializable


def fixed_kmax(records: list[dict[str, Any]], predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for target in (45.0, 48.0, 50.0):
        for r_value in R_VALUES:
            indices = [i for i, row in enumerate(records) if row["R"] == r_value]
            values = np.asarray([records[i]["DeltaK_MPa_sqrt_m"]/(1.0-r_value) for i in indices])
            index = indices[int(np.argmin(abs(values-target)))]
            row = records[index]
            rows.append({
                "target_Kmax_MPa_sqrt_m": target, "R": r_value,
                "source_series_id": row["series_id"], "source_order": row["source_order"],
                "reported_DeltaK_MPa_sqrt_m": row["DeltaK_MPa_sqrt_m"],
                "derived_Kmax_MPa_sqrt_m": row["DeltaK_MPa_sqrt_m"]/(1-r_value),
                "observed_dadn_m_per_cycle": row["observed_dadn_m_per_cycle"],
                **{name: float(values_[index]) for name, values_ in predictions.items()},
            })
    return rows


def make_plots(records: list[dict[str, Any]], finalists: dict[str, Any],
               states: list[dict[str, Any]], base_surface: Any) -> None:
    r_values = np.asarray([row["R"] for row in records])
    dk = np.asarray([row["DeltaK_MPa_sqrt_m"] for row in records])
    observed = np.asarray([row["observed_dadn_m_per_cycle"] for row in records])
    cases = [
        ("restored best absolute", np.asarray(finalists["absolute"]["prediction"]), "#0072B2", "-"),
        ("restored best shape", np.asarray(finalists["shape"]["prediction"]), "#D55E00", "--"),
        ("absolute frozen", np.asarray(finalists["absolute"]["frozen_prediction"]), "#56B4E9", ":"),
    ]
    for axis in ("DeltaK", "Kmax"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
        for panel, r_value in zip(axes.flat, R_VALUES):
            mask = r_values == r_value
            x = dk[mask] if axis == "DeltaK" else dk[mask]/(1-r_value)
            order = np.argsort(x)
            panel.scatter(x, observed[mask], s=10, c="#333333", alpha=.45, label="measured")
            for label, prediction, color, style in cases:
                panel.plot(x[order], prediction[mask][order], color=color, ls=style, lw=1.8, label=label)
            panel.set(xscale="log", yscale="log", title=f"R={r_value:.1f}",
                      xlabel=(r"$\Delta K$ (MPa$\sqrt{m}$)" if axis == "DeltaK" else r"$K_{max}$ (MPa$\sqrt{m}$)"))
            panel.grid(True, which="both", alpha=.2)
        axes[0,0].set_ylabel("da/dN (m/cycle)"); axes[1,0].set_ylabel("da/dN (m/cycle)")
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(.5,.98), frameon=False)
        fig.tight_layout(rect=(0,0,1,.92)); fig.savefig(OUT/f"restored_fit_{axis}.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for panel, r_value in zip(axes.flat, R_VALUES):
        mask = r_values == r_value
        for label, prediction, color, style in cases[:2]:
            residual = np.log10(prediction[mask]/observed[mask])
            panel.plot(dk[mask], residual, color=color, ls=style, lw=1.6, label=label)
        panel.axhline(0, color="#555555", lw=.8); panel.set(title=f"R={r_value:.1f}", xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)")
        panel.grid(True, alpha=.2)
    axes[0,0].set_ylabel("log10 prediction/data"); axes[1,0].set_ylabel("log10 prediction/data")
    fig.tight_layout(); fig.savefig(OUT/"restored_residuals.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for panel, r_value in zip(axes.flat, R_VALUES):
        mask = r_values == r_value
        x = np.asarray([row["source_order"] for row in records])[mask]
        mobile = np.asarray([row["mobile_count"] for row in states])[mask]
        retained = np.asarray([row["retained_count"] for row in states])[mask]
        slip = np.asarray([row["net_slip_count"] for row in states])[mask]
        returned = np.asarray([row["returned_slip_count"] for row in states])[mask]
        panel.plot(x, mobile, label="mobile"); panel.plot(x, retained, label="retained")
        panel.plot(x, slip, label="net slip"); panel.plot(x, returned, label="returned slip")
        panel.set(title=f"R={r_value:.1f}", xlabel="source order", yscale="symlog", ylabel="native count")
        panel.grid(True, alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(OUT/"restored_state_histories.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    stress = np.linspace(0.0, 30.0e9, 500)
    fig, axis = plt.subplots(figsize=(7, 5))
    for mode, color in (("absolute", "#0072B2"), ("shape", "#D55E00")):
        vector = np.asarray(finalists[mode]["opening_coordinates_vector"])
        surface = opening_surface(base_surface, unpack_opening(vector))
        axis.plot(stress/1e9, surface.G_eV(stress, TEMPERATURE_K), color=color, label=mode)
    axis.set(xlabel="production opening stress (GPa)", ylabel=r"$G^*$ (eV)", title="Restored-runtime alternative opening barriers at 298.15 K")
    axis.grid(True, alpha=.2); axis.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/"restored_Gstar.png", dpi=240, bbox_inches="tight"); plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-candidates", type=int, default=16)
    parser.add_argument("--local-candidates", type=int, default=4)
    parser.add_argument("--coarse-support", type=int, default=4)
    parser.add_argument("--coarse-phase", type=int, default=8)
    parser.add_argument("--refined-support", type=int, default=6)
    parser.add_argument("--refined-phase", type=int, default=16)
    parser.add_argument("--candidate-wall-s", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_records("index_rank", premeasurement=True)
    parent = load_parent_binding(ALT_REGISTRY, PARENT_REGISTRY, candidate_id=PARENT_CANDIDATE)
    initial = []
    for candidate_id, plastic in candidate_vectors(args.initial_candidates):
        result = evaluate_candidate(candidate_id, plastic, parent, records, args.coarse_support, args.coarse_phase, force=args.force, max_wall_s=args.candidate_wall_s)
        initial.append(result)
        atomic_json(OUT/"search_progress.json", {"stage":"initial", "completed":len(initial), "results":initial})
    complete = [row for row in initial if row.get("status") == "complete"]
    if not complete:
        raise RuntimeError("no initial native candidate completed")
    best_abs = min(complete, key=lambda row: row["fits"]["absolute"]["metrics"]["overall_rmse_log10_rate"])
    best_shape = min(complete, key=lambda row: row["fits"]["shape"]["metrics"]["shape_rmse_log10"])
    local = []
    for candidate_id, plastic in local_vectors([best_abs,best_shape], args.local_candidates):
        result = evaluate_candidate(candidate_id, plastic, parent, records, args.coarse_support, args.coarse_phase, force=args.force, max_wall_s=args.candidate_wall_s)
        local.append(result)
        atomic_json(OUT/"search_progress.json", {"stage":"local", "initial":initial, "local":local})
    all_results = initial+local
    complete = [row for row in all_results if row.get("status") == "complete"]
    best_abs = min(complete, key=lambda row: row["fits"]["absolute"]["metrics"]["overall_rmse_log10_rate"])
    best_shape = min(complete, key=lambda row: row["fits"]["shape"]["metrics"]["shape_rmse_log10"])

    refined = {}
    runtime_trajectories = {}
    for mode, selected in (("absolute",best_abs),("shape",best_shape)):
        plastic = PlasticCoordinates(**selected["plastic_coordinates"])
        candidate = evaluate_candidate(
            f"refined_{mode}_{selected['candidate_id']}", plastic, parent, records,
            args.refined_support, args.refined_phase, force=args.force,
            max_wall_s=max(args.candidate_wall_s*2, 900),
        )
        fit = candidate["fits"][mode]
        opening = OpeningCoordinates(**fit["coordinates"])
        qualification, runtime_trajectory = exact_qualification(
            parent, plastic, opening, records, args.refined_support, args.refined_phase,
        )
        prediction = np.asarray(fit["prediction"])
        frozen_prediction = predict(
            parent.manifest.cleavage.surface,
            np.asarray(fit["coordinates_vector"]),
            frozen_stress(records, args.refined_phase),
        )[0]
        refined[mode] = {
            "source_candidate_id": selected["candidate_id"],
            "plastic_coordinates": selected["plastic_coordinates"],
            "opening_coordinates": fit["coordinates"],
            "opening_coordinates_vector": fit["coordinates_vector"],
            "prediction": prediction.tolist(),
            "metrics": metrics(records,prediction),
            "frozen_prediction": frozen_prediction.tolist(),
            "frozen_metrics": metrics(records,frozen_prediction),
            "exact_native_qualification": qualification,
            "refined_candidate": candidate,
        }
        runtime_trajectories[mode] = runtime_trajectory

    # Decision-relevant history-coordinate sensitivities use the selected
    # absolute plastic vector without refitting plasticity.
    sensitivity = {}
    selected_plastic = PlasticCoordinates(**refined["absolute"]["plastic_coordinates"])
    for name, alignment, premeasurement in (
        ("deltaK_crack_length", "deltaK_crack_length", True),
        ("no_premeasurement", "index_rank", False),
    ):
        sensitivity_records = build_records(alignment, premeasurement=premeasurement)
        trajectory = run_native_history(
            parent, selected_plastic, sensitivity_records,
            points_per_R=args.coarse_support, phase_count=args.coarse_phase,
        )
        fit = fit_opening("absolute", parent.manifest.cleavage.surface, sensitivity_records, trajectory["stress"])
        fit["metrics"] = metrics(sensitivity_records,np.asarray(fit["prediction"]))
        sensitivity[name] = {"fit":fit,"support_rows":trajectory["support_rows"],"elapsed_s":trajectory["elapsed_s"]}

    predictions = {f"{mode}_prediction":np.asarray(value["prediction"]) for mode,value in refined.items()}
    contrasts = fixed_kmax(records,predictions)
    payload = {
        "schema": "qste340tm-restored-native-joint-refit/1",
        "starting_commit": "07ca1070",
        "temperature_K": TEMPERATURE_K,
        "frequency_Hz": FREQUENCY_HZ,
        "frequency_status": "provisional assumption",
        "event_length_m": EVENT_LENGTH_M,
        "runtime_model_id": MODEL_ID,
        "parent_candidate": PARENT_CANDIDATE,
        "history_alignment": {
            "primary": "exact source-index pairing for R=.1,.3,.7; normalized-rank interpolation for 171 versus 178 rows at R=.5",
            "sensitivity": "monotone DeltaK mapped to normalized measured crack length, then cycles interpolated on a(N)",
            "shared_identifier_available": False,
        },
        "search": {
            "initial_requested": args.initial_candidates,
            "local_requested": args.local_candidates,
            "completed": sum(row.get("status")=="complete" for row in all_results),
            "rejected": [row for row in all_results if row.get("status")!="complete"],
            "coarse_support_per_R": args.coarse_support,
            "coarse_phase_count": args.coarse_phase,
            "refined_support_per_R": args.refined_support,
            "refined_phase_count": args.refined_phase,
            "all_candidates": all_results,
            "pareto_front": pareto(all_results),
        },
        "finalists": refined,
        "runtime_trajectories": runtime_trajectories,
        "history_sensitivity": sensitivity,
        "fixed_Kmax_contrasts": contrasts,
        "claim_limit": "prescribed measured-history native replay; not autonomous crack-growth validation",
    }
    atomic_json(OUT/"restored_native_refit.json",payload)

    prediction_rows = []
    for index,row in enumerate(records):
        prediction_rows.append(dict(row)|{
            "absolute_prediction":refined["absolute"]["prediction"][index],
            "absolute_frozen":refined["absolute"]["frozen_prediction"][index],
            "shape_prediction":refined["shape"]["prediction"][index],
            "shape_frozen":refined["shape"]["frozen_prediction"][index],
        })
    write_csv(OUT/"restored_predictions.csv",prediction_rows)
    write_csv(OUT/"fixed_Kmax_contrasts.csv",contrasts)
    state_rows = []
    for row,state in zip(records,runtime_trajectories["absolute"]["states"]):
        state_rows.append({key:value for key,value in (dict(row)|dict(state)).items() if key!="interval_process"})
    write_csv(OUT/"restored_state_history.csv",state_rows)
    make_plots(records,refined,runtime_trajectories["absolute"]["states"],parent.manifest.cleavage.surface)
    print(json.dumps({mode:{"absolute_rmse":value["metrics"]["overall_rmse_log10_rate"],"shape_rmse":value["metrics"]["shape_rmse_log10"]} for mode,value in refined.items()},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
