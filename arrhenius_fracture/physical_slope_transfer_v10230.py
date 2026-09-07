"""Analysis-only diagnostics for the v10.2.30 physical slope transfer.

Archived observations and derived counterfactuals are kept in separate columns.
The module never instantiates or modifies the production solver.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.special import gammainc

from .inverse_fatigue_barrier_design_v10230 import (
    ExpFloorBounds,
    RenewalControls,
    barrier_from_vector,
    cycle_growth_and_slope,
)
from .material_manifest import ExpFloorBarrier, KB_EV_PER_K


MODEL_ID = "v10.2.30_physical_slope_transfer_analysis_v1"
R0_M = 1.0e-6
T_K = 300.0
F_HZ = 1000.0
HITS = 3.0
TAU_S = 1.0e-6
NU0_S = 1.0e12
DEVELOPMENT_M = 20.0e-6
TRANSFER_K_GRID = np.asarray((12.0, 12.75, 13.5, 15.0, 18.0, 21.0, 24.3))
CORRECTED_OPTION = "INV_PHYS_M4_TRANSFER_V1"


def read_json(path: Path):
    return json.loads(path.read_text())


def local_log_slopes(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    xv = np.asarray(tuple(x), dtype=float)
    yv = np.asarray(tuple(y), dtype=float)
    if xv.size < 3 or np.any(xv <= 0.0) or np.any(yv <= 0.0):
        raise ValueError("local logarithmic slopes require positive data")
    return np.gradient(np.log(yv), np.log(xv), edge_order=2)


def interval_log_slope(x0: float, x1: float, y0: float, y1: float) -> float:
    return math.log(float(y1) / float(y0)) / math.log(float(x1) / float(x0))


def exp_floor_barrier_eV(sigma_Pa: np.ndarray, row: pd.Series) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma_Pa, dtype=float), 0.0)
    G0 = float(row.cleave_G00_eV)
    floor = G0 * float(row.cleave_floor_frac)
    u = float(row.cleave_exp_a) * np.power(
        sigma / (float(row.cleave_sigc0_GPa) * 1.0e9), float(row.cleave_exp_n)
    )
    return floor + (G0 - floor) * np.exp(-u)


def raw_and_gamma_rates(sigma_Pa: np.ndarray, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    barrier = exp_floor_barrier_eV(sigma_Pa, row)
    raw = NU0_S * np.exp(-barrier / (KB_EV_PER_K * T_K))
    return raw, gammainc(HITS, raw * TAU_S) / TAU_S


def phase_fractions(R: float, n_phase: int = 4096) -> np.ndarray:
    phi = 2.0 * math.pi * (np.arange(n_phase, dtype=float) + 0.5) / n_phase
    signed = 0.5 * (1.0 + R) + 0.5 * (1.0 - R) * np.cos(phi)
    return np.maximum(signed, 0.0)


def reduced_held_state_action(
    Kmax_Pa_sqrt_m: float,
    R: float,
    archived_sigma_peak_Pa: float,
    archived_K_shield_Pa_sqrt_m: float,
    material: pd.Series,
) -> float:
    """Exact quadrature of an explicitly reduced, held-event-state mapping.

    This is not an exact replay of the missing phase-resolved MPZ trajectory.
    It preserves the archived peak stress and signed shielding, holds that state
    fixed through a cycle, and evaluates the immutable cleavage surface exactly.
    """
    h = phase_fractions(float(R))
    Kmax = float(Kmax_Pa_sqrt_m)
    shield = float(archived_K_shield_Pa_sqrt_m)
    peak = max(Kmax - shield, 1.0e-300)
    sigma = float(archived_sigma_peak_Pa) * np.maximum(Kmax * h - shield, 0.0) / peak
    _raw, gamma = raw_and_gamma_rates(sigma, material)
    return float(np.mean(gamma) / F_HZ)


def _history(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "high_cycle_live_history.jsonl").read_text().splitlines()]


def extract_event_history(source_root: Path) -> pd.DataFrame:
    registry = pd.read_csv(source_root / "inverse_design_candidate_registry.csv").set_index("option_key")
    records: list[dict] = []
    for run in sorted((source_root / "physical" / "developed").glob("INV_OPENING_M*_V1/K_*_R_0.1")):
        summary = read_json(run / "developed_fatigue_growth_summary.json")
        events = summary["event_measurements"]
        kinetic = read_json(run / "kinetic_tip_cell_audit_v101.json")["records"]
        history = _history(run)
        passages = [x for x in history if x.get("reason") == "first_passage"]
        starts = [x for x in history if x.get("reason") == "outer_driver_initial_committed_state"]
        starts += [x for x in history if x.get("reason") == "outer_driver_geometry_committed"]
        gates = read_json(run / "hazard_energy_gated_events_v10_2_30.json")
        geometry = read_json(run / "stochastic_avalanche_geometry_events.json")
        steps = pd.read_csv(run / "steps_0300K.csv")
        lengths = tuple(map(len, (events, kinetic, passages, starts, gates, geometry)))
        if lengths != (18, 18, 18, 18, 18, 18):
            raise RuntimeError(f"incomplete event archive {lengths}: {run}")
        option = str(events[0]["parameter_option"])
        material = registry.loc[option]
        M = int(option.split("_M", 1)[1].split("_", 1)[0])
        Kmax = float(events[0]["Kmax_MPa_sqrt_m"])
        R = float(events[0]["R"])
        for i, event in enumerate(events):
            audit, start, passage = kinetic[i], starts[i], passages[i]
            gate, geom = gates[i], geometry[i]
            sigma = float(passage["diagnostics"]["sigma_tip_Pa"])
            radius = float(audit["persistent_tip_radius_m"])
            shield = float(audit["state_active_K_shield_signed_Pa_sqrt_m"])
            raw, gamma = raw_and_gamma_rates(np.asarray([sigma]), material)
            radius_free_sigma = sigma * math.sqrt(max(radius / R0_M, 1.0e-300))
            zero_shield_scale = Kmax * 1.0e6 / max(Kmax * 1.0e6 - shield, 1.0e-300)
            records.append({
                "option_key": option, "M_target": M, "Kmax_MPa_sqrt_m": Kmax, "R": R,
                "event_index": i + 1,
                "developed_event": bool(float(event["projected_extension_post_m"]) > DEVELOPMENT_M),
                "cycles_between_events": float(event["cycles_between_events"]),
                "cycles_pre": float(event["cycles_pre"]), "cycles_post": float(event["cycles_post"]),
                "K_signed_max_Pa_sqrt_m": Kmax * 1.0e6,
                "K_signed_min_Pa_sqrt_m": R * Kmax * 1.0e6,
                "K_open_max_Pa_sqrt_m": Kmax * 1.0e6,
                "front_event_K_Pa_sqrt_m": float(gate["event_K_Pa_sqrt_m"]),
                "probe_K_Pa_sqrt_m": float(gate["probe_K_Pa_sqrt_m"]),
                "K_shield_pre_Pa_sqrt_m": float(audit["state_active_K_shield_signed_Pa_sqrt_m_pre"]),
                "K_shield_post_Pa_sqrt_m": shield,
                "r_eff_pre_m": float(start["diagnostics"]["tip_radius_m"]),
                "r_eff_post_m": radius,
                "sigma_back_pre_Pa": float(audit["state_sigma_back_Pa_pre"]),
                "sigma_back_post_Pa": float(audit["persistent_sigma_back_Pa"]),
                "cleavage_sigma_peak_Pa": sigma,
                "archived_raw_barrier_eV": math.nan,
                "archived_raw_cleavage_rate_s": math.nan,
                "derived_EXP_floor_barrier_peak_eV": float(exp_floor_barrier_eV(np.asarray([sigma]), material)[0]),
                "derived_raw_cleavage_rate_peak_s": float(raw[0]),
                "derived_gamma_cleavage_rate_peak_s": float(gamma[0]),
                "reduced_held_state_action_per_cycle": reduced_held_state_action(Kmax*1e6, R, sigma, shield, material),
                "archived_last_outer_action_block": float(event["physical_hazard_action"]),
                "archived_residual_action_pre": float(start["stochastic"]["hazard_action_current"]),
                "derived_total_interval_action_from_first_passage_clock": float(event["threshold_action"])-float(start["stochastic"]["hazard_action_current"]),
                "first_passage_threshold": float(event["threshold_action"]),
                "proposed_event_length_m": float(event["stochastic_proposed_advance_m"]),
                "energy_admissible_event_length_m": float(event["energy_admissible_advance_m"]),
                "accepted_projected_event_length_m": float(event["projected_advance_m"]),
                "accepted_path_event_length_m": float(event["path_advance_m"]),
                "event_phase_archived": False, "event_cycle": float(event["cycles_post"]),
                "pre_mobile_count": float(audit["state_mobile_count_pre"]),
                "post_mobile_count": float(audit["state_mobile_count"]),
                "pre_retained_count": float(audit["state_retained_count_pre"]),
                "post_retained_count": float(audit["state_retained_count"]),
                "geometry_commit_inserted": bool(event["geometry_commit_inserted"]),
                "geometry_transaction_mode": str(event["geometry_transaction_mode"]),
                "geometry_rebuild_invalidated_reason": str(passage["high_cycle_cache"].get("invalidated_reason")),
                "first_passage_action_closes_to_threshold": bool(math.isclose(float(event["threshold_action"])-float(start["stochastic"]["hazard_action_current"]), float(event["threshold_action"]), rel_tol=2e-12, abs_tol=2e-14)),
                "proposal_matches_gate": bool(math.isclose(float(event["stochastic_proposed_advance_m"]), float(event["energy_admissible_advance_m"]), rel_tol=2e-12, abs_tol=2e-14)),
                "gate_matches_path_commit": bool(math.isclose(float(event["energy_admissible_advance_m"]), float(event["path_advance_m"]), rel_tol=2e-12, abs_tol=2e-14)),
                "csv_raw_barrier_is_zero_placeholder": bool(float(steps.iloc[i]["G_cleave_raw_eV"]) == 0.0),
                "csv_shield_is_zero_placeholder": bool(float(steps.iloc[i]["mpz_K_shield_Pa_sqrt_m"]) == 0.0),
                "diagnostic_freeze_r_eff_action": reduced_held_state_action(Kmax*1e6, R, radius_free_sigma, shield, material),
                "diagnostic_zero_K_shield_action": reduced_held_state_action(Kmax*1e6, R, sigma*zero_shield_scale, 0.0, material),
                "diagnostic_freeze_emission_action": reduced_held_state_action(Kmax*1e6, R, radius_free_sigma*zero_shield_scale, 0.0, material),
                "run_path": str(run.resolve()),
            })
    result = pd.DataFrame(records).sort_values(["M_target", "Kmax_MPa_sqrt_m", "event_index"])
    if len(result) != 378:
        raise RuntimeError(f"expected 378 events, found {len(result)}")
    return result


def _reduced_replay(group: pd.DataFrame, action: str, length: str, expected_xi: bool = False) -> float:
    action_values = np.maximum(group[action].to_numpy(dtype=float), 1.0e-300)
    thresholds = np.ones(len(group)) if expected_xi else group.first_passage_threshold.to_numpy(dtype=float)
    return float(group[length].sum() / np.sum(thresholds / action_values))


def aggregate_stage_points(events: pd.DataFrame, prospective: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (option, K), all_events in events.groupby(["option_key", "Kmax_MPa_sqrt_m"], sort=True):
        group = all_events[all_events.developed_event]
        prediction = prospective[(prospective.option_key == option) & np.isclose(prospective.Kmax_MPa_sqrt_m, K) & np.isclose(prospective.R, 0.1)]
        if group.empty or len(prediction) != 1:
            raise RuntimeError(f"incomplete developed/prediction data for {option} K={K}")
        cycles = float(group.cycles_between_events.sum())
        accepted = float(group.accepted_projected_event_length_m.sum())
        n_events = len(group)
        rows.append({
            "option_key": option, "M_target": int(group.M_target.iloc[0]), "Kmax_MPa_sqrt_m": float(K),
            "developed_event_count": n_events,
            "T0_exact_A0_da_dN": float(prediction.iloc[0].EXP_floor_A0_da_dN),
            "D1_reduced_archived_peak_state_da_dN": _reduced_replay(group, "reduced_held_state_action_per_cycle", "accepted_projected_event_length_m", True),
            "D2_reduced_archived_threshold_da_dN": _reduced_replay(group, "reduced_held_state_action_per_cycle", "accepted_projected_event_length_m"),
            "T4_archived_identity_da_dN": accepted / cycles,
            "archived_action_per_cycle": float(group.first_passage_threshold.sum() / cycles),
            "archived_first_passage_frequency": n_events / cycles,
            "archived_accepted_event_frequency": int((group.accepted_projected_event_length_m > 0).sum()) / cycles,
            "mean_accepted_event_length_m": float(group.accepted_projected_event_length_m.mean()),
            "cycle_weighted_r_eff_m": float(np.average(group.r_eff_post_m, weights=group.cycles_between_events)),
            "event_mean_r_eff_m": float(group.r_eff_post_m.mean()),
            "mean_sigma_peak_Pa": float(group.cleavage_sigma_peak_Pa.mean()),
            "mean_derived_raw_rate_peak_s": float(group.derived_raw_cleavage_rate_peak_s.mean()),
            "mean_derived_gamma_rate_peak_s": float(group.derived_gamma_cleavage_rate_peak_s.mean()),
            "mean_K_shield_Pa_sqrt_m": float(group.K_shield_post_Pa_sqrt_m.mean()),
            "diagnostic_freeze_r_eff_da_dN": _reduced_replay(group, "diagnostic_freeze_r_eff_action", "accepted_projected_event_length_m"),
            "diagnostic_zero_K_shield_da_dN": _reduced_replay(group, "diagnostic_zero_K_shield_action", "accepted_projected_event_length_m"),
            "diagnostic_freeze_emission_da_dN": _reduced_replay(group, "diagnostic_freeze_emission_action", "accepted_projected_event_length_m"),
            "diagnostic_freeze_event_length_da_dN": float(group.accepted_projected_event_length_m.mean() * n_events / cycles),
            "diagnostic_remove_energy_truncation_da_dN": float(np.sum(
                group.proposed_event_length_m
                * group.accepted_projected_event_length_m
                / np.maximum(group.energy_admissible_event_length_m, 1.0e-300)
            ) / cycles),
            "diagnostic_expected_threshold_da_dN": _reduced_replay(group, "reduced_held_state_action_per_cycle", "accepted_projected_event_length_m", True),
            "all_event_window_da_dN": float(all_events.accepted_projected_event_length_m.sum() / all_events.cycles_between_events.sum()),
        })
    points = pd.DataFrame(rows).sort_values(["M_target", "Kmax_MPa_sqrt_m"])
    slope_inputs = [name for name in points if name.endswith("da_dN") or name in {
        "archived_action_per_cycle", "archived_first_passage_frequency",
        "archived_accepted_event_frequency", "mean_accepted_event_length_m",
        "cycle_weighted_r_eff_m", "mean_sigma_peak_Pa",
        "mean_derived_raw_rate_peak_s", "mean_derived_gamma_rate_peak_s",
    }]
    for _M, group in points.groupby("M_target"):
        idx, K = group.index, group.Kmax_MPa_sqrt_m.to_numpy(dtype=float)
        for name in slope_inputs:
            points.loc[idx, f"m_{name}"] = local_log_slopes(K, group[name])
    points["A_phys"] = points.m_T4_archived_identity_da_dN / points.m_T0_exact_A0_da_dN
    points["A_reduced_peak_state"] = points.m_D1_reduced_archived_peak_state_da_dN / points.m_T0_exact_A0_da_dN
    points["m_K_to_sigma"] = points.m_mean_sigma_peak_Pa
    points["m_sigma_to_raw"] = points.m_mean_derived_raw_rate_peak_s / points.m_mean_sigma_peak_Pa
    points["m_raw_to_gamma"] = points.m_mean_derived_gamma_rate_peak_s / points.m_mean_derived_raw_rate_peak_s
    points["m_gamma_to_action"] = points.m_archived_action_per_cycle / points.m_mean_derived_gamma_rate_peak_s
    points["m_action_to_first_passage"] = points.m_archived_first_passage_frequency / points.m_archived_action_per_cycle
    points["m_first_passage_to_accepted"] = points.m_archived_accepted_event_frequency / points.m_archived_first_passage_frequency
    points["m_accepted_to_da_dN"] = points.m_T4_archived_identity_da_dN / points.m_archived_accepted_event_frequency
    return points


def cross_target_fit(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for K, group in points.groupby("Kmax_MPa_sqrt_m"):
        by_M = group.set_index("M_target")
        x2, x6 = float(by_M.loc[2].m_T0_exact_A0_da_dN), float(by_M.loc[6].m_T0_exact_A0_da_dN)
        y2, y6 = float(by_M.loc[2].m_T4_archived_identity_da_dN), float(by_M.loc[6].m_T4_archived_identity_da_dN)
        A = (y6 - y2) / (x6 - x2)
        B = y2 - A * x2
        held = float(by_M.loc[4].m_T4_archived_identity_da_dN)
        predicted = A * float(by_M.loc[4].m_T0_exact_A0_da_dN) + B
        rows.append({"Kmax_MPa_sqrt_m": float(K), "A_K_from_M2_M6": A, "B_K_from_M2_M6": B,
                     "M4_heldout_physical_slope": held, "M4_heldout_predicted_slope": predicted,
                     "M4_heldout_residual": predicted-held, "A_phys_spread": float(group.A_phys.max()-group.A_phys.min())})
    return pd.DataFrame(rows).sort_values("Kmax_MPa_sqrt_m")


def seed_check_predictions(points: pd.DataFrame) -> pd.DataFrame:
    group = points[points.M_target == 4].set_index("Kmax_MPa_sqrt_m")
    rows = []
    for lo, hi in ((12.0, 18.0), (18.0, 24.3)):
        m0 = interval_log_slope(lo, hi, group.loc[lo].T0_exact_A0_da_dN, group.loc[hi].T0_exact_A0_da_dN)
        mp = interval_log_slope(lo, hi, group.loc[lo].T4_archived_identity_da_dN, group.loc[hi].T4_archived_identity_da_dN)
        rows.append({"seed":1001723,"M_target":4,"K_low_MPa_sqrt_m":lo,"K_high_MPa_sqrt_m":hi,
                     "reference_seed":1720,"reference_A_interval":mp/m0,
                     "reference_physical_interval_slope":mp,"reference_A0_interval_slope":m0,
                     "maximum_allowed_absolute_A_difference":0.15})
    return pd.DataFrame(rows)


def second_seed_results(physical_root: Path, frozen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the three fresh prospective controls without changing the transfer fit."""
    rows = []
    for K in (12.0, 18.0, 24.3):
        run = physical_root / "INV_OPENING_M4_V1" / f"K_{K:g}_R_0.1_seed_1001723"
        summary_path = run / "developed_fatigue_growth_summary.json"
        manifest_path = run / "high_cycle_run_manifest.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"missing terminal second-seed record: {run}")
        summary, manifest = read_json(summary_path), read_json(manifest_path)
        developed = summary.get("developed_interval") or {}
        environment = manifest.get("environment") or {}
        if not summary.get("target_reached") or int(developed.get("event_count", 0)) < 1:
            raise RuntimeError(f"nonterminal second-seed trajectory: {run}")
        rows.append({
            "option_key": "INV_OPENING_M4_V1", "Kmax_MPa_sqrt_m": K,
            "R": 0.1, "seed": int(manifest["hazard_seed"]),
            "n_bins": 80, "fresh": True,
            "resume_environment_present": "V10230_RESTART_CHECKPOINT_DIR" in environment,
            "solver_head": str(manifest["git_head"]),
            "target_reached": bool(summary["target_reached"]),
            "cycles_consumed": float(summary["cycles_consumed"]),
            "event_count": int(summary["event_count"]),
            "developed_event_count": int(developed["event_count"]),
            "developed_da_dN": float(developed["da_dN"]),
            "result_path": str(run.resolve()),
        })
    results = pd.DataFrame(rows).sort_values("Kmax_MPa_sqrt_m")
    if not results.seed.eq(1001723).all() or results.resume_environment_present.any():
        raise RuntimeError("second-seed provenance is not fresh")
    values = results.set_index("Kmax_MPa_sqrt_m")
    checks = []
    for prediction in frozen.itertuples(index=False):
        measured_slope = interval_log_slope(
            prediction.K_low_MPa_sqrt_m, prediction.K_high_MPa_sqrt_m,
            values.loc[prediction.K_low_MPa_sqrt_m].developed_da_dN,
            values.loc[prediction.K_high_MPa_sqrt_m].developed_da_dN,
        )
        measured_A = measured_slope / float(prediction.reference_A0_interval_slope)
        difference = measured_A - float(prediction.reference_A_interval)
        checks.append({
            **prediction._asdict(),
            "measured_physical_interval_slope": measured_slope,
            "measured_A_interval": measured_A,
            "A_interval_difference_from_frozen_reference": difference,
            "absolute_A_interval_difference": abs(difference),
            "acceptance_pass": abs(difference) <= float(prediction.maximum_allowed_absolute_A_difference),
            "operator_refit_performed": False,
        })
    validation = pd.DataFrame(checks)
    return results, validation


def _integrated_growth(K: np.ndarray, slope: np.ndarray, anchor_K: float,
                       anchor_growth: float) -> np.ndarray:
    logK = np.log(np.asarray(K, dtype=float))
    integral = cumulative_trapezoid(np.asarray(slope, dtype=float), logK, initial=0.0)
    anchor = int(np.flatnonzero(np.isclose(K, anchor_K))[0])
    return np.exp(math.log(float(anchor_growth)) + integral - integral[anchor])


def _candidate_evaluations(vector: np.ndarray, template: ExpFloorBarrier,
                           K: np.ndarray, R: float = 0.1) -> tuple[ExpFloorBarrier, list[dict[str, float]]]:
    barrier = barrier_from_vector(vector, template)
    controls = RenewalControls()
    return barrier, [cycle_growth_and_slope(barrier, float(load), R, controls) for load in K]


def corrected_inverse_design(points: pd.DataFrame, fit: pd.DataFrame,
                             source_registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict, pd.DataFrame]:
    """Invert the frozen physical transfer and project it into EXP-floor.

    The empirical M2/M6 fit is retained as a diagnostic.  Candidate construction
    uses only the independently identified K-to-stress map, m_phys=m_A0*m_Ksigma.
    """
    m4 = points[points.M_target == 4].sort_values("Kmax_MPa_sqrt_m").reset_index(drop=True)
    fit = fit.sort_values("Kmax_MPa_sqrt_m").reset_index(drop=True)
    K = m4.Kmax_MPa_sqrt_m.to_numpy(dtype=float)
    if not np.allclose(K, TRANSFER_K_GRID):
        raise RuntimeError("corrected inverse grid differs from frozen transfer grid")
    empirical_required = (4.0 - fit.B_K_from_M2_M6.to_numpy(dtype=float)) / fit.A_K_from_M2_M6.to_numpy(dtype=float)
    reduced_required = 4.0 / m4.m_K_to_sigma.to_numpy(dtype=float)
    anchor_index = int(np.flatnonzero(np.isclose(K, 18.0))[0])
    target_A0_growth = _integrated_growth(
        K, reduced_required, 18.0, float(m4.T0_exact_A0_da_dN.iloc[anchor_index])
    )
    target_physical_growth = float(m4.T4_archived_identity_da_dN.iloc[anchor_index]) * (K / 18.0) ** 4.0
    original = source_registry[source_registry.option_key == "INV_OPENING_M4_V1"].iloc[0]
    template = ExpFloorBarrier(
        G00_eV=float(original.cleave_G00_eV), gT_eV_per_K=float(original.cleave_gT_eV_per_K),
        sigc0_Pa=float(original.cleave_sigc0_GPa) * 1.0e9,
        sT_Pa_per_K=float(original.cleave_sT_GPa_per_K) * 1.0e9,
        alpha=float(original.cleave_exp_a), exponent=float(original.cleave_exp_n),
        floor_fraction=float(original.cleave_floor_frac), attempt_frequency_s=NU0_S,
    )
    bounds = ExpFloorBounds()
    starts = [
        [template.G00_eV, template.sigc0_Pa / 1e9, template.alpha, template.exponent, template.floor_fraction],
        [0.46, 7.4, 0.122, 1.33, 0.001], [0.48, 8.0, 0.13, 1.29, 0.001],
        [0.50, 8.0, 0.10, 1.00, 0.001], [0.80, 8.0, 0.05, 2.00, 0.001],
        [1.20, 6.0, 0.05, 2.50, 0.02], [0.60, 3.0, 0.30, 1.25, 0.01],
    ]

    def residual(vector: np.ndarray) -> np.ndarray:
        _barrier, evaluated = _candidate_evaluations(vector, template, K)
        growth = np.asarray([item["da_dN"] for item in evaluated])
        slope = np.asarray([item["local_slope"] for item in evaluated])
        return np.r_[np.log(growth) - np.log(target_A0_growth), slope - reduced_required]

    trials = [least_squares(
        residual, np.clip(np.asarray(start), bounds.lower(), bounds.upper()),
        bounds=(bounds.lower(), bounds.upper()), xtol=1e-12, ftol=1e-12,
        gtol=1e-12, max_nfev=3000,
    ) for start in starts]
    solution = min(trials, key=lambda trial: float(np.dot(residual(trial.x), residual(trial.x))))
    barrier, evaluated = _candidate_evaluations(solution.x, template, K)
    A0_growth = np.asarray([item["da_dN"] for item in evaluated])
    A0_slope = np.asarray([item["local_slope"] for item in evaluated])
    old_rate_scale = (
        m4.T4_archived_identity_da_dN.to_numpy(dtype=float) /
        m4.T0_exact_A0_da_dN.to_numpy(dtype=float)
    )
    predicted_physical_growth = old_rate_scale * A0_growth
    predicted_empirical_slope = (
        fit.A_K_from_M2_M6.to_numpy(dtype=float) * A0_slope +
        fit.B_K_from_M2_M6.to_numpy(dtype=float)
    )
    predicted_reduced_slope = m4.m_K_to_sigma.to_numpy(dtype=float) * A0_slope
    design = pd.DataFrame({
        "Kmax_MPa_sqrt_m": K,
        "desired_physical_slope": 4.0,
        "empirical_fit_required_A0_slope": empirical_required,
        "reduced_operator_required_A0_slope": reduced_required,
        "required_slope_difference_empirical_minus_reduced": empirical_required - reduced_required,
        "target_A0_da_dN": target_A0_growth,
        "projected_EXP_floor_A0_da_dN": A0_growth,
        "projected_EXP_floor_A0_local_slope": A0_slope,
        "target_physical_da_dN": target_physical_growth,
        "predicted_physical_da_dN": predicted_physical_growth,
        "predicted_physical_slope_empirical_fit": predicted_empirical_slope,
        "predicted_physical_slope_reduced_operator": predicted_reduced_slope,
        "frozen_original_physical_to_A0_rate_ratio": old_rate_scale,
        "frozen_m_K_to_sigma": m4.m_K_to_sigma.to_numpy(dtype=float),
        "frozen_A_fit": fit.A_K_from_M2_M6.to_numpy(dtype=float),
        "frozen_B_fit": fit.B_K_from_M2_M6.to_numpy(dtype=float),
    })
    row = original.copy()
    row["option_key"] = CORRECTED_OPTION
    row["candidate_id"] = CORRECTED_OPTION
    row["role"] = "prospective physical-transfer-corrected opening barrier"
    row["mechanism_summary"] = "M=4 physical slope target; frozen reduced K-to-stress transfer; A_NATIVE common physics"
    row["validation_status"] = "PROSPECTIVE_FROZEN"
    row["cleave_G00_eV"] = barrier.G00_eV
    row["cleave_sigc0_GPa"] = barrier.sigc0_Pa / 1.0e9
    row["cleave_exp_a"] = barrier.alpha
    row["cleave_exp_n"] = barrier.exponent
    row["cleave_floor_frac"] = barrier.floor_fraction
    slope_rms = float(np.sqrt(np.mean((A0_slope - reduced_required) ** 2)))
    physical_rms = float(np.sqrt(np.mean((predicted_reduced_slope - 4.0) ** 2)))
    physical_max = float(np.max(np.abs(predicted_reduced_slope - 4.0)))
    sigma_knee = barrier.sigc0_Pa * (1.0 / barrier.alpha) ** (1.0 / barrier.exponent)
    K_knee = sigma_knee * math.sqrt(2.0 * math.pi * R0_M) / 1.0e6
    classification = (
        "EXP_FLOOR_CAN_COMPENSATE_PHYSICAL_TRANSFER"
        if slope_rms <= 0.75 and physical_rms <= 0.50 and physical_max <= 0.75
        else "EXP_FLOOR_TOO_RESTRICTIVE_AFTER_TRANSFER"
    )
    summary = {
        "schema": "v10.2.30_corrected_physical_transfer_candidate_v1",
        "option_key": CORRECTED_OPTION,
        "classification": classification,
        "construction_operator": "m_physical=m_A0*m_K_to_sigma (reduced physical operator)",
        "empirical_fit_role": "diagnostic comparison only; not used in candidate optimization",
        "target_physical_slope": 4.0,
        "projection_A0_slope_RMSE": slope_rms,
        "predicted_reduced_physical_slope_RMSE": physical_rms,
        "predicted_reduced_physical_slope_maximum_absolute_error": physical_max,
        "projection_log_A0_growth_RMSE": float(np.sqrt(np.mean((np.log(A0_growth)-np.log(target_A0_growth))**2))),
        "optimizer_success": bool(solution.success), "optimizer_nfev": int(solution.nfev),
        "EXP_floor_lower_bound_active": bool(math.isclose(barrier.floor_fraction, bounds.floor_fraction[0], rel_tol=0, abs_tol=1e-8)),
        "analytical_knee_Kmax_MPa_sqrt_m": K_knee,
        "high_K_behavior": "bounded barrier reaches its floor; cooperative hazard saturates and local slope tends to zero",
        "minimum_additional_flexibility_if_exact_profile_required": "second bounded EXP component or integrated-logistic slope window",
        "changed_constitutive_fields": ["cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a", "cleave_exp_n", "cleave_floor_frac"],
        "noncleavage_common_physics_unchanged": True,
    }
    R_rows = []
    q18 = float(old_rate_scale[anchor_index])
    for R in (-0.95, 0.1, 0.5):
        item = cycle_growth_and_slope(barrier, 18.0, R, RenewalControls())
        R_rows.append({
            "Kmax_MPa_sqrt_m": 18.0, "R": R,
            "analytical_A0_da_dN": item["da_dN"], "analytical_A0_local_slope": item["local_slope"],
            "conditional_physical_da_dN_if_R01_rate_scale_held": q18 * item["da_dN"],
            "physical_transfer_at_this_R_validated": R == 0.1,
            "material_barrier_retuned_by_R": False,
        })
    return design, row, summary, pd.DataFrame(R_rows)


__all__ = ["CORRECTED_OPTION", "MODEL_ID", "TRANSFER_K_GRID", "aggregate_stage_points",
           "corrected_inverse_design", "cross_target_fit", "extract_event_history",
           "interval_log_slope", "local_log_slopes", "second_seed_results",
           "seed_check_predictions"]
