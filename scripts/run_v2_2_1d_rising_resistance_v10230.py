#!/usr/bin/env python3
"""Run the frozen reload-separated reduced 1-D resistance screen."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes

from arrhenius_fracture.canonical_v2_registry_v10230 import REGISTRY, load_rows
from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import write_checkpoint
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import capture_ledgers
import scripts.complete_corrected_thermodynamic_joint_search_v10230 as v2


OUT = ROOT / "analysis_outputs/v2_2_1d_rising_resistance_and_fatigue"
T_K = 900.0
KDOT = 0.005
K_CENSOR = 80.0
XI = math.log(2.0)
DK = 0.25
EVENT_M = 5.0e-6
MAX_EVENTS = 10


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def state_record(engine) -> dict[str, float]:
    result = v2.snapshot_summary(engine)
    result.update({
        "time_s": float(engine.t),
        "n_adv": int(engine.n_adv),
        "micro_advance_m": float(engine.micro_advance_total_m),
        "checkpoint_advance_m": float(engine.checkpoint_advance_total_m),
        "hazard_action": float(engine.hazard_action_current),
        "threshold_action": float(engine.hazard_threshold_action),
    })
    return result


def conservation(state: dict[str, float]) -> float:
    accounted = sum(state[name] for name in (
        "mobile", "retained", "wake_mobile", "wake_retained", "escaped", "recovered"
    ))
    return abs(accounted - state["emitted_total"]) / max(state["emitted_total"], 1.0)


def prepare_engine(row: dict[str, str], mode: str):
    # The authoritative loader preserves CSV field bytes.  The inherited V2
    # constructor consumes pandas' explicit numeric parsing, as it did in V2.1.
    table = pd.read_csv(REGISTRY).set_index("candidate_id", drop=False)
    candidate = table.loc[row["candidate_id"]].to_dict()
    engine, cleavage, emission, parent = v2.build_engine(candidate, v2.ROWS, v2.ACTIVE)
    # The frozen screen uses the accepted fixed 5 um reduced convention.
    engine.avalanche_cfg.mode = "fixed"
    engine.f.da = EVENT_M
    engine.hazard_threshold_action = XI
    engine.hazard_action_current = 0.0
    engine.B = 0.0
    engine._set_current_event_length()
    if mode == "OPENING_ONLY_FROZEN_STATE":
        # Existing qualified diagnostic flag; the opening barrier is untouched.
        engine.tip_cfg.plasticity_enabled = False
        engine.tip_cfg.active_shielding = False
    return engine, cleavage, emission, parent


def run_case(row: dict[str, str], mode: str, output: Path) -> tuple[list[dict], list[dict], dict]:
    engine, cleavage, _emission, parent = prepare_engine(row, mode)
    events: list[dict] = []
    states: list[dict] = []
    status = "COMPLETE"
    reason = ""
    maximum_conservation = 0.0
    for event_index in range(MAX_EVENTS):
        # Complete unload with zero dwell. Calling step makes the applied state
        # explicit without advancing time or physical state.
        engine.step(0.0, T_K, 0.0)
        engine.hazard_threshold_action = XI
        engine.hazard_action_current = 0.0
        engine.B = 0.0
        engine._set_current_event_length()
        initial_time = float(engine.t)
        before_reload = state_record(engine)
        before_ledgers = capture_ledgers(engine)
        last_left = 0.0
        final_info = None
        try:
            for left in np.arange(0.0, K_CENSOR, DK):
                mid = float(left + 0.5 * DK)
                dt = DK / KDOT
                engine.mpz._reversible_transport_K_signed_Pa_sqrt_m = mid * 1.0e6
                info = engine.step(mid * 1.0e6, T_K, dt)
                if bool(info.get("fired", False)):
                    final_info = info
                    last_left = float(left)
                    break
        except Exception as exc:  # fail-closed terminal record
            status = "STATE_OR_NUMERICAL_FAILURE"
            reason = f"{type(exc).__name__}: {exc}"
            break
        if final_info is None:
            status = "K_CENSOR_REACHED"
            reason = "no first passage below 80 MPa sqrt(m)"
            break
        consumed = float(final_info.get("kinetic_dt_consumed_s", final_info.get("dt_consumed", DK / KDOT)))
        K_event = last_left + KDOT * min(max(consumed, 0.0), DK / KDOT)
        after = state_record(engine)
        cres = conservation(after)
        maximum_conservation = max(maximum_conservation, cres)
        raw_rate = float(final_info.get("lambda_c_raw", 0.0))
        ceiling = bool(raw_rate * float(parent["physics__cleavage_correlation_time_s"]) >= 20.0)
        event = {
            "candidate_id": row["candidate_id"], "parent_id": row["parent_id"],
            "mode": mode, "temperature_K": T_K, "event_index": event_index,
            "cumulative_advance_um": (event_index + 1) * 5.0,
            "event_advance_um": 5.0, "K_reinit_MPa_sqrt_m": K_event,
            "loading_time_s": float(engine.t) - initial_time,
            "threshold_action": XI,
            "physical_hazard_action": float(final_info.get("hazard_action_completed", XI)),
            "elementary_cleavage_rate_s": raw_rate,
            "constitutive_ceiling_occupied": ceiling,
            "sigma_cap_active": bool(final_info.get("sigma_cap_active", False)),
            "dN_cap_active": bool(final_info.get("dN_cap_active", False)),
            "N_sat_active": bool(final_info.get("N_sat_active", False)),
            "conservation_relative": cres,
            "complete_bound_row_sha256": row["complete_bound_row_sha256"],
            **after,
        }
        event["J_equiv_J_m2"] = (K_event * 1.0e6) ** 2 / (
            float(engine.G) * 2.0 * (1.0 + float(engine.nu)) / (1.0 - float(engine.nu) ** 2)
        )
        events.append(event)
        states.extend([
            {"candidate_id": row["candidate_id"], "mode": mode,
             "event_index": event_index, "state_position": "PRE_RELOAD", **before_reload},
            {"candidate_id": row["candidate_id"], "mode": mode,
             "event_index": event_index, "state_position": "POST_EVENT", **after,
             "ledger_delta_json": json.dumps({k: capture_ledgers(engine).get(k, 0.0) - value
                                                for k, value in before_ledgers.items()}, sort_keys=True)},
        ])
        checkpoint_dir = output / "checkpoints" / row["candidate_id"] / mode
        old = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR")
        os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(checkpoint_dir)
        try:
            write_checkpoint(engine, temperature_K=T_K, reason="first_passage", metadata={
                "candidate_id": row["candidate_id"], "mode": mode,
                "event_index": event_index, "K_reinit_MPa_sqrt_m": K_event,
            })
        finally:
            if old is None:
                os.environ.pop("V10230_HIGH_CYCLE_CHECKPOINT_DIR", None)
            else:
                os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = old
    terminal = {
        "candidate_id": row["candidate_id"], "mode": mode, "status": status,
        "reason": reason, "events": len(events),
        "cumulative_advance_um": 5.0 * len(events),
        "maximum_conservation_relative": maximum_conservation,
        "exact_renewal_m": float(parent["physics__cleavage_hits"]),
        "exact_renewal_tau_s": float(parent["physics__cleavage_correlation_time_s"]),
        "opening_surface_sha256": row["opening_surface_sha256"],
        "emission_surface_sha256": row["emission_surface_sha256"],
    }
    atomic_json(output / "case_records" / row["candidate_id"] / f"{mode}.json",
                {"terminal": terminal, "events": events, "states": states})
    return events, states, terminal


def classify(points: pd.DataFrame, opening: pd.DataFrame) -> dict:
    candidate_id = str(points.candidate_id.iloc[0])
    K = points.K_reinit_MPa_sqrt_m.to_numpy(float)
    advance = points.cumulative_advance_um.to_numpy(float)
    slope = float(theilslopes(K, advance).slope) if len(K) >= 2 else math.nan
    rho = float(spearmanr(advance, K).statistic) if len(K) >= 2 else math.nan
    increments = np.diff(K)
    nonnegative = int(np.count_nonzero(increments >= -1.0e-12))
    rise = float(K[-1] - K[0]) if len(K) else math.nan
    rise_fraction = rise / K[0] if len(K) and K[0] > 0.0 else math.nan
    Ko = opening.K_reinit_MPa_sqrt_m.to_numpy(float)
    opening_rise = float(Ko[-1] - Ko[0]) if len(Ko) else math.nan
    state_rise = rise - opening_rise
    state_fraction = state_rise / rise if rise > 0.0 else math.nan
    pathology = bool(points[["constitutive_ceiling_occupied", "sigma_cap_active",
                              "dN_cap_active", "N_sat_active"]].any().any()
                     or points.conservation_relative.max() >= 1.0e-7)
    transient = len(K) >= 4 and np.max(K[:2] - K[0]) > 0 and K[-1] < np.max(K[:2])
    strong = bool(len(K) >= 8 and slope > 0 and rho >= 0.8 and nonnegative >= 7
                  and rise_fraction >= 0.2 and rise > opening_rise
                  and state_fraction >= 0.5 and not pathology)
    moderate = bool(len(K) >= 8 and slope > 0 and rise_fraction >= 0.1
                    and rise > opening_rise and not pathology)
    if strong: label = "1D_STRONG_RISING_RESISTANCE"
    elif moderate: label = "1D_MODERATE_RISING_RESISTANCE"
    elif transient: label = "1D_TRANSIENT_RISE_NOT_SUSTAINED"
    elif not np.all(np.isfinite(K)) or len(K) < 8 or pathology: label = "1D_EFFECTIVE_RESISTANCE_UNRESOLVED"
    elif rise < -1.0e-10: label = "1D_EFFECTIVE_RESISTANCE_SOFTENING"
    else: label = "1D_EFFECTIVE_RESISTANCE_FLAT"
    return {
        "candidate_id": candidate_id, "classification": label,
        "finite_points": int(len(K)), "theil_sen_slope_MPa_sqrt_m_per_um": slope,
        "spearman_rho": rho, "nonnegative_successive_increments": nonnegative,
        "K_first_MPa_sqrt_m": float(K[0]) if len(K) else math.nan,
        "K_last_MPa_sqrt_m": float(K[-1]) if len(K) else math.nan,
        "total_rise_MPa_sqrt_m": rise, "normalized_rise": rise_fraction,
        "opening_only_rise_MPa_sqrt_m": opening_rise,
        "state_mediated_rise_MPa_sqrt_m": state_rise,
        "state_mediated_fraction": state_fraction,
        "numerical_or_constitutive_pathology": pathology,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.out.resolve(); output.mkdir(parents=True, exist_ok=True)
    v2.ROWS = v2.source_rows(); v2.ACTIVE = v2.active_stresses(v2.ROWS)
    rows = load_rows()
    all_events, all_states, terminals = [], [], []
    for candidate_id, row in rows.items():
        for mode in ("FULL_PRODUCTION_STATE", "OPENING_ONLY_FROZEN_STATE"):
            print(f"SCREEN_START {candidate_id} {mode}", flush=True)
            events, states, terminal = run_case(row, mode, output)
            all_events += events; all_states += states; terminals.append(terminal)
            print(f"SCREEN_END {candidate_id} {mode} {terminal['status']} events={len(events)}", flush=True)
    points = pd.DataFrame(all_events)
    states = pd.DataFrame(all_states)
    points.to_csv(output / "one_d_reload_separated_resistance_points.csv", index=False)
    states.to_parquet(output / "one_d_reload_separated_state_table.parquet", index=False, compression="zstd")
    classes = []
    for candidate_id in rows:
        full = points[(points.candidate_id == candidate_id) & (points["mode"] == "FULL_PRODUCTION_STATE")]
        opening = points[(points.candidate_id == candidate_id) & (points["mode"] == "OPENING_ONLY_FROZEN_STATE")]
        if len(full): classes.append(classify(full, opening))
        else: classes.append({"candidate_id": candidate_id, "classification": "1D_EFFECTIVE_RESISTANCE_UNRESOLVED", "finite_points": 0})
    classification = pd.DataFrame(classes)
    classification.to_csv(output / "one_d_rising_resistance_classification.csv", index=False)
    classification.to_csv(output / "one_d_full_vs_opening_only_ablation.csv", index=False)
    atomic_json(output / "screen_terminal_records.json", {"rows": terminals})
    if len(terminals) != 16 or any(row["status"] != "COMPLETE" for row in terminals):
        raise RuntimeError("all sixteen resistance screen records must complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
