#!/usr/bin/env python3
"""Continue preregistered best P25/P40 rows from event checkpoints to 100 um."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from arrhenius_fracture.canonical_v2_registry_v10230 import load_rows
from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint, write_checkpoint,
)
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import capture_ledgers
import scripts.complete_corrected_thermodynamic_joint_search_v10230 as v2
from scripts.run_v2_2_1d_rising_resistance_v10230 import (
    DK, EVENT_M, KDOT, K_CENSOR, T_K, XI, atomic_json, conservation,
    prepare_engine, state_record,
)


OUT = ROOT / "analysis_outputs/v2_2_1d_rising_resistance_and_fatigue"
SELECTED = ("P25_TJBSV2_S_002987", "P40_TJBSV2_S_043821")
MODES = ("FULL_PRODUCTION_STATE", "OPENING_ONLY_FROZEN_STATE")


def restore_geometry(engine, payload: dict) -> None:
    sig = payload["geometry_signature"]
    engine.n_adv = int(sig[0]); engine.a_adv = float(sig[1])
    engine.micro_advance_total_m = float(sig[2])
    engine.checkpoint_advance_total_m = float(sig[3])
    engine.mpz.advance_total_m = float(sig[4])


def continue_case(row: dict[str, str], mode: str) -> tuple[list[dict], list[dict]]:
    case_path = OUT / "case_records" / row["candidate_id"] / f"{mode}.json"
    saved_case = json.loads(case_path.read_text())
    events = list(saved_case["events"]); states = list(saved_case["states"])
    if len(events) != 10:
        raise RuntimeError("100 um continuation requires an exact 10-event checkpoint")
    checkpoint_dir = OUT / "checkpoints" / row["candidate_id"] / mode
    payload = json.loads((checkpoint_dir / "high_cycle_live_checkpoint.json").read_text())
    engine, _cleavage, _emission, parent = prepare_engine(row, mode)
    restore_geometry(engine, payload)
    restore_checkpoint(engine, checkpoint_dir)

    # Deterministic restart: two independent restorations must produce exactly
    # the same next event.  The first becomes the physical continuation.
    def one_event(restored_engine, event_index: int):
        restored_engine.step(0.0, T_K, 0.0)
        restored_engine.hazard_threshold_action = XI
        restored_engine.hazard_action_current = 0.0
        restored_engine.B = 0.0
        restored_engine._set_current_event_length()
        initial_time = float(restored_engine.t)
        pre = state_record(restored_engine); before_ledgers = capture_ledgers(restored_engine)
        for left in np.arange(0.0, K_CENSOR, DK):
            mid = float(left + 0.5 * DK)
            restored_engine.mpz._reversible_transport_K_signed_Pa_sqrt_m = mid * 1.0e6
            info = restored_engine.step(mid * 1.0e6, T_K, DK / KDOT)
            if bool(info.get("fired", False)):
                consumed = float(info.get("kinetic_dt_consumed_s", info.get("dt_consumed", DK / KDOT)))
                K = float(left) + KDOT * min(max(consumed, 0.0), DK / KDOT)
                post = state_record(restored_engine)
                record = {
                    "candidate_id": row["candidate_id"], "parent_id": row["parent_id"],
                    "mode": mode, "temperature_K": T_K, "event_index": event_index,
                    "cumulative_advance_um": (event_index + 1) * 5.0,
                    "event_advance_um": 5.0, "K_reinit_MPa_sqrt_m": K,
                    "loading_time_s": float(restored_engine.t) - initial_time,
                    "threshold_action": XI,
                    "physical_hazard_action": float(info.get("hazard_action_completed", XI)),
                    "elementary_cleavage_rate_s": float(info.get("lambda_c_raw", 0.0)),
                    "constitutive_ceiling_occupied": bool(float(info.get("lambda_c_raw", 0.0)) * float(parent["physics__cleavage_correlation_time_s"]) >= 20.0),
                    "sigma_cap_active": bool(info.get("sigma_cap_active", False)),
                    "dN_cap_active": bool(info.get("dN_cap_active", False)),
                    "N_sat_active": bool(info.get("N_sat_active", False)),
                    "conservation_relative": conservation(post),
                    "complete_bound_row_sha256": row["complete_bound_row_sha256"],
                    "continued_from_50um_checkpoint": True, **post,
                }
                Eprime = float(restored_engine.G) * 2.0 * (1.0 + float(restored_engine.nu)) / (1.0 - float(restored_engine.nu) ** 2)
                record["J_equiv_J_m2"] = (K * 1.0e6) ** 2 / Eprime
                state_rows = [
                    {"candidate_id": row["candidate_id"], "mode": mode, "event_index": event_index, "state_position": "PRE_RELOAD", **pre},
                    {"candidate_id": row["candidate_id"], "mode": mode, "event_index": event_index, "state_position": "POST_EVENT", **post,
                     "ledger_delta_json": json.dumps({k: capture_ledgers(restored_engine).get(k, 0.0) - value for k, value in before_ledgers.items()}, sort_keys=True)},
                ]
                return record, state_rows
        raise RuntimeError(f"{row['candidate_id']}:{mode} continuation reached K censor")

    # Exact restart parity at event 10.
    control, *_ = prepare_engine(row, mode)
    restore_geometry(control, payload); restore_checkpoint(control, checkpoint_dir)
    event_a, state_a = one_event(engine, 10)
    event_b, _state_b = one_event(control, 10)
    if json.dumps(event_a, sort_keys=True, default=str) != json.dumps(event_b, sort_keys=True, default=str):
        raise RuntimeError("deterministic checkpoint/restart parity failed")
    events.append(event_a); states.extend(state_a)
    for event_index in range(11, 20):
        event, state_rows = one_event(engine, event_index)
        events.append(event); states.extend(state_rows)
    old = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR")
    os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(checkpoint_dir)
    try:
        write_checkpoint(engine, temperature_K=T_K, reason="first_passage", metadata={
            "candidate_id": row["candidate_id"], "mode": mode, "event_index": 19,
            "K_reinit_MPa_sqrt_m": events[-1]["K_reinit_MPa_sqrt_m"],
            "continued_from_50um_checkpoint": True,
        })
    finally:
        if old is None: os.environ.pop("V10230_HIGH_CYCLE_CHECKPOINT_DIR", None)
        else: os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = old
    saved_case["events"] = events; saved_case["states"] = states
    saved_case["terminal"].update(status="COMPLETE_100UM_CONTINUATION", events=20,
                                  cumulative_advance_um=100.0,
                                  deterministic_restart_parity="PASS_EXACT")
    atomic_json(case_path, saved_case)
    return events, states


def main() -> int:
    v2.ROWS = v2.source_rows(); v2.ACTIVE = v2.active_stresses(v2.ROWS)
    rows = load_rows(); points = pd.read_csv(OUT / "one_d_reload_separated_resistance_points.csv")
    states = pd.read_parquet(OUT / "one_d_reload_separated_state_table.parquet")
    additions, state_additions = [], []
    for candidate_id in SELECTED:
        for mode in MODES:
            print("CONTINUE_START", candidate_id, mode, flush=True)
            events, state_rows = continue_case(rows[candidate_id], mode)
            additions.extend(events[10:]); state_additions.extend(state_rows[20:])
            print("CONTINUE_END", candidate_id, mode, flush=True)
    points = pd.concat([points, pd.DataFrame(additions)], ignore_index=True)
    states = pd.concat([states, pd.DataFrame(state_additions)], ignore_index=True)
    points.to_csv(OUT / "one_d_reload_separated_resistance_points.csv", index=False)
    states.to_parquet(OUT / "one_d_reload_separated_state_table.parquet", index=False, compression="zstd")
    atomic_json(OUT / "one_d_100um_continuation.json", {
        "selected_by_frozen_50um_metrics": list(SELECTED),
        "modes": list(MODES), "status": "PASS", "checkpoint_restart_parity": "PASS_EXACT",
    })
    return 0


if __name__ == "__main__": raise SystemExit(main())
