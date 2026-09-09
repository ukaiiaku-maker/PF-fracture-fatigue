#!/usr/bin/env python3
"""Bounded behavioral parity and historical impact; no PF or FEM execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from arrhenius_fracture.current_source_runtime_bindings import runtime_binding_inventory
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
from arrhenius_fracture.unified_mpz import UnifiedMPZState
from scripts.v13_frozen_support import (
    ATLAS, initialized_engine, saved_owner, recapture, restore_complete_current_source_engine,
    field_differences, frozen_process_step, rng_hash,
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_state(engine):
    mpz = engine.mpz
    fields = {}
    for name, value in vars(mpz).items():
        if any(term in name for term in (
            "mobile", "retained", "slip", "available", "escaped", "recovered",
            "emitted", "activations", "line_content", "multiplicity", "back", "rate",
        )) and isinstance(value, (float, int, np.ndarray)):
            fields[name] = value.tolist() if isinstance(value, np.ndarray) else value
    fields.update({
        "engine_time_s": engine.t, "process_time_s": mpz.time_s,
        "radius_m": engine.r_eff(), "shielding_Pa_sqrt_m": engine.K_shield(),
        "B": engine.B, "hazard_action": engine.hazard_action_current,
        "hazard_threshold": engine.hazard_threshold_action,
        "hazard_ordinal": engine.hazard_event_index, "rng_sha256": rng_hash(engine),
        "persistent_geometry": mpz.persistent_site_last_geometry,
    })
    return fields


def historical_fallback(saved):
    """The exact old allocation/field-copy path, isolated for impact only."""
    from arrhenius_fracture.persistent_site_audited_engine_v10221 import AuditedPersistentSiteStateResolvedTipEngine
    payload = saved.complete_checkpoint_payload()
    engine = AuditedPersistentSiteStateResolvedTipEngine.__new__(AuditedPersistentSiteStateResolvedTipEngine)
    engine.mpz = UnifiedMPZState.__new__(UnifiedMPZState)
    data = copy.deepcopy({"engine": payload["engine_fields"], "mpz": payload["mpz_fields"]})
    vars(engine).update(data["engine"])
    vars(engine.mpz).update(data["mpz"])
    return engine


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cases = [f"{material}_{temperature}K" for material in ("Peak", "DBTT", "weakT", "ceramic") for temperature in (300, 1000)]
    parity, impact = [], []
    for case in cases:
        path = ATLAS / "continued_1000um_runs_v2" / case / "checkpoint/latest.v12.pkl"
        before_hash = sha(path)
        checkpoint, saved = saved_owner(case)
        payload = saved.complete_checkpoint_payload()
        T = float(case.rsplit("_", 1)[1][:-1])
        for label, duration, active in (("zero_duration", 0.0, True), ("negligible_emission", 1e-10, False), ("emission_active", 1e-10, True)):
            with initialized_engine(payload) as a:
                b = restore_complete_current_source_engine(recapture(a))
                start = selected_state(a)
                binding_a, binding_b = runtime_binding_inventory(a), runtime_binding_inventory(b)
                initial_diff = field_differences(_capture_shared_engine(a), _capture_shared_engine(b))
                K = a._signed_current_K_Pa_sqrt_m if active else 0.0
                out_a = frozen_process_step(a, T, duration, K)
                out_b = frozen_process_step(b, T, duration, K)
                state_diff = field_differences(_capture_shared_engine(a), _capture_shared_engine(b))
                output_diff = field_differences(out_a, out_b)
                row = {
                    "case": case, "interval": (
                        "loaded_emission_blocked" if label == "emission_active" and out_a.get("dN_emit", 0.0) == 0.0 else label
                    ), "checkpoint_sha256": before_hash,
                    "accepted_state_id": checkpoint.runtime.accepted_state_id,
                    "duration_s": duration, "K_Pa_sqrt_m": K,
                    "binding_identities": binding_a, "bindings_exact": binding_a == binding_b,
                    "initial_state_differences": initial_diff,
                    "complete_final_state_differences": state_diff,
                    "complete_output_differences": output_diff,
                    "starting_selected_state": start,
                    "final_selected_state": selected_state(a),
                    "step_output": out_a,
                    "initialization_reference": "actual_production_constructor_then_identical_physical_state_transfer_no_rehydration_on_A",
                    "pass": binding_a == binding_b and not (initial_diff or state_diff or output_diff),
                }
                if not row["pass"]:
                    raise RuntimeError(f"behavioral restore parity failed: {case}/{label}")
                parity.append(row)
        if case in ("Peak_300K", "DBTT_1000K"):
            with initialized_engine(payload):
                old, corrected = historical_fallback(saved), restore_complete_current_source_engine(saved)
                dt = 1e-10
                K = corrected._signed_current_K_Pa_sqrt_m
                before_old, before_new = selected_state(old), selected_state(corrected)
                old_result = frozen_process_step(old, T, dt, K)
                new_result = frozen_process_step(corrected, T, dt, K)
                after_old, after_new = selected_state(old), selected_state(corrected)
                differences = field_differences(_capture_shared_engine(old), _capture_shared_engine(corrected))
                impact.append({
                    "case": case, "source_checkpoint": str(path), "checkpoint_sha256": before_hash,
                    "duration_s": dt, "K_Pa_sqrt_m": K,
                    "old_initial": before_old, "corrected_initial": before_new,
                    "old_final": after_old, "corrected_final": after_new,
                    "old_output": old_result, "corrected_output": new_result,
                    "different_fields": differences, "process_evolution_differs": bool(differences),
                    "old_emitted_increment": old_result.get("dN_emit"),
                    "corrected_emitted_increment": new_result.get("dN_emit"),
                    "both_results_discarded_no_checkpoint_commit": True,
                })
        if sha(path) != before_hash:
            raise RuntimeError("historical checkpoint changed")
        print(case, "behavioral parity passed", flush=True)
    result = {
        "schema": "v13.process-runtime-rehydration-qualification/1",
        "interpretation_boundary": "BRANCHING_KINETICS_MODEL_UNCALIBRATED",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "phase12_status": "PASS", "phase3_status": "COMPLETE",
        "nonzero_emission_interval_count": sum(r["step_output"].get("dN_emit", 0.0) > 0.0 for r in parity),
        "parity": parity, "historical_impact": impact,
        "historical_atlas_use": "TOPOLOGY_SOFTWARE_CAPABILITY_ONLY_PROCESS_LAW_PARITY_NOT_ESTABLISHED",
        "new_pf_trajectories": 0, "new_fem_solves": 0,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "process_restore_qualification.json").write_text(json.dumps(safe_json(result), indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
