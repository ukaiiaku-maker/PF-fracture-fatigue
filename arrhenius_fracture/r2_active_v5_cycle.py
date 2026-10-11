"""Active-cavity positive-R V5 cycle and exact native restart fixture.

The 1% initial vacancy field is the retained R1 functional fixture, not a
material nucleation claim.  All subsequent equilibrium, scheduler, cavity,
field, and checkpoint operations are native production owners.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_v11 import restore_checkpoint, write_checkpoint
from .global_physical_scheduler_v1 import advance_event_aware_coupled_interval
from .shared_v5_cavity_adapter import _apply_load_factor_pair, _directional_provider, _finite
from .topology_transaction_v11 import complete_accepted_state_fingerprint
from .unified_fracture_material_v5 import material_bundle
from .voiding_production_v5 import deterministic_trajectory, observables


MODEL_ID = "r2.active-cavity-positive-R-native-v5/1"


def _array_hash(state: Any) -> str:
    digest = hashlib.sha256()
    for name in ("displacement", "ep_gp", "rho_gp", "damage", "vacancy_fraction_gp"):
        value = np.ascontiguousarray(getattr(state, name))
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_active_v5_cycle(output: str | Path, *, duration_s: float = 1.0e-8) -> dict[str, Any]:
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    state, lifecycle = deterministic_trajectory(
        bundle=material_bundle("DBTT"), stop_before_ligament=True,
        sentinel_initial_vacancy_fraction_gp=0.01, vacancy_coupled=True,
        boundary_segments=32, radial_layers=12, subgrid_radius_m=1.0e-6,
        promotion_radius_m=2.0e-6, resolved_radius_m=3.0e-6,
    )
    observer = _directional_provider(900.0)
    history = []
    factor = 1.0

    def accept(current, target):
        nonlocal factor
        if target != factor:
            current = _apply_load_factor_pair({"active": current}, target, factor)["active"]
            factor = target
        before = complete_accepted_state_fingerprint(current)
        before_arrays = _array_hash(current)
        advanced, scheduler = advance_event_aware_coupled_interval(
            current, duration_s, temperature_K=900.0,
            directional_observation_provider=observer,
        )
        raw_observables = observables(advanced, f"active_cycle_{target:g}")
        compact_observables = {
            key: raw_observables.get(key)
            for key in (
                "operation", "applied_opening_m", "void_phase",
                "cavity_radius_m", "cavity_area_m2", "energy_J_per_m",
                "compliance_m2_per_N", "top_reaction_N_per_m",
                "bottom_reaction_N_per_m", "vacancy_fraction_min",
                "vacancy_fraction_max", "vacancy_conservation_residual_m2",
                "event_counters", "active_crack_branch_ids", "fingerprint",
            )
        }
        compact_scheduler = {
            key: scheduler.get(key)
            for key in (
                "schema", "outcome", "requested_duration_s",
                "total_consumed_duration_s", "total_unconsumed_duration_s",
                "trial_evaluations", "native_topology_transition_count",
                "nonlinear_radius_replay_total", "localization_replays",
            )
        }
        compact_scheduler["accepted_event_segment_count"] = len(
            scheduler.get("accepted_event_segments", ())
        )
        history.append(_finite({
            "normalized_load_factor": target,
            "input_fingerprint": before,
            "output_fingerprint": complete_accepted_state_fingerprint(advanced),
            "mechanical_field_hash_before": before_arrays,
            "mechanical_field_hash_after": _array_hash(advanced),
            "scheduler": compact_scheduler,
            "observables": compact_observables,
        }))
        return advanced

    state = accept(state, 1.0)
    state = accept(state, 0.2)
    midpoint_fingerprint = complete_accepted_state_fingerprint(state)
    checkpoint = root / "active_midcycle_checkpoint.json"
    write_checkpoint(state, checkpoint, compression="gzip")
    restored = restore_checkpoint(checkpoint)
    midpoint_restart_exact = (
        complete_accepted_state_fingerprint(restored) == midpoint_fingerprint
    )
    if not midpoint_restart_exact:
        raise RuntimeError("active V5 midcycle checkpoint changed accepted state")

    uninterrupted = copy.deepcopy(state)
    uninterrupted = accept(uninterrupted, 1.0)
    uninterrupted_history = history[-1]
    # Reapply the same final stage from the independently restored midpoint.
    factor = 0.2
    history.pop()
    restarted = accept(restored, 1.0)
    restarted_history = history[-1]
    restart_continuation_exact = (
        complete_accepted_state_fingerprint(uninterrupted)
        == complete_accepted_state_fingerprint(restarted)
    )
    if not restart_continuation_exact:
        raise RuntimeError("active V5 restart continuation diverged")

    cavity = restarted.void_state.cavities[0]
    payload = {
        "schema": MODEL_ID,
        "classification": "INITIAL_CONDITION_DRIVEN_FULL_ORDER_FUNCTIONAL_FIXTURE",
        "production_eligible_material_evidence": False,
        "natural_nucleation_claim": False,
        "initial_vacancy_fraction": 0.01,
        "load_factors": [1.0, 0.2, 1.0],
        "duration_s_per_factor": duration_s,
        "native_positive_equilibrium_only": True,
        "nonpositive_structural_load_status": "FAIL_CLOSED_UNSUPPORTED_CONTACT_COHESIVE_COMBINATION",
        "active_cavity": True,
        "terminal_cavity_phase": cavity.phase.value,
        "terminal_cavity_radius_m": float(cavity.radius_m),
        "lifecycle_operations": [row["operation"] for row in lifecycle],
        "history": history,
        "uninterrupted_final_stage": uninterrupted_history,
        "restarted_final_stage": restarted_history,
        "midpoint_restart_exact": midpoint_restart_exact,
        "restart_continuation_exact": restart_continuation_exact,
        "terminal_fingerprint": complete_accepted_state_fingerprint(restarted),
        "midcycle_checkpoint": str(checkpoint),
        "field_feedback_exercised": all(
            row["mechanical_field_hash_before"] != row["mechanical_field_hash_after"]
            for row in history
        ),
        "limits": [
            "positive-R piecewise load history, not a continuous sinusoid",
            "no cavity-radius change was forced during the short cycle",
            "fixture proves active-state coupling and restart, not material nucleation",
        ],
    }
    (root / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


__all__ = ["MODEL_ID", "run_active_v5_cycle"]
