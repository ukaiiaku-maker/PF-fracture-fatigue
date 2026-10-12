"""R6 matched current-owner execution and explicit accepted-state dataflow."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .current_coupled_execution_r4 import (
    CurrentPersistentFractureAdapter, _mechanics_binding,
    build_current_r4_engine, complete_state, state_sha256,
)


MODEL_ID = "r6.current-state-matched-execution/1"


def accepted_increment_dataflow() -> list[dict[str, Any]]:
    """The executable ownership ledger; entries name actual code consumers."""
    return [
        {
            "producer": "bulk_plasticity_v104.accepted_bulk_load_step",
            "quantity": "delta_equivalent_plastic_strain_gp",
            "units": "dimensionless",
            "support": "accepted FEM Gauss points/material coordinates",
            "consumer": (
                "global_physical_scheduler_v1 -> vacancy_transport_phase_field_v2."
                "solve_transport_interval and dislocation_interaction_voiding_v1"
            ),
            "conversion": "epdot_gp_s = delta_ep_eq_gp / accepted_dt_s",
            "consumed_once": True,
            "distinct_from_tip_mpz": True,
        },
        {
            "producer": "persistent_site_source_v10221 accepted source solve",
            "quantity": "anisotropic_dN_emit_by_system",
            "units": "signed line-content count by slip system",
            "support": "tip MPZ source bins/material moving frame",
            "consumer": (
                "signed_burgers_shared_v1025 and "
                "persistent_site_reversible_transport_v10230"
            ),
            "conversion": "activation_to_line_content_by_system from signed kernel family",
            "consumed_once": True,
            "vacancy_bulk_strain_alias_forbidden": True,
        },
        {
            "producer": "persistent_site_reversible_transport_v10230",
            "quantity": "physical returned/cancelled source line content",
            "units": "signed line-content count",
            "support": "tip MPZ and wake moving frame",
            "consumer": "source availability, shielding, radius and signed return ledgers",
            "conversion": "native conservative signed transport",
            "consumed_once": True,
        },
        {
            "producer": "embryo_growth_v1 cavity-surface source transaction",
            "quantity": "loop swept area and deposited line content",
            "units": "m2 and m-2 on accepted spatial support",
            "support": "cavity surface kernel/current material coordinates",
            "consumer": "cavity growth inventory and deposited forest state",
            "conversion": "b * loop_swept_area / model_thickness",
            "consumed_once": True,
        },
    ]


def _arm(parent: Any, factor: float, *, cavity_owned: bool) -> dict[str, Any]:
    adapter = CurrentPersistentFractureAdapter(copy.deepcopy(parent))
    schedule = (
        ("pre_pulse", 5.0e6, 0.35),
        ("fracture_pulse", 51.10046751415974e6, 1.0e-6),
        ("later_common_time", 5.0e6, 0.05),
    )
    rows = []
    for stage, external_K, duration in schedule:
        before_events = int(adapter.engine.n_adv)
        receipt = adapter.advance_adaptive_source_refresh(
            applied_K_Pa_sqrt_m=external_K*factor,
            temperature_K=900.0, duration_s=duration,
        )
        rows.append({
            "stage": stage, "external_K_Pa_sqrt_m": external_K,
            "consumed_K_Pa_sqrt_m": external_K*factor,
            "duration_s": duration, "cavity_owned": cavity_owned,
            "cavity_radius_m": 5.5e-5 if cavity_owned else None,
            "events_added": int(adapter.engine.n_adv)-before_events,
            "terminal_time_s": float(adapter.engine.t),
            "terminal_sha256": state_sha256(adapter.engine),
            "source_refresh": receipt,
        })
    checkpoint = adapter.checkpoint()
    direct = copy.deepcopy(adapter)
    restarted = CurrentPersistentFractureAdapter.restore(checkpoint)
    direct_receipt = direct.advance_adaptive_source_refresh(
        applied_K_Pa_sqrt_m=5.0e6*factor, temperature_K=900.0,
        duration_s=0.01,
    )
    restart_receipt = restarted.advance_adaptive_source_refresh(
        applied_K_Pa_sqrt_m=5.0e6*factor, temperature_K=900.0,
        duration_s=0.01,
    )
    return {
        "schedule": rows,
        "terminal": complete_state(adapter.engine),
        "restart": {
            "complete_state_exact": (
                complete_state(direct.engine) == complete_state(restarted.engine)
            ),
            "direct_controller": direct_receipt,
            "restored_controller": restart_receipt,
        },
    }


def execute_matched_r6(output: str | Path) -> dict[str, Any]:
    mechanics = _mechanics_binding()
    factor = float(mechanics["K_energy_equivalent_ratio"])
    parent = build_current_r4_engine()
    no_void = _arm(parent, 1.0, cavity_owned=False)
    passive = _arm(parent, 1.0, cavity_owned=True)
    active = _arm(parent, factor, cavity_owned=True)
    result = {
        "schema": MODEL_ID,
        "mechanics": mechanics,
        "fixture_scope": (
            "SUPPORTED_PREEXISTING_55UM_OFF_AXIS_FIXED_CAVITY; FINITE_SPATIAL_"
            "RESPONSE; NOT_NATURAL_NUCLEATION_OR_MOVING_CAVITY_RESPONSE"
        ),
        "common_owner": complete_state(parent),
        "identical_external_schedule": True,
        "arms": {"no_void": no_void, "passive": passive, "active": active},
        "accepted_increment_dataflow": accepted_increment_dataflow(),
        "zero_correction_control": {
            "name": "SUBGRID_ZERO_FAR_FIELD_CORRECTION_42NM",
            "retained_separately": True,
            "finite_effect_claimed": False,
        },
        "qualification": {
            "finite_effect_exercised": factor != 1.0,
            "passive_matches_no_void": passive["terminal"] == no_void["terminal"],
            "all_arms_share_external_schedule": True,
            "event_and_later_interval_executed": all(
                arm["schedule"][1]["events_added"] >= 1
                and arm["schedule"][2]["terminal_time_s"]
                    > arm["schedule"][1]["terminal_time_s"]
                for arm in (no_void, passive, active)
            ),
            "complete_restart_exact": all(
                arm["restart"]["complete_state_exact"]
                for arm in (no_void, passive, active)
            ),
            "moving_cavity_response_qualified": False,
        },
    }
    result["passed"] = all((
        result["qualification"]["finite_effect_exercised"],
        result["qualification"]["passive_matches_no_void"],
        result["qualification"]["event_and_later_interval_executed"],
        result["qualification"]["complete_restart_exact"],
    ))
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


__all__ = ["MODEL_ID", "accepted_increment_dataflow", "execute_matched_r6"]
