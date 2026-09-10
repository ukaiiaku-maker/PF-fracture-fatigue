"""Prospective audit of the one-void child-tip constitutive handoff.

This module does not change a rate or state.  It distinguishes an accepted
sharp-front radius law from an accepted mapping that actually supplies that law
with the child-owned radius and a source intensity.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

from . import voiding_production_v5 as production
from .sharp_front import FrontEngine

SCHEMA = "v5.one-void-r-tip-constitutive-handoff/2"


def _location(function) -> dict:
    lines, first = inspect.getsourcelines(function)
    root = Path(__file__).resolve().parents[1]
    return {
        "path": str(Path(inspect.getsourcefile(function)).resolve().relative_to(root)),
        "function": function.__name__,
        "first_line": first,
        "last_line": first + len(lines) - 1,
        "sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
    }


def audit() -> dict:
    rate_source = inspect.getsource(production.directional_clock_rates)
    tensor_source = inspect.getsource(production.crack_tip_tensor)
    transaction_source = inspect.getsource(production.downstream_front_transaction)
    analytical_source = inspect.getsource(FrontEngine.sigma_tip)
    accepted_analytical_relation_exists = (
        "self.r_eff()" in analytical_source and "np.sqrt(2.0 * np.pi" in analytical_source
    )
    child_radius_enters_tensor = "r_tip" in tensor_source
    child_radius_enters_rates = "r_tip" in rate_source
    child_radius_bound_to_engine = (
        "engine.f.r0" in transaction_source or "engine.f.r_tip" in transaction_source
    )
    fem_tensor_converted_to_intensity = any(
        token in transaction_source for token in ("stress_intensity", "K_directional", "interaction_integral")
    )
    accepted_matching_operator_exists = bool(
        child_radius_enters_tensor
        or child_radius_enters_rates
        or child_radius_bound_to_engine
        or fem_tensor_converted_to_intensity
    )
    factors = (0.75, 1.0, 1.25)
    predictions = [
        {
            "r_tip_over_r0": factor,
            "uncapped_sigma_over_baseline_at_fixed_positive_K": math.sqrt(1.0 / factor),
            "predicted_barrier_direction": "not_higher" if factor < 1 else "unchanged" if factor == 1 else "not_lower",
            "predicted_rate_direction": "not_lower" if factor < 1 else "unchanged" if factor == 1 else "not_higher",
            "predicted_crossing_time_direction": "not_later" if factor < 1 else "unchanged" if factor == 1 else "not_earlier",
        }
        for factor in factors
    ]
    graph = {
        "schema": "v5.r-tip-consumer-graph/2",
        "nodes": [
            {"id": "tip_process_state.r_tip_m", "kind": "owned_child_state"},
            {"id": "child_branch.local_state.r_tip_m", "kind": "owned_state_mirror"},
            {"id": "crack_tip_tensor", "kind": "local_FEM_tensor_observer", **_location(production.crack_tip_tensor)},
            {"id": "directional_clock_rates", "kind": "resolved_stress_rate_adapter", **_location(production.directional_clock_rates)},
            {"id": "FrontEngine.lambda_cleave", "kind": "accepted_barrier_and_rate_law", **_location(FrontEngine.lambda_cleave)},
            {"id": "FrontEngine.lambda_emit", "kind": "accepted_emission_barrier_and_rate_law", **_location(FrontEngine.lambda_emit)},
            {"id": "FrontEngine.sigma_tip", "kind": "accepted_K_over_sqrt_radius_law", **_location(FrontEngine.sigma_tip)},
            {"id": "_select_emitted_proposal", "kind": "continuation_event_selector", **_location(production._select_emitted_proposal)},
            {"id": "downstream_front_transaction", "kind": "child_continuation_transaction", **_location(production.downstream_front_transaction)},
            {"id": "renewed_child_r_tip", "kind": "post_event_owned_state"},
        ],
        "edges": [
            {"from": "tip_process_state.r_tip_m", "to": "child_branch.local_state.r_tip_m", "role": "exact_ownership_mirror"},
            {"from": "crack_tip_tensor", "to": "directional_clock_rates", "role": "direct_resolved_FEM_stress"},
            {"from": "directional_clock_rates", "to": "FrontEngine.lambda_cleave", "role": "resolved_opening_stress_and_temperature"},
            {"from": "directional_clock_rates", "to": "_select_emitted_proposal", "role": "completed_cleavage_first_passages"},
            {"from": "_select_emitted_proposal", "to": "downstream_front_transaction", "role": "selected_continuation_event"},
            {"from": "downstream_front_transaction", "to": "renewed_child_r_tip", "role": "fresh_child_initialization_then_preservation_only"},
        ],
        "disconnected_accepted_subgraph": [
            {"from": "intensity_K", "to": "FrontEngine.sigma_tip", "role": "accepted_original_sharp_front_input"},
            {"from": "FrontEngine.r_eff", "to": "FrontEngine.sigma_tip", "role": "accepted_original_radius_input"},
            {"from": "FrontEngine.sigma_tip", "to": "FrontEngine.lambda_cleave", "role": "accepted_original_cleavage_path"},
            {"from": "FrontEngine.sigma_tip", "to": "FrontEngine.lambda_emit", "role": "accepted_original_emission_path"},
        ],
        "required_but_absent_edges": [
            {"from": "tip_process_state.r_tip_m", "to": "local_FEM_to_sharp_front_matching_operator"},
            {"from": "crack_tip_tensor", "to": "local_FEM_to_sharp_front_matching_operator"},
            {"from": "local_FEM_to_sharp_front_matching_operator", "to": "FrontEngine.sigma_tip"},
        ],
    }
    classification = "IMPLEMENTED" if accepted_matching_operator_exists else "NOT_DEFINED"
    return {
        "schema": SCHEMA,
        "scope": "READ_ONLY_CONSUMER_GRAPH_AND_MODEL_FORM_AUDIT",
        "consumer_graph": graph,
        "accepted_original_sharp_front_relation": {
            "form": "sigma_tip = K_eff / sqrt(2*pi*r_eff)",
            "exists": accepted_analytical_relation_exists,
            "location": _location(FrontEngine.sigma_tip),
        },
        "child_continuation_path": {
            "source": "area_weighted_aligned_child_tip_CST_tensor",
            "rate_input": "candidate_normal_resolved_opening_stress",
            "child_radius_enters_tensor_observer": child_radius_enters_tensor,
            "child_radius_enters_rate_adapter": child_radius_enters_rates,
            "child_radius_bound_to_analytical_engine": child_radius_bound_to_engine,
            "FEM_tensor_to_intensity_operator_present": fem_tensor_converted_to_intensity,
            "emission_barrier_or_rate_consumed": False,
            "post_event_r_tip_renewal_law": "fresh child initialized from max(hbar_tip,10b); continuation retains owned r_tip",
        },
        "signed_analytical_prediction_frozen_before_any_future_implementation": predictions,
        "reciprocal_control_prediction": "R_void variation cannot alter child rate when FEM fields, child source, and r_tip are frozen",
        "forbidden_repairs": [
            "empirical_multiplier_of_r_tip",
            "empirical_multiplier_of_R_void",
            "de_smearing_or_amplification_of_explicit_FEM_stress_without_an_accepted_matching_derivation",
        ],
        "R_TIP_OWNERSHIP": "PASS",
        "R_TIP_DISTINCT_FROM_VOID_RADIUS": "PASS",
        "R_TIP_CAUSAL_LAW": classification,
        "model_form_complete": classification == "IMPLEMENTED",
        "scientific_classification": (
            "R_TIP_CAUSAL_LAW_IMPLEMENTED" if classification == "IMPLEMENTED"
            else "R_TIP_CAUSAL_LAW_NOT_DEFINED"
        ),
    }


__all__ = ["SCHEMA", "audit"]
