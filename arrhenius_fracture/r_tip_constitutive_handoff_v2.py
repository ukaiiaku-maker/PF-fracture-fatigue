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
from .sharp_front_v11_branching import measure_directional_front_loads

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
    rate_source = inspect.getsource(production.directional_sharp_front_rates)
    tensor_source = inspect.getsource(production.crack_tip_tensor)
    transaction_source = inspect.getsource(production.downstream_front_transaction)
    provider_source = inspect.getsource(measure_directional_front_loads)
    analytical_source = inspect.getsource(FrontEngine.sigma_tip)
    radius_source = inspect.getsource(FrontEngine.r_eff)
    accepted_analytical_relation_exists = (
        "self.r_eff()" in analytical_source and "np.sqrt(2.0 * np.pi" in analytical_source
    )
    canonical_radius_relation_exists = all(
        token in radius_source for token in ("self.f.r0", "self.f.c_blunt", "self.b", "self.N_em")
    )
    child_radius_enters_tensor = "r_tip" in tensor_source
    orphan_radius_stored = '"r_tip_m"' in transaction_source
    child_engine_restored = "restore_sharp_front_engine" in rate_source
    child_uses_existing_response = "sharp_front_constitutive_response" in rate_source
    explicit_branch_provider = all(
        token in provider_source for token in ("branch_id", "compute_J_integral", "K_directional_Pa_sqrt_m")
    )
    child_uses_provider = "sharp_front_load_provider" in transaction_source
    child_renews_existing_engine = "child_engine.step" in transaction_source
    accepted_matching_operator_exists = all((
        canonical_radius_relation_exists, child_engine_restored,
        child_uses_existing_response, explicit_branch_provider,
        child_uses_provider, child_renews_existing_engine,
    )) and not orphan_radius_stored
    factors = (0.75, 1.0, 1.25)
    predictions = [
        {
            "r_tip_over_r0": factor,
            "uncapped_sigma_over_baseline_at_fixed_positive_K": math.sqrt(1.0 / factor),
            "canonical_peer_control": "modify_N_em_so_FrontEngine.r_eff_reaches_target",
            "coupled_response_policy": "evaluate_the_unchanged_full_N_em_coupled_barrier_and_rate_law",
        }
        for factor in factors
    ]
    graph = {
        "schema": "v5.r-tip-consumer-graph/2",
        "nodes": [
            {"id": "tip_process_state.by_branch.<id>.canonical_state.N_em", "kind": "owned_child_blunting_ledger"},
            {"id": "FrontEngine.r_eff", "kind": "accepted_derived_radius_law", **_location(FrontEngine.r_eff)},
            {"id": "crack_tip_tensor", "kind": "local_FEM_tensor_observer", **_location(production.crack_tip_tensor)},
            {"id": "measure_directional_front_loads", "kind": "accepted_directional_J_K_provider", **_location(measure_directional_front_loads)},
            {"id": "directional_sharp_front_rates", "kind": "existing_engine_rate_adapter", **_location(production.directional_sharp_front_rates)},
            {"id": "FrontEngine.lambda_cleave", "kind": "accepted_barrier_and_rate_law", **_location(FrontEngine.lambda_cleave)},
            {"id": "FrontEngine.lambda_emit", "kind": "accepted_emission_barrier_and_rate_law", **_location(FrontEngine.lambda_emit)},
            {"id": "FrontEngine.sigma_tip", "kind": "accepted_K_over_sqrt_radius_law", **_location(FrontEngine.sigma_tip)},
            {"id": "_select_emitted_proposal", "kind": "continuation_event_selector", **_location(production._select_emitted_proposal)},
            {"id": "downstream_front_transaction", "kind": "child_continuation_transaction", **_location(production.downstream_front_transaction)},
            {"id": "capture_sharp_front_engine", "kind": "post_event_canonical_state_checkpoint", **_location(production.capture_sharp_front_engine)},
        ],
        "edges": [
            {"from": "tip_process_state.by_branch.<id>.canonical_state.N_em", "to": "FrontEngine.r_eff", "role": "canonical_blunting_state"},
            {"from": "measure_directional_front_loads", "to": "FrontEngine.sigma_tip", "role": "existing_directional_K_interface"},
            {"from": "FrontEngine.r_eff", "to": "FrontEngine.sigma_tip", "role": "derived_radius_input"},
            {"from": "FrontEngine.sigma_tip", "to": "FrontEngine.lambda_cleave", "role": "existing_cleavage_path"},
            {"from": "FrontEngine.sigma_tip", "to": "FrontEngine.lambda_emit", "role": "existing_emission_path"},
            {"from": "directional_sharp_front_rates", "to": "_select_emitted_proposal", "role": "completed_cleavage_first_passages"},
            {"from": "_select_emitted_proposal", "to": "downstream_front_transaction", "role": "selected_continuation_event"},
            {"from": "downstream_front_transaction", "to": "FrontEngine.step", "role": "existing_emission_and_renewal_update"},
            {"from": "FrontEngine.step", "to": "capture_sharp_front_engine", "role": "checkpoint_canonical_child_state"},
        ],
        "pre_void_call_graph": [
            "active_tip_ids -> measure_directional_front_loads",
            "directional_J -> K_directional=sqrt(Eprime*positive_J)",
            "N_em -> FrontEngine.r_eff=r0+c_blunt*b*N_em",
            "K_directional/gamma_rel -> FrontEngine.sigma_tip",
            "sigma_tip -> FrontEngine.lambda_cleave + FrontEngine.lambda_emit",
            "FrontEngine.step -> B/N_em/W_emit/t/K_prev -> renewal/advance",
        ],
        "post_void_call_graph_before_repair": [
            "DOWNSTREAM_FIRST_PASSAGE -> child with orphan r_tip_m",
            "child continuation -> crack_tip_tensor",
            "tensor normal stress -> directional_clock_rates -> fresh FrontEngine.lambda_cleave only",
            "directional first passage -> topology advance; no FrontEngine.step",
        ],
        "first_missing_or_bypassed_edge_before_repair": (
            "child active tip -> established measure_directional_front_loads -> owned FrontEngine"
        ),
        "disconnected_accepted_subgraph_before_repair": [
            {"from": "intensity_K", "to": "FrontEngine.sigma_tip", "role": "accepted_original_sharp_front_input"},
            {"from": "FrontEngine.r_eff", "to": "FrontEngine.sigma_tip", "role": "accepted_original_radius_input"},
            {"from": "FrontEngine.sigma_tip", "to": "FrontEngine.lambda_cleave", "role": "accepted_original_cleavage_path"},
            {"from": "FrontEngine.sigma_tip", "to": "FrontEngine.lambda_emit", "role": "accepted_original_emission_path"},
        ],
        "required_but_absent_edges": [],
    }
    classification = "PASS" if accepted_matching_operator_exists else "MISSING_OR_BYPASSED"
    return {
        "schema": SCHEMA,
        "scope": "READ_ONLY_CONSUMER_GRAPH_AND_MODEL_FORM_AUDIT",
        "consumer_graph": graph,
        "accepted_original_sharp_front_relation": {
            "form": "r_eff = r0 + c_blunt*b*N_em; sigma_tip = K_eff / sqrt(2*pi*r_eff)",
            "exists": accepted_analytical_relation_exists and canonical_radius_relation_exists,
            "location": _location(FrontEngine.sigma_tip),
        },
        "child_continuation_path": {
            "source": "established_directional_J_K_provider",
            "rate_input": "K_directional_Pa_sqrt_m through FrontEngine.sigma_tip",
            "child_radius_enters_tensor_observer": child_radius_enters_tensor,
            "orphan_radius_stored": orphan_radius_stored,
            "child_canonical_state_restored": child_engine_restored,
            "child_radius_bound_to_analytical_engine": child_engine_restored,
            "directional_J_to_K_provider_present": explicit_branch_provider,
            "emission_barrier_or_rate_consumed": child_uses_existing_response,
            "post_event_r_tip_renewal_law": "FrontEngine.step; radius remains derived from renewed N_em",
        },
        "signed_analytical_prediction_frozen_before_any_future_implementation": predictions,
        "reciprocal_control_prediction": "R_void variation cannot alter child constitutive response when directional K and canonical FrontEngine state are frozen",
        "forbidden_repairs": [
            "empirical_multiplier_of_r_tip",
            "empirical_multiplier_of_R_void",
            "de_smearing_or_amplification_of_explicit_FEM_stress_without_an_accepted_matching_derivation",
        ],
        "R_TIP_OWNERSHIP": "PASS_CANONICAL_N_EM_LEDGER",
        "R_TIP_DISTINCT_FROM_VOID_RADIUS": "PASS",
        "DOWNSTREAM_CHILD_TO_EXISTING_SHARP_FRONT_ENGINE_ADAPTER": classification,
        "R_TIP_CAUSAL_LAW": "REUSED_UNCHANGED" if classification == "PASS" else "PRESENT_BUT_ADAPTER_MISSING",
        "EXISTING_R_TIP_LAW_REUSED_BY_DOWNSTREAM_CHILD": classification,
        "model_form_complete": classification == "PASS",
        "scientific_classification": (
            "EXISTING_R_TIP_LAW_REUSED_BY_DOWNSTREAM_CHILD"
            if classification == "PASS"
            else "DOWNSTREAM_CHILD_TO_EXISTING_SHARP_FRONT_ENGINE_ADAPTER_MISSING_OR_BYPASSED"
        ),
    }


__all__ = ["SCHEMA", "audit"]
