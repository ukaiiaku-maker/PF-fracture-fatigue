"""Audited v10.4.3 entry with converged stagger-consistent diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_4_bulk_peierls_taylor_audited as _v1041
from .plastic_flow_terminal_v1042 import MODEL_ID as TERMINAL_MODEL_ID
from .plastic_flow_accepted_work_v1042 import MODEL_ID as PLASTIC_WORK_MODEL_ID
from .directional_j_positive_v1042 import MODEL_ID as DIRECTIONAL_J_MODEL_ID
from .plastic_flow_stagger_consistent_v1043 import MODEL_ID as STAGGER_MODEL_ID
from .plastic_flow_fixed_point_converged_v1043 import MODEL_ID as FIXED_POINT_MODEL_ID
from .plastic_flow_adaptive_timestep_v1043 import (
    MODEL_ID as ADAPTIVE_TIMESTEP_MODEL_ID,
)
from .plastic_flow_path_work_startup_v1043 import (
    MODEL_ID as PATH_WORK_MODEL_ID,
)
from .plastic_flow_physical_progress_v1043 import (
    MODEL_ID as PHYSICAL_PROGRESS_MODEL_ID,
    load_transformed_sharp_front,
)

# The public entry path is retained so existing launcher contracts remain valid.
# The model audit records the constitutive-time, strict convergence,
# rejected-trial adaptive-timestep, endpoint-path work, and physical-progress
# corrections.
MODEL_ID = (
    "v10.4.3_bulk_detailed_balance_adaptive_converged_stagger_"
    "path_work_physical_progress_terminal"
)


def _has_option(args: list[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in args)


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _prepare_args(args: list[str]) -> None:
    if _has_option(args, "--fatigue-cycles"):
        raise SystemExit("v10.4.3 plastic-flow terminal is monotonic-only")
    if not _has_option(args, "--plastic-flow-terminal"):
        args.append("--plastic-flow-terminal")


def _rewrite_model_audit(root: Path) -> None:
    path = root / "v10_4_bulk_coupled_model_audit.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    payload.update(
        {
            "schema": MODEL_ID,
            "plastic_flow_terminal_model": TERMINAL_MODEL_ID,
            "plastic_work_ledger_base_model": PLASTIC_WORK_MODEL_ID,
            "plastic_work_path_model": PATH_WORK_MODEL_ID,
            "physical_progress_model": PHYSICAL_PROGRESS_MODEL_ID,
            "directional_J_model": DIRECTIONAL_J_MODEL_ID,
            "stagger_consistency_model": STAGGER_MODEL_ID,
            "stagger_fixed_point_model": FIXED_POINT_MODEL_ID,
            "stagger_adaptive_timestep_model": ADAPTIVE_TIMESTEP_MODEL_ID,
            "directional_J_sign_convention": (
                "positive_raw_signed_J_is_forward_configurational_work"
            ),
            "directional_J_effective_definition": "max(J_signed,0)",
            "directional_J_first_nonzero_sign_latch_used": False,
            "directional_J_absolute_value_used": False,
            "negative_directional_J_is_non_driving": True,
            "plastic_flow_terminal_enabled": True,
            "plastic_flow_status": "plastic_flow_no_sharp_fracture",
            "plastic_flow_is_successful_campaign_terminal": True,
            "ductile_fracture_simulated": False,
            "fracture_measure": "positive_signed_configurational_J_only",
            "bulk_plastic_work_enters_fracture_measure": False,
            "bulk_plastic_work_enters_cleavage_hazard": False,
            "bulk_plastic_work_enters_energy_gate": False,
            "mechanics_plasticity_stagger_role": (
                "under_relaxed_fixed_point_iteration_for_one_dt"
            ),
            "mechanics_plasticity_stagger_convergence_required": True,
            "mechanics_plasticity_unconverged_trial_policy": (
                "rollback_reduce_dt_and_dU_at_fixed_loading_rate_then_retry"
            ),
            "mechanics_plasticity_unconverged_min_dt_policy": (
                "raise_before_acceptance"
            ),
            "adaptive_substep_outer_loop_target": (
                "requested_nominal_loading_progress_not_accepted_row_count"
            ),
            "requested_steps_semantics": (
                "nominal_dt_and_dU_increments_preserved_under_subdivision"
            ),
            "accepted_row_step_semantics": "accepted_substep_index",
            "terminal_window_coordinate": "nominal_loading_increment_span",
            "terminal_remaining_horizon_coordinate": (
                "remaining_nominal_loading_progress_times_nominal_dt"
            ),
            "plastic_state_rebased_each_stagger": True,
            "plastic_state_physical_time_advance_per_step": "accepted_dt_cur_only",
            "rejected_stagger_trials_advance_physical_state": False,
            "accepted_mechanics_re_equilibrated_after_final_plastic_iterate": True,
            "accepted_mechanics_state_role": (
                "J_force_stiffness_and_terminal_diagnostics_use_final_accepted_state"
            ),
            "accepted_plastic_work_stagger_policy": "converged_iterate_only",
            "bulk_plastic_work_primary_ledger": (
                "equilibrated_endpoint_trapezoid_sigma_colon_delta_ep"
            ),
            "bulk_plastic_work_constitutive_comparison_ledger": (
                "constitutive_dWp_accepted_gp_converged_stagger_rebased_state"
            ),
            "bulk_plastic_work_endpoint_states": (
                "beginning_and_end_equilibrated_accepted_step_stresses"
            ),
            "bulk_plastic_work_endpoint_increment": (
                "actual_accepted_ep_end_minus_ep_begin"
            ),
            "bulk_plastic_work_startup_stress_policy": (
                "zero_only_before_first_bound_compatible_sigma_gp_else_previous_accepted"
            ),
            "bulk_plastic_work_event_step_policy": (
                "constitutive_fallback_to_avoid_mixing_pre_and_post_crack_geometries"
            ),
            "negative_accepted_plastic_work_policy": (
                "raise_if_materially_negative_clamp_roundoff_only"
            ),
            "post_update_sigma_dot_ep_role": "compatibility_fallback_only",
            "J_pl_diss_definition": (
                "W_bulk_plastic/(unit_thickness*initial_ligament)"
            ),
            "J_pl_diss_role": (
                "temperature_dependent_plastic_dissipation_diagnostic"
            ),
            "contour_directional_J_definition": "max(J_signed,0)",
            "contour_sign_reference_must_equal_one": True,
            "contour_shielding_definition": (
                "max_over_peak_load_and_terminal_states_of_"
                "max(J_outer_positive-J_tip_positive,0)"
            ),
            "contour_states": ["historical_peak_reaction_force", "terminal"],
            "contour_shielding_role": "diagnostic_only",
            "contour_shielding_enters_fracture_hazard": False,
            "v10_4_1_native_complete_cases_physics_compatible": False,
            "v10_4_1_reuse_disabled_reason": (
                "preexisting_stagger_loop_advanced_plastic_state_n_stagger_times_per_dt"
            ),
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    _prepare_args(args)
    transformed = load_transformed_sharp_front()

    bulk_entry = _v1041._entry
    original_sharp_front = bulk_entry._v101.sharp_front
    bulk_entry._v101.sharp_front = transformed
    try:
        print(
            "  v10.4.3 terminal model: plastic_flow_no_sharp_fracture; "
            "directional J=max(J_signed,0); each stagger is re-based to the "
            "beginning-of-step plastic state; the stagger map is under-relaxed "
            "and must converge before a step is accepted; unconverged trials "
            "are rolled back and retried with smaller dt and dU at fixed loading "
            "rate; adaptive accepted substeps continue until the requested nominal "
            "loading time and opening are reached; mechanics is then re-equilibrated "
            "against the accepted plastic state; accepted bulk Wp uses equilibrated "
            "endpoint-average stress contracted with the actual accepted plastic-"
            "strain increment; Wp, J_pl and contour shielding are diagnostic only"
        )
        result = _v1041.main(args)
        out = _option_value(args, "--out")
        if out:
            _rewrite_model_audit(Path(out))
        return result
    finally:
        bulk_entry._v101.sharp_front = original_sharp_front


if __name__ == "__main__":
    main()
