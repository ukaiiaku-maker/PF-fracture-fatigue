"""Audited v10.4.3 entry with stagger-consistent plastic-flow diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_4_bulk_peierls_taylor_audited as _v1041
from .plastic_flow_terminal_v1042 import MODEL_ID as TERMINAL_MODEL_ID
from .plastic_flow_accepted_work_v1042 import MODEL_ID as PLASTIC_WORK_MODEL_ID
from .directional_j_positive_v1042 import MODEL_ID as DIRECTIONAL_J_MODEL_ID
from .plastic_flow_stagger_consistent_v1043 import (
    MODEL_ID as STAGGER_MODEL_ID,
    load_transformed_sharp_front,
)

# The public entry path is retained so existing launcher contracts remain valid.
# The model audit records the v10.4.3 constitutive-time correction explicitly.
MODEL_ID = "v10.4.3_bulk_detailed_balance_stagger_consistent_plastic_terminal"


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
            "directional_J_model": DIRECTIONAL_J_MODEL_ID,
            "stagger_consistency_model": STAGGER_MODEL_ID,
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
            "mechanics_plasticity_stagger_role": "fixed_point_iteration_for_one_dt",
            "plastic_state_rebased_each_stagger": True,
            "plastic_state_physical_time_advance_per_step": "dt_cur",
            "accepted_plastic_work_stagger_policy": "converged_last_iterate_only",
            "bulk_plastic_work_primary_ledger": (
                "constitutive_dWp_accepted_gp_converged_stagger_rebased_state"
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
            "beginning-of-step plastic state; converged accepted Wp, J_pl and "
            "contour shielding are diagnostic only"
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
