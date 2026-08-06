"""Audited v10.4.3 entry for fracture versus plastic-dominance censoring."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_4_bulk_peierls_taylor_audited as _v1041
from .plastic_dominance_runtime_v1043 import (
    MODEL_ID as PLASTIC_DOMINANCE_MODEL_ID,
    load_transformed_sharp_front,
)

MODEL_ID = "v10.4.3_bulk_detailed_balance_plastic_dominance_censor"


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


def _append_default(args: list[str], name: str, value: str) -> None:
    if not _has_option(args, name):
        args.extend([name, value])


def _prepare_args(args: list[str]) -> None:
    if _has_option(args, "--fatigue-cycles"):
        raise SystemExit("v10.4.3 plastic-dominance censor is monotonic-only")
    if not _has_option(args, "--plastic-flow-terminal"):
        args.append("--plastic-flow-terminal")

    _append_default(args, "--plastic-flow-min-plastic-fraction", "0.50")
    _append_default(
        args,
        "--plastic-flow-min-cumulative-plastic-fraction",
        "0.10",
    )
    _append_default(args, "--plastic-flow-max-elastic-fraction", "0.50")
    _append_default(args, "--plastic-flow-max-tangent-fraction", "0.50")
    _append_default(args, "--plastic-flow-energy-balance-tolerance", "0.01")


def _rewrite_model_audit(root: Path) -> None:
    path = root / "v10_4_bulk_coupled_model_audit.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    payload.update(
        {
            "schema": MODEL_ID,
            "plastic_dominance_model": PLASTIC_DOMINANCE_MODEL_ID,
            "directional_J_sign_convention": (
                "positive_raw_signed_J_is_forward_configurational_work"
            ),
            "directional_J_effective_definition": "max(J_signed,0)",
            "directional_J_first_nonzero_sign_latch_used": False,
            "directional_J_absolute_value_used": False,
            "negative_directional_J_is_non_driving": True,
            "sharp_fracture_process": "thermally_activated_first_passage",
            "absolute_athermal_Gc_used": False,
            "bulk_plasticity_mode": "full_field",
            "bulk_net_slip_model": "detailed_balance_forward_minus_reverse",
            "zero_stress_net_plastic_rate_exactly_zero": True,
            "stagger_constitutive_time_contract": (
                "all_nonlinear_iterations_reintegrate_one_physical_dt_"
                "from_the_same_start_of_step_state"
            ),
            "plastic_work_primary_ledger": (
                "constitutive_dWp_accepted_gp_final_stagger_iterate"
            ),
            "post_update_sigma_dot_ep_role": "compatibility_fallback_only",
            "final_mechanics_solve_after_constitutive_update": True,
            "plastic_flow_terminal_enabled": True,
            "plastic_flow_status": "plastic_flow_no_sharp_fracture",
            "plastic_terminal_is_model_limit_censor": True,
            "plastic_terminal_interpretation": (
                "no_sharp_fracture_before_sustained_plastic_dominance"
            ),
            "future_fracture_beyond_terminal_resolved": False,
            "ductile_fracture_simulated": False,
            "post_terminal_ductile_failure_modeled": False,
            "plastic_dominance_primary_metric": (
                "area_weighted_axial_plastic_strain_increment_divided_by_"
                "imposed_axial_strain_increment"
            ),
            "plastic_dominance_threshold": 0.50,
            "maximum_elastic_accommodation_ratio": 0.50,
            "maximum_normalized_tangent_stiffness": 0.50,
            "minimum_cumulative_plastic_activity_fraction": 0.10,
            "energy_balance_relative_tolerance": 0.01,
            "bulk_plastic_work_enters_fracture_measure": False,
            "bulk_plastic_work_enters_cleavage_hazard": False,
            "bulk_plastic_work_enters_energy_gate": False,
            "J_pl_diss_role": (
                "temperature_dependent_plastic_dissipation_diagnostic"
            ),
            "contour_shielding_role": "diagnostic_only",
            "contour_shielding_enters_fracture_hazard": False,
            "v10_4_1_native_complete_cases_physics_compatible": (
                "only_after_positive_directional_J_history_audit"
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
            "  v10.4.3 outcome competition: sharp-fracture first passage "
            "versus sustained plastic-dominance model-limit censor; "
            "J=max(J_signed,0); plastic work and contour shielding remain "
            "diagnostic only"
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
