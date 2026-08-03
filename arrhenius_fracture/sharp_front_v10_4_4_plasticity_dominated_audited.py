"""Audited v10.4.4 entry for bulk-plasticity-dominated fracture campaigns."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from . import sharp_front_v10_4_2_plastic_flow_audited as _v1043
from .plastic_flow_campaign_terminal_v1044 import (
    MODEL_ID as CAMPAIGN_TERMINAL_MODEL_ID,
    load_transformed_sharp_front,
)

MODEL_ID = (
    "v10.4.4_full_field_bulk_plasticity_with_"
    "plasticity_dominated_campaign_terminal"
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
    _v1043._prepare_args(args)
    if not _has_option(args, "--plastic-flow-window-steps"):
        args.extend(["--plastic-flow-window-steps", "32"])
    if not _has_option(args, "--plastic-flow-min-step"):
        args.extend(["--plastic-flow-min-step", "32"])


def _rewrite_terminal_outputs(root: Path) -> None:
    audit_path = root / "plastic_flow_terminal_audit.json"
    if not audit_path.is_file():
        return

    audit = json.loads(audit_path.read_text())
    J_elastic = max(
        float(
            audit.get(
                "J_elastic_positive_terminal_J_per_m2",
                audit.get("J_tip_positive_final_J_per_m2", 0.0),
            )
        ),
        0.0,
    )
    J_plastic = max(float(audit.get("J_pl_diss_J_per_m2", 0.0)), 0.0)
    Eprime = max(float(audit.get("Eprime_Pa", 0.0)), 0.0)
    if Eprime <= 0.0 and J_plastic > 0.0:
        K_pl = max(float(audit.get("K_pl_equivalent_MPa_sqrt_m", 0.0)), 0.0)
        Eprime = (K_pl * 1.0e6) ** 2 / J_plastic if K_pl > 0.0 else 0.0
    Eprime = max(Eprime, 1.0e-30)

    J_apparent = J_elastic + J_plastic
    K_elastic = math.sqrt(Eprime * J_elastic) / 1.0e6
    K_apparent = math.sqrt(Eprime * J_apparent) / 1.0e6

    classification = str(
        audit.get("classification", "plasticity_dominated_no_crack_growth")
    )
    audit.update(
        {
            "schema": "v10.4.4_plasticity_dominated_campaign_terminal_audit_v2",
            "campaign_model_id": MODEL_ID,
            "campaign_terminal_model_id": CAMPAIGN_TERMINAL_MODEL_ID,
            "campaign_classification": classification,
            "J_elastic_positive_J_per_m2": J_elastic,
            "J_plastic_dissipation_J_per_m2": J_plastic,
            "J_apparent_total_J_per_m2": J_apparent,
            "K_elastic_equivalent_MPa_sqrt_m": K_elastic,
            "K_plastic_equivalent_MPa_sqrt_m": math.sqrt(
                Eprime * J_plastic
            )
            / 1.0e6,
            "K_apparent_plasticity_limited_MPa_sqrt_m": K_apparent,
            "apparent_toughness_label": (
                "plasticity_limited_apparent_toughness_not_K_IC"
            ),
            "J_apparent_definition": "J_elastic_positive_terminal_plus_J_pl_diss",
            "J_plastic_definition": (
                "cumulative_accepted_bulk_plastic_work_divided_by_"
                "unit_thickness_initial_ligament"
            ),
            "J_apparent_enters_cleavage_hazard": False,
            "J_apparent_enters_fracture_energy_gate": False,
            "projected_cleavage_action_role": "diagnostic_only",
        }
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    marker = root / "PLASTICITY_DOMINATED"
    marker.write_text(classification + "\n")

    summary_path = root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if isinstance(summary, list):
            for record in summary:
                if not isinstance(record, dict):
                    continue
                if record.get("campaign_terminal") is True or record.get("mode") == "plastic-flow":
                    record.update(
                        {
                            "terminal_status": classification,
                            "campaign_classification": classification,
                            "J_elastic_positive_J_per_m2": J_elastic,
                            "J_plastic_dissipation_J_per_m2": J_plastic,
                            "J_apparent_total_J_per_m2": J_apparent,
                            "K_elastic_equivalent_MPa_sqrt_m": K_elastic,
                            "K_plastic_equivalent_MPa_sqrt_m": math.sqrt(
                                Eprime * J_plastic
                            )
                            / 1.0e6,
                            "K_apparent_plasticity_limited_MPa_sqrt_m": K_apparent,
                        }
                    )
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )


def _rewrite_model_audit(root: Path) -> None:
    _v1043._rewrite_model_audit(root)
    path = root / "v10_4_bulk_coupled_model_audit.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    payload.update(
        {
            "schema": MODEL_ID,
            "plasticity_dominated_campaign_terminal_model": (
                CAMPAIGN_TERMINAL_MODEL_ID
            ),
            "plasticity_dominated_terminal_acceptance": [
                "no_crack_event_in_recent_nominal_loading_window",
                "negligible_crack_extension_in_recent_window",
                "bulk_plastic_work_dominates_recent_external_work",
                "recent_elastic_storage_is_flat",
                "incremental_load_carrying_stiffness_is_collapsed",
            ],
            "plasticity_dominated_terminal_allows_prior_first_passage": True,
            "plasticity_dominated_terminal_allows_partial_crack_growth": True,
            "plasticity_dominated_terminal_projected_hazard_role": (
                "diagnostic_only_not_acceptance_criterion"
            ),
            "plasticity_dominated_terminal_cleavage_clock_role": (
                "diagnostic_only_not_acceptance_criterion"
            ),
            "plasticity_dominated_terminal_J_and_sigma_role": (
                "diagnostic_only_not_zero_gates"
            ),
            "default_plasticity_terminal_window_nominal_increments": 32,
            "apparent_toughness_output": (
                "K_apparent=sqrt(Eprime*(J_elastic_positive+J_pl_diss))"
            ),
            "apparent_toughness_is_K_IC": False,
            "fracture_hazard_unchanged": True,
            "fracture_event_energy_gate_unchanged": True,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    _prepare_args(args)
    transformed = load_transformed_sharp_front()

    bulk_entry = _v1043._v1041._entry
    original_sharp_front = bulk_entry._v101.sharp_front
    bulk_entry._v101.sharp_front = transformed
    try:
        print(
            "  v10.4.4 campaign model: full-field bulk plasticity remains "
            "coupled to the FEM stress and directional J; sharp fracture follows "
            "the unchanged Arrhenius first-passage and event-energy gate; a case "
            "terminates as plasticity dominated after a recent no-growth window "
            "when bulk plastic work dominates, elastic storage is flat, and the "
            "incremental load-carrying stiffness has collapsed; projected future "
            "cleavage action, finite elastic J, and tip stress are diagnostics only; "
            "the terminal may occur before first passage or after partial crack growth"
        )
        result = _v1043._v1041.main(args)
        out = _option_value(args, "--out")
        if out:
            root = Path(out)
            _rewrite_model_audit(root)
            _rewrite_terminal_outputs(root)
        return result
    finally:
        bulk_entry._v101.sharp_front = original_sharp_front


if __name__ == "__main__":
    main()
