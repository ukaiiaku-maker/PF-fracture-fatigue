"""Audited v10.4.2 entry with plastic-flow termination and contour-J diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_4_bulk_peierls_taylor_audited as _v1041
from .plastic_flow_terminal_v1042 import (
    MODEL_ID as TERMINAL_MODEL_ID,
    load_transformed_sharp_front,
)

MODEL_ID = "v10.4.2_bulk_detailed_balance_plastic_flow_terminal"


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
        raise SystemExit("v10.4.2 plastic-flow terminal is monotonic-only")
    if not _has_option(args, "--plastic-flow-terminal"):
        args.append("--plastic-flow-terminal")


def _rewrite_model_audit(root: Path) -> None:
    path = root / "v10_4_bulk_coupled_model_audit.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    payload.update(
        {
            "schema": MODEL_ID,
            "plastic_flow_terminal_model": TERMINAL_MODEL_ID,
            "plastic_flow_terminal_enabled": True,
            "plastic_flow_status": "plastic_flow_no_sharp_fracture",
            "plastic_flow_is_successful_campaign_terminal": True,
            "ductile_fracture_simulated": False,
            "fracture_measure": "positive_signed_configurational_J_only",
            "bulk_plastic_work_enters_fracture_measure": False,
            "bulk_plastic_work_enters_cleavage_hazard": False,
            "bulk_plastic_work_enters_energy_gate": False,
            "J_pl_diss_definition": "W_bulk_plastic/(unit_thickness*initial_ligament)",
            "J_pl_diss_role": "temperature_dependent_plastic_dissipation_diagnostic",
            "contour_shielding_definition": "max(J_outer_positive-J_tip_positive,0)",
            "contour_shielding_role": "diagnostic_only",
            "contour_shielding_enters_fracture_hazard": False,
            "v10_4_1_native_complete_cases_physics_compatible": True,
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
            "  v10.4.2 terminal model: plastic_flow_no_sharp_fracture; "
            "fracture J unchanged; J_pl and contour shielding diagnostic only"
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
