"""Audited entry for v10.4.1 full-field bulk Peierls--Taylor coupling."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_4_bulk_peierls_taylor as _entry
from .anisotropic_front_direction_fix_v10227 import install_front_direction_fix
from .energy_ledger_output_v10227 import (
    install_energy_ledger_output,
    restore_energy_ledger_output,
    write_energy_ledger_audit,
)
from .geometry_override_v10227 import install_geometry_override, restore_geometry_override
from .persistent_site_bracket_fix_v10221 import install_backstress_complementarity_fix
from .persistent_site_physical_width_v10222 import install_physical_front_width
from .thermodynamic_net_slip_v1041 import (
    MODEL_ID as DETAILED_BALANCE_MODEL_ID,
    install_detailed_balance_net_slip,
    restore_detailed_balance_net_slip,
    write_detailed_balance_audit,
)


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _rewrite_v1041_model_audit(root: Path) -> None:
    path = root / "v10_4_bulk_coupled_model_audit.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    payload.update(
        {
            "schema": "v10.4.1_bulk_detailed_balance_coupled_model",
            "bulk_net_slip_model": DETAILED_BALANCE_MODEL_ID,
            "one_way_arrhenius_rate_used_as_net_slip": False,
            "net_slip_rate": "Gamma_forward_minus_Gamma_reverse",
            "forward_reverse_barriers": "symmetric_about_zero_stress_barrier",
            "zero_stress_net_plastic_rate_exactly_zero": True,
            "v10_4_0_outputs_physics_compatible": False,
            "new_fitted_parameters_for_detailed_balance": 0,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    original_rate_method = install_detailed_balance_net_slip()
    install_front_direction_fix()
    install_backstress_complementarity_fix()
    install_physical_front_width()
    install_geometry_override()
    install_energy_ledger_output()
    try:
        print(
            "  v10.4.1 bulk net slip: detailed_balance "
            "Gamma_net=Gamma_forward-Gamma_reverse zero_stress_net=0"
        )
        result = _entry.main(args)
        out = _option_value(args, "--out")
        if out:
            root = Path(out)
            write_energy_ledger_audit(root)
            write_detailed_balance_audit(root)
            _rewrite_v1041_model_audit(root)
        return result
    finally:
        restore_detailed_balance_net_slip(original_rate_method)
        restore_energy_ledger_output()
        restore_geometry_override()


if __name__ == "__main__":
    main()
