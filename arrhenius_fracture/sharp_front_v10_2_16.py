"""v10.2.16 Stage-3 parameter overlay on the accepted 2-D continuum model.

The v9.11.1 four-option registry is retained verbatim. This entry repairs only
one wiring defect in v10.2.15: the anisotropic installer had replaced the
accepted continuum tip-source activity law with a finite-site depletion law.
No material parameter, FEM/J calculation, tensor drive, transport operator,
shielding law, or crack-geometry transaction is changed here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from . import anisotropic_emission_v10174 as _anisotropic
from . import sharp_front_v10_2_15 as _base
from .anisotropic_continuum_source_v10216 import (
    MODEL_ID as SOURCE_MODEL_ID,
    SOURCE_MODEL,
    audit_payload as source_audit_payload,
    install_anisotropic_continuum_emission,
)

MODEL_ID = "v10.2.16_stage3_existing_2d_continuum_source_parameter_overlay"
FINAL_2D_ENTRY = _base.FINAL_2D_ENTRY


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _write_v10216_audit(args: list[str], selected: Any, manifest_path: Path,
                          selection_audit_path: Path) -> None:
    out_value = _option_value(args, "--out")
    if not out_value:
        return
    root = Path(out_value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MODEL_ID,
        "final_2d_entry": FINAL_2D_ENTRY,
        "parameter_overlay_only": True,
        "selected_option": selected.audit_payload(),
        "selected_material_manifest": str(manifest_path),
        "parameter_selection_audit": str(selection_audit_path),
        "continuum_source": source_audit_payload(),
        "preserved_physics": {
            "four_option_parameter_rows": True,
            "tensor_resolved_anisotropic_emission_drive": True,
            "validated_scalar_peierls_taylor_transport": True,
            "local_mobile_retained_backstress": True,
            "continuum_activity_and_clearing": True,
            "crack_advance_geometric_source_renewal": True,
            "sharp_wake_geometry_backend": True,
            "tip_engine_substitution_beyond_source_wiring": False,
            "material_barrier_refit": False,
            "transport_operator_replaced": False,
            "shielding_law_replaced": False,
            "geometry_backend_replaced": False,
        },
    }
    (root / "v10_2_16_continuum_source_parameter_overlay.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    source_path = root / "v10_1_1_source_model.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    source.update({
        "schema": MODEL_ID,
        "tip_source_model": SOURCE_MODEL,
        "tip_source_model_id": SOURCE_MODEL_ID,
        "finite_distributed_source_inventory": False,
        "source_sites_per_system_role": "low_rate_arrhenius_hazard_multiplicity",
        "source_multiplicity_consumed": False,
        "source_activity_state": "dimensionless_per_crystallographic_tip_channel",
        "stationary_activity_recovery": "peierls_clearing_over_current_tip_radius",
        "crack_advance_activity_recovery": "geometric_renewal_over_current_tip_radius",
        "available_sites_field_semantics": "derived_M_ref_times_activity_proxy",
    })
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")

    transport_path = root / "v10_1_7_5_transport_mode.json"
    transport = json.loads(transport_path.read_text()) if transport_path.exists() else {}
    transport.update({
        "schema": MODEL_ID,
        "continuum_source_lifecycle_preserved": True,
        "finite_source_budget_active": False,
        "source_multiplicity_consumed": False,
        "transport_operator_changed_by_v10_2_16": False,
    })
    transport_path.write_text(json.dumps(transport, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None):
    args = list(sys.argv[1:] if argv is None else argv)
    original_install = _anisotropic.install_anisotropic_campaign_emission
    original_model_id = _base.MODEL_ID
    _anisotropic.install_anisotropic_campaign_emission = install_anisotropic_continuum_emission
    _base.MODEL_ID = MODEL_ID
    try:
        selected, manifest_path, selection_audit_path = _base._prepare_parameter_option(args)
        _base._force_stage3_validity_envelope(args)
        print(
            "  v10.2.16 parameter overlay only: "
            f"entry={FINAL_2D_ENTRY} option={selected.option_key} "
            f"candidate={selected.candidate_id} "
            f"mpz={selected.mpz_length_um:g}um/{selected.mpz_n_bins}bins "
            "source=continuum_activity_clearing "
            "finite_inventory=0 multiplicity_consumed=0"
        )
        result = _base._final_2d.main(args)
        _write_v10216_audit(args, selected, manifest_path, selection_audit_path)
        return result
    finally:
        _anisotropic.install_anisotropic_campaign_emission = original_install
        _base.MODEL_ID = original_model_id


if __name__ == "__main__":
    main()


__all__ = ["MODEL_ID", "FINAL_2D_ENTRY", "main"]
