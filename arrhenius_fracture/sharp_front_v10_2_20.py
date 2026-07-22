"""v10.2.20: v9.12 top-five DBTT-peak screen on frozen v10.2.17 physics.

The five parameterizations are evaluated in the exact v10.2.18/v10.2.17
2-D tip-only model.  This wrapper changes only the selected material row and
its recommended 50 um / 80-bin MPZ discretization.

The v9.12 campaign also exported ``rho_source0_m2`` and Arrhenius recovery
barrier fields.  Those fields are retained in the registry and case audit, but
they are not silently activated because the frozen v10.2.17 executable has no
corresponding constitutive interface.  The executable uses the explicit
``source_sites_per_system`` and ``retained_recovery_rate_s`` fields instead.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_17 as _base
from .parameter_registry_v9111 import (
    SelectedResponseOption,
    select_option,
    write_compatibility_manifest,
    write_selection_audit,
)

MODEL_ID = "v10.2.20_v912_top5_peak_screen"
BASE_ENTRY = "arrhenius_fracture.sharp_front_v10_2_17"
V912_OPTIONS = (
    "v912_peak_0368",
    "v912_peak_0314",
    "v912_peak_0162",
    "v912_late_0118",
    "v912_plateau_0403",
)


def default_v912_registry_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "materials"
        / "v9_12"
        / "v9_12_top5_v10220_registry.csv"
    )


def _optional_number(row: dict[str, str], key: str) -> float | None:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None
    return float(raw)


def _prepare_v912_option(
    args: list[str],
) -> tuple[SelectedResponseOption, Path, Path]:
    out_value = _base._option_value(args, "--out")
    if not out_value:
        raise SystemExit("v10.2.20 requires --out")
    output_root = Path(out_value).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    registry_value = _base._pop_value(
        args,
        "--parameter-registry",
        str(default_v912_registry_path()),
    )
    option_key = _base._pop_value(args, "--parameter-option")
    if option_key not in V912_OPTIONS:
        allowed = ", ".join(V912_OPTIONS)
        raise SystemExit(f"v10.2.20 option must be one of: {allowed}")

    selected = select_option(
        option_key,
        registry_value,
        canonical_stage3_only=False,
    )
    if selected.material_class.strip().lower() != "dbtt":
        raise SystemExit(
            f"v10.2.20 accepts only DBTT rows; {option_key!r} is "
            f"{selected.material_class!r}"
        )
    if selected.mpz_length_um != 50.0 or selected.mpz_n_bins != 80:
        raise SystemExit(
            "v10.2.20 requires the v9.12 50 um / 80-bin MPZ contract; "
            f"got {selected.mpz_length_um:g} um / {selected.mpz_n_bins} bins"
        )

    row = selected.row
    inactive_campaign_fields = {
        "rho_source0_m2": _optional_number(
            row, "v912_rho_source0_m2_audit_only"
        ),
        "recovery_nu0_s": _optional_number(
            row, "v912_recovery_nu0_s_audit_only"
        ),
        "recovery_H0_eV": _optional_number(
            row, "v912_recovery_H0_eV_audit_only"
        ),
        "recovery_activation_entropy_kB": _optional_number(
            row, "v912_recovery_activation_entropy_kB_audit_only"
        ),
    }
    active_source_recovery = {
        "source_sites_per_system": float(row["source_sites_per_system"]),
        "source_refresh_length_um": float(row["source_refresh_length_um"]),
        "retained_recovery_rate_s": float(row["retained_recovery_rate_s"]),
        "source_recovery_rate_s": float(row["source_recovery_rate_s"]),
    }

    _base._remove_value_option(args, "--material-class")
    manifest_path = write_compatibility_manifest(
        selected,
        output_root / "selected_material_manifest_v9_12.csv",
    )
    _base._set_value_option(args, "--material-manifest", manifest_path)
    _base._set_value_option(args, "--mpz-length-um", selected.mpz_length_um)
    _base._set_value_option(args, "--mpz-n-bins", selected.mpz_n_bins)

    audit_path = write_selection_audit(
        selected,
        output_root / "v10_2_20_v912_parameter_selection.json",
        compatibility_manifest=manifest_path,
        extra={
            "model_id": MODEL_ID,
            "base_entry": BASE_ENTRY,
            "parameter_overlay_only": True,
            "mechanics_changed": False,
            "source_model_changed": False,
            "signed_kernel_changed": False,
            "stochastic_law_changed": False,
            "dbtt_interpretation": (
                "peak cleavage resistance followed by high-temperature ductile "
                "softening; high-temperature K-like values are not automatically K_IC"
            ),
            "active_source_and_recovery_fields": active_source_recovery,
            "campaign_fields_retained_for_audit_but_inactive_in_frozen_solver": (
                inactive_campaign_fields
            ),
            "inactive_field_policy": "fail_visible_not_silent",
            "source_exhausted_control": str(
                row.get("v912_source_exhausted_control", "false")
            ).strip().lower() == "true",
            "intended_peak_delta_K_micro_MPa_sqrt_m": _optional_number(
                row, "v912_intended_peak_delta_K_micro_MPa_sqrt_m"
            ),
            "intended_peak_temperature_K": _optional_number(
                row, "v912_intended_peak_temperature_K"
            ),
            "intended_delta_K_micro_1200K_MPa_sqrt_m": _optional_number(
                row, "v912_intended_delta_K_micro_1200K_MPa_sqrt_m"
            ),
            "J_reporting_policy": {
                "J_c": "mechanical J at first accepted cleavage event",
                "K_Jc": "sqrt(Eprime*J_c), equivalent only",
                "J_init_stable_tearing": "not represented by cleavage-only geometry law",
                "CTOD": "not emitted by frozen v10.2.17 solver",
                "load_displacement_work": "computed by postprocessor from Ftop(Uapp)",
            },
        },
    )
    return selected, manifest_path, audit_path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    original_prepare = _base._prepare_parameter_option
    _base._prepare_parameter_option = _prepare_v912_option
    try:
        result = _base.main(args)
    finally:
        _base._prepare_parameter_option = original_prepare

    out = _base._option_value(args, "--out")
    option = _base._option_value(args, "--parameter-option")
    if out:
        payload = {
            "schema": MODEL_ID,
            "base_entry": BASE_ENTRY,
            "selected_option": option,
            "candidate_screen_only": True,
            "existing_v10_2_17_physics_preserved": True,
            "bulk_plasticity_mode": "tip_only",
            "high_temperature_K_label": "cleavage-equivalent K, not presumed K_IC",
        }
        (Path(out) / "v10_2_20_v912_peak_screen.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return result


if __name__ == "__main__":
    main()
