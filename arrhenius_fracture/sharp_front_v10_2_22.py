"""v10.2.22 top-five DBTT screen with persistent sites and physical front width."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

from . import sharp_front_v10_2_17 as _base
from .parameter_registry_v9111 import (
    SelectedResponseOption,
    select_option,
    write_compatibility_manifest,
    write_selection_audit,
)
from .persistent_site_physical_width_v10222 import (
    MODEL_ID as WIDTH_MODEL_ID,
    install_physical_front_width,
)
from .persistent_site_source_v10221 import (
    MODEL_ID as SOURCE_MODEL_ID,
    PersistentSiteConfig,
    PersistentSiteStateResolvedTipEngine,
)


MODEL_ID = "v10.2.22_v912_top5_persistent_sites_physical_width"
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_22_v912_top5_persistent_site_registry.csv"
)
VALID_OPTIONS = {
    "v912_top1_peak_persistent_sites": "v912_targeted_local_peak_013476_0368",
    "v912_peak_0314_persistent_sites": "v912_targeted_local_peak_013476_0314",
    "v912_peak_0162_persistent_sites": "v912_targeted_local_peak_013476_0162",
    "v912_peak_0118_persistent_sites": "v912_targeted_local_peak_005518_0118",
    "v912_plateau_0403_persistent_sites": "v912_targeted_local_plateau_010759_0403",
}


def _number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"v10.2.22 registry field {key!r} is missing or nonnumeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"v10.2.22 registry field {key!r} is not finite")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None else raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default if raw is None else raw)


def _prepare_option(
    args: list[str],
) -> tuple[SelectedResponseOption, Path, Path]:
    out_value = _base._option_value(args, "--out")
    if not out_value:
        raise SystemExit("v10.2.22 requires --out")
    output_root = Path(out_value).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    registry_value = _base._pop_value(
        args,
        "--parameter-registry",
        os.environ.get("PARAMETER_REGISTRY", str(DEFAULT_REGISTRY)),
    )
    option_key = _base._pop_value(
        args,
        "--parameter-option",
        os.environ.get("PARAMETER_OPTION"),
    )
    if not option_key or option_key not in VALID_OPTIONS:
        allowed = ", ".join(sorted(VALID_OPTIONS))
        raise SystemExit(f"v10.2.22 requires one of --parameter-option: {allowed}")

    selected = select_option(option_key, registry_value, canonical_stage3_only=False)
    expected = VALID_OPTIONS[option_key]
    if selected.candidate_id != expected:
        raise SystemExit(
            f"v10.2.22 candidate fingerprint mismatch for {option_key!r}: "
            f"expected {expected!r}; got {selected.candidate_id!r}"
        )
    if selected.material_class.strip().lower() != "dbtt":
        raise SystemExit("v10.2.22 rows must have material_class=DBTT")

    row = selected.row
    zero_fields = (
        "source_recovery_rate_s",
        "retained_recovery_rate_s",
        "recovery_nu0_s",
        "legacy_source_sites_active",
        "legacy_source_refresh_active",
        "explicit_recovery_active",
    )
    for key in zero_fields:
        if abs(_number(row, key)) > 1.0e-30:
            raise SystemExit(f"v10.2.22 requires {key}=0; got {row[key]!r}")

    cfg = PersistentSiteConfig(
        rho_site0_m2=_number(row, "rho_source0_m2"),
        reference_source_area_m2=1.0e-12
        * _env_float(
            "PERSISTENT_SOURCE_REFERENCE_AREA_UM2",
            _number(row, "reference_source_area_um2"),
        ),
        reference_front_width_m=1.0e-6
        * _env_float(
            "PERSISTENT_SOURCE_REFERENCE_WIDTH_UM",
            _number(row, "reference_front_width_um"),
        ),
        reference_density_m2=_env_float(
            "PERSISTENT_SOURCE_REFERENCE_DENSITY_M2",
            _number(row, "rho_forest_floor_m2"),
        ),
        source_zone_length_m=1.0e-6
        * _env_float(
            "PERSISTENT_SOURCE_ZONE_LENGTH_UM",
            _number(row, "source_zone_length_um"),
        ),
        minimum_front_width_m=1.0e-6
        * _env_float("PERSISTENT_SOURCE_MIN_WIDTH_UM", 0.0),
        maximum_front_width_m=1.0e-6
        * _env_float("PERSISTENT_SOURCE_MAX_WIDTH_UM", selected.mpz_length_um),
        implicit_tolerance=_env_float("PERSISTENT_SOURCE_IMPLICIT_TOL", 1.0e-10),
        implicit_max_iterations=_env_int("PERSISTENT_SOURCE_IMPLICIT_MAX_ITER", 96),
    ).validate()

    install_physical_front_width()
    PersistentSiteStateResolvedTipEngine.configure_persistent_sites(cfg)

    _base._remove_value_option(args, "--material-class")
    manifest_path = write_compatibility_manifest(
        selected,
        output_root / "selected_material_manifest_v10_2_22.csv",
    )
    _base._set_value_option(args, "--material-manifest", manifest_path)
    _base._set_value_option(args, "--mpz-length-um", selected.mpz_length_um)
    _base._set_value_option(args, "--mpz-n-bins", selected.mpz_n_bins)

    audit_path = write_selection_audit(
        selected,
        output_root / "v10_2_22_parameter_selection.json",
        compatibility_manifest=manifest_path,
        extra={
            "model_id": MODEL_ID,
            "source_model_id": SOURCE_MODEL_ID,
            "front_width_model_id": WIDTH_MODEL_ID,
            "base_entry": "arrhenius_fracture.sharp_front_v10_2_17",
            "mechanics_changed": False,
            "signed_kernel_changed": False,
            "stochastic_cleavage_law_changed": False,
            "source_closure_changed": True,
            "finite_source_inventory": False,
            "legacy_source_sites_per_system_active": False,
            "legacy_source_refresh_length_active": False,
            "explicit_recovery_active": False,
            "persistent_site_config": cfg.__dict__,
            "front_width_grid_independent": True,
            "front_width_minimum_semantics": "max(explicit_physical_minimum,b)",
            "ahead_of_tip_dx_used_as_front_width_floor": False,
            "rho_source0_semantics": "persistent_areal_nucleation_site_density_m2",
        },
    )
    return selected, manifest_path, audit_path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    original_prepare = _base._prepare_parameter_option
    original_engine = _base.StateResolvedSignedBurgersTipEngine
    _base._prepare_parameter_option = _prepare_option
    _base.StateResolvedSignedBurgersTipEngine = PersistentSiteStateResolvedTipEngine
    try:
        result = _base.main(args)
        out = _base._option_value(args, "--out")
        if out:
            selection = json.loads(
                (Path(out) / "v10_2_22_parameter_selection.json").read_text()
            )
            selected_payload = selection.get("selected_option", {})
            payload = {
                "schema": MODEL_ID,
                "base_commit_model": "v10.2.21/v10.2.18 signed 2-D stack",
                "selected_option": selected_payload.get("option_key"),
                "selected_candidate": selected_payload.get("candidate_id"),
                "persistent_sites": True,
                "finite_source_inventory": False,
                "source_refresh": False,
                "backstress_limited_emission": True,
                "dynamic_tip_blunting": True,
                "moving_frame_resharpening": True,
                "front_width_grid_independent": True,
                "ahead_of_tip_dx_used_as_front_width_floor": False,
            }
            (Path(out) / "v10_2_22_persistent_site_model.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        return result
    finally:
        _base._prepare_parameter_option = original_prepare
        _base.StateResolvedSignedBurgersTipEngine = original_engine


if __name__ == "__main__":
    main()
