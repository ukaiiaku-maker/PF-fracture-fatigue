"""v10.2.15 Stage 3 four-option monotonic production entry.

Combines the mechanically measured v10.2.14 active-only signed FEM kernel
family with one exact row from the v9.11.1 MPZ parameter registry. Restricted
to deterministic, fixed-length, single-front, branching-disabled monotonic
calculations. Wake state may remain in kinetic bookkeeping, but wake shielding
is disabled because no measured signed 2-D wake operator is available.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from . import anisotropic_emission_v10174 as _anisotropic
from . import sharp_front_v10_1_7_4 as _entry74
from . import sharp_front_v10_1_7_5 as _transport
from .parameter_registry_v9111 import (
    CANONICAL_STAGE3_OPTIONS,
    SelectedResponseOption,
    default_registry_path,
    select_option,
    write_compatibility_manifest,
    write_selection_audit,
)
from .signed_kernel_family_v10214 import ActiveOnlySigned2DShieldingKernelFamily
from .state_equivalence_trace_v1025 import capture_exact_signed_trace, write_exact_signed_trace
from .state_resolved_signed_engine_v10214 import (
    MODEL_ID as ENGINE_MODEL_ID,
    StateResolvedSignedBurgersTipEngine,
)

MODEL_ID = "v10.2.15_stage3_four_option_active_only_signed"


def _pop_value(args: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for index, token in enumerate(list(args)):
        if token.startswith(prefix):
            value = token[len(prefix):]
            del args[index]
            return value
        if token == name:
            if index + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[index + 1]
            del args[index:index + 2]
            return value
    return default


def _option_value(args: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return default


def _remove_value_option(args: list[str], name: str) -> None:
    prefix = name + "="
    kept: list[str] = []
    skip = False
    for token in args:
        if skip:
            skip = False
            continue
        if token == name:
            skip = True
            continue
        if token.startswith(prefix):
            continue
        kept.append(token)
    args[:] = kept


def _set_value_option(args: list[str], name: str, value: Any) -> None:
    _remove_value_option(args, name)
    args.extend([name, str(value)])


def _set_toggle(args: list[str], positive: str, negative: str, enabled: bool) -> None:
    args[:] = [token for token in args if token not in {positive, negative}]
    args.append(positive if enabled else negative)


def _has_option(args: list[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in args)


def _require_env_value(name: str, expected: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower().replace("-", "_")
    normalized_expected = expected.strip().lower().replace("-", "_")
    if value != normalized_expected:
        raise SystemExit(f"v10.2.15 Stage 3 requires {name}={expected}; got {value!r}")
    return value


def _force_stage3_validity_envelope(args: list[str]) -> None:
    mode = _option_value(args, "--mode", "2d")
    if mode != "2d":
        raise SystemExit("v10.2.15 Stage 3 supports only --mode 2d")
    _set_value_option(args, "--mode", "2d")
    if _has_option(args, "--fatigue-cycles"):
        raise SystemExit("v10.2.15 Stage 3 is monotonic; fatigue options are not permitted")
    if "--crystal-branch" in args:
        raise SystemExit("v10.2.15 Stage 3 requires branching disabled")
    supplied_fronts = _option_value(args, "--max-fronts")
    if supplied_fronts is not None and int(supplied_fronts) != 1:
        raise SystemExit("v10.2.15 Stage 3 requires --max-fronts 1")
    _set_value_option(args, "--max-fronts", 1)
    if "--wake-shielding" in args:
        raise SystemExit("v10.2.15 has no measured wake kernel; use --no-wake-shielding")
    _set_toggle(args, "--wake-shielding", "--no-wake-shielding", False)
    mobile_fraction = _option_value(args, "--mobile-shield-fraction")
    if mobile_fraction is not None and abs(float(mobile_fraction)) > 1.0e-15:
        raise SystemExit("v9.11.1 Stage 3 requires --mobile-shield-fraction 0")
    _set_value_option(args, "--mobile-shield-fraction", 0.0)
    controls = {
        "--bulk-plasticity-mode": "tip_only",
        "--directional-j-mode": "root_signed",
        "--tip-kinetics-mode": "moving_velocity",
        "--tip-source-model": "continuum",
        "--front-state-model": "moving_pz",
    }
    for name, expected in controls.items():
        supplied = _option_value(args, name)
        if supplied is not None and supplied.strip().lower() != expected:
            raise SystemExit(f"v10.2.15 Stage 3 requires {name} {expected}")
        _set_value_option(args, name, expected)
    if "--crystal-aniso" not in args:
        raise SystemExit("v10.2.15 Stage 3 requires --crystal-aniso")
    if "--crystal-compete" not in args:
        raise SystemExit("v10.2.15 Stage 3 requires --crystal-compete")
    _require_env_value("CLEAVAGE_HAZARD_MODE", "deterministic", "deterministic")
    _require_env_value("CLEAVAGE_EVENT_LENGTH_MODE", "fixed", "fixed")
    _require_env_value("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar", "validated_scalar")


def _prepare_parameter_option(args: list[str]) -> tuple[SelectedResponseOption, Path, Path]:
    out_value = _option_value(args, "--out")
    if not out_value:
        raise SystemExit("v10.2.15 requires --out so the selected registry row is persistent")
    output_root = Path(out_value).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    registry_value = _pop_value(
        args, "--parameter-registry",
        os.environ.get("PARAMETER_REGISTRY", str(default_registry_path())),
    )
    option_key = _pop_value(args, "--parameter-option", os.environ.get("PARAMETER_OPTION"))
    if not option_key:
        allowed = ", ".join(CANONICAL_STAGE3_OPTIONS)
        raise SystemExit(f"v10.2.15 requires --parameter-option; Stage 3 options: {allowed}")
    selected = select_option(option_key, registry_value, canonical_stage3_only=True)
    _remove_value_option(args, "--material-class")
    manifest_path = write_compatibility_manifest(
        selected, output_root / "selected_material_manifest_v9_11_1.csv"
    )
    _set_value_option(args, "--material-manifest", manifest_path)
    _set_value_option(args, "--mpz-length-um", selected.mpz_length_um)
    _set_value_option(args, "--mpz-n-bins", selected.mpz_n_bins)
    audit_path = write_selection_audit(
        selected,
        output_root / "v10_2_15_parameter_selection.json",
        compatibility_manifest=manifest_path,
        extra={
            "model_id": MODEL_ID,
            "canonical_stage3_option": True,
            "single_front_unbranched_monotonic": True,
            "mobile_shield_fraction": 0.0,
            "wake_shielding_enabled": False,
            "parameter_values_refit_in_2d": False,
        },
    )
    return selected, manifest_path, audit_path


def _write_run_audit(
    args: list[str],
    family: ActiveOnlySigned2DShieldingKernelFamily,
    selected: SelectedResponseOption,
    manifest_path: Path,
    selection_audit_path: Path,
    transport_mode: str,
) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    payload = StateResolvedSignedBurgersTipEngine.audit_payload()
    payload.update({
        "schema": MODEL_ID,
        "engine_model_id": ENGINE_MODEL_ID,
        "loading_path": "monotonic",
        "same_engine_for_monotonic_and_fatigue": True,
        "transport_mode": transport_mode,
        "single_front_unbranched_geometry_only": True,
        "maximum_fronts_forced": 1,
        "active_kernel_mechanically_measured": True,
        "wake_kernel_mechanically_measured": False,
        "wake_shielding_supported": False,
        "wake_shielding_enabled": False,
        "constitutive_K_shield_cap_applied": False,
        "mobile_shield_fraction": 0.0,
        "parameter_option": selected.audit_payload(),
        "selected_material_manifest": str(manifest_path),
        "parameter_selection_audit": str(selection_audit_path),
        "state_resolved_kernel_family": family.audit_payload(),
    })
    (root / "v10_2_15_stage3_shared_physics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    selected, manifest_path, selection_audit_path = _prepare_parameter_option(args)
    _force_stage3_validity_envelope(args)
    family_path = _pop_value(
        args, "--signed-kernel-family", os.environ.get("SIGNED_KERNEL_FAMILY_JSON")
    )
    if not family_path:
        raise SystemExit(
            "v10.2.15 requires --signed-kernel-family PATH or SIGNED_KERNEL_FAMILY_JSON"
        )
    family = ActiveOnlySigned2DShieldingKernelFamily.from_json(family_path)
    transport_mode = _transport.normalize_transport_mode(os.environ.get("ANISOTROPIC_TRANSPORT_MODE"))
    StateResolvedSignedBurgersTipEngine.configure_state_resolved_physics(
        family,
        transport_mode,
        fixed_point_tolerance=float(os.environ.get("SIGNED_KERNEL_FIXED_POINT_TOL", "1e-8")),
        fixed_point_max_iterations=int(os.environ.get("SIGNED_KERNEL_FIXED_POINT_MAX_ITER", "80")),
        fixed_point_damping=float(os.environ.get("SIGNED_KERNEL_FIXED_POINT_DAMPING", "0.5")),
    )
    original_anisotropic = _anisotropic.AnisotropicStochasticAvalancheTipEngine
    original_entry = _entry74.AnisotropicStochasticAvalancheTipEngine
    _anisotropic.AnisotropicStochasticAvalancheTipEngine = StateResolvedSignedBurgersTipEngine
    _entry74.AnisotropicStochasticAvalancheTipEngine = StateResolvedSignedBurgersTipEngine
    capture = os.environ.get("SIGNED_STATE_TRACE", "0").strip().lower() not in {
        "0", "false", "no", "off"
    }
    trace_context = capture_exact_signed_trace() if capture else None
    try:
        print(
            "  v10.2.15 Stage 3: "
            f"option={selected.option_key} candidate={selected.candidate_id} "
            f"mpz={selected.mpz_length_um:g}um/{selected.mpz_n_bins}bins "
            f"transport={transport_mode} fronts=1 wake_shield=0 mobile_shield=0"
        )
        if trace_context is None:
            result = _transport.main(args)
            trace = None
        else:
            with trace_context as trace:
                result = _transport.main(args)
        _write_run_audit(
            args, family, selected, manifest_path, selection_audit_path, transport_mode
        )
        out = _option_value(args, "--out")
        if capture and trace is not None and out:
            write_exact_signed_trace(trace, out)
        return result
    finally:
        _anisotropic.AnisotropicStochasticAvalancheTipEngine = original_anisotropic
        _entry74.AnisotropicStochasticAvalancheTipEngine = original_entry


if __name__ == "__main__":
    main()
