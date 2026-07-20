"""v10.2.17 Stage 3 overlay on the final v10.2.14 signed 2-D stack.

Only the exact selected v9.11.1 material row and its recommended MPZ grid are
changed.  The active-only FEM kernel, signed-Burgers state, uncapped shielding,
effective-opening fixed point, anisotropic emission, moving MPZ, and stochastic
cleavage renewal/event-distance law are inherited from the final v10.2 stack.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from typing import Any

from . import anisotropic_emission_v10174 as _anisotropic
from . import sharp_front_v10_1_7_4 as _entry74
from . import sharp_front_v10_1_7_5 as _transport
from . import zero_event_summary_v10215 as _zero_event_summary  # noqa: F401
from .parameter_registry_v9111 import (
    CANONICAL_STAGE3_OPTIONS,
    SelectedResponseOption,
    default_registry_path,
    select_option,
    write_compatibility_manifest,
    write_selection_audit,
)
from .signed_kernel_family_v10214 import ActiveOnlySigned2DShieldingKernelFamily
from .state_equivalence_trace_v1025 import (
    capture_exact_signed_trace,
    write_exact_signed_trace,
)
from .state_resolved_signed_engine_v10214 import (
    MODEL_ID as ENGINE_MODEL_ID,
    StateResolvedSignedBurgersTipEngine,
)

MODEL_ID = "v10.2.17_stage3_final_signed_stochastic_parameter_overlay"
FINAL_ENGINE = "arrhenius_fracture.state_resolved_signed_engine_v10214"
FINAL_FAMILY = "arrhenius_fracture.signed_kernel_family_v10214"
LOW_LEVEL_DRIVER = "arrhenius_fracture.sharp_front_v10_1_7_5"


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


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None else raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default if raw is None else raw)


def _require_env_value(name: str, expected: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower().replace("-", "_")
    normalized_expected = expected.strip().lower().replace("-", "_")
    if value != normalized_expected:
        raise SystemExit(f"v10.2.17 Stage 3 requires {name}={expected}; got {value!r}")
    return value


def _require_stochastic_seed() -> int:
    raw = os.environ.get("CLEAVAGE_HAZARD_SEED", "").strip()
    if not raw:
        raise SystemExit("v10.2.17 Stage 3 requires explicit CLEAVAGE_HAZARD_SEED")
    try:
        seed = int(raw)
    except ValueError as exc:
        raise SystemExit("CLEAVAGE_HAZARD_SEED must be an integer") from exc
    if seed < 0:
        raise SystemExit("CLEAVAGE_HAZARD_SEED must be nonnegative")
    return seed


def _force_stage3_validity_envelope(args: list[str]) -> int:
    mode = _option_value(args, "--mode", "2d")
    if mode != "2d":
        raise SystemExit("v10.2.17 Stage 3 supports only --mode 2d")
    _set_value_option(args, "--mode", "2d")
    if _has_option(args, "--fatigue-cycles"):
        raise SystemExit("v10.2.17 Stage 3 is monotonic; fatigue options are not permitted")
    if "--crystal-branch" in args:
        raise SystemExit("v10.2.17 Stage 3 requires branching disabled")
    supplied_fronts = _option_value(args, "--max-fronts")
    if supplied_fronts is not None and int(supplied_fronts) != 1:
        raise SystemExit("v10.2.17 Stage 3 requires --max-fronts 1")
    _set_value_option(args, "--max-fronts", 1)
    if "--wake-shielding" in args:
        raise SystemExit("v10.2.17 active-only atlas does not support wake shielding")
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
            raise SystemExit(f"v10.2.17 Stage 3 requires {name} {expected}")
        _set_value_option(args, name, expected)
    if "--crystal-aniso" not in args:
        raise SystemExit("v10.2.17 Stage 3 requires --crystal-aniso")
    if "--crystal-compete" not in args:
        raise SystemExit("v10.2.17 Stage 3 requires --crystal-compete")
    _require_env_value("CLEAVAGE_HAZARD_MODE", "exponential", "exponential")
    _require_env_value(
        "CLEAVAGE_EVENT_LENGTH_MODE", "threshold_scaled", "threshold_scaled"
    )
    _require_env_value(
        "ANISOTROPIC_TRANSPORT_MODE", "validated_scalar", "validated_scalar"
    )
    if not _bool_env("ANISOTROPIC_USE_AVALANCHE_BACKEND", True):
        raise SystemExit("v10.2.17 requires ANISOTROPIC_USE_AVALANCHE_BACKEND=1")
    return _require_stochastic_seed()


def _prepare_parameter_option(
    args: list[str],
) -> tuple[SelectedResponseOption, Path, Path]:
    out_value = _option_value(args, "--out")
    if not out_value:
        raise SystemExit("v10.2.17 requires --out so the selected registry row is persistent")
    output_root = Path(out_value).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    registry_value = _pop_value(
        args,
        "--parameter-registry",
        os.environ.get("PARAMETER_REGISTRY", str(default_registry_path())),
    )
    option_key = _pop_value(args, "--parameter-option", os.environ.get("PARAMETER_OPTION"))
    if not option_key:
        allowed = ", ".join(CANONICAL_STAGE3_OPTIONS)
        raise SystemExit(f"v10.2.17 requires --parameter-option; options: {allowed}")
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
        output_root / "v10_2_17_parameter_selection.json",
        compatibility_manifest=manifest_path,
        extra={
            "model_id": MODEL_ID,
            "final_engine": FINAL_ENGINE,
            "final_family": FINAL_FAMILY,
            "low_level_driver": LOW_LEVEL_DRIVER,
            "parameter_overlay_only": True,
            "material_parameter_refit_in_2d": False,
        },
    )
    return selected, manifest_path, audit_path


def _write_audit(
    args: list[str],
    family: ActiveOnlySigned2DShieldingKernelFamily,
    selected: SelectedResponseOption,
    manifest_path: Path,
    selection_audit_path: Path,
    transport_mode: str,
    seed: int,
) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = StateResolvedSignedBurgersTipEngine.audit_payload()
    payload.update(
        {
            "schema": MODEL_ID,
            "engine_model_id": ENGINE_MODEL_ID,
            "loading_path": "monotonic",
            "same_signed_engine_for_monotonic_and_fatigue": True,
            "final_engine": FINAL_ENGINE,
            "final_family": FINAL_FAMILY,
            "low_level_driver": LOW_LEVEL_DRIVER,
            "transport_mode": transport_mode,
            "cleavage_hazard_mode": "exponential",
            "cleavage_hazard_seed": seed,
            "event_length_mode": "threshold_scaled",
            "event_length_uses_same_integrated_hazard_threshold": True,
            "constitutive_K_shield_cap_applied": False,
            "signed_burgers_population_required": True,
            "effective_opening_fixed_point_enabled": True,
            "active_kernel_mechanically_measured": True,
            "wake_kernel_mechanically_measured": False,
            "wake_shielding_enabled": False,
            "physical_kernel_axes": ["cumulative_crack_path_extension_m"],
            "selected_option": selected.audit_payload(),
            "selected_material_manifest": str(manifest_path),
            "parameter_selection_audit": str(selection_audit_path),
            "state_resolved_kernel_family": family.audit_payload(),
            "production_parameterization_allowed": bool(
                family.metadata.get("production_parameterization_allowed", False)
            ),
        }
    )
    (root / "v10_2_17_final_signed_stochastic_stack.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    selected, manifest_path, selection_audit_path = _prepare_parameter_option(args)
    seed = _force_stage3_validity_envelope(args)
    family_path = _pop_value(
        args,
        "--signed-kernel-family",
        os.environ.get("SIGNED_KERNEL_FAMILY_JSON"),
    )
    if not family_path:
        raise SystemExit(
            "v10.2.17 requires --signed-kernel-family PATH or SIGNED_KERNEL_FAMILY_JSON"
        )
    family = ActiveOnlySigned2DShieldingKernelFamily.from_json(family_path)
    if family.metadata.get("production_parameterization_allowed") is not True:
        raise SystemExit("v10.2.17 requires a production-authorized v10.2.14 family")
    transport_mode = _transport.normalize_transport_mode(
        os.environ.get("ANISOTROPIC_TRANSPORT_MODE")
    )
    StateResolvedSignedBurgersTipEngine.configure_state_resolved_physics(
        family,
        transport_mode,
        fixed_point_tolerance=_float_env("SIGNED_KERNEL_FIXED_POINT_TOL", 1.0e-8),
        fixed_point_max_iterations=_int_env("SIGNED_KERNEL_FIXED_POINT_MAX_ITER", 80),
        fixed_point_damping=_float_env("SIGNED_KERNEL_FIXED_POINT_DAMPING", 0.5),
    )

    original_anisotropic = _anisotropic.AnisotropicStochasticAvalancheTipEngine
    original_entry = _entry74.AnisotropicStochasticAvalancheTipEngine
    _anisotropic.AnisotropicStochasticAvalancheTipEngine = StateResolvedSignedBurgersTipEngine
    _entry74.AnisotropicStochasticAvalancheTipEngine = StateResolvedSignedBurgersTipEngine
    capture = _bool_env("SIGNED_STATE_TRACE", False)
    trace_context = capture_exact_signed_trace() if capture else nullcontext(None)
    try:
        print(
            "  v10.2.17 final signed stochastic Stage 3: "
            f"option={selected.option_key} candidate={selected.candidate_id} "
            f"mpz={selected.mpz_length_um:g}um/{selected.mpz_n_bins}bins "
            f"seed={seed} states={len(family.states)} Kcap=off wake=off"
        )
        with trace_context as trace:
            result = _transport.main(args)
        _write_audit(
            args,
            family,
            selected,
            manifest_path,
            selection_audit_path,
            transport_mode,
            seed,
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
