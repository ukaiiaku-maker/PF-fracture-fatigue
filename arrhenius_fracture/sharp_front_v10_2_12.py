"""v10.2.12 real signed 2-D atlas entry point for fracture and fatigue."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

from . import anisotropic_emission_v10174 as _anisotropic
from . import sharp_front_v10_1_7_4 as _entry74
from . import sharp_front_v10_1_7_5 as _transport
from .signed_kernel_family_v10212 import RealSigned2DShieldingKernelFamily
from .state_equivalence_trace_v1025 import (
    capture_exact_signed_trace,
    write_exact_signed_trace,
)
from .state_resolved_signed_engine_v10212 import (
    MODEL_ID,
    StateResolvedSignedBurgersTipEngine,
)


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


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


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


def _write_audit(args: list[str], family, transport_mode: str) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = StateResolvedSignedBurgersTipEngine.audit_payload()
    payload.update(
        {
            "schema": MODEL_ID,
            "loading_path": "fatigue" if "--fatigue-cycles" in args else "monotonic",
            "same_engine_for_monotonic_and_fatigue": True,
            "transport_mode": transport_mode,
            "kernel_state_uses_effective_local_opening": True,
            "kernel_radius_axis_policy": "disabled_constant_compatibility",
            "analytical_r_eff_used_for_interpolation": False,
            "physical_kernel_axes": [
                "opening_strength_fraction",
                "crack_extension_m",
            ],
            "local_cohesive_strength_sigma_cap_preserved": True,
            "local_strength_limit_is_not_Kshield_cap": True,
            "constitutive_K_shield_cap_applied": False,
            "state_resolved_kernel_family": family.audit_payload(),
            "production_parameterization_allowed": bool(
                family.metadata.get("production_parameterization_allowed", False)
            ),
        }
    )
    (root / "v10_2_12_real_signed_shared_physics.json").write_text(
        json.dumps(payload, indent=2)
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    family_path = _pop_value(
        args,
        "--signed-kernel-family",
        os.environ.get("SIGNED_KERNEL_FAMILY_JSON"),
    )
    if not family_path:
        raise SystemExit(
            "v10.2.12 requires --signed-kernel-family PATH or "
            "SIGNED_KERNEL_FAMILY_JSON. Static, unsigned, isotropic-shortcut, or "
            "unreviewed fallback operators are not permitted."
        )
    family = RealSigned2DShieldingKernelFamily.from_json(family_path)
    if not bool(family.metadata.get("production_parameterization_allowed", False)):
        if _bool_env("PARAMETER_CAMPAIGN", False):
            raise SystemExit(
                "v10.2.12 atlas has not authorized production parameterization; "
                "complete physical station, spatial projection, normalization, "
                "replay, and full 2-D fracture/fatigue validation gates"
            )
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
    _anisotropic.AnisotropicStochasticAvalancheTipEngine = (
        StateResolvedSignedBurgersTipEngine
    )
    _entry74.AnisotropicStochasticAvalancheTipEngine = (
        StateResolvedSignedBurgersTipEngine
    )
    capture = _bool_env("SIGNED_STATE_TRACE", False)
    trace_context = capture_exact_signed_trace() if capture else nullcontext(None)
    try:
        loading = "fatigue" if "--fatigue-cycles" in args else "monotonic"
        print(
            "  v10.2.12 real signed 2-D atlas: "
            f"loading={loading} transport={transport_mode} states={len(family.states)} "
            "axes=opening+extension analytical_r_axis=disabled Kcap=off"
        )
        with trace_context as trace:
            result = _transport.main(args)
        _write_audit(args, family, transport_mode)
        out = _option_value(args, "--out")
        if capture and trace is not None and out:
            write_exact_signed_trace(trace, out)
        return result
    finally:
        _anisotropic.AnisotropicStochasticAvalancheTipEngine = original_anisotropic
        _entry74.AnisotropicStochasticAvalancheTipEngine = original_entry


if __name__ == "__main__":
    main()
