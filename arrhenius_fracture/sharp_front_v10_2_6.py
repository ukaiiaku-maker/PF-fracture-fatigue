"""v10.2.6 state-resolved signed-kernel entry point for fracture and fatigue."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

from . import anisotropic_emission_v10174 as _anisotropic
from . import sharp_front_v10_1_7_4 as _entry74
from . import sharp_front_v10_1_7_5 as _transport
from .state_equivalence_trace_v1025 import (
    capture_exact_signed_trace,
    write_exact_signed_trace,
)
from .state_resolved_signed_engine_v1026 import (
    MODEL_ID,
    StateResolvedSignedBurgersTipEngine,
)
from .signed_kernel_family_v1026 import (
    StateResolvedSignedShieldingKernelFamily,
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
            "local_cohesive_strength_sigma_cap_preserved": True,
            "local_strength_limit_is_not_Kshield_cap": True,
            "constitutive_K_shield_cap_applied": False,
            "state_resolved_kernel_family": family.audit_payload(),
            "production_parameterization_allowed": bool(
                family.metadata.get("production_parameterization_allowed", False)
            ),
        }
    )
    (root / "v10_2_6_state_resolved_signed_physics.json").write_text(
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
            "v10.2.6 requires --signed-kernel-family PATH or "
            "SIGNED_KERNEL_FAMILY_JSON. A one-state or unsigned shielding fallback "
            "is not permitted."
        )
    family = StateResolvedSignedShieldingKernelFamily.from_json(family_path)
    if not bool(family.metadata.get("production_parameterization_allowed", False)):
        # Mechanics/kernels may be exercised in explicit validation runs, but a
        # parameter campaign must not use an artifact that has not passed the
        # complete state-envelope assessment.
        if _bool_env("PARAMETER_CAMPAIGN", False):
            raise SystemExit(
                "state-resolved kernel family has not authorized production "
                "parameterization; complete the interaction-integral, amplitude, "
                "normalization, and state-envelope gates first"
            )
    transport_mode = _transport.normalize_transport_mode(
        os.environ.get("ANISOTROPIC_TRANSPORT_MODE")
    )
    StateResolvedSignedBurgersTipEngine.configure_state_resolved_physics(
        family, transport_mode
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
            "  v10.2.6 state-resolved signed physics: "
            f"loading={loading} transport={transport_mode} "
            f"states={len(family.states)} interpolation="
            f"{family.interpolation.get('method')} Kcap=off"
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
