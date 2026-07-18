"""v10.2.5 shared signed-Burgers entry point for monotonic fracture and fatigue."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

from . import anisotropic_emission_v10174 as _anisotropic
from . import sharp_front_v10_1_7_4 as _entry74
from . import sharp_front_v10_1_7_5 as _transport
from .signed_burgers_shared_v1025 import (
    MODEL_ID,
    SignedBurgersAnisotropicTipEngine,
    SignedShieldingKernel,
)
from .state_equivalence_trace_v1025 import (
    capture_exact_signed_trace,
    write_exact_signed_trace,
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


def _write_audit(args: list[str], kernel: SignedShieldingKernel, transport_mode: str) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = SignedBurgersAnisotropicTipEngine.audit_payload()
    payload.update({
        "schema": MODEL_ID,
        "loading_path": "fatigue" if "--fatigue-cycles" in args else "monotonic",
        "same_engine_for_monotonic_and_fatigue": True,
        "transport_mode": transport_mode,
        "local_cohesive_strength_sigma_cap_preserved": True,
        "local_strength_limit_is_not_Kshield_cap": True,
        "constitutive_K_shield_cap_applied": False,
        "source_sites_are_nucleation_opportunities": True,
        "emitted_population_is_signed_line_content": True,
        "kernel": kernel.audit_payload(),
    })
    (root / "v10_2_5_signed_shared_physics.json").write_text(
        json.dumps(payload, indent=2)
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    kernel_path = _pop_value(
        args,
        "--signed-shielding-kernel",
        os.environ.get("SIGNED_SHIELDING_KERNEL_JSON"),
    )
    if not kernel_path:
        raise SystemExit(
            "v10.2.5 requires --signed-shielding-kernel PATH or "
            "SIGNED_SHIELDING_KERNEL_JSON. The old unsigned +1/+1 operator is "
            "not an allowed fallback."
        )
    transport_mode = _transport.normalize_transport_mode(
        os.environ.get("ANISOTROPIC_TRANSPORT_MODE")
    )
    kernel = SignedShieldingKernel.from_json(kernel_path)
    SignedBurgersAnisotropicTipEngine.configure_signed_physics(
        kernel, transport_mode
    )

    original_anisotropic = _anisotropic.AnisotropicStochasticAvalancheTipEngine
    original_entry = _entry74.AnisotropicStochasticAvalancheTipEngine
    _anisotropic.AnisotropicStochasticAvalancheTipEngine = (
        SignedBurgersAnisotropicTipEngine
    )
    _entry74.AnisotropicStochasticAvalancheTipEngine = (
        SignedBurgersAnisotropicTipEngine
    )

    capture = _bool_env("SIGNED_STATE_TRACE", False)
    trace_context = capture_exact_signed_trace() if capture else nullcontext(None)
    try:
        loading = "fatigue" if "--fatigue-cycles" in args else "monotonic"
        print(
            "  v10.2.5 signed shared physics: "
            f"loading={loading} transport={transport_mode} "
            "population=positive+negative_Burgers_species "
            "shielding=2d_unit_response_kernel Kcap=off"
        )
        with trace_context as trace:
            result = _transport.main(args)
        _write_audit(args, kernel, transport_mode)
        out = _option_value(args, "--out")
        if capture and trace is not None and out:
            write_exact_signed_trace(trace, out)
        return result
    finally:
        _anisotropic.AnisotropicStochasticAvalancheTipEngine = original_anisotropic
        _entry74.AnisotropicStochasticAvalancheTipEngine = original_entry


if __name__ == "__main__":
    main()
