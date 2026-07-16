"""Safeguarded v10.1 entry point for the kinetic moving-tip MPZ solver."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

from . import plasticity, sharp_front
from .fractional_moving_frame import fractional_moving_frame_advance
from .kinetic_tip_cell import KineticMovingTipFrontEngine, KineticTipConfig
from .unified_mpz import UnifiedMPZState


def _pop_value(args: list[str], option: str, default: str) -> str:
    prefix = option + "="
    for i, token in enumerate(list(args)):
        if token.startswith(prefix):
            value = token[len(prefix):]
            del args[i]
            return value
        if token == option:
            if i + 1 >= len(args):
                raise SystemExit(f"{option} requires a value")
            value = args[i + 1]
            del args[i:i + 2]
            return value
    return default


def _pop_toggle(args: list[str], positive: str, negative: str, default: bool) -> bool:
    value = bool(default)
    kept: list[str] = []
    for token in args:
        if token == positive:
            value = True
        elif token == negative:
            value = False
        else:
            kept.append(token)
    args[:] = kept
    return value


def _option_value(args: list[str], option: str, default: str | None = None) -> str | None:
    prefix = option + "="
    for i, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == option and i + 1 < len(args):
            return args[i + 1]
    return default


def _resolved_wake_shielding(args: list[str]) -> bool:
    value = True
    for token in args:
        if token == "--wake-shielding":
            value = True
        elif token == "--no-wake-shielding":
            value = False
    return value


def _tip_only_update_plasticity(
    ep_gp, rho_gp, sigma_gp, mat, T, dt, plast_model, disl_cfg,
    return_info: bool = False,
):
    """Transactional no-op for the unparameterized surrounding bulk field."""
    ep_out = np.asarray(ep_gp, dtype=float).copy()
    rho_out = np.asarray(rho_gp, dtype=float).copy()
    dot_ep = np.zeros_like(rho_out)
    if return_info:
        info = {
            "dWp_accepted_gp": np.zeros_like(rho_out),
            "dWp_requested_gp": np.zeros_like(rho_out),
            "dep_eq_accepted_gp": np.zeros_like(rho_out),
            "bulk_plasticity_mode": "tip_only",
        }
        return ep_out, rho_out, dot_ep, info
    return ep_out, rho_out, dot_ep


def _prepare_args_v101(argv: Iterable[str]) -> tuple[list[str], str, str, str, KineticTipConfig]:
    args = list(argv)
    bulk_mode = _pop_value(args, "--bulk-plasticity-mode", "tip_only").strip().lower()
    j_mode = _pop_value(args, "--directional-j-mode", "root_signed").strip().lower()
    kinetics_mode = _pop_value(args, "--tip-kinetics-mode", "moving_velocity").strip().lower()
    if bulk_mode not in {"tip_only", "full_field"}:
        raise SystemExit("--bulk-plasticity-mode must be tip_only or full_field")
    if j_mode not in {"abs_forward", "root_signed"}:
        raise SystemExit("--directional-j-mode must be abs_forward or root_signed")
    if kinetics_mode not in {"moving_velocity", "legacy_jump"}:
        raise SystemExit("--tip-kinetics-mode must be moving_velocity or legacy_jump")
    if j_mode == "abs_forward" and "--allow-abs-directional-J" not in args:
        args.append("--allow-abs-directional-J")

    cfg = KineticTipConfig(
        enabled=kinetics_mode == "moving_velocity",
        plasticity_enabled=_pop_toggle(args, "--tip-plasticity", "--no-tip-plasticity", True),
        active_shielding=_pop_toggle(args, "--active-shielding", "--no-active-shielding", True),
        signed_active_shielding=_pop_toggle(
            args, "--signed-active-shielding", "--no-signed-active-shielding", True
        ),
        mobile_shield_fraction=float(_pop_value(args, "--mobile-shield-fraction", "1.0")),
        packet_length_m=float(_pop_value(args, "--kinetic-packet-length-m", "2.5e-10")),
        velocity_scale=float(_pop_value(args, "--kinetic-velocity-scale", "1.0")),
        max_action_substep=float(_pop_value(args, "--kinetic-max-action-substep", "0.02")),
        max_translation_substep_m=float(
            _pop_value(args, "--kinetic-max-translation-substep-m", "1e-7")
        ),
        min_substep_s=float(_pop_value(args, "--kinetic-min-substep-s", "1e-15")),
        max_internal_steps=int(_pop_value(args, "--kinetic-max-internal-steps", "20000")),
        coupling_scheme=_pop_value(args, "--kinetic-coupling-scheme", "strang").strip().lower(),
    ).validate()
    return args, bulk_mode, j_mode, kinetics_mode, cfg


def _prepare_args(argv: Iterable[str]) -> tuple[list[str], str, str]:
    """Backward-compatible v10.0.1 parser interface used by existing tests/tools."""
    args, bulk_mode, j_mode, _kinetics_mode, _cfg = _prepare_args_v101(argv)
    return args, bulk_mode, j_mode


_ORIGINAL_MPZ_DIAGNOSTICS = UnifiedMPZState.diagnostics


def _diagnostics_with_csv_aliases(self, G: float, nu: float, b: float, r0: float):
    """Map current active/wake fields onto the inherited long-run CSV keys."""
    data = _ORIGINAL_MPZ_DIAGNOSTICS(self, G, nu, b, r0)
    data["mpz_K_shield_Pa_sqrt_m"] = float(data["mpz_total_K_shield_Pa_sqrt_m"])
    data["mpz_wake_retained_total"] = float(data["mpz_wake_retained_count"])
    data["mpz_local_slip_count"] = float(self.local_slip_count())
    return data


def _write_mode_audit(args: list[str], bulk_mode: str, j_mode: str,
                      kinetics_mode: str, cfg: KineticTipConfig) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    wake_mode = _resolved_wake_shielding(args)
    legacy_payload = {
        "schema": "v10_0_2_1_driver_modes",
        "bulk_plasticity_mode": bulk_mode,
        "directional_j_mode": j_mode,
        "wake_shielding": wake_mode,
        "legacy_full_field_enabled": False,
        "dependency_closed_sharp_backend": True,
        "mpz_csv_diagnostic_aliases_enabled": True,
    }
    (path / "v10_0_1_driver_modes.json").write_text(
        json.dumps(legacy_payload, indent=2)
    )
    payload = {
        **legacy_payload,
        "schema": "v10.1_driver_modes",
        "tip_kinetics_mode": kinetics_mode,
        "tip_plasticity_enabled": cfg.plasticity_enabled,
        "active_shielding_enabled": cfg.active_shielding,
        "signed_active_shielding": cfg.signed_active_shielding,
        "mobile_shield_fraction": cfg.mobile_shield_fraction,
        "packet_length_m": cfg.packet_length_m,
        "kinetic_velocity_scale": cfg.velocity_scale,
        "kinetic_max_action_substep": cfg.max_action_substep,
        "kinetic_max_translation_substep_m": cfg.max_translation_substep_m,
        "fractional_moving_frame": kinetics_mode == "moving_velocity",
    }
    (path / "v10_1_driver_modes.json").write_text(json.dumps(payload, indent=2))
    if kinetics_mode == "moving_velocity":
        (path / "kinetic_tip_cell_audit_v101.json").write_text(
            json.dumps(KineticMovingTipFrontEngine.audit_payload(), indent=2)
        )


def main(argv=None):
    args, bulk_mode, j_mode, kinetics_mode, tip_cfg = _prepare_args_v101(
        sys.argv[1:] if argv is None else argv
    )
    if "--material-class" not in args and "--material-manifest" not in args:
        raise SystemExit(
            "v10 requires --material-class {ceramic,weakT,DBTT} "
            "or --material-manifest PATH"
        )
    if bulk_mode == "full_field":
        raise SystemExit(
            "v10 blocks --bulk-plasticity-mode full_field: the inherited bulk "
            "kinetics are not yet mapped to the promoted material manifest."
        )

    original_update = plasticity.update_plasticity
    original_diag = UnifiedMPZState.diagnostics
    original_advance = UnifiedMPZState.advance
    original_engine = sharp_front.UnifiedMPZFrontEngine
    try:
        plasticity.update_plasticity = _tip_only_update_plasticity
        UnifiedMPZState.diagnostics = _diagnostics_with_csv_aliases
        if kinetics_mode == "moving_velocity":
            UnifiedMPZState.advance = fractional_moving_frame_advance
            KineticMovingTipFrontEngine.configure_default(tip_cfg)
            KineticMovingTipFrontEngine.reset_audit()
            sharp_front.UnifiedMPZFrontEngine = KineticMovingTipFrontEngine
        wake_mode = _resolved_wake_shielding(args)
        print(
            f"  v10.1 driving modes: bulk_plasticity={bulk_mode}, "
            f"directional_J={j_mode}, tip_kinetics={kinetics_mode}, "
            f"tip_plasticity={int(tip_cfg.plasticity_enabled)}, "
            f"active_shielding={int(tip_cfg.active_shielding)}, "
            f"wake_shielding={int(wake_mode)}"
        )
        result = sharp_front.main(args)
        _write_mode_audit(args, bulk_mode, j_mode, kinetics_mode, tip_cfg)
        return result
    finally:
        plasticity.update_plasticity = original_update
        UnifiedMPZState.diagnostics = original_diag
        UnifiedMPZState.advance = original_advance
        sharp_front.UnifiedMPZFrontEngine = original_engine


if __name__ == "__main__":
    main()
