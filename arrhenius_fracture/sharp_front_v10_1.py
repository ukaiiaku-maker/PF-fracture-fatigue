"""v10.0.1+ production entry point for the unified sharp-front MPZ solver.

The safeguarded entry point keeps the surrounding FEM elastic during the
initial transfer gates, preserves production root-signed directional J, records
resolved run modes, and supplies a compatibility mapping between the current
unified-MPZ diagnostics and the inherited long-run CSV schema.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

from . import plasticity, sharp_front
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
    ep_gp,
    rho_gp,
    sigma_gp,
    mat,
    T,
    dt,
    plast_model,
    disl_cfg,
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


def _prepare_args(argv: Iterable[str]) -> tuple[list[str], str, str]:
    args = list(argv)
    bulk_mode = _pop_value(args, "--bulk-plasticity-mode", "tip_only").strip().lower()
    j_mode = _pop_value(args, "--directional-j-mode", "root_signed").strip().lower()
    if bulk_mode not in {"tip_only", "full_field"}:
        raise SystemExit("--bulk-plasticity-mode must be tip_only or full_field")
    if j_mode not in {"abs_forward", "root_signed"}:
        raise SystemExit("--directional-j-mode must be abs_forward or root_signed")
    if j_mode == "abs_forward" and "--allow-abs-directional-J" not in args:
        args.append("--allow-abs-directional-J")
    return args, bulk_mode, j_mode


def _diagnostics_with_csv_aliases(self, G: float, nu: float, b: float, r0: float):
    """Expose current MPZ fields under the inherited long-run CSV key names."""
    data = _ORIGINAL_MPZ_DIAGNOSTICS(self, G, nu, b, r0)
    data["mpz_K_shield_Pa_sqrt_m"] = float(data["mpz_total_K_shield_Pa_sqrt_m"])
    data["mpz_wake_retained_total"] = float(data["mpz_wake_retained_count"])
    data["mpz_local_slip_count"] = float(self.local_slip_count())
    return data


_ORIGINAL_MPZ_DIAGNOSTICS = UnifiedMPZState.diagnostics


def _write_mode_audit(args: list[str], bulk_mode: str, j_mode: str) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    (path / "v10_0_1_driver_modes.json").write_text(json.dumps({
        "schema": "v10_0_2_1_driver_modes",
        "bulk_plasticity_mode": bulk_mode,
        "directional_j_mode": j_mode,
        "wake_shielding": _resolved_wake_shielding(args),
        "legacy_full_field_enabled": False,
        "dependency_closed_sharp_backend": True,
        "mpz_csv_diagnostic_aliases_enabled": True,
    }, indent=2))


def main(argv=None):
    args, bulk_mode, j_mode = _prepare_args(
        sys.argv[1:] if argv is None else argv
    )
    if "--material-class" not in args and "--material-manifest" not in args:
        raise SystemExit(
            "v10 requires --material-class {ceramic,weakT,DBTT} "
            "or --material-manifest PATH"
        )
    if bulk_mode == "full_field":
        raise SystemExit(
            "v10 blocks --bulk-plasticity-mode full_field: the inherited "
            "bulk kinetics are not yet mapped to the promoted material manifest. "
            "Use tip_only for the validated unified-MPZ baseline."
        )

    original_update = plasticity.update_plasticity
    original_diag = UnifiedMPZState.diagnostics
    try:
        plasticity.update_plasticity = _tip_only_update_plasticity
        UnifiedMPZState.diagnostics = _diagnostics_with_csv_aliases
        wake_mode = _resolved_wake_shielding(args)
        print(
            f"  v10 driving modes: bulk_plasticity={bulk_mode}, "
            f"directional_J={j_mode}, wake_shielding={int(wake_mode)}"
        )
        result = sharp_front.main(args)
        _write_mode_audit(args, bulk_mode, j_mode)
        return result
    finally:
        plasticity.update_plasticity = original_update
        UnifiedMPZState.diagnostics = original_diag


if __name__ == "__main__":
    main()
