"""v10.0.1 sharp-front entry point with explicit bulk-plasticity and J-sign modes.

The v10.0 material-transfer gate unintentionally advanced the legacy full-field
plasticity model even though that model was not parameterized from the promoted
ceramic/weakT/DBTT manifest.  It also inherited the legacy directional-J sign
tracker, whose sign was fixed by the first numerically nonzero J value.

This wrapper keeps the complete anisotropic/multifront solver, but makes those
two choices explicit:

* ``tip_only``: the unified active/wake MPZ is the only plastic state advanced.
  The FEM remains elastic outside the MPZ.  This is the required first material-
  transfer gate and is not a removal of the full-field capability.
* ``full_field``: retain the existing full-field plasticity update for audit and
  later manifest-coupled development.
* ``abs_forward``: use |J| only after the existing global-forward candidate
  filter.  This mode is restricted to the one-front transfer gate.
* ``root_signed``: retain the signed directional-J convention used for branching.
"""
from __future__ import annotations

import sys
from typing import Iterable

import numpy as np

from . import plasticity, sharp_front


def _pop_value(args: list[str], option: str, default: str) -> str:
    """Remove one ``--option value`` or ``--option=value`` pair from argv."""
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
    """No-op bulk update used while the unified MPZ supplies tip plasticity.

    Arrays are copied because the production update mutates ``ep_gp`` in place;
    returning independent arrays preserves the transaction semantics expected by
    the adaptive load-step rollback.
    """
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


def main(argv=None):
    args, bulk_mode, j_mode = _prepare_args(
        sys.argv[1:] if argv is None else argv
    )
    if "--material-class" not in args and "--material-manifest" not in args:
        raise SystemExit(
            "v10.0.1 requires --material-class {ceramic,weakT,DBTT} "
            "or --material-manifest PATH"
        )

    original_update = plasticity.update_plasticity
    try:
        if bulk_mode == "tip_only":
            plasticity.update_plasticity = _tip_only_update_plasticity
        print(
            f"  v10.0.1 driving modes: bulk_plasticity={bulk_mode}, "
            f"directional_J={j_mode}"
        )
        return sharp_front.main(args)
    finally:
        plasticity.update_plasticity = original_update


if __name__ == "__main__":
    main()
