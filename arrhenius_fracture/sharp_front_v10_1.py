"""v10.0.1 production entry point for the unified sharp-front MPZ solver.

The v10.0 two-dimensional gate could silently mix this repository with modules
from the older ``arrhenius-fem-czm`` editable installation because both projects
used the ``arrhenius_fracture`` namespace.  It also advanced a legacy full-field
bulk plasticity model that was not parameterized from the promoted material
manifest.

v10.0.1 therefore makes the validated baseline explicit:

* the unified active/wake MPZ is the plastic state used by monotonic and fatigue
  fracture during the initial transfer gates;
* the surrounding FEM is elastic until a manifest-coupled full-field model is
  implemented and audited;
* signed directional J remains the production convention for anisotropy and
  branching.  ``abs_forward`` is retained only as a one-front diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

from . import plasticity, sharp_front


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


def _write_mode_audit(args: list[str], bulk_mode: str, j_mode: str) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    (path / "v10_0_1_driver_modes.json").write_text(json.dumps({
        "schema": "v10_0_1_driver_modes",
        "bulk_plasticity_mode": bulk_mode,
        "directional_j_mode": j_mode,
        "legacy_full_field_enabled": False,
        "dependency_closed_sharp_backend": True,
    }, indent=2))


def main(argv=None):
    args, bulk_mode, j_mode = _prepare_args(
        sys.argv[1:] if argv is None else argv
    )
    if "--material-class" not in args and "--material-manifest" not in args:
        raise SystemExit(
            "v10.0.1 requires --material-class {ceramic,weakT,DBTT} "
            "or --material-manifest PATH"
        )
    if bulk_mode == "full_field":
        raise SystemExit(
            "v10.0.1 blocks --bulk-plasticity-mode full_field: the inherited "
            "bulk kinetics are not yet mapped to the promoted material manifest. "
            "Use tip_only for the validated unified-MPZ baseline."
        )

    original_update = plasticity.update_plasticity
    try:
        plasticity.update_plasticity = _tip_only_update_plasticity
        print(
            f"  v10.0.1 driving modes: bulk_plasticity={bulk_mode}, "
            f"directional_J={j_mode}"
        )
        result = sharp_front.main(args)
        _write_mode_audit(args, bulk_mode, j_mode)
        return result
    finally:
        plasticity.update_plasticity = original_update


if __name__ == "__main__":
    main()
