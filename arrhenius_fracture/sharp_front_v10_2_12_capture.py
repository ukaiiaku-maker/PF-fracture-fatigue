"""v10.2.12 entry point for physical signed-kernel FEM state capture.

This is a mechanics-data collection mode, not a fracture parameterization run.
It observes the same 2-D sharp-front FEM and anisotropic tensor-probe path while
explicitly disabling the inherited unsigned shielding operator. Captured states
are subsequently perturbed offline with the hardened signed interaction integral.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import anisotropic_emission_v10174 as _anisotropic
from . import fem as _fem
from . import sharp_front_v10_1_7_4 as _entry74
from . import sharp_front_v10_1_7_5 as _transport
from .physical_fem_capture_v10212 import MODEL_ID, load_capture_requests
from .physical_fem_capture_trace_v10212 import PhysicalFEMCapture


def _pop_value(args: list[str], option: str) -> str | None:
    prefix = option + "="
    for index, token in enumerate(list(args)):
        if token.startswith(prefix):
            value = token[len(prefix):]
            del args[index]
            return value
        if token == option:
            if index + 1 >= len(args):
                raise SystemExit(f"{option} requires a value")
            value = args[index + 1]
            del args[index:index + 2]
            return value
    return None


def _remove_option(args: list[str], option: str, takes_value: bool = False) -> None:
    prefix = option + "="
    index = 0
    while index < len(args):
        token = args[index]
        if token.startswith(prefix):
            del args[index]
            continue
        if token == option:
            del args[index]
            if takes_value and index < len(args):
                del args[index]
            continue
        index += 1


def _force_capture_modes(args: list[str]) -> None:
    for option in (
        "--active-shielding",
        "--no-active-shielding",
        "--wake-shielding",
        "--no-wake-shielding",
        "--crystal-branch",
    ):
        _remove_option(args, option)
    _remove_option(args, "--max-fronts", takes_value=True)
    args.extend(["--no-active-shielding", "--no-wake-shielding", "--max-fronts", "1"])


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    state_table = _pop_value(args, "--atlas-state-table")
    outroot = _pop_value(args, "--atlas-outroot")
    trajectory_only = "--atlas-trajectory-only" in args
    if trajectory_only:
        args.remove("--atlas-trajectory-only")
    allow_incomplete = "--allow-incomplete-atlas-capture" in args
    if allow_incomplete:
        args.remove("--allow-incomplete-atlas-capture")
    if not outroot:
        raise SystemExit("v10.2.12 capture requires --atlas-outroot PATH")
    if trajectory_only:
        if state_table:
            raise SystemExit(
                "--atlas-trajectory-only must not be combined with --atlas-state-table"
            )
        requests = []
        allow_incomplete = True
    else:
        if not state_table:
            raise SystemExit(
                "snapshot capture requires --atlas-state-table PATH; use "
                "--atlas-trajectory-only for discovery"
            )
        requests = load_capture_requests(state_table)
    capture = PhysicalFEMCapture(requests, outroot)
    _force_capture_modes(args)

    engine_type = _anisotropic.AnisotropicStochasticAvalancheTipEngine
    original_step = engine_type.step
    original_factory = _entry74.wrap_assemble_mechanics
    original_solve = _fem.solve_dirichlet
    _entry74.wrap_assemble_mechanics = capture.wrap_assemble_factory(original_factory)
    _fem.solve_dirichlet = capture.wrap_solve_dirichlet(original_solve)
    engine_type.step = capture.wrap_engine_step(original_step)

    try:
        print(
            "  v10.2.12 physical FEM atlas capture: "
            f"mode={'trajectory_only' if trajectory_only else 'snapshot_capture'} "
            f"requests={len(requests)} unsigned_shielding=disabled "
            "kinetics_path=production fem_path=production parameterization=blocked"
        )
        result = _transport.main(args)
        audit = capture.finalize(require_complete=not allow_incomplete)
        root = Path(outroot)
        (root / "v10_2_12_capture_entry.json").write_text(
            json.dumps(
                {
                    "schema": MODEL_ID,
                    "capture_mode": (
                        "trajectory_only" if trajectory_only else "snapshot_capture"
                    ),
                    "state_table": (
                        str(Path(state_table).resolve()) if state_table else None
                    ),
                    "atlas_outroot": str(root.resolve()),
                    "allow_incomplete": allow_incomplete,
                    "capture": audit,
                    "reachable_state_trace": audit.get("reachable_state_trace"),
                    "next_step": (
                        "select a Cartesian opening/extension state grid from the reachable trace"
                        if trajectory_only
                        else "evaluate each captured snapshot with scripts/evaluate_v10_2_12_signed_snapshot.py"
                    ),
                },
                indent=2,
            )
        )
        return result
    finally:
        engine_type.step = original_step
        _entry74.wrap_assemble_mechanics = original_factory
        _fem.solve_dirichlet = original_solve


if __name__ == "__main__":
    main()
