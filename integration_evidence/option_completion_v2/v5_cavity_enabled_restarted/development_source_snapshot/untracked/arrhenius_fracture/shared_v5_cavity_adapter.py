"""Bounded public adapter for the native full-order V5 cavity state.

The adapter stops after resolved cavity growth.  It exercises the production
FEM/remesh/field-transfer/cavity transactions while deliberately withholding
ligament rupture and child-front creation, so cavity enablement is independent
of physical branch birth.  Its checkpoint is the native atomic V11 full-state
family used by the production topology model.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np

from .checkpoint_v11 import restore_checkpoint, write_checkpoint
from .topology_transaction_v11 import (
    complete_accepted_state_fingerprint,
    equilibrate_fixed_load_with_production_fem,
)
from .unified_fracture_material_v5 import material_bundle
from .voiding_production_v5 import (
    build_production_void_state, deterministic_trajectory, observables,
)

MODEL_ID = "shared.native_v5_cavity_branch_birth_disabled/1"


def _mechanical_arrays_equal(left, right) -> bool:
    fields = (
        "displacement", "ep_gp", "rho_gp", "damage", "vacancy_fraction_gp",
    )
    return (
        np.array_equal(left.mesh.nodes, right.mesh.nodes)
        and np.array_equal(left.mesh.elems, right.mesh.elems)
        and all(np.array_equal(getattr(left, name), getattr(right, name))
                for name in fields)
    )


def _atomic_checkpoint(state, root: Path) -> Path:
    directory = Path(os.environ.get(
        "V10230_HIGH_CYCLE_CHECKPOINT_DIR", root / "checkpoint"
    ))
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "accepted_v5.json"
    write_checkpoint(state, target, compression="gzip")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cavity-enabled", action="store_true")
    parser.add_argument("--cavity-disabled", action="store_true")
    parser.add_argument("--restart-checkpoint")
    parser.add_argument("--save-snapshots")
    parser.add_argument("--no-plots", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise SystemExit("unsupported V5 adapter arguments: " + " ".join(unknown))
    if args.cavity_enabled == args.cavity_disabled:
        raise SystemExit("select exactly one of --cavity-enabled/--cavity-disabled")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    restarted = args.restart_checkpoint is not None
    initial_match = None
    rows = []
    if restarted:
        state = restore_checkpoint(args.restart_checkpoint)
        if bool(state.void_state is not None) != bool(args.cavity_enabled):
            raise RuntimeError("V5 restart cavity mode does not match selected options")
        before = complete_accepted_state_fingerprint(state)
        state = equilibrate_fixed_load_with_production_fem(state)
        rows.append(observables(state, "restart_re_equilibrated"))
        continued_from = before
    else:
        bundle = material_bundle("DBTT")
        disabled, _ = build_production_void_state(
            bundle=bundle, enabled=False, seed=3621, temperature_K=900.0,
        )
        enabled, _ = build_production_void_state(
            bundle=bundle, enabled=True, seed=3621, temperature_K=900.0,
        )
        initial_match = _mechanical_arrays_equal(disabled, enabled)
        if not initial_match:
            raise RuntimeError("V5 enabled/disabled initial mechanical fields differ")
        if args.cavity_enabled:
            state, rows = deterministic_trajectory(
                bundle=bundle, stop_before_ligament=True,
            )
        else:
            state = disabled
            rows = [observables(state, "disabled_control")]
        continued_from = None

    active_tips = tuple(state.crack_network.active_tip_ids)
    branch_births = int(state.event_counters.get("branch_births", 0))
    if branch_births != 0 or len(active_tips) > 1:
        raise RuntimeError("V5 cavity adapter activated physical branch birth")
    checkpoint = _atomic_checkpoint(state, out)
    restored = restore_checkpoint(checkpoint)
    exact = (
        complete_accepted_state_fingerprint(restored)
        == complete_accepted_state_fingerprint(state)
    )
    if not exact:
        raise RuntimeError("V5 native checkpoint round trip changed accepted state")
    final = observables(state, "terminal")
    payload = {
        "schema": MODEL_ID,
        "cavity_enabled": bool(args.cavity_enabled),
        "physical_branch_birth_enabled": False,
        "physical_branch_birth_count": branch_births,
        "active_tip_ids": list(active_tips),
        "matching_enabled_disabled_initial_mechanical_arrays": initial_match,
        "restart_input": args.restart_checkpoint,
        "restarted": restarted,
        "continued_from_fingerprint": continued_from,
        "terminal_fingerprint": complete_accepted_state_fingerprint(state),
        "native_checkpoint": str(checkpoint),
        "native_checkpoint_round_trip_exact": exact,
        "terminal": final,
        "history": rows,
    }
    (out / "shared_v5_cavity_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    return payload


if __name__ == "__main__":
    main()


__all__ = ["MODEL_ID", "main"]
