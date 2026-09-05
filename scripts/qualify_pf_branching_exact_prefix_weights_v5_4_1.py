#!/usr/bin/env python3
"""Publish explicit V5.4.1 resolver weights without running mechanics."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.restart_family_migration_v11 import canonical_hash
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Grid:
    pass


def runtime_bound(family, mpz):
    grid = Grid()
    for name in ("n_systems", "n_bins", "wake_n_bins", "x", "wake_x", "site_capacity"):
        setattr(grid, name, copy.deepcopy(mpz[name]))
    bound = family.clone_for_engine()
    bound.validate_state(grid)
    return bound


def evaluate(family, extension_m: float) -> dict:
    active, wake = family.resolve(
        r_eff_over_r0=3.25,
        opening_strength_fraction=0.625,
        crack_extension_m=extension_m,
    )
    operator = {
        "active_I": np.asarray(active),
        "wake_I": np.asarray(wake),
        "active_II": np.asarray(family.active_kernel_II),
        "wake_II": np.asarray(family.wake_kernel_II),
    }
    return {
        "operator_digest": canonical_hash(operator),
        "state_ids": list(family._last_state_ids),
        "weights": np.asarray(family._last_weights, dtype=float),
        "boundary_action": str(family._last_boundary_action),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-family", type=Path, required=True)
    parser.add_argument("--target-family", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = ActiveOnlySigned2DShieldingKernelFamily.from_json(args.source_family)
    target = ActiveOnlySigned2DShieldingKernelFamily.from_json(args.target_family)
    checkpoint = restore_branch_checkpoint(args.checkpoint)
    mpz = checkpoint.shared_process_state["mpz_fields"]
    source_bound = runtime_bound(source, mpz)
    loaded_bound = runtime_bound(target, mpz)
    variants = {
        "loaded_then_runtime_bound": loaded_bound,
        "clone_after_runtime_binding": loaded_bound.clone_for_engine(),
        "deepcopy_after_runtime_binding": copy.deepcopy(loaded_bound),
        "independent_runtime_binding": runtime_bound(target, mpz),
        "pickle_after_runtime_binding": pickle.loads(pickle.dumps(loaded_bound, protocol=5)),
    }
    legacy_ids = [state.state_id for state in source_bound.states]
    legacy_set = set(legacy_ids)
    all_ids = [state.state_id for state in loaded_bound.states]
    tolerance = float(loaded_bound.interpolation["envelope_relative_tolerance"])
    interpolation_error = {
        key: loaded_bound.metadata.get(key)
        for key in (
            "maximum_relative_spatial_cross_validation_error",
            "spatial_cross_validation_tolerance",
            "spatial_cross_validation_not_required_for_two_endpoint_active_curves",
        )
    }
    extensions_um = (0.0, 2.5, 5.0, 200.0, 400.0, 415.0, 420.0, 425.0, 600.0, 745.0)
    rows = []
    references = {
        value: evaluate(source_bound, value * 1.0e-6)
        for value in extensions_um if value <= 415.0
    }
    for extension_um in extensions_um:
        delegated = extension_um <= 415.0
        for mode, family in variants.items():
            observed = evaluate(family, extension_um * 1.0e-6)
            weights_by_id = {
                state_id: float(weight)
                for state_id, weight in zip(all_ids, observed["weights"])
            }
            legacy_sum = sum(weight for state_id, weight in weights_by_id.items() if state_id in legacy_set)
            appended_sum = sum(weight for state_id, weight in weights_by_id.items() if state_id not in legacy_set)
            if delegated:
                reference = references[extension_um]
                if observed["operator_digest"] != reference["operator_digest"]:
                    raise RuntimeError(f"operator prefix mismatch at {extension_um} um ({mode})")
                if observed["state_ids"] != reference["state_ids"]:
                    raise RuntimeError(f"state-id prefix mismatch at {extension_um} um ({mode})")
                if any(weights_by_id[state_id] != 0.0 for state_id in set(all_ids) - legacy_set):
                    raise RuntimeError(f"appended weight entered legacy prefix at {extension_um} um")
            rows.append({
                "resolver_mode": (
                    "LEGACY_EXACT_PREFIX_DELEGATED" if delegated
                    else "FULL_APPEND_ONLY_FAMILY"
                ),
                "legacy_prefix_delegation_flag": delegated,
                "runtime_variant": mode,
                "extension_um": extension_um,
                "resolved_state_ids": json.dumps(observed["state_ids"], separators=(",", ":")),
                "weights_by_state_id": json.dumps(weights_by_id, sort_keys=True, separators=(",", ":")),
                "legacy_weight_sum": format(legacy_sum, ".17g"),
                "appended_weight_sum": format(appended_sum, ".17g"),
                "boundary_action": observed["boundary_action"],
                "operator_digest_sha256": observed["operator_digest"],
                "qualification": "PASS",
            })

    beyond_um = 745.0 + 2.0 * tolerance * 1.0e6
    for mode, family in variants.items():
        try:
            evaluate(family, beyond_um * 1.0e-6)
        except (RuntimeError, ValueError):
            pass
        else:
            raise RuntimeError(f"{mode} did not fail closed beyond 745 um")
        rows.append({
            "resolver_mode": "BEYOND_QUALIFIED_ENVELOPE_FAIL_CLOSED",
            "legacy_prefix_delegation_flag": False,
            "runtime_variant": mode,
            "extension_um": beyond_um,
            "resolved_state_ids": "[]",
            "weights_by_state_id": "{}",
            "legacy_weight_sum": "NOT_EVALUATED",
            "appended_weight_sum": "NOT_EVALUATED",
            "boundary_action": "FAIL_CLOSED",
            "operator_digest_sha256": "NOT_EVALUATED",
            "qualification": "PASS_FAIL_CLOSED",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema": "pf_branching_exact_prefix_weight_audit_v5_4_1/1",
        "qualification": "PASS",
        "source_family_sha256": sha256(args.source_family),
        "target_family_sha256": sha256(args.target_family),
        "runtime_checkpoint": str(args.checkpoint.resolve()),
        "runtime_active_bins": int(mpz["n_bins"]),
        "envelope_relative_tolerance": tolerance,
        "interpolation_error_metadata": interpolation_error,
        "envelope_tolerance_and_interpolation_error_separately_represented": True,
        "legacy_state_ids": legacy_ids,
        "appended_state_ids": [value for value in all_ids if value not in legacy_set],
        "variants": list(variants),
        "beyond_745_um_fails_closed": True,
        "mechanics_solve_performed": False,
        "stochastic_update_performed": False,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
