#!/usr/bin/env python3
"""Qualify recursive owner transitions without manufacturing a branch event.

This is an ownership/checkpoint fixture.  It does not claim that native
mechanics or hazards spontaneously selected either represented branch.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.qualification_manifest import capacity_preflight
from arrhenius_fracture.process_state_ownership_v11 import (
    ProcessStateOwner, ProcessStateOwnerRegistry,
)


def qualification_fixture() -> dict:
    registry = ProcessStateOwnerRegistry(
        {"root": "root"},
        {"root": ProcessStateOwner(
            "root", "single_tip_engine", process_engine_id="root"
        )},
    )
    first = registry.branch("root", "cluster-1", ("daughter-1", "daughter-2"))
    first.validate(
        ("daughter-1", "daughter-2"), engine_ids=("cluster-1",),
        cluster_ids=("cluster-1",), reservoir_ids=(),
    )
    resolved = first.resolve(
        "cluster-1", ("daughter-1", "daughter-2"), "reservoir-1"
    )
    resolved.validate(
        ("daughter-1", "daughter-2"),
        engine_ids=("daughter-1", "daughter-2"), cluster_ids=(),
        reservoir_ids=("reservoir-1",),
    )
    second = resolved.branch(
        "daughter-1", "cluster-2", ("granddaughter-1", "granddaughter-2")
    )
    active = ("daughter-2", "granddaughter-1", "granddaughter-2")
    second.validate(
        active, engine_ids=("daughter-2", "cluster-2"),
        cluster_ids=("cluster-2",), reservoir_ids=("reservoir-1",),
    )
    serialized = second.to_dict()
    restored = ProcessStateOwnerRegistry.from_dict(
        json.loads(json.dumps(serialized, sort_keys=True))
    )
    restored.validate(
        active, engine_ids=("daughter-2", "cluster-2"),
        cluster_ids=("cluster-2",), reservoir_ids=("reservoir-1",),
    )
    if restored.to_dict() != serialized:
        raise RuntimeError("recursive owner registry changed across checkpoint encoding")
    return {
        "schema": "recursive-branch-owner-adapter-qualification/1",
        "fixture_classification": "INJECTED_OWNER_TRANSITIONS_NOT_NATURAL_BRANCH_PREDICTION",
        "transitions": [
            "single_owner_to_unresolved_first_generation_cluster",
            "first_generation_cluster_to_independent_daughter_owners",
            "resolved_daughter_to_unresolved_second_generation_cluster",
            "checkpoint_round_trip",
        ],
        "active_tips": list(active),
        "owner_registry": serialized,
        "checkpoint_exact": True,
        "mutable_independent_owner_aliasing": False,
        "native_birth_demonstrated": False,
        "status": "PARTIAL_ADAPTER_QUALIFIED_NATIVE_BIRTH_PENDING",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--second-generation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output")
    args = p.parse_args()
    if not args.second_generation:
        raise SystemExit("recursive readiness requires a second-generation birth")
    payload = {"qualification": "Q-BRANCH-RECURSIVE",
               "native_birth_entry": "scripts/run_v11_branching_case.py",
               "capacity": capacity_preflight(ROOT, 24.0).__dict__,
               "production_execution_authorized": False,
               **qualification_fixture()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not args.dry_run and not args.output:
        raise SystemExit(
            "native branch execution remains disabled; use --output to retain "
            "the ownership-adapter qualification"
        )


if __name__ == "__main__":
    main()
