"""V12 generic production entrypoint and no-solve restore preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .general_multifront_v12 import ResourcePolicy
from .production_multifront_v12 import prepare_pre_solve_inventory


BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


def run_production_interval(runtime, accepted_fem_state, hooks, *, duration_s: float):
    """Actual programmatic V12 production entrypoint.

    Callers must provide the reviewed current-source adapters collected in a
    ``ProductionHooks`` instance.  Qualification invokes only the separate
    pre-solve path below; this function is therefore wired but not executed by
    the V6.1 seal.
    """
    context = getattr(hooks, "context", None)
    if context is not None:
        if runtime is not context.runtime or accepted_fem_state is not context.accepted_fem_state:
            raise RuntimeError("V6.4 production entry requires the authoritative context objects")
        from .stateful_multifront_production_v12 import run_stateful_accepted_interval_v12
        return run_stateful_accepted_interval_v12(context, duration_s)

    # Compatibility only for source-level V6.1 dependency-injection tests.
    # Every current-source stateful hook factory supplies a context and must
    # therefore take the authoritative integrated path above.
    from .live_topology_kernel_v12 import DynamicExactTopologyProviderV12
    from .production_multifront_v12 import GenericMultiFrontProductionDriver
    return GenericMultiFrontProductionDriver(
        hooks, DynamicExactTopologyProviderV12(runtime.resource_policy),
    ).run_accepted_interval(
        runtime, accepted_fem_state, duration_s=duration_s,
    )


def optional_limit(value: str) -> int | None:
    if value.strip().lower() == "none":
        return None
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("resource limits must be positive or 'none'")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--branching-mode", choices=("disabled", "mechanistic"), required=True)
    result.add_argument("--front-resource-limit", type=optional_limit, default=None)
    result.add_argument("--branch-transaction-limit", type=optional_limit, default=None)
    result.add_argument("--compatibility-checkpoint")
    result.add_argument("--source-only-preflight", action="store_true")
    result.add_argument("--production-preflight", action="store_true")
    result.add_argument("--v11-compatibility-checkpoint")
    result.add_argument("--v12-checkpoint")
    result.add_argument("--v6-2-production-adapter-dry-run", action="store_true")
    result.add_argument("--v6-3-stateful-production-dry-run", action="store_true")
    result.add_argument("--v6-4-integrated-interval-dry-run", action="store_true")
    result.add_argument("--v5-4-2-n1-checkpoint")
    result.add_argument("--v5-4-2-n2-checkpoint")
    result.add_argument("--fresh-output-destination")
    result.add_argument("--fresh-cache-destination")
    result.add_argument(
        "--shared-region-branching", choices=("forbid", "experimental"),
        default="forbid",
    )
    result.add_argument(
        "--scheduler-policy", choices=(
            "v11_correlated_proposal_compatibility",
            "v12_global_earliest_proposal",
        ), default="v11_correlated_proposal_compatibility",
    )
    return result


def validate_args(argv: Sequence[str]) -> ResourcePolicy:
    args = parser().parse_args(tuple(argv))
    if not (
        args.source_only_preflight or args.production_preflight
        or args.v6_2_production_adapter_dry_run
        or args.v6_3_stateful_production_dry_run
        or args.v6_4_integrated_interval_dry_run
    ):
        raise RuntimeError("V12_SOURCE_ONLY_QUALIFICATION_NO_PF_FEM_EXECUTION")
    if sum(bool(x) for x in (
        args.source_only_preflight, args.production_preflight,
        args.v6_2_production_adapter_dry_run,
        args.v6_3_stateful_production_dry_run,
        args.v6_4_integrated_interval_dry_run,
    )) != 1:
        raise RuntimeError("select exactly one V12 preflight mode")
    if args.branching_mode == "disabled" and args.front_resource_limit not in (None, 1):
        raise ValueError("branching-disabled matched control permits limit 1 or none")
    return ResourcePolicy(
        branching_mode=args.branching_mode,
        front_resource_limit=args.front_resource_limit,
        branch_transaction_limit=args.branch_transaction_limit,
        shared_region_branching=args.shared_region_branching,
        scheduler_policy=args.scheduler_policy,
    )


def v6_4_integrated_interval_dry_run(argv: Sequence[str]) -> dict:
    """Invoke the authoritative executor against immutable cache-only N1/N2.

    Missing archived coverage is a scientific fail-closed result.  This path
    assembles the checkpoint stress only; it cannot call a Dirichlet solve,
    provider solve, PF worker, or output publication.
    """
    import hashlib
    import pickle
    import numpy as np
    from .branch_checkpoint_v11 import restore_branch_checkpoint
    from .fem import assemble_mechanics
    from .multifront_checkpoint_v12 import load_v11_checkpoint_as_v12
    from .production_multifront_v12 import (
        AdaptedAcceptedState, SolvedAcceptedState,
        accepted_fem_state_fingerprint,
    )
    from .stateful_multifront_production_v12 import (
        CurrentSourceMultiFrontProductionContextV12,
        run_stateful_accepted_interval_v12,
        load_cached_provider_result_for_topology,
    )

    args = parser().parse_args(tuple(argv))
    policy = validate_args(argv)
    if not args.v6_4_integrated_interval_dry_run:
        raise RuntimeError("V6.4 integrated interval dry-run mode required")
    if not args.v5_4_2_n1_checkpoint or not args.v5_4_2_n2_checkpoint:
        raise ValueError("V6.4 dry run requires native N1 and N2 checkpoints")
    destinations = {
        "output": args.fresh_output_destination,
        "cache": args.fresh_cache_destination,
    }
    if any(value is None for value in destinations.values()):
        raise ValueError("V6.4 dry run requires fresh output and cache destinations")
    if any(Path(value).exists() for value in destinations.values()):
        raise FileExistsError("V6.4 dry-run destinations must be fresh")

    cases = {}
    for name, source_path, expected, native_policy in (
        ("native_N1", args.v5_4_2_n1_checkpoint, 1,
         ResourcePolicy("disabled", 1, None, "forbid",
                        "v11_correlated_proposal_compatibility")),
        ("native_N2", args.v5_4_2_n2_checkpoint, 2, policy),
    ):
        source = restore_branch_checkpoint(source_path)
        runtime = load_v11_checkpoint_as_v12(source, resource_policy=native_policy)
        if len(runtime.active_front_ids) != expected:
            raise RuntimeError(f"{name} checkpoint front cardinality mismatch")
        state = source.state
        sigma = assemble_mechanics(
            state.mesh, state.displacement, state.ep_gp, state.rho_gp,
            state.damage, state.elasticity_D, state.material,
            cohesive_network=state.cohesive_network,
        )[2]
        candidate_map = {
            item.candidate_id: item
            for competition in source.front_competitions.values()
            for item in competition.candidates
        }
        context = None

        def adapt(current, inventory, owned_context):
            return AdaptedAcceptedState(current, {
                "kind": "immutable_checkpoint_no_remesh",
                "front_candidate_inventory": {
                    key: list(value) for key, value in inventory.items()
                },
            })

        def load_mechanics(adapted, owned_context):
            return SolvedAcceptedState(
                adapted.fem_state, sigma,
                {"accepted_fem_state_sha256": accepted_fem_state_fingerprint(
                    adapted.fem_state
                ), "stress_reconstruction": "assemble_mechanics_only"},
                source.accepted_load, source.physical_time_s + 8.4,
                "immutable_v5_4_2_checkpoint_assembly_only",
            )

        def request(solved, provisional_runtime, owned_context):
            return {"topology_fingerprint": provisional_runtime.topology_fingerprint}

        def lookup(request_value, provisional_runtime, owned_context):
            source_cache_topology = provisional_runtime.compatibility_provenance.get(
                "source_topology_fingerprint"
            )
            if not source_cache_topology:
                raise RuntimeError("immutable V11 cache topology identity is absent")
            result, manifest = load_cached_provider_result_for_topology(
                source.provider_cache_identity,
                source_cache_topology,
            )
            return result

        context = CurrentSourceMultiFrontProductionContextV12(
            args={"temperature_K": 700.0, "duration_s": 8.4},
            mechanical_configuration={"source": "immutable_v5_4_2"},
            accepted_fem_state=state, accepted_stress_field=sigma,
            runtime=runtime, provider_runtime=source.provider_runtime,
            destination_cache_root=destinations["cache"],
            output_root=Path(destinations["output"]) / name,
            checkpoint_path=Path(destinations["output"]) / name / "latest.json",
            candidates=tuple(candidate_map[key] for key in sorted(candidate_map)),
            clusters={item.cluster_id: item for item in source.branch_clusters},
            physical_time_s=source.physical_time_s,
            accepted_opening_m=source.accepted_load,
            step_count=int(source.state.event_counters.get("accepted_steps", 0)),
            stress_available=True,
            mechanics_source_identity="immutable_v5_4_2_checkpoint_assembly_only",
            adapter_configuration={
                "accepted_mesh_adapter": adapt,
                "accepted_mechanics_loader": load_mechanics,
                "exact_request_builder": request,
                "provider_lookup": lookup,
                "da_phys_m": 5.0e-6,
                "temperature_K": 700.0,
                "correlation_interval_s": 0.0,
            },
        )
        before = context.fingerprint
        # Closed V6.4 evidence result: the 171 immutable archive checkpoints
        # do not contain the next accepted mechanics/provider state needed to
        # replay an interval.  Retrying the integrated executor would neither
        # add evidence nor authorize reconstruction from incomplete scalars.
        status = "FAIL_CLOSED"
        reason = (
            "CLOSED_EVIDENCE_LIMITATION: immutable V6.4 archive lacks the "
            "next accepted FEM/provider state; no archive replay attempted"
        )
        transaction = None
        cases[name] = {
            "status": status, "reason": reason,
            "source_checkpoint": str(Path(source_path).resolve()),
            "source_checkpoint_sha256": hashlib.sha256(
                Path(source_path).read_bytes()
            ).hexdigest(),
            "active_front_ids": list(runtime.active_front_ids),
            "accepted_stress_sha256": hashlib.sha256(
                pickle.dumps(sigma, protocol=5)
            ).hexdigest(),
            "stress_reconstruction": "assemble_mechanics_only_no_solve_dirichlet",
            "context_restored_after_fail_closed": context.fingerprint == before,
            "provider_lookup_attempted": False,
            "provider_lookup_count": context.provider_lookup_count,
            "provider_solve_count": context.provider_solve_count,
            "mechanics_solve_count": context.mechanics_solve_count,
            "workers_started": context.pf_workers_started,
            "transaction": transaction,
        }
    passed = all(value["status"] == "PASS" for value in cases.values())
    return {
        "schema": "v12.general-multifront-integrated-interval-dry-run/1",
        "boundary": BOUNDARY,
        "qualification": "PASS" if passed else "FAIL_CLOSED",
        "integrated_interval_transaction": "PASS" if passed else "FAIL_CLOSED",
        "trajectory_execution_authorized": False,
        "cases": cases,
        "provider_solve_count": 0, "mechanics_solve_count": 0,
        "workers_started": 0, "output_transaction_published": False,
        "source_checkpoints_unchanged": True,
    }


def v6_3_stateful_production_dry_run(argv: Sequence[str]) -> dict:
    """Build real persistent hooks for native N1/N2 and stop before cache miss."""
    import numpy as np
    args = parser().parse_args(tuple(argv))
    policy = validate_args(argv)
    if not args.v6_3_stateful_production_dry_run:
        raise RuntimeError("V6.3 stateful production dry-run mode required")
    if not args.v5_4_2_n1_checkpoint or not args.v5_4_2_n2_checkpoint:
        raise ValueError("V6.3 dry run requires native N1 control and enabled N2 checkpoints")
    destinations = {"output": args.fresh_output_destination, "cache": args.fresh_cache_destination}
    if any(value is None for value in destinations.values()):
        raise ValueError("V6.3 dry run requires fresh output and cache destinations")
    if any(Path(value).exists() for value in destinations.values()):
        raise FileExistsError("V6.3 dry-run destinations must be fresh")

    from .branch_checkpoint_v11 import restore_branch_checkpoint
    from .current_source_multifront_hooks_v12 import (
        build_current_source_multifront_production_hooks,
        restore_complete_current_source_engine,
    )
    from .multifront_checkpoint_v12 import load_v11_checkpoint_as_v12
    from .stateful_multifront_production_v12 import (
        CurrentSourceMultiFrontProductionContextV12,
        StatefulProductionInterlock, load_cached_provider_result_for_topology,
    )
    cases = {}
    for name, source_path, expected, native_policy in (
        ("branch_disabled_N1_control", args.v5_4_2_n1_checkpoint, 1,
         ResourcePolicy("disabled", 1, None, "forbid", "v11_correlated_proposal_compatibility")),
        ("enabled_N2_terminal", args.v5_4_2_n2_checkpoint, 2, policy),
    ):
        source = restore_branch_checkpoint(source_path)
        runtime = load_v11_checkpoint_as_v12(source, resource_policy=native_policy)
        if len(runtime.active_front_ids) != expected:
            raise RuntimeError(f"{name} checkpoint front cardinality mismatch")
        candidate_map = {}
        for competition in source.front_competitions.values():
            for candidate in competition.candidates:
                candidate_map[candidate.candidate_id] = candidate
        # V5.4.2 did not archive sigma_gp.  The dry run owns an explicit
        # unavailable sentinel of the correct shape and therefore cannot call
        # the process/tensor hook without failing closed.
        sigma_unavailable = np.zeros((3, source.state.mesh.ne), dtype=float)
        context = CurrentSourceMultiFrontProductionContextV12(
            args={"dry_run": True},
            mechanical_configuration={"source": "immutable_v5_4_2_checkpoint"},
            accepted_fem_state=source.state,
            accepted_stress_field=sigma_unavailable,
            runtime=runtime, provider_runtime=source.provider_runtime,
            destination_cache_root=args.fresh_cache_destination,
            output_root=args.fresh_output_destination,
            checkpoint_path=Path(args.fresh_output_destination) / "latest.json",
            candidates=tuple(candidate_map[key] for key in sorted(candidate_map)),
            clusters={item.cluster_id: item for item in source.branch_clusters},
            physical_time_s=source.physical_time_s,
            accepted_opening_m=source.accepted_load,
            step_count=int(source.state.event_counters.get("accepted_steps", 0)),
            adapter_configuration={"stress_field_status": "not_archived_fail_closed"},
        )
        hooks = build_current_source_multifront_production_hooks(context)
        cached = None
        try:
            cached_result, cached_manifest = load_cached_provider_result_for_topology(
                source.provider_cache_identity, source.topology_fingerprint,
            )
        except StatefulProductionInterlock as exc:
            cache_status = "NOT_APPLICABLE_PREBRANCH" if (
                source.provider_runtime.routing.active_mechanics_provider
                == source.provider_runtime.routing.initial_mechanics_provider
            ) else "FAIL_CLOSED"
            cache_reason = str(exc)
        else:
            context.provider_lookup_count += 1
            cache_status = "EXACT_CACHE_HIT"
            cache_reason = None
            cached = {
                "manifest": cached_manifest,
                "result_sha256": __import__("hashlib").sha256(
                    __import__("pickle").dumps(cached_result, protocol=5)
                ).hexdigest(),
                "tip_count": len(cached_result.get("tips", ())),
            }
        restored_engines = {
            owner_id: type(restore_complete_current_source_engine(
                runtime.process_engines[region.process_engine_id]
            )).__name__
            for owner_id, region in runtime.process_regions.items()
        }
        cases[name] = {
            "source_checkpoint": str(Path(source_path).resolve()),
            "native_branching_mode": native_policy.branching_mode,
            "active_front_ids": list(runtime.active_front_ids),
            "owner_ids": sorted(runtime.process_regions),
            "restored_engines": restored_engines,
            "context_sha256": context.fingerprint,
            "hook_context_identity_exact": hooks.context is context,
            "first_interval_inventory": prepare_pre_solve_inventory(
                runtime, accepted_fem_state_reference={
                    "source_checkpoint": str(Path(source_path).resolve()),
                    "mesh_identity": source.mesh_identity,
                },
            ).to_dict(),
            "accepted_stress_field_status": "not_archived_fail_closed",
            "cached_provider_status": cache_status,
            "cached_provider_reason": cache_reason,
            "cached_provider": cached,
        }
        if cache_status == "FAIL_CLOSED":
            raise RuntimeError(f"{name} exact provider cache lookup failed closed")
    return {
        "schema": "v12.general-multifront-stateful-production-dry-run/1",
        "boundary": BOUNDARY,
        "qualification": "PASS",
        "trajectory_execution_authorized": False,
        "cases": cases,
        "fresh_destinations_verified_absent": {
            key: str(Path(value).resolve()) for key, value in destinations.items()
        },
        "provider_lookup_count": sum(
            case["cached_provider_status"] == "EXACT_CACHE_HIT"
            for case in cases.values()
        ), "provider_solve_count": 0,
        "mechanics_solve_count": 0, "pf_worker_started": False,
        "fem_worker_started": False, "destination_created": False,
    }


def v6_2_production_adapter_dry_run(argv: Sequence[str]) -> dict:
    """Restore N=1/N=2 exact owners and stop before any physical operation."""
    args = parser().parse_args(tuple(argv))
    policy = validate_args(argv)
    if not args.v6_2_production_adapter_dry_run:
        raise RuntimeError("V6.2 production-adapter dry-run mode required")
    if not args.v5_4_2_n1_checkpoint or not args.v5_4_2_n2_checkpoint:
        raise ValueError("V6.2 dry run requires exact N=1 and N=2 checkpoints")
    destinations = {
        "output": args.fresh_output_destination,
        "cache": args.fresh_cache_destination,
    }
    if any(value is None for value in destinations.values()):
        raise ValueError("V6.2 dry run requires output and cache destinations")
    if any(Path(value).exists() for value in destinations.values()):
        raise FileExistsError("V6.2 dry-run destinations must be fresh")

    from .branch_checkpoint_v11 import restore_branch_checkpoint
    from .current_source_multifront_hooks_v12 import (
        CurrentSourceHookBindingError, build_current_source_production_hooks,
        current_source_hook_map, restore_complete_current_source_engine,
    )
    from .multifront_checkpoint_v12 import load_v11_checkpoint_as_v12
    from .restart_family_migration_v11 import process_state_digest, rng_digest
    from .sharp_front_v11_branching import _capture_shared_engine

    inventories = {}
    for cardinality, source_path in (
        ("N1", args.v5_4_2_n1_checkpoint),
        ("N2", args.v5_4_2_n2_checkpoint),
    ):
        source = restore_branch_checkpoint(source_path)
        runtime = load_v11_checkpoint_as_v12(source, resource_policy=policy)
        if len(runtime.active_front_ids) != int(cardinality[1:]):
            raise RuntimeError(f"{cardinality} checkpoint front cardinality mismatch")
        owner_rows = []
        for owner_id, region in runtime.process_regions.items():
            state = runtime.process_engines[region.process_engine_id]
            before = state.complete_checkpoint_payload()
            restored = restore_complete_current_source_engine(state)
            after = _capture_shared_engine(restored)
            owner_rows.append({
                "owner_id": owner_id,
                "engine_class": state.engine_class,
                "checkpoint_payload_sha256": state.checkpoint_payload_sha256,
                "field_count": len(state.checkpoint_field_inventory),
                "process_state_sha256_before": process_state_digest(before),
                "process_state_sha256_after": process_state_digest(after),
                "rng_sha256_before": rng_digest(before),
                "rng_sha256_after": rng_digest(after),
                "family_migration_count": int(getattr(
                    restored, "_restart_family_migration_count", 0
                )),
                "engine_mpz_family_alias_exact": (
                    restored._state_kernel_family is restored.mpz._signed_kernel
                ),
            })
        inventories[cardinality] = {
            "source_checkpoint": str(Path(source_path).resolve()),
            "active_front_ids": list(runtime.active_front_ids),
            "pre_solve_inventory": prepare_pre_solve_inventory(
                runtime,
                accepted_fem_state_reference={
                    "source_checkpoint": str(Path(source_path).resolve()),
                    "mesh_identity": source.mesh_identity,
                },
            ).to_dict(),
            "owners": owner_rows,
        }
    try:
        build_current_source_production_hooks()
    except CurrentSourceHookBindingError as exc:
        adapter_status = "FAIL_CLOSED"
        adapter_reason = str(exc)
    else:  # pragma: no cover - construction remains deliberately sealed
        adapter_status = "QUALIFIED"
        adapter_reason = None
    return {
        "schema": "v12.general-multifront-production-adapter-dry-run/1",
        "boundary": BOUNDARY,
        "qualification": adapter_status,
        "qualification_reason": adapter_reason,
        "hook_map": current_source_hook_map(),
        "inventories": inventories,
        "fresh_destinations_verified_absent": {
            key: str(Path(value).resolve()) for key, value in destinations.items()
        },
        "provider_lookup_count": 0,
        "mechanics_solve_count": 0,
        "pf_worker_started": False,
        "fem_worker_started": False,
        "destination_created": False,
    }


def production_preflight(argv: Sequence[str]) -> dict:
    args = parser().parse_args(tuple(argv))
    policy = validate_args(argv)
    if not args.production_preflight:
        raise RuntimeError("production preflight mode required")
    sources = [
        item for item in (args.v11_compatibility_checkpoint, args.v12_checkpoint) if item
    ]
    if len(sources) != 1:
        raise ValueError("production preflight requires exactly one checkpoint source")
    if args.v11_compatibility_checkpoint:
        from .branch_checkpoint_v11 import restore_branch_checkpoint
        from .multifront_checkpoint_v12 import load_v11_checkpoint_as_v12

        source = restore_branch_checkpoint(args.v11_compatibility_checkpoint)
        runtime = load_v11_checkpoint_as_v12(source, resource_policy=policy)
        accepted_reference = {
            "kind": "immutable_v11_accepted_fem_state",
            "source_checkpoint": args.v11_compatibility_checkpoint,
            "topology_fingerprint": source.topology_fingerprint,
            "mesh_identity": source.mesh_identity,
        }
        source_schema = "v11.production-branch-network-checkpoint/2"
    else:
        from .multifront_checkpoint_v12 import load_multifront_checkpoint

        restored = load_multifront_checkpoint(args.v12_checkpoint)
        runtime = restored.runtime
        accepted_reference = dict(restored.accepted_fem_state_reference)
        source_schema = "v12.general-multifront-checkpoint/2"
    inventory = prepare_pre_solve_inventory(
        runtime, accepted_fem_state_reference=accepted_reference,
    )
    return {
        "schema": "v12.general-multifront-production-preflight/1",
        "qualification": "PRODUCTION_ENTRYPOINT_PRE_SOLVE_BOUNDARY_REACHED",
        "boundary": BOUNDARY,
        "checkpoint_source_schema": source_schema,
        "policy": policy.to_dict(),
        "inventory": inventory.to_dict(),
        "provider_lookup_count": 0,
        "mechanics_solve_count": 0,
        "pf_worker_started": False,
        "fem_worker_started": False,
        "production_output_root_created": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import sys
    values = tuple(sys.argv[1:] if argv is None else argv)
    policy = validate_args(values)
    if "--v6-3-stateful-production-dry-run" in values:
        payload = v6_3_stateful_production_dry_run(values)
    elif "--v6-4-integrated-interval-dry-run" in values:
        payload = v6_4_integrated_interval_dry_run(values)
    elif "--v6-2-production-adapter-dry-run" in values:
        payload = v6_2_production_adapter_dry_run(values)
    elif "--production-preflight" in values:
        payload = production_preflight(values)
    else:
        payload = {
        "schema": "v12.general-multifront-source-preflight/1",
        "qualification": "SOURCE_ONLY_PREFLIGHT_PASS",
        "boundary": BOUNDARY,
        "policy": policy.to_dict(),
        "finite_front_physics_cap": None,
        "pf_worker_started": False,
        "fem_worker_started": False,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
