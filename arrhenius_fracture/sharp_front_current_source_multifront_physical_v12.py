"""Bounded, resumable current-source V12 physical multifront qualification."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from .general_multifront_v12 import ResourcePolicy
from .network_metrics_v11 import crack_growth_metrics


CLAIM_LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


def _value(argv: Sequence[str], name: str, default=None):
    values = []
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    if len(values) > 1:
        raise SystemExit(f"duplicate argument: {name}")
    return values[0] if values else default


def _remove(argv: Sequence[str], *names: str) -> list[str]:
    result = []
    index = 0
    names = set(names)
    while index < len(argv):
        token = argv[index]
        key = token.split("=", 1)[0]
        if key in names:
            index += 1 if "=" in token else 2
            continue
        result.append(token)
        index += 1
    return result


def run_v12_2d(args):
    """Continue an immutable V11 or accepted V12 boundary with V12 lifecycle."""
    from . import sharp_front as base
    from .branch_checkpoint_v11 import restore_branch_checkpoint
    from .branch_scale_identity_v11 import resolve_branch_scale_identity
    from .directional_competition_v11 import tungsten_cleavage_candidates
    from .fem import assemble_mechanics
    from .multifront_checkpoint_v12 import (
        load_accepted_boundary_checkpoint_v12, load_v11_checkpoint_as_v12,
    )
    from .current_source_multifront_hooks_v12 import (
        restore_complete_current_source_engine,
    )
    from .stateful_multifront_production_v12 import (
        CurrentSourceMultiFrontProductionContextV12,
        evaluate_candidate_marginal_kinetic_v12,
        production_directional_rate_adapter_v12,
        recompute_physical_process_connectivity_v12,
        run_stateful_accepted_interval_v12,
    )
    from .production_multifront_v12 import accepted_fem_state_fingerprint

    source_v11 = os.environ.get("V12_SOURCE_V11_CHECKPOINT", "").strip()
    source_v12 = os.environ.get("V12_SOURCE_ACCEPTED_CHECKPOINT", "").strip()
    maximum_fronts = int(os.environ["V12_MAXIMUM_FRONTS"])
    if bool(source_v11) == bool(source_v12):
        raise SystemExit("select exactly one V11 or V12 accepted source checkpoint")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint" / "latest.v12.pkl"
    resume = checkpoint_path if checkpoint_path.is_file() else (
        Path(source_v12) if source_v12 else None
    )
    cfg = base.make_emergent_config()
    mat = cfg.material
    candidates = tungsten_cleavage_candidates(
        theta_deg=float(getattr(args, "crystal_theta_deg", 0.0) or 0.0),
        include_110=bool(getattr(args, "crystal_include_110", False)),
        gamma_110_rel=float(getattr(args, "gamma_110_rel", 1.3) or 1.3),
    )
    policy = ResourcePolicy(
        "mechanistic", maximum_fronts, None,
        "forbid", "v11_correlated_proposal_compatibility",
    )
    if resume is not None:
        restored = load_accepted_boundary_checkpoint_v12(resume)
        runtime = replace(restored.runtime, resource_policy=policy)
        state = restored.accepted_fem_state
        sigma = restored.accepted_stress_field
        physical_time = restored.physical_time_s
        accepted_load = restored.accepted_opening_m
        step_count = restored.step_count
        source_record = {
            "kind": "v12_accepted_boundary", "path": str(resume.resolve()),
            "payload_sha256": restored.payload_sha256,
        }
    else:
        source = restore_branch_checkpoint(source_v11)
        runtime = load_v11_checkpoint_as_v12(source, resource_policy=policy)
        state = source.state
        sigma = assemble_mechanics(
            state.mesh, state.displacement, state.ep_gp, state.rho_gp,
            state.damage, state.elasticity_D, state.material,
            cohesive_network=state.cohesive_network,
        )[2]
        physical_time = source.physical_time_s
        accepted_load = source.accepted_load
        step_count = int(source.state.event_counters.get("accepted_steps", 0))
        source_record = {
            "kind": "immutable_v11_checkpoint",
            "path": str(Path(source_v11).resolve()),
        }
    scale = resolve_branch_scale_identity(args, state.mesh)
    source_is_pre_adapted = bool(
        source_v11 and "_mesh_adaptation_" in Path(source_v11).name
    )
    actual_engines = {
        engine_id: restore_complete_current_source_engine(engine)
        for engine_id, engine in runtime.process_engines.items()
    }
    if not actual_engines:
        raise SystemExit("accepted restart contains no physical process engine")
    correlation_intervals = {
        float(engine.f.tau_c) for engine in actual_engines.values()
    }
    if len(correlation_intervals) != 1:
        raise SystemExit("accepted process engines disagree on correlation interval")
    context = CurrentSourceMultiFrontProductionContextV12(
        args=args,
        mechanical_configuration={
            "theta_deg": float(getattr(args, "crystal_theta_deg", 0.0) or 0.0),
            "temperature_K": float(args.temperatures[0]),
            "crack_backend": str(args.crack_backend),
            "bulk_plasticity_mode": str(
                getattr(args, "bulk_plasticity_mode", "tip_only")
            ),
        },
        accepted_fem_state=state, accepted_stress_field=sigma,
        runtime=runtime, provider_runtime={},
        destination_cache_root=out / "live_kernel_cache",
        output_root=out / "accepted_intervals",
        checkpoint_path=checkpoint_path, candidates=candidates,
        physical_time_s=physical_time, accepted_opening_m=accepted_load,
        step_count=step_count, stress_available=True,
        mechanics_source_identity="current_source_internal_fem_v12",
        adapter_configuration={
            "cfg": cfg, "base": base, "material": mat,
            "elasticity_D": state.elasticity_D,
            "da_phys_m": float(args.da_phys),
            "tip_h_fine_m": float(getattr(args, "tip_h_fine", 0.0) or 1.0e-6),
            "contour_radius_m": float(getattr(args, "rJ", None) or args.L_pz),
            "crack_band_radius_m": 0.5e-6,
            "temperature_K": float(args.temperatures[0]),
            "correlation_interval_s": correlation_intervals.pop(),
            "requested_interval_duration_s": float(args.dt),
            "accepted_pre_adapted_source_sha256": (
                accepted_fem_state_fingerprint(state)
                if source_is_pre_adapted else None
            ),
            "mechanics_source_identity": "current_source_internal_fem_v12",
            "directional_rate_adapter": production_directional_rate_adapter_v12,
            "marginal_kinetic_evaluator": evaluate_candidate_marginal_kinetic_v12,
            "connectivity_recomputer": recompute_physical_process_connectivity_v12,
            "branch_handoff_length_m": scale.branch_handoff_length_m,
            "process_zone_length_m": scale.physical_process_zone_length_m,
            "permitted_physical_hazard_action": float(
                getattr(args, "adaptive_event_target", 0.075) or 0.075
            ),
        },
    )
    context.actual_engines = actual_engines
    initial_growth = crack_growth_metrics(
        context.runtime.crack_network, initial_crack_length_m=cfg.geometry.a0,
    )
    initial_extension = initial_growth.max_root_to_tip_path_extension_m
    initial_births = context.runtime.cumulative_branch_births
    target_added = float(os.environ.get("V12_TARGET_ADDED_EXTENSION_UM", "25")) * 1e-6
    target_forward_text = os.environ.get("V12_TARGET_MAXIMUM_FORWARD_REACH_UM", "").strip()
    target_forward = None if not target_forward_text else float(target_forward_text) * 1e-6
    target_births_text = os.environ.get("V12_STOP_AFTER_TOTAL_BIRTHS", "").strip()
    target_births = None if not target_births_text else int(target_births_text)
    bounded_birth_mode = target_forward is None
    if bounded_birth_mode and target_births is None:
        target_births = initial_births + 1
    post_birth_events = int(os.environ.get("V12_POST_BIRTH_EVENTS", "3"))
    events_after_target_birth = 0
    reached_target_birth = bool(
        target_births is not None
        and context.runtime.cumulative_branch_births >= target_births
    )
    historical_front_counts = [len(context.runtime.active_front_ids)]
    historical_front_counts.extend(
        item.post_active_front_count for item in context.runtime.transaction_records
    )
    maximum_concurrent_fronts = max(historical_front_counts)
    milestone_targets_um = (250.0, 500.0, 750.0, 1000.0)
    milestone_root = out / "milestone_morphology"
    already_recorded = {
        float(path.stem.split("_", 1)[1])
        for path in milestone_root.glob("reach_*.json")
        if "_" in path.stem
    } if milestone_root.is_dir() else set()
    launch = {
        "schema": "v12.bounded-physical-multifront-launch/1",
        "claim_label": CLAIM_LABEL, "source": source_record,
        "maximum_fronts": maximum_fronts,
        "branch_transaction_limit": None,
        "target_added_extension_um": target_added * 1e6,
        "target_total_births": target_births,
        "target_maximum_network_forward_reach_um": (
            None if target_forward is None else target_forward * 1e6
        ),
        "post_birth_events": post_birth_events,
        "argv": vars(args),
    }
    (out / "v12_launch.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True, default=str) + "\n"
    )
    while True:
        try:
            result = run_stateful_accepted_interval_v12(
                context, float(args.dt)
            )
        except (ValueError, RuntimeError) as exc:
            if not (
                str(exc) == "owner_local_kernel_coordinate_outside_qualified_family_domain"
                or "outside the validated signed-kernel envelope" in str(exc)
            ):
                raise
            termination = "owner_local_kernel_coordinate_outside_qualified_family_domain"
            growth = crack_growth_metrics(
                context.runtime.crack_network, initial_crack_length_m=cfg.geometry.a0,
            )
            progress = {
                "step": context.step_count,
                "disposition": "fail_closed_owner_kernel_envelope",
                "fail_closed_detail": str(exc),
                "active_front_count": len(context.runtime.active_front_ids),
                "branch_birth_count": context.runtime.cumulative_branch_births,
                "maximum_network_forward_reach_um": (
                    growth.max_forward_projected_extension_m * 1.0e6
                ),
                "checkpoint": str(checkpoint_path.resolve()),
            }
            break
        growth = crack_growth_metrics(
            context.runtime.crack_network, initial_crack_length_m=cfg.geometry.a0,
        )
        event = result.disposition == "accepted_event"
        if (
            target_births is not None
            and context.runtime.cumulative_branch_births >= target_births
        ):
            if reached_target_birth and event:
                events_after_target_birth += 1
            reached_target_birth = True
        progress = {
            "step": context.step_count, "disposition": result.disposition,
            "accepted_duration_s": result.accepted_duration_s,
            "physical_time_s": context.physical_time_s,
            "accepted_opening_m": context.accepted_opening_m,
            "active_front_count": len(context.runtime.active_front_ids),
            "branch_birth_count": context.runtime.cumulative_branch_births,
            "physical_extension_um": (
                growth.max_root_to_tip_path_extension_m * 1.0e6
            ),
            "maximum_network_forward_reach_um": (
                growth.max_forward_projected_extension_m * 1.0e6
            ),
            "added_extension_um": (
                growth.max_root_to_tip_path_extension_m - initial_extension
            ) * 1.0e6,
            "events_after_target_birth": events_after_target_birth,
            "checkpoint": str(checkpoint_path.resolve()),
            "owner_local_coordinates_um": {
                owner_id: region.cumulative_process_advance_m * 1.0e6
                for owner_id, region in sorted(context.runtime.process_regions.items())
            },
            "topology_fingerprint": context.runtime.topology_fingerprint,
            "registry_fingerprint": context.runtime.registry_fingerprint,
        }
        maximum_concurrent_fronts = max(
            maximum_concurrent_fronts, len(context.runtime.active_front_ids)
        )
        print(json.dumps(progress, sort_keys=True), flush=True)
        progress_path = out / "v12_progress.jsonl"
        with progress_path.open("a") as stream:
            stream.write(json.dumps(progress, sort_keys=True) + "\n")
        current_forward_um = growth.max_forward_projected_extension_m * 1.0e6
        for milestone_um in milestone_targets_um:
            if milestone_um in already_recorded or current_forward_um < milestone_um:
                continue
            milestone_root.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "v12.maximum-forward-reach-milestone/1",
                "claim_label": CLAIM_LABEL,
                "milestone_um": milestone_um,
                "actual_maximum_network_forward_reach_um": current_forward_um,
                "step": context.step_count,
                "physical_time_s": context.physical_time_s,
                "accepted_opening_m": context.accepted_opening_m,
                "active_front_ids": list(context.runtime.active_front_ids),
                "owner_by_front": dict(context.runtime.owner_by_front),
                "owner_local_coordinates_um": progress["owner_local_coordinates_um"],
                "cumulative_binary_branch_births": context.runtime.cumulative_branch_births,
                "crack_network": context.runtime.crack_network.to_dict(),
                "topology_fingerprint": context.runtime.topology_fingerprint,
                "registry_fingerprint": context.runtime.registry_fingerprint,
                "checkpoint": str(checkpoint_path.resolve()),
            }
            (milestone_root / f"reach_{int(milestone_um):04d}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            already_recorded.add(milestone_um)
        if result.disposition == "resource_stop":
            termination = "configured_front_resource_limit_reached"
            break
        if target_forward is not None and (
            growth.max_forward_projected_extension_m >= target_forward
        ):
            termination = "maximum_network_forward_reach_target_reached"
            break
        if bounded_birth_mode and (
            reached_target_birth
            and events_after_target_birth >= post_birth_events
            and growth.max_root_to_tip_path_extension_m - initial_extension
            >= target_added
        ):
            termination = "bounded_physical_target_reached"
            break
        if context.step_count >= int(args.steps):
            termination = "step_limit_reached"
            break
    final = {
        **progress, "schema": "v12.bounded-physical-multifront-result/1",
        "claim_label": CLAIM_LABEL, "termination": termination,
        "source": source_record,
        "runtime_sha256": context.runtime.registry_fingerprint,
        "target_1000um_reached": bool(
            growth.max_forward_projected_extension_m >= 1000.0e-6
        ),
        "cumulative_binary_branch_births": context.runtime.cumulative_branch_births,
        "maximum_concurrent_active_fronts": maximum_concurrent_fronts,
        "at_least_four_cumulative_births_observed": bool(
            context.runtime.cumulative_branch_births >= 4
        ),
        "front_resource_limit_bound": termination == "configured_front_resource_limit_reached",
        "predictive_recursive_branching_physics": "NOT_VALIDATED",
    }
    (out / "v12_run_complete.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n"
    )
    return 0


def main(argv=None):
    from . import sharp_front as base
    from . import sharp_front_v10_2_27 as paper
    from . import sharp_front_v10_2_28_audited as entry

    values = list(sys.argv[1:] if argv is None else argv)
    maximum = int(_value(values, "--maximum-fronts", "2"))
    if maximum < 1 or maximum > 6:
        raise SystemExit("V12 capability continuation permits one through six fronts")
    source_v11 = _value(values, "--v11-restart-checkpoint")
    source_v12 = _value(values, "--v12-restart-checkpoint")
    if bool(source_v11) == bool(source_v12):
        raise SystemExit("select exactly one V11 or V12 restart checkpoint")
    os.environ["V12_MAXIMUM_FRONTS"] = str(maximum)
    if source_v11:
        os.environ["V12_SOURCE_V11_CHECKPOINT"] = source_v11
        os.environ.pop("V12_SOURCE_ACCEPTED_CHECKPOINT", None)
    else:
        os.environ["V12_SOURCE_ACCEPTED_CHECKPOINT"] = source_v12
        os.environ.pop("V12_SOURCE_V11_CHECKPOINT", None)
    os.environ["V12_TARGET_ADDED_EXTENSION_UM"] = str(
        _value(values, "--v12-target-added-extension-um", "25")
    )
    target_forward = _value(values, "--v12-target-maximum-forward-reach-um")
    if target_forward is None:
        os.environ.pop("V12_TARGET_MAXIMUM_FORWARD_REACH_UM", None)
    else:
        os.environ["V12_TARGET_MAXIMUM_FORWARD_REACH_UM"] = str(target_forward)
    os.environ["V12_POST_BIRTH_EVENTS"] = str(
        _value(values, "--v12-post-birth-events", "3")
    )
    total_births = _value(values, "--v12-stop-after-total-births")
    if total_births is not None:
        os.environ["V12_STOP_AFTER_TOTAL_BIRTHS"] = str(total_births)
    forwarded = _remove(
        values, "--maximum-fronts", "--v11-restart-checkpoint",
        "--v12-restart-checkpoint", "--v12-target-added-extension-um",
        "--v12-post-birth-events", "--v12-stop-after-total-births",
        "--v12-target-maximum-forward-reach-um",
    )
    forwarded.extend(["--max-fronts", "1"])
    original = base.run_2d
    original_registry = paper.DEFAULT_REGISTRY
    original_selection = paper.SELECTION_RECORD
    original_options = paper.VALID_OPTIONS
    root = Path(__file__).resolve().parents[1]
    paper.DEFAULT_REGISTRY = root / (
        "runtime_inputs/pf_current_source_branching/"
        "pf_v2_four_class_pf_transfer_registry.csv"
    )
    paper.SELECTION_RECORD = root / (
        "runtime_inputs/pf_current_source_branching/"
        "pf_v2_four_class_pf_transfer_selection.json"
    )
    paper.VALID_OPTIONS = {
        "v913_paper_peak01_0242980_persistent_sites": "v913_zeroD_sobol_0242980",
        "v913_paper_dbtt01_0202500_persistent_sites": "v913_zeroD_sobol_0202500",
        "v913_paper_weakT01_0129902_persistent_sites": "oneD_v2_focused_weak_T_0016",
        "v913_paper_ceramic01_0077080_persistent_sites": "oneD_v2_focused_ceramic_like_0018",
    }
    base.run_2d = run_v12_2d
    try:
        try:
            return entry.main(forwarded)
        except RuntimeError as exc:
            output = Path(str(_value(forwarded, "--out", "")))
            if (
                str(exc) == "no stochastic avalanche backend was constructed"
                and (output / "v12_run_complete.json").is_file()
            ):
                return 0
            raise
    finally:
        base.run_2d = original
        paper.DEFAULT_REGISTRY = original_registry
        paper.SELECTION_RECORD = original_selection
        paper.VALID_OPTIONS = original_options


if __name__ == "__main__":
    raise SystemExit(main())
