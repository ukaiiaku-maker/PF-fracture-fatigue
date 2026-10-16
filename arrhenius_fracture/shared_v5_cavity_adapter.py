"""Paired physical V5 cavity control using the native coupled scheduler.

The default path evolves enabled and suppressed arms from matching admissible
states under the same nonzero physical exposure. The former extreme-reservoir
lifecycle demonstration is retained only behind explicit test authorization.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .checkpoint_v11 import restore_checkpoint, write_checkpoint
from .crack_network_v11 import ROOT_BRANCH_ID
from .global_physical_scheduler_v1 import advance_event_aware_coupled_interval
from .topology_transaction_v11 import (
    complete_accepted_state_fingerprint,
    equilibrate_fixed_load_with_production_fem,
)
from .unified_fracture_material_v5 import (
    material_bundle, registry_material_bundle, require_bound_identity,
)
from .voiding_production_v5 import (
    _bundle_from_state, build_production_void_state, crack_tip_tensor,
    deterministic_trajectory,
    directional_sharp_front_rates, dispatch_cavity_intersecting_candidate_drives,
    fresh_sharp_front_engine, local_incremental_cavity_candidate_drives,
    observables, remesh_cavity, sharp_front_load_provider,
)
from .voiding_v5 import initialize_preexisting_resolved_cavity

MODEL_ID = "shared.native_v5_physical_paired_cavity/2"
FIXTURE_MODEL_ID = "shared.test_only_v5_extreme_reservoir_lifecycle_fixture/1"
PAIR_CHECKPOINT_SCHEMA = "shared-v5-physical-pair-checkpoint/2"
LEGACY_PAIR_CHECKPOINT_SCHEMA = "shared-v5-physical-pair-checkpoint/1"
DEFAULT_SITES_M = ((6.0e-4, 6.0e-5), (7.5e-4, -6.0e-5))
DEFAULT_PARTITION = (0.50, 0.10, 0.40)
TRANSPORT_BOUND_FAILURE = "vacancy transport solve exceeded fail-closed concentration bounds"
TRANSPORT_PICARD_FAILURE = "vacancy transport Picard iteration did not converge"
RETRIABLE_TRANSPORT_FAILURES = {
    TRANSPORT_BOUND_FAILURE,
    TRANSPORT_PICARD_FAILURE,
}


def _finite(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _finite(asdict(value))
    if isinstance(value, Enum):
        return _finite(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def _accepted_fingerprint(state) -> str:
    """Production fingerprint with a deterministic scalar-test fallback."""
    try:
        return complete_accepted_state_fingerprint(state)
    except AttributeError:
        return hashlib.sha256(repr(state).encode()).hexdigest()


def _common_start_audit(disabled, enabled) -> dict:
    left_transport = disabled.vacancy_transport_state_v2
    right_transport = enabled.vacancy_transport_state_v2
    return {
        "mechanical_and_transport_arrays_exact": _mechanical_arrays_equal(
            disabled, enabled
        ),
        "fracture_rng_state_exact": dict(disabled.rng_state) == dict(enabled.rng_state),
        "vacancy_transport_field_exact": np.array_equal(
            left_transport.vacancy_fraction_gp,
            right_transport.vacancy_fraction_gp,
        ),
        "transport_parameter_row_exact": (
            left_transport.parameter_row == right_transport.parameter_row
        ),
        "enabled_cavity_rng_owned_independently": bool(
            enabled.void_state is not None
            and getattr(enabled.void_state, "rng_state", None) is not None
        ),
        "suppressed_arm_has_no_cavity_owner": disabled.void_state is None,
    }


def _directional_provider(temperature_K: float):
    """Return the qualified native 4 then 4/5-level directional observer."""
    def observe(state):
        tensor, support = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
        provider = sharp_front_load_provider(
            state, branch_id=ROOT_BRANCH_ID, matched_discretization=True,
            marginal_delta_a_m=(3.0e-5, 2.0e-5),
            marginal_mesh_levels=(4,),
            virtual_mechanical_trial_transport="deferred_read_only",
        )
        if not any(row["directional_drive_status"] == "AVAILABLE"
                   for row in provider["directional"]):
            provider = sharp_front_load_provider(
                state, branch_id=ROOT_BRANCH_ID, matched_discretization=True,
                marginal_delta_a_m=(3.0e-5, 2.0e-5),
                marginal_mesh_levels=(4, 5),
                virtual_mechanical_trial_transport="deferred_read_only",
                mechanical_refinement_policy="legacy_accumulated_generations",
            )
        physical_event_advance_m = float(
            fresh_sharp_front_engine(state.material, _bundle_from_state(state)).f.da
        )
        local_incremental = local_incremental_cavity_candidate_drives(
            state, branch_id=ROOT_BRANCH_ID,
            maximum_event_advance_m=physical_event_advance_m,
            refinement_levels=(3, 4),
        )
        provider["directional"] = list(
            dispatch_cavity_intersecting_candidate_drives(
                state,
                branch_id=ROOT_BRANCH_ID,
                directional_rows=provider["directional"],
                maximum_event_advance_m=physical_event_advance_m,
                refinement_levels=(3, 4),
                local_incremental_rows=local_incremental,
            )
        )
        rates = directional_sharp_front_rates(
            state, provider["directional"], branch_id=ROOT_BRANCH_ID,
            temperature_K=float(temperature_K),
        )
        available = [row for row in rates
                     if row["directional_drive_status"] == "AVAILABLE"]
        if not available:
            raise RuntimeError(
                "REQUIRED_FRACTURE_OBSERVATION_UNAVAILABLE_AFTER_BOUNDED_REFINEMENT"
            )
        effective_k = max(float(row["K_effective_Pa_sqrt_m"])
                          for row in available)
        audit = {
            "adjacent_element_tensor_Pa": tensor,
            "adjacent_element_ids": support,
            "canonical_directional_rows": provider["directional"],
            "fracture_engine_rows": rates,
            "state_bound_K_effective_Pa_sqrt_m": effective_k,
        }
        return tensor, effective_k, audit, provider["directional"]
    return observe


def _requested_bundle(args):
    registry = getattr(args, "material_registry", None)
    option = getattr(args, "parameter_option", None)
    historical = bool(getattr(args, "historical_v5_material", False))
    if historical:
        if registry is not None or option is not None:
            raise SystemExit(
                "historical V5 material is mutually exclusive with a current registry request"
            )
        return material_bundle(args.material_class)
    if registry is None or option is None:
        raise SystemExit(
            "physical V5 execution requires --material-registry and --parameter-option; "
            "use --historical-v5-material only for the named retained alternative"
        )
    return registry_material_bundle(
        registry, option, material_class=args.material_class,
    )


def _build_pair(args, bundle=None):
    bundle = _requested_bundle(args) if bundle is None else bundle
    common = dict(
        bundle=bundle, temperature_K=float(args.temperature_K),
        vacancy_energy_variant=args.vacancy_energy_variant,
        void_site_centers_m=DEFAULT_SITES_M, stochastic=True, seed=int(args.seed),
        direct_interaction_birth_enabled=True,
        cavity_surface_loop_emission_enabled=True,
        interaction_population_partition=DEFAULT_PARTITION,
        bulk_initial_density_m2=float(bundle.fracture_row["rho_forest_floor_m2"]),
        initial_plastic_strain_gp=0.0,
        boundary_segments=int(getattr(args, "boundary_segments", 32)),
        radial_layers=int(getattr(args, "radial_layers", 12)),
        shape_regular_local_level=getattr(args, "shape_regular_local_level", None),
        source_conforming_geometry=bool(getattr(
            args, "source_conforming_geometry", False
        )),
    )
    preexisting = args.preexisting_cavity_radius_pz is not None
    if preexisting:
        if args.preexisting_cavity_separation_pz is None:
            raise SystemExit(
                "preexisting cavity radius and surface-to-tip separation must be supplied together"
            )
        pz = float(args.process_zone_length_um) * 1.0e-6
        radius = float(args.preexisting_cavity_radius_pz) * pz
        separation = float(args.preexisting_cavity_separation_pz) * pz
        if not all(math.isfinite(value) and value > 0.0 for value in (pz, radius, separation)):
            raise SystemExit("preexisting cavity geometry must be finite and positive")
        tip_x = float(args.initial_crack_length_um) * 1.0e-6
        center = (tip_x + separation + radius, 0.0)
        if center[0] + radius >= 1.0e-3:
            raise SystemExit("preexisting cavity lies outside the native specimen")
        common.update({
            "cavity_center_m": center,
            "void_site_centers_m": (center,),
            "crack_path_m": ((0.0, 0.0), (tip_x, 0.0)),
            "cavity_geometry_radius_m": radius,
        })
    disabled, _ = build_production_void_state(enabled=False, **common)
    enabled, hole = build_production_void_state(enabled=True, **common)
    if preexisting:
        voids = initialize_preexisting_resolved_cavity(
            enabled.void_state, "site-1", radius, seed=int(args.seed) + 84731,
        )
        enabled = remesh_cavity(
            enabled, hole, voids, "preexisting-cavity-initial-condition",
        )
        disabled = equilibrate_fixed_load_with_production_fem(disabled)
        enabled = equilibrate_fixed_load_with_production_fem(enabled)
        audit = {
            "conditional_preexisting_geometry": True,
            "preexisting_radius_m": radius,
            "surface_to_tip_separation_m": separation,
            "initial_crack_tip_x_m": tip_x,
            "preexisting_geometry_vacancy_debit_zero": (
                enabled.void_state.event_history[-1].get(
                    "vacancy_inventory_debit_area_m2"
                ) == 0.0
            ),
            "both_arms_native_fixed_load_equilibrated": True,
        }
        return disabled, enabled, audit
    return disabled, enabled, _common_start_audit(disabled, enabled)


def _write_pair_checkpoint(states, root: Path, metadata: dict) -> Path:
    directory = Path(os.environ.get(
        "V10230_HIGH_CYCLE_CHECKPOINT_DIR", root / "checkpoint"
    )).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    fingerprints = {
        arm: complete_accepted_state_fingerprint(state)
        for arm, state in states.items()
    }
    identity = hashlib.sha256(json.dumps({
        "physical_time_s": metadata.get("physical_time_s"),
        "interval_count": metadata.get("interval_count"),
        "fingerprints": fingerprints,
    }, sort_keys=True).encode()).hexdigest()[:16]
    paths = {}
    for arm, state in states.items():
        path = directory / f"accepted_v5_{identity}_{arm}.json"
        write_checkpoint(state, path, compression="gzip")
        restored = restore_checkpoint(path)
        if complete_accepted_state_fingerprint(restored) != fingerprints[arm]:
            raise RuntimeError(f"paired V5 {arm} checkpoint verification failed")
        paths[arm] = path.name
    payload = {
        "schema": PAIR_CHECKPOINT_SCHEMA,
        "arms": paths,
        "generation": identity,
        "fingerprints": fingerprints,
        **metadata,
    }
    target = directory / "accepted_v5_pair.json"
    temporary = directory / "accepted_v5_pair.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)
    return target


def _restore_pair(path: str | Path):
    source = Path(path).resolve()
    payload = json.loads(source.read_text())
    if payload.get("schema") not in {
        PAIR_CHECKPOINT_SCHEMA, LEGACY_PAIR_CHECKPOINT_SCHEMA,
    }:
        raise RuntimeError(
            "physical V5 restart requires a paired native checkpoint manifest"
        )
    states = {}
    for arm in ("disabled", "enabled"):
        checkpoint = Path(payload["arms"][arm])
        if not checkpoint.is_absolute():
            checkpoint = source.parent / checkpoint
        states[arm] = restore_checkpoint(checkpoint)
        observed = complete_accepted_state_fingerprint(states[arm])
        if observed != payload["fingerprints"][arm]:
            raise RuntimeError(f"paired V5 {arm} checkpoint fingerprint changed")
    return states, payload


class _PairedRefinementFailure(RuntimeError):
    def __init__(self, states, segments, diagnostic, reason=TRANSPORT_BOUND_FAILURE):
        super().__init__(reason)
        self.states = states
        self.segments = segments
        self.diagnostic = diagnostic
        self.reason = reason


def _advance_paired_with_refinement(
    states, duration_s: float, advance, *, minimum_duration_s: float,
    requested_duration_s: float | None = None, depth: int = 0,
    trial_counter: list[int] | None = None,
):
    """Advance both controls on one shared, fail-closed native time partition."""
    trial = {}
    schedulers = {}
    if trial_counter is None:
        trial_counter = [0]
    if requested_duration_s is None:
        requested_duration_s = float(duration_s)
    try:
        for arm in ("disabled", "enabled"):
            trial_counter[0] += 1
            trial[arm], schedulers[arm] = advance(
                copy.deepcopy(states[arm]), float(duration_s), arm
            )
        return trial, [(float(duration_s), schedulers, trial)]
    except RuntimeError as error:
        if isinstance(error, _PairedRefinementFailure):
            raise
        reason = str(error)
        if reason not in RETRIABLE_TRANSPORT_FAILURES:
            raise
        diagnostic = dict(getattr(error, "diagnostic", {}))
        projected_raw = getattr(states[arm], "vacancy_fraction_gp", None)
        mechanics_projected = (None if projected_raw is None else
                               np.asarray(projected_raw, float))
        void_state = getattr(states[arm], "void_state", None)
        void_rng = None if void_state is None else getattr(
            void_state, "rng_state", None)
        transport_state = getattr(states[arm], "vacancy_transport_state_v2", None)
        parent_fingerprints = {
            name: _accepted_fingerprint(state)
            for name, state in states.items()
        }
        diagnostic.update({
            "transport_failure_reason": reason,
            "arm": arm,
            "requested_interval_s": float(requested_duration_s),
            "actual_subinterval_s": float(duration_s),
            "refinement_depth": int(depth),
            "trial_number": int(trial_counter[0]),
            "parent_accepted_state_fingerprints": parent_fingerprints,
            "rollback_result": "exact-paired-parent-retained",
            "mechanics_projected_concentration_min": (None if mechanics_projected is None
                else float(np.min(mechanics_projected))),
            "mechanics_projected_concentration_max": (None if mechanics_projected is None
                else float(np.max(mechanics_projected))),
            "fracture_rng_sha256": hashlib.sha256(
                repr(getattr(states[arm], "rng_state", None)).encode()).hexdigest(),
            "cavity_rng_sha256": (None if void_rng is None else hashlib.sha256(
                repr(void_rng).encode()).hexdigest()),
            "transport_clock_start_s": (None if transport_state is None else
                float(transport_state.transport_time_s)),
        })
        half = 0.5 * float(duration_s)
        if half < float(minimum_duration_s):
            diagnostic["minimum_duration_s"] = float(minimum_duration_s)
            raise _PairedRefinementFailure(
                states, [], diagnostic, reason=reason,
            ) from error
        midpoint, first = _advance_paired_with_refinement(
            states, half, advance, minimum_duration_s=minimum_duration_s,
            requested_duration_s=requested_duration_s, depth=depth+1,
            trial_counter=trial_counter,
        )
        try:
            terminal, second = _advance_paired_with_refinement(
                midpoint, half, advance, minimum_duration_s=minimum_duration_s,
                requested_duration_s=requested_duration_s, depth=depth+1,
                trial_counter=trial_counter,
            )
        except _PairedRefinementFailure as rejection:
            raise _PairedRefinementFailure(
                rejection.states, [*first, *rejection.segments],
                rejection.diagnostic, reason=rejection.reason,
            ) from rejection
        return terminal, [*first, *second]


def _physical_main(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    requested_bundle = _requested_bundle(args)
    if args.restart_checkpoint:
        states, ancestry = _restore_pair(args.restart_checkpoint)
        observed_hashes = {
            arm: require_bound_identity(state)["material_bundle_sha256"]
            for arm, state in states.items()
        }
        if set(observed_hashes.values()) != {requested_bundle.bundle_hash}:
            raise ValueError(
                "paired restart material does not match requested registry option"
            )
        common_start = ancestry.get("common_start")
        physical_time = float(ancestry["physical_time_s"])
        interval_index = int(ancestry["interval_count"])
    else:
        disabled, enabled, common_start = _build_pair(args, requested_bundle)
        if not all(common_start.values()):
            raise RuntimeError("V5 enabled/suppressed common start failed")
        states = {"disabled": disabled, "enabled": enabled}
        ancestry = None
        physical_time = 0.0
        interval_index = 0

    duration = float(args.duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise SystemExit("--duration-s must be finite and positive")
    endpoint_controller = (
        args.target_extension_um is not None or args.exposure_limit_s is not None
    )
    if endpoint_controller:
        if args.target_extension_um is None or args.exposure_limit_s is None:
            raise SystemExit(
                "endpoint control requires both --target-extension-um and --exposure-limit-s"
            )
        if not math.isfinite(args.target_extension_um) or args.target_extension_um <= 0.0:
            raise SystemExit("--target-extension-um must be finite and positive")
        if not math.isfinite(args.exposure_limit_s) or args.exposure_limit_s <= 0.0:
            raise SystemExit("--exposure-limit-s must be finite and positive")
        intervals = int(args.endpoint_max_intervals)
        if intervals < 1:
            raise SystemExit("--endpoint-max-intervals must be at least one")
    else:
        intervals = int(args.steps) + int(bool(args.continue_one))
        if intervals < 1:
            raise SystemExit("--steps must be at least one")
    observer = _directional_provider(float(args.temperature_K))
    history = {"disabled": [], "enabled": []}
    minimum_duration = float(args.transport_refinement_min_duration_s)
    if not math.isfinite(minimum_duration) or minimum_duration <= 0.0:
        raise SystemExit("--transport-refinement-min-duration-s must be positive")
    requested_interval_count = 0
    terminal_rejection = None
    call_start_time = physical_time
    selected_arm = "enabled" if args.cavity_enabled else "disabled"
    if ancestry is None:
        initial_graph_lengths = {
            arm: float(state.crack_network.total_physical_crack_length_m)
            for arm, state in states.items()
        }
    else:
        initial_graph_lengths = dict(
            ancestry.get("configuration", {}).get(
                "endpoint_initial_graph_length_m_by_arm", {}
            )
        )
        if endpoint_controller and set(initial_graph_lengths) != {"disabled", "enabled"}:
            raise ValueError("endpoint restart lacks initial paired crack lengths")

    target_extension_m = (
        None if not endpoint_controller else float(args.target_extension_um) * 1.0e-6
    )
    exposure_limit_s = (
        None if not endpoint_controller else float(args.exposure_limit_s)
    )

    if ancestry is not None:
        prior_configuration = ancestry.get("configuration", {})
        requested_restart_configuration = {
            "material_bundle_sha256": requested_bundle.bundle_hash,
            "material_class": args.material_class,
            "temperature_K": float(args.temperature_K),
            "vacancy_energy_variant": args.vacancy_energy_variant,
            "opening_rate_m_per_s": float(args.opening_rate_m_per_s),
            "endpoint_controller": endpoint_controller,
            "target_extension_m": target_extension_m,
            "exposure_limit_s": exposure_limit_s,
        }
        disagreements = {
            key: {"checkpoint": prior_configuration.get(key), "request": value}
            for key, value in requested_restart_configuration.items()
            if prior_configuration.get(key) != value
        }
        if disagreements:
            raise ValueError(
                "paired restart changes immutable physical request: "
                + json.dumps(disagreements, sort_keys=True)
            )

    def endpoint_status():
        if not endpoint_controller:
            return None
        extensions = {
            arm: (
                float(state.crack_network.total_physical_crack_length_m)
                - float(initial_graph_lengths[arm])
            )
            for arm, state in states.items()
        }
        if all(
            extension >= float(target_extension_m) * (1.0 - 1.0e-12)
            for extension in extensions.values()
        ):
            return "paired_developed_extension_reached"
        if physical_time >= float(exposure_limit_s) - max(1.0e-12, 1.0e-12 * float(exposure_limit_s)):
            return "declared_exposure_censor_reached"
        return None

    requested_exposure = (
        duration * intervals if not endpoint_controller
        else max(float(exposure_limit_s) - physical_time, 0.0)
    )
    terminal_endpoint = endpoint_status()
    for _ in range(intervals):
        terminal_endpoint = endpoint_status()
        if terminal_endpoint is not None:
            break
        requested_interval_count += 1
        starts = {
            arm: complete_accepted_state_fingerprint(state)
            for arm, state in states.items()
        }
        def advance(state, accepted_duration, _arm):
            return advance_event_aware_coupled_interval(
                state, accepted_duration,
                temperature_K=float(args.temperature_K),
                directional_observation_provider=observer,
                opening_rate_m_per_s=float(args.opening_rate_m_per_s),
            )
        interval_duration = duration
        if endpoint_controller:
            interval_duration = min(
                duration, max(float(exposure_limit_s) - physical_time, 0.0)
            )
            if interval_duration <= 0.0:
                terminal_endpoint = "declared_exposure_censor_reached"
                break
        try:
            states, accepted_segments = _advance_paired_with_refinement(
                states, interval_duration, advance, minimum_duration_s=minimum_duration,
            )
        except _PairedRefinementFailure as rejection:
            states = rejection.states
            accepted_segments = rejection.segments
            terminal_rejection = rejection.diagnostic
        segment_start = physical_time
        prior = starts
        for accepted_duration, schedulers, segment_states in accepted_segments:
            interval_index += 1
            segment_end = segment_start + accepted_duration
            for arm in ("disabled", "enabled"):
                fingerprint = complete_accepted_state_fingerprint(
                    segment_states[arm]
                )
                history[arm].append(_finite({
                    "interval": interval_index,
                    "requested_interval": requested_interval_count,
                    "physical_time_start_s": segment_start,
                    "physical_time_end_s": segment_end,
                    "duration_s": accepted_duration,
                    "input_fingerprint": prior[arm],
                    "output_fingerprint": fingerprint,
                    "scheduler": schedulers[arm],
                    "observables": observables(
                        segment_states[arm], f"physical_interval_{interval_index}"
                    ),
                }))
                prior[arm] = fingerprint
            segment_start = segment_end
        physical_time = segment_start
        if terminal_rejection is not None:
            terminal_rejection.update({
                "last_successfully_accepted_pair_time_s": physical_time,
                "last_successfully_accepted_pair_fingerprints": {
                    arm: complete_accepted_state_fingerprint(state)
                    for arm, state in states.items()
                },
                "rejected_interval_start_s": physical_time,
                "rejected_interval_end_s": physical_time + float(
                    terminal_rejection["actual_subinterval_s"]),
                "unused_requested_duration_s": max(
                    requested_exposure-(physical_time-call_start_time), 0.0),
            })
            break

    checkpoint = _write_pair_checkpoint(
        states, out,
        {
            "physical_time_s": physical_time,
            "interval_count": interval_index,
            "common_start": common_start,
            "configuration": {
                "material_class": args.material_class,
                "temperature_K": float(args.temperature_K),
                "seed": int(args.seed),
                "vacancy_energy_variant": args.vacancy_energy_variant,
                "loading_history": ("linear_total_opening_ramp" if
                    float(args.opening_rate_m_per_s) > 0.0 else
                    "hold_at_native_opening"),
                "opening_rate_m_per_s": float(args.opening_rate_m_per_s),
                "interval_duration_s": duration,
                "transport_refinement_min_duration_s": minimum_duration,
                "test_only_extreme_reservoir": False,
                "full_field_plasticity": True,
                "earliest_event_and_remainder_scheduler": True,
                "material_registry": (
                    None if args.material_registry is None
                    else str(Path(args.material_registry).expanduser().resolve())
                ),
                "parameter_option": args.parameter_option,
                "material_bundle_sha256": requested_bundle.bundle_hash,
                "historical_v5_material": bool(args.historical_v5_material),
                "endpoint_controller": endpoint_controller,
                "target_extension_m": target_extension_m,
                "exposure_limit_s": exposure_limit_s,
                "endpoint_initial_graph_length_m_by_arm": initial_graph_lengths,
            },
        },
    )
    restored, checkpoint_payload = _restore_pair(checkpoint)
    round_trip = all(
        complete_accepted_state_fingerprint(restored[arm])
        == complete_accepted_state_fingerprint(states[arm])
        for arm in states
    )
    terminal_endpoint = endpoint_status() if terminal_rejection is None else None
    extensions = {
        arm: (
            float(state.crack_network.total_physical_crack_length_m)
            - float(initial_graph_lengths[arm])
        )
        for arm, state in states.items()
    }
    payload = _finite({
        "schema": MODEL_ID,
        "production_eligible": True,
        "selected_arm": selected_arm,
        "both_arms_evolved": True,
        "status": (
            "transport_rejected_after_accepted_prefix"
            if terminal_rejection is not None else terminal_endpoint
            if terminal_endpoint is not None else
            "endpoint_iteration_guard_reached" if endpoint_controller else
            "requested_exposure_completed"
        ),
        "endpoint_controller_active": endpoint_controller,
        "selected_arm_extension_m": extensions[selected_arm],
        "arm_extensions_m": extensions,
        "target_extension_m": target_extension_m,
        "declared_exposure_limit_s": exposure_limit_s,
        "physical_exposure_s_per_arm_this_call": physical_time-call_start_time,
        "requested_physical_exposure_s_per_arm_this_call": requested_exposure,
        "unused_requested_duration_s": max(
            requested_exposure-(physical_time-call_start_time), 0.0),
        "physical_time_s": physical_time,
        "interval_count": interval_index,
        "requested_interval_count_this_call": requested_interval_count,
        "restart_input": args.restart_checkpoint,
        "restart_ancestry": ancestry,
        "common_start": common_start,
        "configuration": checkpoint_payload["configuration"],
        "branch_birth_permission_changed_by_cavity_toggle": False,
        "arms": {
            arm: {
                "cavity_enabled": arm == "enabled",
                "terminal_fingerprint": complete_accepted_state_fingerprint(state),
                "terminal": observables(state, "terminal"),
                "history": history[arm],
                "active_tip_ids": list(state.crack_network.active_tip_ids),
                "branch_birth_count": int(state.event_counters.get("branch_births", 0)),
            }
            for arm, state in states.items()
        },
        "native_pair_checkpoint": str(checkpoint),
        "native_checkpoint_round_trip_exact": round_trip,
        "transport_rejection": terminal_rejection,
    })
    (out / "shared_v5_cavity_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _fixture_main(args):
    """Retained historical lifecycle fixture; never selected by a preset."""
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    restarted = args.restart_checkpoint is not None
    initial_match = None
    if restarted:
        state = restore_checkpoint(args.restart_checkpoint)
        before = complete_accepted_state_fingerprint(state)
        state = equilibrate_fixed_load_with_production_fem(state)
        rows = [observables(state, "restart_re_equilibrated")]
    else:
        bundle = material_bundle("DBTT")
        disabled, _ = build_production_void_state(
            bundle=bundle, enabled=False, seed=3621, temperature_K=900.0,
        )
        enabled, _ = build_production_void_state(
            bundle=bundle, enabled=True, seed=3621, temperature_K=900.0,
        )
        initial_match = _mechanical_arrays_equal(disabled, enabled)
        if args.cavity_enabled:
            state, rows = deterministic_trajectory(
                bundle=bundle, stop_before_ligament=True,
            )
        else:
            state, rows = disabled, [observables(disabled, "disabled_control")]
        before = None
    path = out / "test_only_fixture_checkpoint.json"
    write_checkpoint(state, path, compression="gzip")
    payload = _finite({
        "schema": FIXTURE_MODEL_ID,
        "production_eligible": False,
        "explicit_test_authorization": True,
        "test_only_extreme_reservoir": True,
        "paired_physical_time_evolution": False,
        "matching_demonstration_initial_arrays": initial_match,
        "continued_from_fingerprint": before,
        "terminal_fingerprint": complete_accepted_state_fingerprint(state),
        "native_checkpoint": str(path),
        "history": rows,
    })
    (out / "shared_v5_lifecycle_fixture_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cavity-enabled", action="store_true")
    parser.add_argument("--cavity-disabled", action="store_true")
    parser.add_argument("--restart-checkpoint")
    parser.add_argument("--continue-one", action="store_true")
    parser.add_argument("--test-lifecycle-fixture", action="store_true")
    parser.add_argument("--material-class", default="DBTT")
    parser.add_argument("--material-registry")
    parser.add_argument("--parameter-option")
    parser.add_argument("--historical-v5-material", action="store_true")
    parser.add_argument("--temperature-K", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=3621)
    parser.add_argument("--duration-s", type=float, default=1.0e-5)
    parser.add_argument("--opening-rate-m-per-s", type=float, default=0.0)
    parser.add_argument("--transport-refinement-min-duration-s", type=float, default=1.0e-8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--target-extension-um", type=float)
    parser.add_argument("--exposure-limit-s", type=float)
    parser.add_argument("--endpoint-max-intervals", type=int, default=1000000)
    parser.add_argument("--vacancy-energy-variant", default="reference")
    parser.add_argument("--preexisting-cavity-radius-pz", type=float)
    parser.add_argument("--preexisting-cavity-separation-pz", type=float)
    parser.add_argument("--process-zone-length-um", type=float, default=50.0)
    parser.add_argument("--initial-crack-length-um", type=float, default=500.0)
    parser.add_argument("--save-snapshots")
    parser.add_argument("--no-plots", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise SystemExit("unsupported V5 adapter arguments: " + " ".join(unknown))
    if args.cavity_enabled == args.cavity_disabled:
        raise SystemExit("select exactly one of --cavity-enabled/--cavity-disabled")
    if args.test_lifecycle_fixture:
        return _fixture_main(args)
    return _physical_main(args)


if __name__ == "__main__":
    main()


__all__ = [
    "MODEL_ID", "FIXTURE_MODEL_ID", "PAIR_CHECKPOINT_SCHEMA", "main",
]
