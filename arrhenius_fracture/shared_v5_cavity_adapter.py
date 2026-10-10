"""Paired physical V5 cavity control using the native coupled scheduler.

The default path evolves enabled and suppressed arms from matching admissible
states under the same nonzero physical exposure. The former extreme-reservoir
lifecycle demonstration is retained only behind explicit test authorization.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
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
from .unified_fracture_material_v5 import material_bundle
from .voiding_production_v5 import (
    build_production_void_state, crack_tip_tensor, deterministic_trajectory,
    directional_sharp_front_rates, observables, sharp_front_load_provider,
)

MODEL_ID = "shared.native_v5_physical_paired_cavity/2"
FIXTURE_MODEL_ID = "shared.test_only_v5_extreme_reservoir_lifecycle_fixture/1"
PAIR_CHECKPOINT_SCHEMA = "shared-v5-physical-pair-checkpoint/1"
DEFAULT_SITES_M = ((6.0e-4, 6.0e-5), (7.5e-4, -6.0e-5))
DEFAULT_PARTITION = (0.50, 0.10, 0.40)


def _finite(value):
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


def _build_pair(args):
    bundle = material_bundle(args.material_class)
    common = dict(
        bundle=bundle, temperature_K=float(args.temperature_K),
        vacancy_energy_variant=args.vacancy_energy_variant,
        void_site_centers_m=DEFAULT_SITES_M, stochastic=True, seed=int(args.seed),
        direct_interaction_birth_enabled=True,
        cavity_surface_loop_emission_enabled=True,
        interaction_population_partition=DEFAULT_PARTITION,
        bulk_initial_density_m2=float(bundle.fracture_row["rho_forest_floor_m2"]),
        initial_plastic_strain_gp=0.0,
    )
    disabled, _ = build_production_void_state(enabled=False, **common)
    enabled, _ = build_production_void_state(enabled=True, **common)
    return disabled, enabled, _common_start_audit(disabled, enabled)


def _write_pair_checkpoint(states, root: Path, metadata: dict) -> Path:
    directory = Path(os.environ.get(
        "V10230_HIGH_CYCLE_CHECKPOINT_DIR", root / "checkpoint"
    )).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for arm, state in states.items():
        path = directory / f"accepted_v5_{arm}.json"
        write_checkpoint(state, path, compression="gzip")
        paths[arm] = str(path)
    payload = {
        "schema": PAIR_CHECKPOINT_SCHEMA,
        "arms": paths,
        "fingerprints": {
            arm: complete_accepted_state_fingerprint(state)
            for arm, state in states.items()
        },
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
    if payload.get("schema") != PAIR_CHECKPOINT_SCHEMA:
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


def _physical_main(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.restart_checkpoint:
        states, ancestry = _restore_pair(args.restart_checkpoint)
        common_start = ancestry.get("common_start")
        physical_time = float(ancestry["physical_time_s"])
        interval_index = int(ancestry["interval_count"])
    else:
        disabled, enabled, common_start = _build_pair(args)
        if not all(common_start.values()):
            raise RuntimeError("V5 enabled/suppressed common start failed")
        states = {"disabled": disabled, "enabled": enabled}
        ancestry = None
        physical_time = 0.0
        interval_index = 0

    duration = float(args.duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise SystemExit("--duration-s must be finite and positive")
    intervals = int(args.steps) + int(bool(args.continue_one))
    if intervals < 1:
        raise SystemExit("--steps must be at least one")
    observer = _directional_provider(float(args.temperature_K))
    history = {"disabled": [], "enabled": []}
    for _ in range(intervals):
        interval_index += 1
        for arm in ("disabled", "enabled"):
            before = complete_accepted_state_fingerprint(states[arm])
            state, scheduler = advance_event_aware_coupled_interval(
                states[arm], duration,
                temperature_K=float(args.temperature_K),
                directional_observation_provider=observer,
            )
            states[arm] = state
            history[arm].append(_finite({
                "interval": interval_index,
                "physical_time_start_s": physical_time,
                "physical_time_end_s": physical_time + duration,
                "duration_s": duration,
                "input_fingerprint": before,
                "output_fingerprint": complete_accepted_state_fingerprint(state),
                "scheduler": scheduler,
                "observables": observables(state, f"physical_interval_{interval_index}"),
            }))
        physical_time += duration

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
                "loading_history": "hold_at_native_opening",
                "interval_duration_s": duration,
                "test_only_extreme_reservoir": False,
                "full_field_plasticity": True,
                "earliest_event_and_remainder_scheduler": True,
            },
        },
    )
    restored, checkpoint_payload = _restore_pair(checkpoint)
    round_trip = all(
        complete_accepted_state_fingerprint(restored[arm])
        == complete_accepted_state_fingerprint(states[arm])
        for arm in states
    )
    selected_arm = "enabled" if args.cavity_enabled else "disabled"
    payload = _finite({
        "schema": MODEL_ID,
        "production_eligible": True,
        "selected_arm": selected_arm,
        "both_arms_evolved": True,
        "physical_exposure_s_per_arm_this_call": duration * intervals,
        "physical_time_s": physical_time,
        "interval_count": interval_index,
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
    parser.add_argument("--temperature-K", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=3621)
    parser.add_argument("--duration-s", type=float, default=1.0e-5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--vacancy-energy-variant", default="reference")
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
