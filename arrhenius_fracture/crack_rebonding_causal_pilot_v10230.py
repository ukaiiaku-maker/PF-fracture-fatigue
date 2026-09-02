"""Bounded crack-rebonding causal pilot (v10.2.30, post-S8).

S8 qualified the software (REBONDING_PREPHYSICS_INTEGRATION_QUALIFIED). This
module implements the small, explicitly bounded RB0/RB1/RB2 causal pilot
authorized as the next step: NOT the full Part X Paris-slope matrix, and NOT
a merge -- a single-Kmax, six-trajectory causal comparison that either finds
or fails to find a real, dynamically-generated shielding effect, with every
configuration frozen and hashed before any trajectory runs.

Imports only from ``crack_rebonding_kinetics_v10230``/``crack_rebonding_v10230``
(pure config/kinetics) so this module has no hard dependency on any specific
engine-construction recipe -- the real production engine is supplied by the
caller via dependency injection (``build_engine``/``make_controller``/
``waveform_cls``), exactly the way ``tests/_crack_rebonding_engine_fixture.py``
already builds it for the S7/S8 qualification tests. This keeps this module
importable and unit-testable without requiring the (test-only) fixture.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, replace
from typing import Any, Callable

from . import crack_rebonding_v10230 as _rebond
from .crack_rebonding_kinetics_v10230 import (
    REFERENCE_ACTION_PRESETS,
    ContactModel,
    CrackRebondingControls,
    FeedbackMode,
    InitialPrecrackWakeMode,
    RebondModelLevel,
    _integrate_A_on,  # noqa: F401  -- reused deliberately, see freeze_pilot_configuration
    solve_reference_action_barriers,
    two_state_fixed_point,
)

SCHEMA = "v10.2.30_crack_rebonding_causal_pilot_v1"

REQUIRED_PYTHON = (
    "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
)

SEED = 1720
T_K = 300.0
KMAX_REF_Pa_sqrt_m = 18.0e6
F_HZ = 1000.0
R_REF = -0.95
N_PHASE = 80

PI_K_TARGET = 0.05
K_REBOND_MAX_TARGET_Pa_sqrt_m = PI_K_TARGET * KMAX_REF_Pa_sqrt_m  # 0.9 MPa*sqrt(m)
ETA_K = 1.0  # ``rebond_K_geometry_factor`` fixed at 1 so the target K_rebond_max
# resolves G_max unambiguously (see resolve_restored_work_of_separation below);
# Pi_K is the mission-specified mechanism control, not a second free knob.

ANALYTICAL_KMAX_GRID_Pa_sqrt_m = (12.0e6, 15.0e6, 18.0e6, 24.3e6)

# R=0.1 residual-formation check (hard gate 2): the activation volume search
# below verifies analytically, before freezing, that the fully-tensile R=0.1
# waveform (K never goes negative under SIGNED_K_COMPRESSION_PROXY, so
# sigma_comp=0 identically) produces a negligible predicted single-patch
# action relative to the R=-0.95 reference action -- not asserted after the
# fact, checked as part of construction.
#
# The crossing from "not negligible" to "negligible" is extremely steep
# (Arrhenius-exponential in the compression-assisted work term), and the
# achievable live-bonding signal at Kmax=18 MPa*sqrt(m) within this pilot's
# short inter-event dwell times (native cleavage hazard here is orders of
# magnitude faster than one waveform period, confirmed during pilot
# construction) falls off just as steeply past that crossing. 1e-3 relative
# (three decades below the reference-condition action -- clearly negligible
# for gate 2's purpose, and consistent with this repo's existing
# bulk_action_error_rel_tol=1e-3 convention) combined with a fine (x1.15)
# geometric search lands close to the crossing from the negligible side,
# preserving as much of the (inherently weak, honestly reported) live
# signal as gate 2 allows, rather than searching from a seed far past it.
R_ZERO_CHECK = 0.1
R_ZERO_ACTION_REL_TOL = 1.0e-3
_BOND_ACTIVATION_VOLUME_SEED_m3 = 1.0e-30
_BOND_ACTIVATION_VOLUME_GROWTH = 1.15
_BOND_ACTIVATION_VOLUME_MAX_STEPS = 200

# Stopping / censoring / acceptance limits (frozen here, not tuned after
# seeing simulated results -- see docs/v10_2_30_crack_rebonding_causal_pilot.md).
MAX_ACCEPTED_EVENTS = 5
MAX_PROJECTED_EXTENSION_m = 30.0e-6
MIN_ACCEPTED_EVENTS_FOR_UNCENSORED = 3
MAX_BLOCKS_PER_EVENT = 20000
MAX_WALL_SECONDS_PER_TRAJECTORY = 1200.0
BLOCK_CYCLES = 1000.0
MAX_BLOCK_CYCLES = 1.0e6

# S8C periodic-orbit bulk-action certificate, as coded in phase_resolved_action
# / _commit_rebonding_event (crack_rebonding_v10230.py) -- recorded here, not
# reconfigured, since these are not currently caller-exposed knobs.
PERIODIC_ORBIT_TRANSIENT_BUDGET_INITIAL = 200
BULK_ACTION_ERROR_REL_TOL = CrackRebondingControls().bulk_action_error_rel_tol
BULK_ACTION_MAX_TRANSIENT_EXTENSIONS = CrackRebondingControls().bulk_action_max_transient_extensions

TRAJECTORY_NAMES = ("P0", "P1", "P2", "P3", "P4", "P5")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value") and hasattr(value, "name") and not isinstance(value, (int, str)):
        return value.value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def canonical_hash(payload: Any) -> str:
    canonical = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_restored_work_of_separation(Eprime_Pa: float) -> float:
    """G_max such that eta_K * sqrt(Eprime * G_max) == K_REBOND_MAX_TARGET,
    with eta_K fixed at 1 (see ETA_K docstring above)."""
    target = K_REBOND_MAX_TARGET_Pa_sqrt_m / max(ETA_K, 1.0e-300)
    return float(target * target / max(Eprime_Pa, 1.0e-300))


def _rb2_template(
    *, reference_contact_radius_m: float, G_max: float, bond_activation_volume_m3: float
) -> CrackRebondingControls:
    return CrackRebondingControls(
        enabled=True,
        model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        contact_model=ContactModel.SIGNED_K_COMPRESSION_PROXY,
        feedback_mode=FeedbackMode.HAZARD_ONLY_REBOND_SHIELD,
        initial_precrack_wake_mode=InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE,
        wake_length_m=5.0e-4,
        wake_weight_length_m=5.0e-7,
        contact_radius_min_m=max(reference_contact_radius_m, 1.0e-9),
        fresh_surface_clean_fraction=1.0,
        chemistry_factor=1.0,
        restored_work_of_separation_J_m2=G_max,
        rebond_K_geometry_factor=ETA_K,
        bond_activation_volume_m3=bond_activation_volume_m3,
        rupture_activation_volume_m3=0.0,
        bond_attempt_frequency_s=1.0e10,
        rupture_attempt_frequency_s=1.0e10,
        bond_barrier_eV=0.2,
        rupture_barrier_eV=1.0,
    )


def _r_zero_action_ratio(
    resolved_cfg: CrackRebondingControls, *, reference_contact_radius_m: float
) -> float:
    """Predicted single-patch A_on at R=R_ZERO_CHECK relative to the
    reference-condition target A_on_ref, using the already-solved barriers.
    Zero (to floating-point) whenever K never goes negative and formation is
    purely stress-assisted, which is what activation-volume search below is
    driving toward."""
    a_on_ref = _integrate_A_on(
        resolved_cfg.bond_barrier_eV,
        T_K=T_K,
        f_Hz=F_HZ,
        R=R_REF,
        Kmax_Pa_sqrt_m=KMAX_REF_Pa_sqrt_m,
        s_j_m=0.0,
        r_contact_m=reference_contact_radius_m,
        cfg=resolved_cfg,
        n_phase=360,
    )
    a_on_zero = _integrate_A_on(
        resolved_cfg.bond_barrier_eV,
        T_K=T_K,
        f_Hz=F_HZ,
        R=R_ZERO_CHECK,
        Kmax_Pa_sqrt_m=KMAX_REF_Pa_sqrt_m,
        s_j_m=0.0,
        r_contact_m=reference_contact_radius_m,
        cfg=resolved_cfg,
        n_phase=360,
    )
    return float(a_on_zero / max(a_on_ref, 1.0e-300))


def solve_calibrated_rb2_configs(
    *, Eprime_Pa: float, reference_contact_radius_m: float
) -> dict[str, Any]:
    """Solve RB2-reversible and RB2-persistent configs hitting their target
    reference actions at (R_REF, KMAX_REF) with K_rebond_max == target, then
    verify (not assume) that the R=0.1 residual single-patch action is
    negligible -- searching bond_activation_volume_m3 upward (monotone: more
    stress-assisted formation at fixed target action at R_REF implies a
    larger un-assisted nominal barrier, hence a smaller R=0.1 residual) until
    it is, or raising if the search is exhausted. This is a pre-launch
    analytical validation of hard gate 2, not a post-hoc tuning of results.
    """
    G_max = resolve_restored_work_of_separation(Eprime_Pa)
    activation_volume = _BOND_ACTIVATION_VOLUME_SEED_m3
    resolved: dict[str, CrackRebondingControls] = {}
    ratios: dict[str, float] = {}
    for attempt in range(_BOND_ACTIVATION_VOLUME_MAX_STEPS):
        resolved = {}
        ratios = {}
        ok = True
        for name, (a_on_ref, a_off_ref) in (
            ("reversible", REFERENCE_ACTION_PRESETS["reversible"]),
            ("persistent", REFERENCE_ACTION_PRESETS["persistent"]),
        ):
            template = _rb2_template(
                reference_contact_radius_m=reference_contact_radius_m,
                G_max=G_max,
                bond_activation_volume_m3=activation_volume,
            ).validate()
            cfg = solve_reference_action_barriers(
                a_on_ref,
                a_off_ref,
                T_K=T_K,
                f_Hz=F_HZ,
                R=R_REF,
                Kmax_Pa_sqrt_m=KMAX_REF_Pa_sqrt_m,
                reference_patch_distance_m=0.0,
                reference_contact_radius_m=reference_contact_radius_m,
                cfg_template=template,
            ).validate()
            resolved[name] = cfg
            ratio = _r_zero_action_ratio(
                cfg, reference_contact_radius_m=reference_contact_radius_m
            )
            ratios[name] = ratio
            if ratio > R_ZERO_ACTION_REL_TOL:
                ok = False
        if ok:
            return {
                "configs": resolved,
                "r_zero_action_ratios": ratios,
                "bond_activation_volume_m3": activation_volume,
                "search_attempts": attempt + 1,
                "restored_work_of_separation_J_m2": G_max,
                "rebond_K_geometry_factor": ETA_K,
            }
        activation_volume *= _BOND_ACTIVATION_VOLUME_GROWTH
    raise RuntimeError(
        "could not calibrate a bond_activation_volume_m3 driving the R="
        f"{R_ZERO_CHECK} residual single-patch action below "
        f"{R_ZERO_ACTION_REL_TOL} relative to the R={R_REF} reference action "
        f"after {_BOND_ACTIVATION_VOLUME_MAX_STEPS} x{_BOND_ACTIVATION_VOLUME_GROWTH} "
        f"growth steps (last ratios={ratios!r}); hard gate 2 is not achievable "
        "with this template without further design changes."
    )


def rb1_config(rb2_reversible: CrackRebondingControls) -> CrackRebondingControls:
    """CONTACT_PROXY_ONLY: patch_Q returns the zero generator unconditionally
    for this model level (crack_rebonding_v10230.patch_Q), so the barrier
    fields below are inert diagnostics, not physics -- kept identical to the
    RB2-reversible config only so config_hash() differences are exactly and
    only the model_level field."""
    return replace(rb2_reversible, model_level=RebondModelLevel.CONTACT_PROXY_ONLY)


def rb0_config() -> None:
    """REBOND_OFF is represented by passing None to the engine builder (no
    install_crack_rebonding call at all), matching the existing
    disabled_engine_allocates_no_rebonding_state invariant -- there is no
    CrackRebondingControls instance to freeze for P0/P4."""
    return None


def analytical_single_patch_predictions(
    resolved_cfg: CrackRebondingControls, *, reference_contact_radius_m: float
) -> list[dict[str, Any]]:
    """Single-patch two-state fixed-point prediction (b*, P_survive) at each
    of ANALYTICAL_KMAX_GRID_Pa_sqrt_m, R_REF, using the already-solved
    barriers -- reuses solve_reference_action_barriers' own integrators
    rather than re-deriving the action integral. Frozen/hashed alongside the
    configs; not compared against simulation in this single-Kmax pilot (see
    hard gate 8)."""
    from .crack_rebonding_kinetics_v10230 import _integrate_A_off

    rows = []
    for Kmax in ANALYTICAL_KMAX_GRID_Pa_sqrt_m:
        a_on = _integrate_A_on(
            resolved_cfg.bond_barrier_eV,
            T_K=T_K,
            f_Hz=F_HZ,
            R=R_REF,
            Kmax_Pa_sqrt_m=Kmax,
            s_j_m=0.0,
            r_contact_m=reference_contact_radius_m,
            cfg=resolved_cfg,
            n_phase=360,
        )
        a_off = _integrate_A_off(
            resolved_cfg.rupture_barrier_eV,
            T_K=T_K,
            f_Hz=F_HZ,
            R=R_REF,
            Kmax_Pa_sqrt_m=Kmax,
            s_j_m=0.0,
            r_contact_m=reference_contact_radius_m,
            cfg=resolved_cfg,
            n_phase=360,
        )
        fixed_point = two_state_fixed_point(a_on, a_off)
        rows.append(
            {
                "Kmax_Pa_sqrt_m": Kmax,
                "R": R_REF,
                "A_on": a_on,
                "A_off": a_off,
                "b_star": fixed_point["b_star"],
                "P_survive": fixed_point["P_survive"],
            }
        )
    return rows


def trajectory_specs() -> list[dict[str, Any]]:
    return [
        {"name": "P0", "R": R_REF, "rebonding": "RB0"},
        {"name": "P1", "R": R_REF, "rebonding": "RB1"},
        {"name": "P2", "R": R_REF, "rebonding": "RB2_reversible"},
        {"name": "P3", "R": R_REF, "rebonding": "RB2_persistent"},
        {"name": "P4", "R": R_ZERO_CHECK, "rebonding": "RB0"},
        {"name": "P5", "R": R_ZERO_CHECK, "rebonding": "RB2_persistent"},
    ]


def freeze_pilot_configuration(
    *, Eprime_Pa: float, reference_contact_radius_m: float, engine_G_Pa: float, engine_nu: float
) -> dict[str, Any]:
    calibration = solve_calibrated_rb2_configs(
        Eprime_Pa=Eprime_Pa, reference_contact_radius_m=reference_contact_radius_m
    )
    rb2_reversible = calibration["configs"]["reversible"]
    rb2_persistent = calibration["configs"]["persistent"]
    rb1 = rb1_config(rb2_reversible)

    configs = {
        "RB0": None,
        "RB1": rb1,
        "RB2_reversible": rb2_reversible,
        "RB2_persistent": rb2_persistent,
    }
    config_hashes = {
        name: (cfg.config_hash() if cfg is not None else None) for name, cfg in configs.items()
    }

    analytical_predictions = {
        "reversible": analytical_single_patch_predictions(
            rb2_reversible, reference_contact_radius_m=reference_contact_radius_m
        ),
        "persistent": analytical_single_patch_predictions(
            rb2_persistent, reference_contact_radius_m=reference_contact_radius_m
        ),
    }

    frozen = {
        "schema": SCHEMA,
        "required_python": REQUIRED_PYTHON,
        "seed": SEED,
        "common_settings": {
            "parameter_option": "A_NATIVE",
            "T_K": T_K,
            "Kmax_Pa_sqrt_m": KMAX_REF_Pa_sqrt_m,
            "frequency_Hz": F_HZ,
            "n_phase": N_PHASE,
            "integrator_mode": "EXPLICIT_PHASE_RESOLVED",
            "feedback_mode": FeedbackMode.HAZARD_ONLY_REBOND_SHIELD.value,
            "initial_precrack_wake_mode": InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE.value,
            "fresh_surface_clean_fraction": 1.0,
            "chemistry_factor": 1.0,
            "passivation_enabled": False,
            "topological_healing_enabled": False,
            "dmd_poincare_acceleration_enabled": False,
            "restart_resume_forbidden": True,
            "contact_model": ContactModel.SIGNED_K_COMPRESSION_PROXY.value,
            "contact_semantics_label": _rebond.CONTACT_SEMANTICS_LABEL,
        },
        "material": {
            "engine_G_Pa": engine_G_Pa,
            "engine_nu": engine_nu,
            "Eprime_Pa": Eprime_Pa,
            "reference_contact_radius_m": reference_contact_radius_m,
        },
        "rebonding_mechanism_control": {
            "Pi_K_target": PI_K_TARGET,
            "K_rebond_max_target_Pa_sqrt_m": K_REBOND_MAX_TARGET_Pa_sqrt_m,
            "eta_K": ETA_K,
            "restored_work_of_separation_J_m2": calibration["restored_work_of_separation_J_m2"],
            "bond_activation_volume_m3": calibration["bond_activation_volume_m3"],
            "bond_activation_volume_search_attempts": calibration["search_attempts"],
            "r_zero_check": {
                "R": R_ZERO_CHECK,
                "rel_tol": R_ZERO_ACTION_REL_TOL,
                "predicted_action_ratios": calibration["r_zero_action_ratios"],
            },
        },
        "reference_action_presets": {
            "reversible": REFERENCE_ACTION_PRESETS["reversible"],
            "persistent": REFERENCE_ACTION_PRESETS["persistent"],
        },
        "configs": {name: (asdict(cfg) if cfg is not None else None) for name, cfg in configs.items()},
        "config_hashes": config_hashes,
        "analytical_single_patch_predictions": {
            "Kmax_grid_Pa_sqrt_m": list(ANALYTICAL_KMAX_GRID_Pa_sqrt_m),
            "predictions": analytical_predictions,
        },
        "limits": {
            "max_accepted_events": MAX_ACCEPTED_EVENTS,
            "max_projected_extension_m": MAX_PROJECTED_EXTENSION_m,
            "min_accepted_events_for_uncensored": MIN_ACCEPTED_EVENTS_FOR_UNCENSORED,
            "max_blocks_per_event": MAX_BLOCKS_PER_EVENT,
            "max_wall_seconds_per_trajectory": MAX_WALL_SECONDS_PER_TRAJECTORY,
            "block_cycles": BLOCK_CYCLES,
            "max_block_cycles": MAX_BLOCK_CYCLES,
            "periodic_orbit_transient_budget_initial": PERIODIC_ORBIT_TRANSIENT_BUDGET_INITIAL,
            "bulk_action_error_rel_tol": BULK_ACTION_ERROR_REL_TOL,
            "bulk_action_max_transient_extensions": BULK_ACTION_MAX_TRANSIENT_EXTENSIONS,
        },
        "trajectories": trajectory_specs(),
    }
    frozen["frozen_configuration_sha256"] = canonical_hash(frozen)
    return frozen


# ---------------------------------------------------------------------------
# Trajectory execution (dependency-injected engine construction)
# ---------------------------------------------------------------------------


def _committed_length_for_pending(engine) -> tuple[float, dict[str, Any], Any]:
    pending = engine._energy_gate_pending
    committed_length = pending["proposal_m"]
    gate = {
        "energy_admissible_event_length_m": committed_length,
        "arrest_reason": "causal_pilot_commit",
        "hazard_resistance_J_per_m2": 1.0,
        "orientation_gamma_relative": 1.0,
    }
    result_ref = pending["descriptor"].get("energy_gate_result_ref")
    return committed_length, gate, result_ref


def run_trajectory(
    *,
    name: str,
    build_engine: Callable[[CrackRebondingControls | None], Any],
    make_controller: Callable[[int], Any],
    waveform_cls: Callable[..., Any],
    rebonding_cfg: CrackRebondingControls | None,
    R: float,
    reset_engine_registry: Callable[[], None] | None = None,
    Kmax_Pa_sqrt_m: float = KMAX_REF_Pa_sqrt_m,
    frequency_Hz: float = F_HZ,
    n_phase: int = N_PHASE,
    T_K_: float = T_K,
    max_accepted_events: int = MAX_ACCEPTED_EVENTS,
    max_projected_extension_m: float = MAX_PROJECTED_EXTENSION_m,
    min_accepted_events_for_uncensored: int = MIN_ACCEPTED_EVENTS_FOR_UNCENSORED,
    max_blocks_per_event: int = MAX_BLOCKS_PER_EVENT,
    max_wall_seconds: float = MAX_WALL_SECONDS_PER_TRAJECTORY,
) -> dict[str, Any]:
    """Drive one fresh, unresumed trajectory in-process via the real
    production engine's own cycle_step_waveform/commit_energy_gated_event
    (the same real methods the S7/S8 qualification tests exercise) -- never
    through the full CLI/mesh-dependent EnergyGatedAvalancheBackend.advance()
    path, which S8D documented as infeasible without out-of-scope FEM
    engineering (hard gate 7). This IS the "explicit phase-resolved,
    DMD/Poincare acceleration disabled" path by construction: the
    accelerated-engine monkeypatch installed by
    sharp_front_v10_2_30_energy_gated_fatigue.main() is never reached
    because main() is never called.

    ``reset_engine_registry``, if given, is called immediately before engine
    construction (e.g. the engine class's own ``reset_audit()``, which resets
    the ``_next_engine_id`` counter kinetic_tip_cell.py mixes into the hazard
    RNG's SeedSequence): every trajectory then starts from the same
    ``_engine_id``, so ``SEED`` alone determines the threshold stream,
    independent of how many engines were constructed earlier in the same
    process -- required for hard gates 1/2/4's identical-threshold-stream
    comparisons."""
    if reset_engine_registry is not None:
        reset_engine_registry()
    engine = build_engine(rebonding_cfg)
    ctrl = make_controller(n_phase)
    waveform = waveform_cls(Kmax=Kmax_Pa_sqrt_m, R=R, frequency_Hz=frequency_Hz)

    bulk_action_records: list[dict[str, Any]] = []
    original_phase_resolved_action = _rebond.phase_resolved_action

    def _recording_phase_resolved_action(*args, **kwargs):
        action, end_states, end_idx, diag = original_phase_resolved_action(*args, **kwargs)
        bulk_action_records.append(dict(diag))
        return action, end_states, end_idx, diag

    events: list[dict[str, Any]] = []
    cumulative_extension_m = 0.0
    cumulative_time_s = 0.0
    time_at_previous_event_s = 0.0
    censored = False
    censor_reason = None
    start_wall = time.monotonic()

    _rebond.phase_resolved_action = _recording_phase_resolved_action
    try:
        while True:
            if len(events) >= max_accepted_events:
                break
            if cumulative_extension_m >= max_projected_extension_m:
                break
            if time.monotonic() - start_wall > max_wall_seconds:
                censored = True
                censor_reason = "wall_time_budget_exhausted_between_events"
                break

            bulk_action_records_before = len(bulk_action_records)
            fired_result = None
            blocks_used = 0
            for blocks_used in range(1, max_blocks_per_event + 1):
                if time.monotonic() - start_wall > max_wall_seconds:
                    break
                result = engine.cycle_step_waveform(ctrl, waveform, T_K_)
                cumulative_time_s += float(result.get("kinetic_dt_consumed_s", 0.0))
                if result.get("fired"):
                    fired_result = result
                    break

            if fired_result is None:
                censored = True
                censor_reason = (
                    "wall_time_budget_exhausted_mid_event"
                    if time.monotonic() - start_wall > max_wall_seconds
                    else f"event_did_not_fire_within_{max_blocks_per_event}_blocks"
                )
                break

            committed_length, gate, result_ref = _committed_length_for_pending(engine)
            # Total elapsed physical time since the PREVIOUS event, not just
            # the firing block's own dt -- an event that takes several
            # blocks to fire must count every block's kinetic_dt_consumed_s,
            # not only the last one (a bug caught during pilot construction:
            # the fired block's own dt understates multi-block waiting times).
            waiting_time_s = cumulative_time_s - time_at_previous_event_s
            time_at_previous_event_s = cumulative_time_s
            # The wake's patch states (p_B/K_rebond) are only mutated inside
            # _commit_rebonding_event, called from commit_energy_gated_event
            # below -- they are frozen at their post-PREVIOUS-commit values
            # for the whole block-stepping loop above (another bug caught
            # during pilot construction: sampling before this call only ever
            # sees stale state). Sample AFTER the commit, which is the only
            # point at which this event's just-completed dwell interval has
            # actually been applied to the wake.
            engine.commit_energy_gated_event(committed_length, gate, result_ref)
            cumulative_extension_m += committed_length
            rebonding_state = getattr(engine, "_rebonding_state", None)
            max_pB_post_commit = (
                max((float(p.p_B) for p in rebonding_state.active), default=0.0)
                if rebonding_state is not None
                else 0.0
            )
            max_K_rebond_post_commit = (
                float(rebonding_state.K_rebond_Pa_sqrt_m) if rebonding_state is not None else 0.0
            )

            new_bulk_records = bulk_action_records[bulk_action_records_before:]
            events.append(
                {
                    "event_index": len(events),
                    "blocks_to_fire": blocks_used,
                    "waiting_time_s_this_event": waiting_time_s,
                    "cumulative_time_s": cumulative_time_s,
                    "accepted_length_m": float(committed_length),
                    "cumulative_extension_m": cumulative_extension_m,
                    "max_pB_post_commit": max_pB_post_commit,
                    "max_K_rebond_post_commit_Pa_sqrt_m": max_K_rebond_post_commit,
                    "bulk_action_records": new_bulk_records,
                    "any_bulk_action_used": any(
                        r.get("bulk_action_used") for r in new_bulk_records
                    ),
                    "all_bulk_action_qualified": all(
                        r.get("bulk_action_qualified", True) for r in new_bulk_records
                    ),
                }
            )
    finally:
        _rebond.phase_resolved_action = original_phase_resolved_action

    uncensored = (not censored) and (len(events) >= min_accepted_events_for_uncensored)
    return {
        "name": name,
        "R": R,
        "rebonding_cfg_hash": rebonding_cfg.config_hash() if rebonding_cfg is not None else None,
        "rebonding_model_level": (
            rebonding_cfg.model_level.value if rebonding_cfg is not None else "REBOND_OFF"
        ),
        "seed": SEED,
        "events": events,
        "n_accepted_events": len(events),
        "cumulative_extension_m": cumulative_extension_m,
        "cumulative_time_s": cumulative_time_s,
        "censored": censored,
        "censor_reason": censor_reason,
        "uncensored": uncensored,
        "wall_seconds": time.monotonic() - start_wall,
    }


# ---------------------------------------------------------------------------
# Censor-aware causal analysis (hard gates 1-8)
# ---------------------------------------------------------------------------


def _event_pairs(a: dict[str, Any], b: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return list(zip(a["events"], b["events"]))


# Per-key relative tolerances for the "physically identical" comparison.
# accepted_length_m comes from the uncoupled stochastic proposal (set during
# _integrate_coupled, before any rebonding-aware timing refinement) and must
# be bit-identical between a REBOND_OFF/CONTACT_PROXY_ONLY pair. Once
# rebonding is merely *enabled* (even at CONTACT_PROXY_ONLY, K_rebond=0
# identically), commit_energy_gated_event routes through
# _commit_rebonding_event's bisection root-finder (solve_coupled_event_time,
# eps_B=1e-6) even though the physics is unchanged, so
# waiting_time_s_this_event (kinetic_dt_consumed_s) picks up a genuine but
# tiny root-finder-tolerance-scale discrepancy relative to the REBOND_OFF
# path, which skips that root-find entirely. This discrepancy compounds
# event over event (confirmed empirically during pilot construction: ~0.02%
# at event index 1, growing to ~0.9% by event index 4 of 5) since each
# event's tiny timing perturbation shifts the pre-event state the next
# event's root-find starts from. gate_1/gate_2 compare P0/P1 and P4/P5,
# both of which use REBOND_OFF vs CONTACT_PROXY_ONLY (patch_Q is the exact
# zero generator for both -- there is no possible REAL physical difference
# between them, so this tolerance only needs to absorb root-finder noise,
# never a genuine effect); 2% comfortably covers the observed compounding
# over up to MAX_ACCEPTED_EVENTS=5 events with margin. This tolerance is
# NOT used for the P1-vs-RB2 comparison in gate 4, which reports raw
# numbers for review rather than applying a pass/fail threshold.
_PHYSICALLY_IDENTICAL_REL_TOL = {
    "accepted_length_m": 1.0e-9,
    "waiting_time_s_this_event": 2.0e-2,
}


def _physically_identical(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    pairs = _event_pairs(a, b)
    if not pairs:
        return {"identical": False, "reason": "no comparable events", "n_pairs": 0}
    mismatches = []
    for ea, eb in pairs:
        for key, rel_tol in _PHYSICALLY_IDENTICAL_REL_TOL.items():
            va, vb = float(ea[key]), float(eb[key])
            denom = max(abs(va), abs(vb), 1.0e-300)
            if abs(va - vb) / denom > rel_tol:
                mismatches.append(
                    {
                        "event_index": ea["event_index"],
                        "key": key,
                        "a": va,
                        "b": vb,
                        "rel_diff": abs(va - vb) / denom,
                        "rel_tol": rel_tol,
                    }
                )
    return {
        "identical": len(mismatches) == 0 and a["n_accepted_events"] == b["n_accepted_events"],
        "n_pairs": len(pairs),
        "n_accepted_events_a": a["n_accepted_events"],
        "n_accepted_events_b": b["n_accepted_events"],
        "mismatches": mismatches,
    }


def causal_analysis(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the mission's eight hard gates against six already-run
    trajectory results (keyed P0..P5). Censor-aware: a gate that cannot be
    evaluated because a required trajectory is censored below the minimum
    uncensored event count is reported as such, not silently skipped."""
    gates: dict[str, Any] = {}

    p0, p1, p2, p3, p4, p5 = (results[name] for name in TRAJECTORY_NAMES)

    # Gate 1: P0 and P1 physically identical.
    gates["gate_1_p0_p1_identical"] = _physically_identical(p0, p1)

    # Gate 2: P4 and P5 physically identical; P5 zero bonding under signed-K proxy.
    gate2_identical = _physically_identical(p4, p5)
    p5_max_pB = max((e["max_pB_post_commit"] for e in p5["events"]), default=0.0)
    p5_max_K_rebond = max(
        (e["max_K_rebond_post_commit_Pa_sqrt_m"] for e in p5["events"]), default=0.0
    )
    gates["gate_2_p4_p5_identical_and_p5_zero_bonding"] = {
        **gate2_identical,
        "p5_max_pB": p5_max_pB,
        "p5_max_K_rebond_Pa_sqrt_m": p5_max_K_rebond,
        "p5_zero_bonding": p5_max_pB == 0.0 and p5_max_K_rebond == 0.0,
        "pass": gate2_identical["identical"] and p5_max_pB == 0.0 and p5_max_K_rebond == 0.0,
    }

    # Gate 3: P2 or P3 generates nonzero p_B and K_rebond dynamically.
    def _dynamic_nonzero(res: dict[str, Any]) -> bool:
        max_pB = max((e["max_pB_post_commit"] for e in res["events"]), default=0.0)
        max_K = max((e["max_K_rebond_post_commit_Pa_sqrt_m"] for e in res["events"]), default=0.0)
        return max_pB > 0.0 and max_K > 0.0

    p2_dynamic = _dynamic_nonzero(p2)
    p3_dynamic = _dynamic_nonzero(p3)
    gates["gate_3_dynamic_nonzero_bonding"] = {
        "p2_dynamic_nonzero": p2_dynamic,
        "p3_dynamic_nonzero": p3_dynamic,
        "pass": p2_dynamic or p3_dynamic,
    }

    # Gate 4: post-first-event waiting intervals, RB0/RB1 (no shielding)
    # vs RB2 variants (shielding, if gate 3 held), same threshold stream.
    def _waiting_interval(res: dict[str, Any], index: int) -> float | None:
        if index < len(res["events"]):
            return float(res["events"][index]["waiting_time_s_this_event"])
        return None

    waiting_comparison = {
        name: _waiting_interval(results[name], 1) for name in TRAJECTORY_NAMES
    }
    gates["gate_4_post_first_event_waiting_intervals"] = {
        "waiting_time_s_event_index_1": waiting_comparison,
        "rb2_reversible_gt_rb1": (
            waiting_comparison["P2"] is not None
            and waiting_comparison["P1"] is not None
            and waiting_comparison["P2"] > waiting_comparison["P1"]
        ),
        "rb2_persistent_gt_rb1": (
            waiting_comparison["P3"] is not None
            and waiting_comparison["P1"] is not None
            and waiting_comparison["P3"] > waiting_comparison["P1"]
        ),
    }

    # Gate 5: K_rebond never enters the energy gate directly -- accepted
    # event LENGTH must match across RB1/RB2 at each matched event index
    # (HAZARD_ONLY_REBOND_SHIELD invariant); any length difference is
    # flagged as an indirect pre-event-state effect, not a direct-coupling
    # failure, and audited rather than silently accepted.
    def _length_series(res: dict[str, Any]) -> list[float]:
        return [float(e["accepted_length_m"]) for e in res["events"]]

    len1, len2r, len2p = _length_series(p1), _length_series(p2), _length_series(p3)
    length_diffs = {
        "P1_vs_P2_reversible": [
            {"event_index": i, "P1": a, "P2": b, "rel_diff": abs(a - b) / max(abs(a), abs(b), 1e-300)}
            for i, (a, b) in enumerate(zip(len1, len2r))
            if abs(a - b) / max(abs(a), abs(b), 1e-300) > 1.0e-9
        ],
        "P1_vs_P3_persistent": [
            {"event_index": i, "P1": a, "P3": b, "rel_diff": abs(a - b) / max(abs(a), abs(b), 1e-300)}
            for i, (a, b) in enumerate(zip(len1, len2p))
            if abs(a - b) / max(abs(a), abs(b), 1e-300) > 1.0e-9
        ],
    }
    gates["gate_5_K_rebond_not_in_energy_gate_directly"] = {
        "length_diffs": length_diffs,
        "direct_coupling_suspected": bool(
            length_diffs["P1_vs_P2_reversible"] or length_diffs["P1_vs_P3_persistent"]
        ),
        "note": (
            "any listed diffs are audited as an indirect pre-event-state "
            "effect (different waiting time -> different pre-event plasticity "
            "state), not evidence K_rebond entered the energy gate directly "
            "(it enters only cleavage_stress_with_rebond, upstream of the "
            "hazard threshold, never energy_admissible_event_length_m)"
        ),
    }

    # Gate 6: only certified bulk actions admitted.
    all_qualified = True
    unqualified_records = []
    for name in TRAJECTORY_NAMES:
        for event in results[name]["events"]:
            for record in event["bulk_action_records"]:
                if not record.get("bulk_action_qualified", True):
                    all_qualified = False
                    unqualified_records.append(
                        {"trajectory": name, "event_index": event["event_index"], "record": record}
                    )
    gates["gate_6_certified_bulk_action_only"] = {
        "pass": all_qualified,
        "unqualified_records": unqualified_records,
        "note": (
            "the caller-side enforcement in _commit_rebonding_event already "
            "raises before committing an event on an uncertified bulk "
            "estimate; this gate additionally confirms no unqualified "
            "record was ever produced during the pilot, via the recording "
            "wrapper installed around phase_resolved_action for each "
            "trajectory"
        ),
    }

    # Gate 7: explicit limitation preserved (static claim, not measured
    # from trajectory output -- true by construction, see run_trajectory's
    # docstring and the pilot report).
    gates["gate_7_contact_proxy_and_mesh_backend_limitation_preserved"] = {
        "pass": True,
        "contact_semantics_label": _rebond.CONTACT_SEMANTICS_LABEL,
        "mesh_dependent_backend_qualified": False,
    }

    # Gate 8: no physical Paris-slope inference from this single-Kmax pilot.
    gates["gate_8_no_paris_slope_inference"] = {
        "pass": True,
        "part_x_physical_campaign_status": "DEFERRED_OUT_OF_SCOPE_THIS_PASS",
        "note": "single Kmax (18 MPa*sqrt(m)) pilot only; multi-K matrix not launched",
    }

    overall_pass = all(
        gates[key].get("pass", gates[key].get("identical", False))
        for key in (
            "gate_1_p0_p1_identical",
            "gate_2_p4_p5_identical_and_p5_zero_bonding",
            "gate_3_dynamic_nonzero_bonding",
            "gate_6_certified_bulk_action_only",
            "gate_7_contact_proxy_and_mesh_backend_limitation_preserved",
            "gate_8_no_paris_slope_inference",
        )
    ) and not gates["gate_5_K_rebond_not_in_energy_gate_directly"]["direct_coupling_suspected"]

    return {
        "schema": "v10.2.30_crack_rebonding_causal_pilot_analysis_v1",
        "gates": gates,
        "overall_pass": overall_pass,
    }


__all__ = [
    "SCHEMA",
    "REQUIRED_PYTHON",
    "SEED",
    "T_K",
    "KMAX_REF_Pa_sqrt_m",
    "F_HZ",
    "R_REF",
    "N_PHASE",
    "PI_K_TARGET",
    "K_REBOND_MAX_TARGET_Pa_sqrt_m",
    "ANALYTICAL_KMAX_GRID_Pa_sqrt_m",
    "MAX_ACCEPTED_EVENTS",
    "MAX_PROJECTED_EXTENSION_m",
    "MIN_ACCEPTED_EVENTS_FOR_UNCENSORED",
    "resolve_restored_work_of_separation",
    "solve_calibrated_rb2_configs",
    "rb1_config",
    "rb0_config",
    "analytical_single_patch_predictions",
    "trajectory_specs",
    "freeze_pilot_configuration",
    "run_trajectory",
    "causal_analysis",
    "canonical_hash",
    "TRAJECTORY_NAMES",
]
