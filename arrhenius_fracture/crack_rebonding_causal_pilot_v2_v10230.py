"""Corrected crack-rebonding causal pilot (v10.2.30, V2).

Supersedes ``crack_rebonding_causal_pilot_v10230.py`` (v1, preserved as a
diagnostic artifact -- see docs/v10_2_30_crack_rebonding_causal_pilot_v2.md
for the four defects it had that this module fixes):

1. Uses the real, provenance-recovered A_NATIVE production engine
   (``a_native_engine_v10230.build_a_native_engine``), not a DBTT
   test-fixture candidate.
2. Relies on the shared kinetics/engine-layer contact gate (exact
   ``K_signed < 0``) and strict RB0/RB1/RB2-zero-cohesion event-timing
   parity fixes (V2-B), not a 2%-tolerance workaround.
3. Adds matching zero-cohesion RB2 controls (``restored_work_of_separation_
   J_m2=0.0``, otherwise identical to their finite-cohesion twin) so the
   cohesive causal effect is isolated as
   ``Delta t = t_finite_cohesion - t_zero_cohesion`` at matched event
   indices, rather than inferred from RB2 minus RB1.
4. Records interval-resolved compression evidence (creation/next-event
   phase, elapsed time/cycles, negative-K contact duration, whether a
   complete negative excursion occurred) using the same external,
   engine-independent K(t) evaluation as the V2-C protocol preflight.

Trajectory set (mission Section 10):

    C0   R=-0.95   RB0
    C1   R=-0.95   RB1 (CONTACT_PROXY_ONLY)
    C2R  R=-0.95   RB2 reversible, zero cohesion
    C3R  R=-0.95   RB2 reversible, finite cohesion
    C2P  R=-0.95   RB2 persistent, zero cohesion
    C3P  R=-0.95   RB2 persistent, finite cohesion
    C4   R=0.1     RB0
    C5   R=0.1     RB2 persistent, finite cohesion
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from typing import Any, Callable

import numpy as np

from . import crack_rebonding_v10230 as _rebond
from .crack_rebonding_kinetics_v10230 import (
    REFERENCE_ACTION_PRESETS,
    ContactModel,
    CrackRebondingControls,
    FeedbackMode,
    InitialPrecrackWakeMode,
    RebondModelLevel,
    _integrate_A_on,
    bond_formation_rate,
    contact_pressure,
    cooperative_hazard,
    solve_reference_action_barriers,
    two_state_fixed_point,
)

SCHEMA = "v10.2.30_crack_rebonding_causal_pilot_v2"

REQUIRED_PYTHON = (
    "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
)

SEED = 1720
T_K = 300.0
KMAX_Pa_sqrt_m = 18.0e6
F_HZ = 1000.0
R_REF = -0.95
R_POSITIVE = 0.1
N_PHASE = 80
MPZ_N_BINS = 80

PI_K_TARGET = 0.05
K_REBOND_MAX_TARGET_Pa_sqrt_m = PI_K_TARGET * KMAX_Pa_sqrt_m
ETA_K = 1.0
_BOND_ACTIVATION_VOLUME_SEED_m3 = 1.0e-30

MAX_ACCEPTED_EVENTS = 8
MAX_PROJECTED_EXTENSION_m = 30.0e-6
MIN_ACCEPTED_EVENTS_FOR_UNCENSORED = 3
MAX_BLOCKS_PER_EVENT = 20000
MAX_WALL_SECONDS_PER_TRAJECTORY = 1800.0
BLOCK_CYCLES = 1000.0
MAX_BLOCK_CYCLES = 1.0e6

EXPANSION_THRESHOLD_LOG10_DECADE = 0.05

TRAJECTORY_NAMES = ("C0", "C1", "C2R", "C3R", "C2P", "C3P", "C4", "C5")


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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def resolve_restored_work_of_separation(Eprime_Pa: float) -> float:
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


def _r_positive_action_ratio(
    resolved_cfg: CrackRebondingControls, *, reference_contact_radius_m: float
) -> float:
    """Predicted single-patch A_on at R_POSITIVE relative to the
    reference-condition A_on_ref. Now provably EXACTLY 0.0 (not merely
    negligible): under the exact K_signed<0 contact gate, R_POSITIVE keeps
    K_signed >= 0 at every phase (Kmin = R_POSITIVE*Kmax > 0), so k_cb is
    forced to exactly 0.0 at every phase point -- hard gate 2's "exactly
    zero bond formation at R=0.1" is a closed-form guarantee of the shared
    kinetics fix, not a calibration target."""
    a_on_ref = _integrate_A_on(
        resolved_cfg.bond_barrier_eV, T_K=T_K, f_Hz=F_HZ, R=R_REF,
        Kmax_Pa_sqrt_m=KMAX_Pa_sqrt_m, s_j_m=0.0,
        r_contact_m=reference_contact_radius_m, cfg=resolved_cfg, n_phase=360,
    )
    a_on_pos = _integrate_A_on(
        resolved_cfg.bond_barrier_eV, T_K=T_K, f_Hz=F_HZ, R=R_POSITIVE,
        Kmax_Pa_sqrt_m=KMAX_Pa_sqrt_m, s_j_m=0.0,
        r_contact_m=reference_contact_radius_m, cfg=resolved_cfg, n_phase=360,
    )
    return float(a_on_pos / max(a_on_ref, 1.0e-300))


def solve_calibrated_rb2_configs(
    *, Eprime_Pa: float, reference_contact_radius_m: float
) -> dict[str, Any]:
    """Solve RB2-reversible and RB2-persistent FINITE-cohesion configs
    hitting their target reference actions at (R_REF, KMAX), then verify the
    R_POSITIVE residual action is exactly 0.0 (closed-form under the exact
    contact gate -- see _r_positive_action_ratio)."""
    G_max = resolve_restored_work_of_separation(Eprime_Pa)
    activation_volume = _BOND_ACTIVATION_VOLUME_SEED_m3

    resolved: dict[str, CrackRebondingControls] = {}
    ratios: dict[str, float] = {}
    for name, (a_on_ref, a_off_ref) in (
        ("reversible", REFERENCE_ACTION_PRESETS["reversible"]),
        ("persistent", REFERENCE_ACTION_PRESETS["persistent"]),
    ):
        template = _rb2_template(
            reference_contact_radius_m=reference_contact_radius_m,
            G_max=G_max, bond_activation_volume_m3=activation_volume,
        ).validate()
        cfg = solve_reference_action_barriers(
            a_on_ref, a_off_ref, T_K=T_K, f_Hz=F_HZ, R=R_REF, Kmax_Pa_sqrt_m=KMAX_Pa_sqrt_m,
            reference_patch_distance_m=0.0, reference_contact_radius_m=reference_contact_radius_m,
            cfg_template=template,
        ).validate()
        ratio = _r_positive_action_ratio(cfg, reference_contact_radius_m=reference_contact_radius_m)
        if ratio != 0.0:
            raise RuntimeError(
                f"R_POSITIVE residual action ratio for {name!r} is {ratio!r}, not exactly "
                "0.0 -- the exact contact gate should make this a closed-form guarantee; "
                "something upstream is not gating on K_signed<0 correctly"
            )
        resolved[name] = cfg
        ratios[name] = ratio

    return {
        "configs_finite_cohesion": resolved,
        "r_positive_action_ratios": ratios,
        "bond_activation_volume_m3": activation_volume,
        "restored_work_of_separation_J_m2": G_max,
        "rebond_K_geometry_factor": ETA_K,
    }


def rb1_config(rb2_reversible_finite: CrackRebondingControls) -> CrackRebondingControls:
    """CONTACT_PROXY_ONLY: patch_Q returns the exact zero generator, and
    the shared engine-layer fix routes it through the identical wake-ledger
    -only commit path as cfg=None -- kept field-identical to the RB2
    reversible finite-cohesion config only so config_hash() differences are
    exactly and only the model_level field."""
    return replace(rb2_reversible_finite, model_level=RebondModelLevel.CONTACT_PROXY_ONLY)


def zero_cohesion_twin(finite_cfg: CrackRebondingControls) -> CrackRebondingControls:
    """Identical to finite_cfg in every field except restored_work_of_
    separation_J_m2=0.0 (hence K_rebond_max=0 regardless of state) -- the
    matching zero-cohesion control mission Section 8 requires."""
    return replace(finite_cfg, restored_work_of_separation_J_m2=0.0)


def trajectory_specs() -> list[dict[str, Any]]:
    return [
        {"name": "C0", "R": R_REF, "rebonding": "RB0"},
        {"name": "C1", "R": R_REF, "rebonding": "RB1"},
        {"name": "C2R", "R": R_REF, "rebonding": "RB2_reversible_zero"},
        {"name": "C3R", "R": R_REF, "rebonding": "RB2_reversible_finite"},
        {"name": "C2P", "R": R_REF, "rebonding": "RB2_persistent_zero"},
        {"name": "C3P", "R": R_REF, "rebonding": "RB2_persistent_finite"},
        {"name": "C4", "R": R_POSITIVE, "rebonding": "RB0"},
        {"name": "C5", "R": R_POSITIVE, "rebonding": "RB2_persistent_finite"},
    ]


def analytical_single_patch_predictions(
    resolved_cfg: CrackRebondingControls, *, reference_contact_radius_m: float
) -> dict[str, Any]:
    from .crack_rebonding_kinetics_v10230 import _integrate_A_off

    a_on = _integrate_A_on(
        resolved_cfg.bond_barrier_eV, T_K=T_K, f_Hz=F_HZ, R=R_REF, Kmax_Pa_sqrt_m=KMAX_Pa_sqrt_m,
        s_j_m=0.0, r_contact_m=reference_contact_radius_m, cfg=resolved_cfg, n_phase=360,
    )
    a_off = _integrate_A_off(
        resolved_cfg.rupture_barrier_eV, T_K=T_K, f_Hz=F_HZ, R=R_REF, Kmax_Pa_sqrt_m=KMAX_Pa_sqrt_m,
        s_j_m=0.0, r_contact_m=reference_contact_radius_m, cfg=resolved_cfg, n_phase=360,
    )
    fixed_point = two_state_fixed_point(a_on, a_off)
    return {"A_on": a_on, "A_off": a_off, **fixed_point}


def barrier_floor_saturation_diagnostics(
    resolved_cfg: CrackRebondingControls, *, reference_contact_radius_m: float
) -> dict[str, Any]:
    """Barrier-floor/cooperative-saturation diagnostics at the deepest
    compression point of the reference protocol (K_signed = R_REF*Kmax,
    s_j_m=0), computed directly from the frozen config -- no engine replay
    needed, since this is a pure function of (cfg, K_min, r_contact_m).

    Shared verbatim between a zero/finite-cohesion twin pair: formation
    kinetics (bond_barrier_eV, bond_activation_volume_m3,
    bond_attempt_frequency_s, healing_cooperative_order,
    healing_correlation_time_s) are identical by construction
    (zero_cohesion_twin only changes restored_work_of_separation_J_m2), so
    this diagnostic is evaluated once per {reversible, persistent} preset
    and applies to both its zero- and finite-cohesion variants.
    """
    K_min_signed = R_REF * KMAX_Pa_sqrt_m
    sigma_comp_Pa = contact_pressure(
        K_min_signed, resolved_cfg.contact_pressure_scale, reference_contact_radius_m,
        0.0, resolved_cfg.contact_pressure_cap_Pa,
    )
    lam_bond_raw, rate_diag = bond_formation_rate(
        sigma_comp_Pa, T_K, resolved_cfg.chemistry_factor, resolved_cfg
    )
    m_h = resolved_cfg.healing_cooperative_order
    k_CB_at_min_K = cooperative_hazard(
        lam_bond_raw, m_h, resolved_cfg.healing_correlation_time_s
    )
    return {
        "K_min_signed_Pa_sqrt_m": K_min_signed,
        "sigma_comp_Pa_at_K_min": sigma_comp_Pa,
        "bond_barrier_floor_fraction_at_K_min": rate_diag["barrier_floor_fraction"],
        "bond_barrier_floored_at_K_min": rate_diag["floored"],
        "bond_rate_saturated_at_K_min": rate_diag["saturated"],
        "healing_cooperative_order": m_h,
        "cooperative_regime": (
            "single_hit_elementary" if m_h <= 1.0 + 1.0e-12 else "gamma_incomplete_saturating"
        ),
        "lambda_bond_raw_at_K_min_s": lam_bond_raw,
        "k_CB_at_K_min_s": k_CB_at_min_K,
    }


def freeze_pilot_configuration(
    *, Eprime_Pa: float, reference_contact_radius_m: float, engine_G_Pa: float, engine_nu: float,
    a_native_provenance_sha256: str,
) -> dict[str, Any]:
    calibration = solve_calibrated_rb2_configs(
        Eprime_Pa=Eprime_Pa, reference_contact_radius_m=reference_contact_radius_m
    )
    rb2_rev_finite = calibration["configs_finite_cohesion"]["reversible"]
    rb2_pers_finite = calibration["configs_finite_cohesion"]["persistent"]
    rb2_rev_zero = zero_cohesion_twin(rb2_rev_finite)
    rb2_pers_zero = zero_cohesion_twin(rb2_pers_finite)
    rb1 = rb1_config(rb2_rev_finite)

    configs = {
        "RB0": None,
        "RB1": rb1,
        "RB2_reversible_zero": rb2_rev_zero,
        "RB2_reversible_finite": rb2_rev_finite,
        "RB2_persistent_zero": rb2_pers_zero,
        "RB2_persistent_finite": rb2_pers_finite,
    }
    config_hashes = {
        name: (cfg.config_hash() if cfg is not None else None) for name, cfg in configs.items()
    }

    analytical_predictions = {
        "reversible_finite": analytical_single_patch_predictions(
            rb2_rev_finite, reference_contact_radius_m=reference_contact_radius_m
        ),
        "persistent_finite": analytical_single_patch_predictions(
            rb2_pers_finite, reference_contact_radius_m=reference_contact_radius_m
        ),
    }

    frozen = {
        "schema": SCHEMA,
        "required_python": REQUIRED_PYTHON,
        "seed": SEED,
        "a_native_provenance_sha256": a_native_provenance_sha256,
        "common_settings": {
            "parameter_option": "A_NATIVE",
            "T_K": T_K,
            "Kmax_Pa_sqrt_m": KMAX_Pa_sqrt_m,
            "frequency_Hz": F_HZ,
            "n_phase": N_PHASE,
            "mpz_n_bins": MPZ_N_BINS,
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
            "restored_work_of_separation_J_m2_finite": calibration["restored_work_of_separation_J_m2"],
            "bond_activation_volume_m3": calibration["bond_activation_volume_m3"],
            "r_positive_check": {
                "R": R_POSITIVE,
                "predicted_action_ratios": calibration["r_positive_action_ratios"],
                "note": "exactly 0.0 by construction of the exact K_signed<0 contact gate",
            },
        },
        "reference_action_presets": {
            "reversible": REFERENCE_ACTION_PRESETS["reversible"],
            "persistent": REFERENCE_ACTION_PRESETS["persistent"],
        },
        "configs": {name: (asdict(cfg) if cfg is not None else None) for name, cfg in configs.items()},
        "config_hashes": config_hashes,
        "analytical_single_patch_predictions": analytical_predictions,
        "limits": {
            "max_accepted_events": MAX_ACCEPTED_EVENTS,
            "max_projected_extension_m": MAX_PROJECTED_EXTENSION_m,
            "min_accepted_events_for_uncensored": MIN_ACCEPTED_EVENTS_FOR_UNCENSORED,
            "max_blocks_per_event": MAX_BLOCKS_PER_EVENT,
            "max_wall_seconds_per_trajectory": MAX_WALL_SECONDS_PER_TRAJECTORY,
            "block_cycles": BLOCK_CYCLES,
            "max_block_cycles": MAX_BLOCK_CYCLES,
        },
        "trajectories": trajectory_specs(),
        "expansion_threshold_log10_decade": EXPANSION_THRESHOLD_LOG10_DECADE,
    }
    frozen["frozen_configuration_sha256"] = canonical_hash(frozen)
    return frozen


# ---------------------------------------------------------------------------
# Trajectory execution
# ---------------------------------------------------------------------------


def signed_K(
    t_s: np.ndarray, *, R: float, frequency_Hz: float = F_HZ, Kmax_Pa_sqrt_m: float = KMAX_Pa_sqrt_m,
) -> np.ndarray:
    Kmin = R * Kmax_Pa_sqrt_m
    Kmean = 0.5 * (Kmax_Pa_sqrt_m + Kmin)
    Kamp = 0.5 * (Kmax_Pa_sqrt_m - Kmin)
    phase = 2.0 * np.pi * frequency_Hz * t_s
    return Kmean + Kamp * np.cos(phase)


def interval_compression_analysis(
    t_creation_s: float, t_next_event_s: float, *, R: float,
    frequency_Hz: float = F_HZ, Kmax_Pa_sqrt_m: float = KMAX_Pa_sqrt_m, n_samples: int = 4000,
) -> dict[str, Any]:
    """Reconstructs a PURE SINUSOIDAL K(t) trace at the given
    ``frequency_Hz``/``Kmax_Pa_sqrt_m`` (both default to this module's own
    R=-0.95/1000Hz/18MPa-sqrt(m) reference trajectory constants -- exact
    prior behavior for every existing caller, which never ran at any other
    condition). Callers with ``minimum_load_hold_s`` > 0 get this
    trajectory's SINUSOIDAL-TRAVERSE-ONLY contact reconstruction: it does
    not model the constant-Kmin dwell segment the real engine's own
    ``cycle_schedule`` adds (PX1.1), so ``negative_contact_duration_s``/
    ``complete_negative_excursion`` under-count true negative-K contact
    time for dwell-enabled trajectories -- a documented approximation, not
    a claim of dwell-aware contact accounting.
    """
    if t_next_event_s <= t_creation_s:
        return {
            "elapsed_time_s": 0.0, "elapsed_cycles": 0.0,
            "negative_contact_duration_s": 0.0, "complete_negative_excursion": False,
        }
    t = np.linspace(t_creation_s, t_next_event_s, n_samples)
    K = signed_K(t, R=R, frequency_Hz=frequency_Hz, Kmax_Pa_sqrt_m=Kmax_Pa_sqrt_m)
    neg = K < 0.0
    dt = t[1] - t[0]
    negative_duration_s = float(np.count_nonzero(neg)) * dt
    complete = False
    if neg.any() and not neg[0] and not neg[-1]:
        edges = np.diff(neg.astype(int))
        complete = bool((edges == 1).any() and (edges == -1).any())
    return {
        "elapsed_time_s": float(t_next_event_s - t_creation_s),
        "elapsed_cycles": float((t_next_event_s - t_creation_s) * frequency_Hz),
        "negative_contact_duration_s": negative_duration_s,
        "complete_negative_excursion": bool(complete),
    }


def run_trajectory(
    *,
    name: str,
    build_engine: Callable[[CrackRebondingControls | None], tuple[Any, dict[str, Any]]],
    make_controller: Callable[[int], Any],
    waveform_cls: Callable[..., Any],
    rebonding_cfg: CrackRebondingControls | None,
    R: float,
    reset_engine_registry: Callable[[], None] | None = None,
    Kmax_Pa_sqrt_m: float = KMAX_Pa_sqrt_m,
    frequency_Hz: float = F_HZ,
    n_phase: int = N_PHASE,
    T_K_: float = T_K,
    max_accepted_events: int = MAX_ACCEPTED_EVENTS,
    max_projected_extension_m: float = MAX_PROJECTED_EXTENSION_m,
    min_accepted_events_for_uncensored: int = MIN_ACCEPTED_EVENTS_FOR_UNCENSORED,
    max_blocks_per_event: int = MAX_BLOCKS_PER_EVENT,
    max_wall_seconds: float = MAX_WALL_SECONDS_PER_TRAJECTORY,
    hazard_rng_seed: int = SEED,
    static_shield_control: dict[str, Any] | None = None,
    minimum_load_hold_s: float = 0.0,
    state_sampler: Callable[[Any, dict[str, Any]], None] | None = None,
    max_cumulative_cycles: float = math.inf,
) -> dict[str, Any]:
    """Drive one fresh, unresumed trajectory in-process via the real
    A_NATIVE production engine's own cycle_step_waveform/
    commit_energy_gated_event (never through the mesh-dependent CLI
    backend).

    ``hazard_rng_seed`` is reporting-only (the actual hazard RNG seed is
    set by the caller via ``Engine.configure_hazard(seed=...)`` before
    ``build_engine``/``reset_engine_registry`` run -- this function never
    touches RNG state itself); it is recorded verbatim in the returned
    dict's ``"seed"`` field so a caller driving multiple seeds through
    this same function gets an honest per-trajectory label instead of the
    module constant ``SEED`` regardless of which seed was actually used
    (a mislabeling caught, and fixed here, while completing the minimal
    slope screen's exposure-unconditioned trajectory -- harmless in every
    prior use since no gate or comparison ever read this field, but
    corrected for provenance honesty going forward).

    ``static_shield_control`` (default ``None``, zero effect on every
    existing caller): opt-in hook for the v10.2.30 static-shield-
    attribution study's PRESCRIBED_POST_FIRST_EVENT_COHESIVE_SHIELD
    mechanism-control ablation (arrhenius_fracture/persistent_site_
    cyclic_v10229.py's ``preview_cycle_waveform``). When provided, it is
    installed as ``engine._static_shield_control`` immediately after
    ``build_engine`` runs, and its ``"first_event_fired"`` key is
    refreshed to ``len(events) >= 1`` once per event-loop iteration
    (i.e. before that event's own block-search begins) -- so K_b applies
    as a pure step function of already-COMMITTED events, never derived
    from any engine-internal counter. This function never reads any other
    key of the dict; its content and interpretation belong entirely to
    the static-shield engine code path.

    ``minimum_load_hold_s`` (Part X PX3, default 0.0 -- exact prior
    behavior for every existing caller) is threaded straight through to
    ``waveform_cls``, letting this same qualified event loop drive the
    PX3 dwell panel (mission section 7.3) without a second
    reimplementation.

    ``max_cumulative_cycles`` (Part X PX4, default ``math.inf`` -- exact
    prior behavior for every existing caller) is an additional right-censor
    condition alongside ``max_accepted_events``/``max_projected_extension_m``,
    needed for mission section 8's developed-campaign budget (max 30
    accepted events, 150 um, 1e12 cycles) -- PX3's screen budget never
    needed a cycle cap since its 12-event/60um limits were always reached
    first.

    PX4.1 fix: this is an EXACT first-passage horizon, not merely checked
    between accepted events. Each block-search call passes the exact
    remaining cycle budget as ``cycle_step_waveform``'s own
    ``requested_cycles`` argument, which threads into ``choose_block_
    cycles_diagnostic``'s already-enforced "requested_cap" mode
    (``base = min(max_block_cycles, requested_cycles)``, and the returned
    block size is clipped to the MINIMUM of every candidate limit, base
    included) -- an already-qualified hard ceiling on a single block's
    size, not a new one invented here. A block that reaches the horizon
    without firing uses the exact same "did not fire" commit path every
    ordinary non-firing block already uses (full MPZ/rebonding-state/
    protocol-cursor advance, no event, no patch, no translation, no new
    threshold draw) -- reusing already-validated machinery rather than a
    second commit path. A remaining budget of exactly 0 skips the engine
    call entirely (no RNG draw at all).

    One inherited, pre-existing precision floor: ``FatigueControllerConfig.
    min_block_cycles`` (default 1e-6) is a hard floor the adaptive search
    refuses to resolve below, so a remaining budget SMALLER than that
    floor can overshoot the horizon by up to the floor's own value. This
    is utterly negligible for this mission's actual 1e12-cycle horizon
    (twelve orders of magnitude above where the floor could ever bind) and
    is not something this fix introduces or could reasonably eliminate
    without changing ``min_block_cycles`` itself, a shared, already-
    qualified system parameter well outside Part X's own scope.
    """
    if reset_engine_registry is not None:
        reset_engine_registry()
    engine, manifest_audit = build_engine(rebonding_cfg)
    if static_shield_control is not None:
        engine._static_shield_control = static_shield_control
    ctrl = make_controller(n_phase)
    waveform = waveform_cls(
        Kmax=Kmax_Pa_sqrt_m, R=R, frequency_Hz=frequency_Hz,
        minimum_load_hold_s=minimum_load_hold_s,
    )

    bulk_action_records: list[dict[str, Any]] = []
    original_phase_resolved_action = _rebond.phase_resolved_action

    def _recording_phase_resolved_action(*args, **kwargs):
        action, end_states, end_idx, diag = original_phase_resolved_action(*args, **kwargs)
        bulk_action_records.append(dict(diag))
        return action, end_states, end_idx, diag

    events: list[dict[str, Any]] = []
    cumulative_extension_m = 0.0
    cumulative_time_s = 0.0
    cumulative_cycles = 0.0
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
            if cumulative_cycles >= max_cumulative_cycles:
                # Reached exactly at a prior event's acceptance (the inner
                # loop's own horizon check normally catches this first, mid-
                # search) -- same classification either way for a single,
                # consistent censor_reason string.
                censored = True
                censor_reason = "complete_physical_cycle_censor"
                break
            if time.monotonic() - start_wall > max_wall_seconds:
                censored = True
                censor_reason = "wall_time_budget_exhausted_between_events"
                break

            if static_shield_control is not None:
                static_shield_control["first_event_fired"] = len(events) >= 1

            bulk_action_records_before = len(bulk_action_records)
            fired_result = None
            reached_cycle_horizon = False
            blocks_used = 0
            for blocks_used in range(1, max_blocks_per_event + 1):
                if time.monotonic() - start_wall > max_wall_seconds:
                    break
                # Exact first-passage cycle horizon: cap THIS block's own
                # candidate size at the remaining cycle budget by passing
                # requested_cycles through to cycle_step_waveform (which
                # threads it into choose_block_cycles_diagnostic's existing
                # "requested_cap" mode -- base = min(max_block_cycles, req),
                # and the returned block size is clipped to the minimum of
                # every candidate limit, base included -- an ALREADY
                # enforced, already-qualified hard ceiling, not a new one
                # built here). A remaining budget of exactly 0 means the
                # horizon is already reached: no engine call, no RNG draw,
                # no event/patch/translation -- censor immediately.
                if math.isfinite(max_cumulative_cycles):
                    remaining = max_cumulative_cycles - cumulative_cycles
                    if remaining <= 0.0:
                        reached_cycle_horizon = True
                        break
                    result = engine.cycle_step_waveform(ctrl, waveform, T_K_, requested_cycles=remaining)
                else:
                    result = engine.cycle_step_waveform(ctrl, waveform, T_K_)
                cumulative_time_s += float(result.get("kinetic_dt_consumed_s", 0.0))
                cumulative_cycles += float(result.get("cycles_consumed", 0.0))
                if state_sampler is not None:
                    state_sampler(engine, result)
                if result.get("fired"):
                    fired_result = result
                    break
                if math.isfinite(max_cumulative_cycles) and cumulative_cycles >= max_cumulative_cycles:
                    reached_cycle_horizon = True
                    break

            if reached_cycle_horizon:
                censored = True
                censor_reason = "complete_physical_cycle_censor"
                break

            if fired_result is None:
                censored = True
                censor_reason = (
                    "wall_time_budget_exhausted_mid_event"
                    if time.monotonic() - start_wall > max_wall_seconds
                    else f"event_did_not_fire_within_{max_blocks_per_event}_blocks"
                )
                break

            pending = engine._energy_gate_pending
            committed_length = pending["proposal_m"]
            gate = {
                "energy_admissible_event_length_m": committed_length,
                "arrest_reason": "causal_pilot_v2_commit",
                "hazard_resistance_J_per_m2": 1.0,
                "orientation_gamma_relative": 1.0,
            }
            result_ref = pending["descriptor"].get("energy_gate_result_ref")
            waiting_time_s = cumulative_time_s - time_at_previous_event_s
            time_at_previous_event_s = cumulative_time_s

            pre_commit_rebonding_state = getattr(engine, "_rebonding_state", None)
            pre_event_max_pB = (
                max((float(p.p_B) for p in pre_commit_rebonding_state.active), default=0.0)
                if pre_commit_rebonding_state is not None
                else 0.0
            )
            pre_event_K_rebond = (
                float(pre_commit_rebonding_state.K_rebond_Pa_sqrt_m)
                if pre_commit_rebonding_state is not None else 0.0
            )
            # RNG/threshold provenance for this event, read before commit
            # mutates any state -- the drawn hazard threshold and the
            # engine/event identifiers that pin down which threshold-stream
            # draw this event corresponds to.
            hazard_threshold_action = getattr(engine, "hazard_threshold_action", None)
            hazard_event_index = getattr(engine, "hazard_event_index", None)
            engine_id = getattr(engine, "_engine_id", None)

            engine.commit_energy_gated_event(committed_length, gate, result_ref)
            cumulative_extension_m += committed_length

            rebonding_state = getattr(engine, "_rebonding_state", None)
            max_pB_post_commit = (
                max((float(p.p_B) for p in rebonding_state.active), default=0.0)
                if rebonding_state is not None else 0.0
            )
            max_K_rebond_post_commit = (
                float(rebonding_state.K_rebond_Pa_sqrt_m) if rebonding_state is not None else 0.0
            )

            new_bulk_records = bulk_action_records[bulk_action_records_before:]
            # The LAST recorded phase_resolved_action call for this event is
            # the one whose (action, end_states) the caller actually
            # committed with (solve_coupled_event_time's bisection loop ends
            # on its final residual() call; the zero-cohesion direct-
            # evaluation path makes exactly one call) -- so it is the
            # converged/final action and K_rebond summary for this event,
            # not an arbitrary bisection trial.
            final_record = new_bulk_records[-1] if new_bulk_records else {}
            max_phase_resolved_K_rebond = max(
                (r.get("max_K_rebond_Pa_sqrt_m", 0.0) for r in new_bulk_records), default=0.0
            )

            # MPZ-state snapshot at the committing block (mission Section
            # 11's "relevant MPZ-state" requirement) -- read directly off
            # the real production engine's own cycle_step_waveform result,
            # never re-derived or approximated.
            mpz_state = {
                key: fired_result.get(key)
                for key in (
                    "mpz_mobile_count", "mpz_retained_count", "mpz_emitted_total",
                    "mpz_escaped_total", "mpz_recovered_total", "r_eff", "sigma_tip",
                    "mpz_total_K_shield_Pa_sqrt_m",
                )
            }

            events.append({
                "event_index": len(events),
                "blocks_to_fire": blocks_used,
                "waiting_time_s_this_event": waiting_time_s,
                "cumulative_time_s": cumulative_time_s,
                "cumulative_cycles": cumulative_cycles,
                "accepted_length_m": float(committed_length),
                "cumulative_extension_m": cumulative_extension_m,
                "pre_event_max_pB": pre_event_max_pB,
                "pre_event_K_rebond_Pa_sqrt_m": pre_event_K_rebond,
                "max_pB_post_commit": max_pB_post_commit,
                "max_K_rebond_post_commit_Pa_sqrt_m": max_K_rebond_post_commit,
                "max_phase_resolved_K_rebond_Pa_sqrt_m": max_phase_resolved_K_rebond,
                "action_weighted_K_rebond_Pa_sqrt_m": final_record.get(
                    "action_weighted_K_rebond_Pa_sqrt_m", 0.0
                ),
                "cleavage_action": final_record.get("action", 0.0),
                "hazard_threshold_action": hazard_threshold_action,
                "hazard_event_index": hazard_event_index,
                "engine_id": engine_id,
                "mpz_state": mpz_state,
                "bulk_action_records": new_bulk_records,
                "any_bulk_action_used": any(r.get("bulk_action_used") for r in new_bulk_records),
                "all_bulk_action_qualified": all(
                    r.get("bulk_action_qualified", True) for r in new_bulk_records
                ),
                "K_b_applied_Pa_sqrt_m": (
                    (
                        float(static_shield_control["K_b_static_Pa_sqrt_m"])
                        if static_shield_control.get("first_event_fired") else 0.0
                    ) if static_shield_control is not None else None
                ),
            })
    finally:
        _rebond.phase_resolved_action = original_phase_resolved_action

    uncensored = (not censored) and (len(events) >= min_accepted_events_for_uncensored)

    intervals = []
    for i in range(1, len(events)):
        t_creation = events[i - 1]["cumulative_time_s"]
        t_next = events[i]["cumulative_time_s"]
        analysis = interval_compression_analysis(
            t_creation, t_next, R=R, frequency_Hz=frequency_Hz, Kmax_Pa_sqrt_m=Kmax_Pa_sqrt_m,
        )
        analysis.update({
            "creation_event_index": i - 1, "next_event_index": i,
            "waiting_time_s_this_event": events[i]["waiting_time_s_this_event"],
        })
        intervals.append(analysis)

    return {
        "name": name, "R": R,
        "rebonding_cfg_hash": rebonding_cfg.config_hash() if rebonding_cfg is not None else None,
        "rebonding_model_level": (
            rebonding_cfg.model_level.value if rebonding_cfg is not None else "REBOND_OFF"
        ),
        "restored_work_of_separation_J_m2": (
            rebonding_cfg.restored_work_of_separation_J_m2 if rebonding_cfg is not None else 0.0
        ),
        "manifest_audit": manifest_audit,
        "seed": hazard_rng_seed,
        "Kmax_Pa_sqrt_m": Kmax_Pa_sqrt_m,
        "T_K": T_K_,
        "frequency_Hz": frequency_Hz,
        "static_shield_control": static_shield_control,
        "events": events,
        "post_first_event_intervals": intervals,
        "n_accepted_events": len(events),
        "cumulative_extension_m": cumulative_extension_m,
        "cumulative_time_s": cumulative_time_s,
        "cumulative_cycles": cumulative_cycles,
        "minimum_load_hold_s": minimum_load_hold_s,
        "censored": censored,
        "censor_reason": censor_reason,
        "uncensored": uncensored,
        "wall_seconds": time.monotonic() - start_wall,
    }


__all__ = [
    "SCHEMA", "REQUIRED_PYTHON", "SEED", "T_K", "KMAX_Pa_sqrt_m", "F_HZ",
    "R_REF", "R_POSITIVE", "N_PHASE", "MPZ_N_BINS",
    "MAX_ACCEPTED_EVENTS", "MAX_PROJECTED_EXTENSION_m",
    "MIN_ACCEPTED_EVENTS_FOR_UNCENSORED", "EXPANSION_THRESHOLD_LOG10_DECADE",
    "TRAJECTORY_NAMES",
    "resolve_restored_work_of_separation", "solve_calibrated_rb2_configs",
    "rb1_config", "zero_cohesion_twin", "trajectory_specs",
    "analytical_single_patch_predictions", "freeze_pilot_configuration",
    "signed_K", "interval_compression_analysis", "run_trajectory",
    "canonical_hash",
]
