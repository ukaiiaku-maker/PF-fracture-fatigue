"""Pure, engine-independent kinetics for optional crack-face rebonding (v10.2.30).

This module contains no dependency on any live solver engine object. It implements:

- the frozen ``CrackRebondingControls`` configuration and its validation;
- log-domain, floor-bounded Arrhenius rate functions for bond formation, bond
  rupture, depassivation, and repassivation;
- the exact three-state conservative Markov generator and its matrix-exponential
  update;
- a generic exact phase-aware propagator ``propagate(p, ..., k0, dt)`` used for
  every wake-state advancement in the engine layer (no rounding, no block-scalar
  shortcuts);
- a Strang-split per-phase reconstruction of one representative cycle, used to
  keep the compression/formation/reopening/rupture correlation intact within a
  cycle rather than smearing it into a single scalar;
- the closed-form two-state analytical bonded-fraction map (RB2), used both as a
  documented regime map and as a unit-test oracle;
- a reference-action parameter generator that inverts target dimensionless
  actions into formation/rupture barriers at a fixed reference condition.

Every equation here is mapped to its source in
``docs/v10_2_30_crack_rebonding_equation_lineage.md``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import gammainc

KB_EV_PER_K = 8.617333262e-5
EV_TO_J = 1.602176634e-19

CONTACT_SEMANTICS_LABEL = "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT"


class RebondModelLevel(Enum):
    REBOND_OFF = "REBOND_OFF"
    CONTACT_PROXY_ONLY = "CONTACT_PROXY_ONLY"
    CLEAN_REVERSIBLE_REBOND = "CLEAN_REVERSIBLE_REBOND"
    PASSIVATION_GATED_REBOND = "PASSIVATION_GATED_REBOND"


class ContactModel(Enum):
    SIGNED_K_COMPRESSION_PROXY = "SIGNED_K_COMPRESSION_PROXY"
    RESOLVED_GAP_TRACTION = "RESOLVED_GAP_TRACTION"


class FeedbackMode(Enum):
    HAZARD_ONLY_REBOND_SHIELD = "HAZARD_ONLY_REBOND_SHIELD"
    COMMON_POSITIVE_LOCAL_K_REDUCTION = "COMMON_POSITIVE_LOCAL_K_REDUCTION"
    HAZARD_AND_ENERGY_GATE_COUPLED = "HAZARD_AND_ENERGY_GATE_COUPLED"


class InitialPrecrackWakeMode(Enum):
    NO_INITIAL_ACTIVE_WAKE = "NO_INITIAL_ACTIVE_WAKE"
    INITIAL_WAKE_PASSIVATED = "INITIAL_WAKE_PASSIVATED"
    INITIAL_WAKE_CLEAN = "INITIAL_WAKE_CLEAN"


_IMPLEMENTED_FEEDBACK_MODES = frozenset({FeedbackMode.HAZARD_ONLY_REBOND_SHIELD})


@dataclass(frozen=True)
class CrackRebondingControls:
    enabled: bool = False
    model_level: RebondModelLevel = RebondModelLevel.REBOND_OFF
    contact_model: ContactModel = ContactModel.SIGNED_K_COMPRESSION_PROXY
    feedback_mode: FeedbackMode = FeedbackMode.HAZARD_ONLY_REBOND_SHIELD

    topological_healing_enabled: bool = False
    negative_crack_advance_enabled: bool = False
    crack_segment_deletion_enabled: bool = False

    wake_length_m: float = 5.0e-6
    wake_weight_length_m: float = 1.0e-6
    contact_radius_min_m: float = 1.0e-9
    contact_pressure_scale: float = 1.0
    contact_pressure_cap_Pa: float = 5.0e9
    opening_stress_cap_Pa: float = 5.0e9

    fresh_surface_clean_fraction: float = 1.0
    initial_precrack_wake_mode: InitialPrecrackWakeMode = (
        InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE
    )

    chemistry_factor: float = 1.0

    bond_attempt_frequency_s: float = 1.0e13
    bond_barrier_eV: float = 0.5
    bond_barrier_floor_eV: float = 0.0
    bond_activation_volume_m3: float = 0.0
    healing_cooperative_order: float = 1.0
    healing_correlation_time_s: float = 1.0e-6

    rupture_attempt_frequency_s: float = 1.0e13
    rupture_barrier_eV: float = 0.5
    rupture_barrier_floor_eV: float = 0.0
    rupture_activation_volume_m3: float = 0.0

    depassivation_attempt_frequency_s: float = 1.0e13
    depassivation_barrier_eV: float = 0.5
    depassivation_barrier_floor_eV: float = 0.0
    depassivation_activation_volume_m3: float = 0.0

    repassivation_attempt_frequency_s: float = 1.0e13
    repassivation_barrier_eV: float = 0.5
    repassivation_barrier_floor_eV: float = 0.0

    restored_work_of_separation_J_m2: float = 0.0
    rebond_K_geometry_factor: float = 1.0

    minimum_load_hold_s: float = 0.0

    stochastic_healing_enabled: bool = False

    contact_oracle_id: str = ""
    contact_oracle_version: str = ""
    contact_oracle_source_hash: str = ""

    rebonding_block_max_dpB: float = 0.05
    rebonding_block_max_dpC: float = 0.05
    rebonding_block_max_dK_rebond_frac: float = 0.05
    rebonding_block_action_consistency_tol: float = 0.05

    bulk_action_error_rel_tol: float = 1.0e-3
    bulk_action_max_transient_extensions: int = 1

    def validate(self) -> "CrackRebondingControls":
        errors: list[str] = []

        def require(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        require(
            not self.topological_healing_enabled,
            "topological_healing_enabled must be False in this campaign",
        )
        require(
            not self.negative_crack_advance_enabled,
            "negative_crack_advance_enabled must be False in this campaign",
        )
        require(
            not self.crack_segment_deletion_enabled,
            "crack_segment_deletion_enabled must be False in this campaign",
        )
        require(
            self.contact_model is ContactModel.SIGNED_K_COMPRESSION_PROXY,
            "contact_model=RESOLVED_GAP_TRACTION has no oracle contract implemented "
            "in this pass; only SIGNED_K_COMPRESSION_PROXY is permitted",
        )
        require(
            self.minimum_load_hold_s == 0.0,
            "CrackRebondingControls.minimum_load_hold_s must be exactly 0.0: "
            "the minimum-load dwell is now implemented (v10.2.30 Part X) as "
            "FatigueWaveform.minimum_load_hold_s, not as a field on this "
            "rebonding-only config -- the dwell must drive ordinary "
            "cleavage/emission hazard and block-cycle selection identically "
            "whether or not rebonding is even enabled, which only the "
            "waveform (threaded through every channel) can reach. Leaving a "
            "second, disconnected minimum_load_hold_s on this config would "
            "be a silent-no-op footgun, so it stays rejected here on "
            "purpose; set the waveform's own field instead.",
        )
        require(
            not self.stochastic_healing_enabled,
            "stochastic_healing_enabled must be False in this pass",
        )
        require(0.0 <= self.chemistry_factor <= 1.0, "chemistry_factor must be in [0, 1]")
        require(self.wake_length_m > 0.0, "wake_length_m must be positive")
        require(
            0.0 < self.wake_weight_length_m <= self.wake_length_m,
            "wake_weight_length_m must be in (0, wake_length_m]",
        )
        require(self.contact_radius_min_m > 0.0, "contact_radius_min_m must be positive")
        require(self.contact_pressure_scale >= 0.0, "contact_pressure_scale must be non-negative")
        require(self.contact_pressure_cap_Pa > 0.0, "contact_pressure_cap_Pa must be positive")
        require(self.opening_stress_cap_Pa > 0.0, "opening_stress_cap_Pa must be positive")
        require(
            0.0 <= self.fresh_surface_clean_fraction <= 1.0,
            "fresh_surface_clean_fraction must be in [0, 1]",
        )

        for name in (
            "bond_barrier_eV",
            "bond_barrier_floor_eV",
            "rupture_barrier_eV",
            "rupture_barrier_floor_eV",
            "depassivation_barrier_eV",
            "depassivation_barrier_floor_eV",
            "repassivation_barrier_eV",
            "repassivation_barrier_floor_eV",
        ):
            require(getattr(self, name) >= 0.0, f"{name} must be non-negative")

        for name in (
            "bond_attempt_frequency_s",
            "rupture_attempt_frequency_s",
            "depassivation_attempt_frequency_s",
            "repassivation_attempt_frequency_s",
        ):
            require(getattr(self, name) >= 0.0, f"{name} must be non-negative")

        for name in (
            "bond_activation_volume_m3",
            "rupture_activation_volume_m3",
            "depassivation_activation_volume_m3",
        ):
            require(getattr(self, name) >= 0.0, f"{name} must be non-negative (unsigned in this pass)")

        require(self.healing_cooperative_order >= 1.0, "healing_cooperative_order must be >= 1")
        require(self.healing_correlation_time_s > 0.0, "healing_correlation_time_s must be positive")
        require(
            self.restored_work_of_separation_J_m2 >= 0.0,
            "restored_work_of_separation_J_m2 must be non-negative",
        )
        require(self.rebond_K_geometry_factor >= 0.0, "rebond_K_geometry_factor must be non-negative")

        for name in (
            "rebonding_block_max_dpB",
            "rebonding_block_max_dpC",
            "rebonding_block_max_dK_rebond_frac",
            "rebonding_block_action_consistency_tol",
            "bulk_action_error_rel_tol",
        ):
            require(getattr(self, name) > 0.0, f"{name} must be positive")
        require(
            self.bulk_action_max_transient_extensions >= 0,
            "bulk_action_max_transient_extensions must be non-negative",
        )

        if errors:
            raise ValueError("; ".join(errors))

        if self.feedback_mode not in _IMPLEMENTED_FEEDBACK_MODES:
            raise NotImplementedError(
                f"feedback_mode={self.feedback_mode.value} is a documented future "
                "interface, not implemented in this pass; only "
                f"{FeedbackMode.HAZARD_ONLY_REBOND_SHIELD.value} is available"
            )

        return self

    def config_hash(self) -> str:
        canonical = json.dumps(_jsonable(asdict(self)), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def crack_rebonding_config_from_environment(
    env: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> CrackRebondingControls:
    """Build a validated ``CrackRebondingControls`` from environment variables.

    ``overrides`` (typically CLI-derived) take precedence over ``env`` values,
    which take precedence over dataclass defaults. Enabled defaults to False.
    """
    import os

    source = dict(os.environ if env is None else env)
    overrides = dict(overrides or {})

    def pick(key: str, cast: Callable[[str], Any], default: Any) -> Any:
        if key in overrides:
            return overrides[key]
        env_key = f"V10230_CRACK_REBONDING_{key.upper()}"
        if env_key in source:
            return cast(source[env_key])
        return default

    def pick_bool(key: str, default: bool) -> bool:
        return bool(pick(key, lambda s: s.strip().lower() in {"1", "true", "yes", "on"}, default))

    def pick_enum(key: str, enum_cls: type[Enum], default: Enum) -> Enum:
        value = pick(key, lambda s: s, default)
        if isinstance(value, enum_cls):
            return value
        return enum_cls(str(value))

    defaults = CrackRebondingControls()
    kwargs: dict[str, Any] = {}
    kwargs["enabled"] = pick_bool("enabled", defaults.enabled)
    kwargs["model_level"] = pick_enum("model_level", RebondModelLevel, defaults.model_level)
    kwargs["contact_model"] = pick_enum("contact_model", ContactModel, defaults.contact_model)
    kwargs["feedback_mode"] = pick_enum("feedback_mode", FeedbackMode, defaults.feedback_mode)
    kwargs["initial_precrack_wake_mode"] = pick_enum(
        "initial_precrack_wake_mode", InitialPrecrackWakeMode, defaults.initial_precrack_wake_mode
    )

    bool_fields = (
        "topological_healing_enabled",
        "negative_crack_advance_enabled",
        "crack_segment_deletion_enabled",
        "stochastic_healing_enabled",
    )
    for name in bool_fields:
        kwargs[name] = pick_bool(name, getattr(defaults, name))

    str_fields = ("contact_oracle_id", "contact_oracle_version", "contact_oracle_source_hash")
    for name in str_fields:
        kwargs[name] = pick(name, str, getattr(defaults, name))

    handled = set(kwargs) | {"enabled"}
    for f in defaults.__dataclass_fields__.values():  # type: ignore[attr-defined]
        if f.name in handled:
            continue
        default_value = getattr(defaults, f.name)
        kwargs[f.name] = pick(f.name, float, default_value)

    return CrackRebondingControls(**kwargs).validate()


_FATIGUE_INTEGRATOR_MODES = ("accelerated", "explicit")


def fatigue_integrator_mode_from_environment(env: dict[str, str] | None = None) -> str:
    """Selects between the default accelerated (DMD/Poincare) fatigue-cycle
    integrator and the explicit phase-resolved one, via
    ``V10230_FATIGUE_INTEGRATOR_MODE`` (default ``"accelerated"``, byte-
    identical to the CLI's behavior before this selector existed).

    S8D (round-3 follow-up): the accelerated integrator is what makes the
    Gate-S0 rebonding/acceleration fail-closed check unconditional -- for a
    real ``--fatigue-cycles`` CLI run, it unconditionally replaces
    ``persistent_site_cyclic_coupled_v10229.integrate_state_coupled_waveform``
    (the confirmed real rebonding injection point,
    ``persistent_site_coupled_hazard_v10229.py``) with the accelerated
    engine, so rebonding's real injection point is never reached once that
    swap is applied. ``"explicit"`` opts out of that one swap only, for a
    genuine outer-driver CLI smoke test with rebonding enabled -- see
    ``sharp_front_v10_2_30_energy_gated_fatigue.py::main``. Every other
    monkeypatch/config path is unaffected by this selector.
    """
    import os

    source = dict(os.environ if env is None else env)
    raw = source.get("V10230_FATIGUE_INTEGRATOR_MODE", "accelerated").strip().lower()
    if raw not in _FATIGUE_INTEGRATOR_MODES:
        raise ValueError(
            f"Unknown V10230_FATIGUE_INTEGRATOR_MODE={raw!r}; expected one of "
            f"{_FATIGUE_INTEGRATOR_MODES}"
        )
    return raw


def _log_domain_rate(
    nu: float, G0_eV: float, G_floor_eV: float, work_eV: float, T_K: float
) -> tuple[float, dict[str, float]]:
    if nu <= 0.0:
        return 0.0, {"barrier_floor_fraction": 0.0, "floored": False, "saturated": False}
    G_nominal = G0_eV - work_eV
    floored = G_nominal < G_floor_eV
    G_eff = max(G_nominal, G_floor_eV)
    kT = max(KB_EV_PER_K * T_K, 1.0e-30)
    ln_lambda = math.log(nu) - G_eff / kT
    saturated = ln_lambda > 700.0
    ln_lambda_clipped = min(ln_lambda, 700.0)
    rate = math.exp(ln_lambda_clipped)
    barrier_floor_fraction = (
        (G_floor_eV - G_nominal) / max(G0_eV, 1.0e-30) if floored else 0.0
    )
    return rate, {
        "barrier_floor_fraction": float(max(barrier_floor_fraction, 0.0)),
        "floored": bool(floored),
        "saturated": bool(saturated),
    }


def cooperative_hazard(lambda_raw: float, m_h: float, tau_h: float) -> float:
    """k = gammainc(m_h, lambda_raw*tau_h)/tau_h for m_h>1, else k = lambda_raw.

    Mirrors ``lambda_cleave``'s existing single-hit special case exactly
    (``unified_front.py:109-114``) so ``m_h=1`` recovers the elementary rate
    identically rather than only in the small-x limit of the unqualified gamma
    expression.
    """
    if lambda_raw <= 0.0:
        return 0.0
    if m_h > 1.0 + 1.0e-12:
        tau = max(float(tau_h), 1.0e-30)
        return float(gammainc(m_h, min(lambda_raw * tau, 1.0e12)) / tau)
    return float(lambda_raw)


def bond_formation_rate(
    sigma_comp_Pa: float, T_K: float, chemistry_factor: float, cfg: CrackRebondingControls
) -> tuple[float, dict[str, float]]:
    if chemistry_factor <= 0.0:
        return 0.0, {"barrier_floor_fraction": 0.0, "floored": False, "saturated": False, "chemistry_zero": True}
    work_eV = float(sigma_comp_Pa) * cfg.bond_activation_volume_m3 / EV_TO_J
    nu = chemistry_factor * cfg.bond_attempt_frequency_s
    rate, diag = _log_domain_rate(nu, cfg.bond_barrier_eV, cfg.bond_barrier_floor_eV, work_eV, T_K)
    diag["chemistry_zero"] = False
    return rate, diag


def bond_rupture_rate(
    sigma_open_Pa: float, T_K: float, cfg: CrackRebondingControls, *, compressive_phase: bool
) -> tuple[float, dict[str, float]]:
    if compressive_phase:
        return 0.0, {"barrier_floor_fraction": 0.0, "floored": False, "saturated": False}
    work_eV = float(sigma_open_Pa) * cfg.rupture_activation_volume_m3 / EV_TO_J
    return _log_domain_rate(
        cfg.rupture_attempt_frequency_s, cfg.rupture_barrier_eV, cfg.rupture_barrier_floor_eV, work_eV, T_K
    )


def depassivation_rate(
    sigma_comp_Pa: float, T_K: float, cfg: CrackRebondingControls, *, compressive_phase: bool
) -> tuple[float, dict[str, float]]:
    if not compressive_phase:
        return 0.0, {"barrier_floor_fraction": 0.0, "floored": False, "saturated": False}
    work_eV = float(sigma_comp_Pa) * cfg.depassivation_activation_volume_m3 / EV_TO_J
    return _log_domain_rate(
        cfg.depassivation_attempt_frequency_s,
        cfg.depassivation_barrier_eV,
        cfg.depassivation_barrier_floor_eV,
        work_eV,
        T_K,
    )


def repassivation_rate(
    T_K: float, cfg: CrackRebondingControls, *, K_s_sign: int
) -> tuple[float, dict[str, float]]:
    if K_s_sign <= 0:
        return 0.0, {"barrier_floor_fraction": 0.0, "floored": False, "saturated": False}
    return _log_domain_rate(
        cfg.repassivation_attempt_frequency_s, cfg.repassivation_barrier_eV, cfg.repassivation_barrier_floor_eV, 0.0, T_K
    )


def contact_pressure(K_s_Pa_sqrt_m: float, eta_p: float, r_contact_m: float, s_j_m: float, cap_Pa: float) -> float:
    neg_part = max(-float(K_s_Pa_sqrt_m), 0.0)
    denom = math.sqrt(2.0 * math.pi * max(r_contact_m + s_j_m, 1.0e-30))
    return float(min(eta_p * neg_part / denom, cap_Pa))


def opening_stress(K_s_Pa_sqrt_m: float, r_contact_m: float, s_j_m: float, cap_Pa: float) -> float:
    pos_part = max(float(K_s_Pa_sqrt_m), 0.0)
    denom = math.sqrt(2.0 * math.pi * max(r_contact_m + s_j_m, 1.0e-30))
    return float(min(pos_part / denom, cap_Pa))


def build_Q(k_CB: float, k_BC: float, k_PC: float, k_CP: float) -> np.ndarray:
    """Conservative generator over state order [P, C, B]; every column sums to zero."""
    return np.array(
        [
            [-k_PC, k_CP, 0.0],
            [k_PC, -(k_CP + k_CB), k_BC],
            [0.0, k_CB, -k_BC],
        ],
        dtype=float,
    )


def advance_markov(p: np.ndarray, Q: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0.0:
        return np.asarray(p, dtype=float).copy()
    return expm(Q * float(dt)) @ np.asarray(p, dtype=float)


def build_phase_factors(
    Q_phase_list: list[np.ndarray], dt_phase: float | Sequence[float]
) -> list[np.ndarray]:
    """Per-bin matrix exponential. ``dt_phase`` may be a uniform scalar (the
    original, pre-Part-X convention -- every bin gets the same duration) or a
    per-bin duration sequence the same length as ``Q_phase_list`` (Part X's
    heterogeneous schedule: ``n_phase`` sinusoidal bins plus one appended
    constant-Kmin dwell bin of its own, generally different, duration)."""
    if np.isscalar(dt_phase):
        return [expm(Q * float(dt_phase)) for Q in Q_phase_list]
    dt_array = np.asarray(dt_phase, dtype=float)
    if dt_array.shape[0] != len(Q_phase_list):
        raise ValueError(
            "per-bin dt_phase length must match Q_phase_list length "
            f"({dt_array.shape[0]} != {len(Q_phase_list)})"
        )
    return [expm(Q * float(dt)) for Q, dt in zip(Q_phase_list, dt_array)]


def _partial_product(phase_factors: list[np.ndarray], k0: int, r: int) -> tuple[np.ndarray, int]:
    n = len(phase_factors)
    M = np.eye(phase_factors[0].shape[0]) if phase_factors else np.eye(3)
    idx = k0 % n
    for _ in range(r):
        M = phase_factors[idx] @ M
        idx = (idx + 1) % n
    return M, idx


def _propagate_uniform(
    p: np.ndarray,
    Q_phase_list: list[np.ndarray],
    phase_factors: list[np.ndarray],
    k0: int,
    dt: float,
    dt_phase: float,
) -> np.ndarray:
    """Original (pre-Part-X) uniform-bin-duration propagator, byte-for-byte
    unchanged. This is the exact hold=0 code path -- kept as its own
    function, never touched by the heterogeneous generalization below, so
    hold=0 trajectories cannot pick up so much as a floating-point
    operation-order change from Part X's dwell support."""
    n = len(phase_factors)
    steps_f = float(dt) / float(dt_phase)
    whole_steps = int(math.floor(steps_f + 1.0e-9))
    frac = max(steps_f - whole_steps, 0.0)
    if frac > 1.0 - 1.0e-9:
        whole_steps += 1
        frac = 0.0

    m_cycles, r_steps = divmod(whole_steps, n)
    if m_cycles > 0:
        M_cycle, _ = _partial_product(phase_factors, k0, n)
        p = np.linalg.matrix_power(M_cycle, m_cycles) @ p

    if r_steps > 0:
        M_r, k_after = _partial_product(phase_factors, k0, r_steps)
        p = M_r @ p
    else:
        k_after = k0 % n

    if frac > 1.0e-12:
        Q_frac = Q_phase_list[k_after % n]
        p = expm(Q_frac * (frac * dt_phase)) @ p

    return p


def _propagate_heterogeneous(
    p: np.ndarray,
    Q_phase_list: list[np.ndarray],
    phase_factors: list[np.ndarray],
    k0: int,
    dt: float,
    dt_array: np.ndarray,
) -> np.ndarray:
    """Exact p(t+dt) = M(k0, dt) @ p for a schedule of ``n`` bins with
    arbitrary, generally unequal, per-bin durations (Part X's sinusoidal-
    traverse-plus-dwell cycle). Any starting bin index, any duration
    (fractional bin, spanning many bins, or many full cycles), no rounding
    and no approximation of the dwell as extra equal-duration phase bins.

    Whole-cycle handling is identical in spirit to the uniform case: the
    full-cycle product starting at ``k0`` returns the state to phase index
    ``k0`` exactly (mod n) regardless of bin-duration heterogeneity, so
    repeated whole cycles are still a matrix power of that one product.
    The remainder (less than one full cycle) is walked bin by bin -- each
    bin either fully consumed (bulk matrix multiply, matching a whole
    sinusoidal bin or, when the remainder reaches it, the entire dwell) or
    partially consumed by one exact sub-bin ``expm`` (a fractional
    sinusoidal bin, or an event firing partway through the dwell) -- so an
    event may localize inside the hold, and a subsequent call starting at
    the resulting (idx, remaining-duration-in-that-bin) position correctly
    consumes only what remains of that same bin, not a fresh one.
    """
    n = len(phase_factors)
    total_cycle = float(np.sum(dt_array))
    if total_cycle <= 0.0:
        return p.copy()

    m_cycles = int(math.floor(dt / total_cycle + 1.0e-12))
    remaining = dt - m_cycles * total_cycle
    if remaining > total_cycle * (1.0 - 1.0e-9):
        m_cycles += 1
        remaining = 0.0

    if m_cycles > 0:
        M_cycle, _ = _partial_product(phase_factors, k0, n)
        p = np.linalg.matrix_power(M_cycle, m_cycles) @ p

    idx = k0 % n
    t_left = remaining
    while t_left > 1.0e-15:
        bin_dt = float(dt_array[idx])
        if bin_dt <= 0.0:
            idx = (idx + 1) % n
            continue
        if t_left >= bin_dt * (1.0 - 1.0e-9):
            p = phase_factors[idx] @ p
            t_left -= bin_dt
            idx = (idx + 1) % n
        else:
            p = expm(Q_phase_list[idx] * t_left) @ p
            t_left = 0.0

    return p


def propagate(
    p: np.ndarray,
    Q_phase_list: list[np.ndarray],
    phase_factors: list[np.ndarray],
    k0: int,
    dt: float,
    dt_phase: float | Sequence[float],
) -> np.ndarray:
    """Exact p(t+dt) = M(k0, dt) @ p for the piecewise-constant-rate phase
    discretization: any starting phase-grid index, any duration (fractional or
    spanning many cycles), no rounding.

    ``dt_phase`` is a uniform scalar for the original (pre-Part-X) equal-
    duration phase grid, or a per-bin duration array for Part X's
    heterogeneous sinusoidal-traverse-plus-dwell schedule -- dispatched to
    ``_propagate_uniform``/``_propagate_heterogeneous`` respectively, so
    every existing (scalar-``dt_phase``) call site is completely unaffected.
    """
    p = np.asarray(p, dtype=float)
    if dt <= 1.0e-300:
        return p.copy()
    if np.isscalar(dt_phase):
        return _propagate_uniform(p, Q_phase_list, phase_factors, k0, dt, float(dt_phase))
    return _propagate_heterogeneous(
        p, Q_phase_list, phase_factors, k0, dt, np.asarray(dt_phase, dtype=float)
    )


def build_augmented_Q(Q: np.ndarray) -> np.ndarray:
    """6x6 augmented generator for exact occupancy time-integral tracking
    alongside the ordinary 3-state propagation: writing the augmented
    state as ``[p; s]`` (``s`` a running integral of ``p``),
    ``d/dt [p;s] = [[Q,0],[I,0]] @ [p;s]`` exactly reproduces
    ``dp/dt = Q p`` in the top block while ``s(t) = integral_0^t p(tau)
    dtau`` accumulates in the bottom block -- so
    ``expm(augmented_Q * dt) @ [p0; 0]`` gives ``(p(dt), integral_0^dt
    p(tau) dtau)`` exactly for a CONSTANT generator, with no averaging or
    trapezoid approximation (PX1.2: mission section 5.2's transition-
    action/flux instrumentation)."""
    n = Q.shape[0]
    Q_aug = np.zeros((2 * n, 2 * n))
    Q_aug[:n, :n] = Q
    Q_aug[n:, :n] = np.eye(n)
    return Q_aug


def transition_actions_and_fluxes(
    p0: np.ndarray,
    Q_phase_list: list[np.ndarray],
    rate_constants: dict[str, np.ndarray],
    k0: int,
    dt: float,
    dt_phase: float | Sequence[float],
) -> dict[str, Any]:
    """Exact transition actions ``A_ij = integral k_ij(tau) dtau`` and
    realized fluxes ``F_ij = integral k_ij(tau) p_source(tau) dtau`` over
    one committed interval, plus the ordinary final state and the P/C/B
    state-balance closure (``dp_P = -F_PC+F_CP``, ``dp_C =
    F_PC-F_CP-F_CB+F_BC``, ``dp_B = F_CB-F_BC``).

    ``rate_constants`` maps ``"CB"``/``"BC"``/``"PC"``/``"CP"`` to a
    per-bin array (same length as ``Q_phase_list``) of that transition's
    instantaneous rate constant -- the same values ``build_Q`` assembled
    into each bin's generator, supplied separately here since the
    generator alone does not expose the individual rate constants needed
    to weight each bin's occupancy integral by the RIGHT transition.

    Deliberately a fully separate, parallel bin-walk from ``propagate``
    (which this function's own returned ``p_final`` can be cross-checked
    against for instrumentation-on-vs-off physical parity): this is
    default-off diagnostic instrumentation and must not alter, or even
    risk perturbing, the real state-evolution code path it audits.
    Because every rate constant is exactly constant within a bin, each
    bin's exact contribution uses ``build_augmented_Q`` once per bin
    (cached across calls at the same schedule via ``phase_factors``-style
    precomputation is left to the caller, matching the existing
    ``build_phase_factors``/``propagate`` split).

    KNOWN LIMITATION (acceptable for a first, correctness-first pass):
    whole cycles are walked bin-by-bin rather than bulk-accelerated via
    matrix power the way ``_propagate_heterogeneous``/
    ``phase_resolved_action`` are for VHCF-scale spans, so this is O(bins
    x cycles), not O(log cycles). Fine for per-event archival on a
    bounded inter-event interval; revisit with a periodic-orbit-style
    bulk shortcut (accumulate the per-cycle A/F once, then scale by the
    whole-cycle count, using the SAME periodic-state argument
    ``_periodic_orbit_certificate`` already makes) if profiling on a real
    VHCF campaign shows this default-off instrumentation is a bottleneck.
    """
    p0 = np.asarray(p0, dtype=float)
    n_state = p0.shape[0]
    if dt <= 1.0e-300:
        zero = np.zeros(())
        return {
            "p_final": p0.copy(), "idx_final": int(k0),
            "A_CB": 0.0, "A_BC": 0.0, "A_PC": 0.0, "A_CP": 0.0,
            "F_CB": 0.0, "F_BC": 0.0, "F_PC": 0.0, "F_CP": 0.0,
        }

    n = len(Q_phase_list)
    if np.isscalar(dt_phase):
        dt_array = np.full(n, float(dt_phase))
    else:
        dt_array = np.asarray(dt_phase, dtype=float)

    k_CB = np.asarray(rate_constants["CB"], dtype=float)
    k_BC = np.asarray(rate_constants["BC"], dtype=float)
    k_PC = np.asarray(rate_constants["PC"], dtype=float)
    k_CP = np.asarray(rate_constants["CP"], dtype=float)

    augmented = [build_augmented_Q(Q) for Q in Q_phase_list]

    p = p0.copy()
    A_CB = A_BC = A_PC = A_CP = 0.0
    F_CB = F_BC = F_PC = F_CP = 0.0

    def _consume_bin(idx: int, duration_s: float) -> None:
        nonlocal p, A_CB, A_BC, A_PC, A_CP, F_CB, F_BC, F_PC, F_CP
        if duration_s <= 0.0:
            return
        y0 = np.concatenate([p, np.zeros(n_state)])
        y1 = expm(augmented[idx] * duration_s) @ y0
        p = y1[:n_state]
        integral_p = y1[n_state:]  # [int p_P dt, int p_C dt, int p_B dt] over this bin
        int_pP, int_pC, int_pB = integral_p[0], integral_p[1], integral_p[2]
        A_CB += k_CB[idx] * duration_s
        A_BC += k_BC[idx] * duration_s
        A_PC += k_PC[idx] * duration_s
        A_CP += k_CP[idx] * duration_s
        F_CB += k_CB[idx] * int_pC
        F_BC += k_BC[idx] * int_pB
        F_PC += k_PC[idx] * int_pP
        F_CP += k_CP[idx] * int_pC

    total_cycle = float(np.sum(dt_array))
    if total_cycle <= 0.0:
        return {
            "p_final": p.copy(), "idx_final": int(k0 % max(n, 1)),
            "A_CB": 0.0, "A_BC": 0.0, "A_PC": 0.0, "A_CP": 0.0,
            "F_CB": 0.0, "F_BC": 0.0, "F_PC": 0.0, "F_CP": 0.0,
        }

    m_cycles = int(math.floor(dt / total_cycle + 1.0e-12))
    remaining = dt - m_cycles * total_cycle
    if remaining > total_cycle * (1.0 - 1.0e-9):
        m_cycles += 1
        remaining = 0.0

    idx = k0 % n
    for _ in range(m_cycles):
        cycle_start_idx = idx
        for _ in range(n):
            _consume_bin(idx, float(dt_array[idx]))
            idx = (idx + 1) % n
        assert idx == cycle_start_idx  # one full cycle returns to the same bin

    t_left = remaining
    while t_left > 1.0e-15:
        bin_dt = float(dt_array[idx])
        if bin_dt <= 0.0:
            idx = (idx + 1) % n
            continue
        if t_left >= bin_dt * (1.0 - 1.0e-9):
            _consume_bin(idx, bin_dt)
            t_left -= bin_dt
            idx = (idx + 1) % n
        else:
            _consume_bin(idx, t_left)
            t_left = 0.0

    return {
        "p_final": p, "idx_final": int(idx),
        "A_CB": float(A_CB), "A_BC": float(A_BC), "A_PC": float(A_PC), "A_CP": float(A_CP),
        "F_CB": float(F_CB), "F_BC": float(F_BC), "F_PC": float(F_PC), "F_CP": float(F_CP),
    }


def strang_cycle_trajectory(
    p_start: np.ndarray, Q_phase_list: list[np.ndarray], dt_phase: float
) -> tuple[np.ndarray, np.ndarray]:
    """One representative cycle via symmetric per-phase-bin splitting.

    Returns (midpoints, p_end): ``midpoints[k]`` is the state at the temporal
    midpoint of phase bin k (half-step in, record, half-step out using the
    same bin-constant generator for both halves — this is exact for the
    piecewise-constant-rate discretization, not an operator-splitting
    approximation, since only one generator applies within a bin). ``p_end``
    is the state after the full cycle, which is bit-identical to
    ``propagate(p_start, ..., k0=0, dt=period)`` since both halves use the
    same generator.
    """
    n = len(Q_phase_list)
    p = np.asarray(p_start, dtype=float)
    midpoints = np.zeros((n, p.shape[0]))
    for k in range(n):
        half = expm(Q_phase_list[k] * (0.5 * dt_phase))
        p = half @ p
        midpoints[k] = p
        p = half @ p
    return midpoints, p


def two_state_fixed_point(A_on: float, A_off: float) -> dict[str, float]:
    A_on = max(float(A_on), 0.0)
    A_off = max(float(A_off), 0.0)
    denom = 1.0 - math.exp(-(A_on + A_off))
    b_star = 0.0 if denom <= 1.0e-300 else math.exp(-A_off) * (1.0 - math.exp(-A_on)) / denom
    p_survive = (1.0 - math.exp(-A_on)) * math.exp(-A_off)
    return {"b_star": float(b_star), "P_survive": float(p_survive)}


def two_state_iterate(b0: float, A_on: float, A_off: float, n_cycles: int) -> list[float]:
    b = float(b0)
    history = [b]
    for _ in range(int(n_cycles)):
        b_contact = 1.0 - (1.0 - b) * math.exp(-A_on)
        b = math.exp(-A_off) * b_contact
        history.append(b)
    return history


REFERENCE_ACTION_PRESETS: dict[str, tuple[float, float]] = {
    "formation_limited": (0.1, 1.0),
    "reversible": (1.0, 1.0),
    "persistent": (10.0, 0.1),
    "ephemeral": (10.0, 10.0),
}


def _reference_phase_grid(n_phase: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * math.pi, n_phase, endpoint=False)


def _reference_K_signed(phase: np.ndarray, Kmax: float, R: float) -> np.ndarray:
    return Kmax * ((1.0 + R) / 2.0 + (1.0 - R) / 2.0 * np.cos(phase))


def _integrate_A_on(
    G_bond0_eV: float,
    *,
    T_K: float,
    f_Hz: float,
    R: float,
    Kmax_Pa_sqrt_m: float,
    s_j_m: float,
    r_contact_m: float,
    cfg: CrackRebondingControls,
    n_phase: int,
) -> float:
    phase = _reference_phase_grid(n_phase)
    dt_phase = (1.0 / f_Hz) / n_phase
    K_s = _reference_K_signed(phase, Kmax_Pa_sqrt_m, R)
    trial_cfg = replace(cfg, bond_barrier_eV=G_bond0_eV)
    total = 0.0
    for Ks in K_s:
        sigma_comp = contact_pressure(
            Ks, trial_cfg.contact_pressure_scale, r_contact_m, s_j_m, trial_cfg.contact_pressure_cap_Pa
        )
        lam_raw, _ = bond_formation_rate(sigma_comp, T_K, trial_cfg.chemistry_factor, trial_cfg)
        # Same exact contact gate as crack_rebonding_v10230.patch_Q: no
        # contact for K_signed >= 0, so k_cb must be exactly zero there (not
        # just the (already-zero) work term inside bond_formation_rate's
        # Arrhenius exponent, which leaves a nonzero unassisted-thermal
        # baseline rate).
        if Ks < 0.0:
            k_cb = cooperative_hazard(lam_raw, trial_cfg.healing_cooperative_order, trial_cfg.healing_correlation_time_s)
        else:
            k_cb = 0.0
        total += k_cb * dt_phase
    return float(total)


def _integrate_A_off(
    G_off0_eV: float,
    *,
    T_K: float,
    f_Hz: float,
    R: float,
    Kmax_Pa_sqrt_m: float,
    s_j_m: float,
    r_contact_m: float,
    cfg: CrackRebondingControls,
    n_phase: int,
) -> float:
    phase = _reference_phase_grid(n_phase)
    dt_phase = (1.0 / f_Hz) / n_phase
    K_s = _reference_K_signed(phase, Kmax_Pa_sqrt_m, R)
    trial_cfg = replace(cfg, rupture_barrier_eV=G_off0_eV)
    total = 0.0
    for Ks in K_s:
        sigma_open = opening_stress(Ks, r_contact_m, s_j_m, trial_cfg.opening_stress_cap_Pa)
        k_bc, _ = bond_rupture_rate(sigma_open, T_K, trial_cfg, compressive_phase=bool(Ks < 0.0))
        total += k_bc * dt_phase
    return float(total)


def _monotone_solve_decreasing(
    f: Callable[[float], float], target: float, lo_eV: float = 0.0, hi_eV: float = 2.0, max_expand: int = 40
) -> float:
    def g(x: float) -> float:
        return f(x) - target

    lo, hi = lo_eV, hi_eV
    flo, fhi = g(lo), g(hi)
    tries = 0
    while flo * fhi > 0.0 and tries < max_expand:
        hi *= 1.5
        fhi = g(hi)
        tries += 1
    if flo * fhi > 0.0:
        raise RuntimeError("could not bracket root for reference-action barrier solve")
    return float(brentq(g, lo, hi, xtol=1.0e-10))


def solve_reference_action_barriers(
    A_on_ref: float,
    A_off_ref: float,
    *,
    T_K: float = 300.0,
    f_Hz: float = 1000.0,
    R: float = -0.95,
    Kmax_Pa_sqrt_m: float = 18.0e6,
    reference_patch_distance_m: float,
    reference_contact_radius_m: float,
    cfg_template: CrackRebondingControls,
    n_phase: int = 360,
) -> CrackRebondingControls:
    """Monotonically solve for bond/rupture barriers giving the target
    dimensionless actions at the stated reference condition, holding every
    other ``cfg_template`` field fixed. Higher barrier -> lower rate -> lower
    integrated action, so each solve is a monotone-decreasing root find.
    """

    def action_on(G_bond0_eV: float) -> float:
        return _integrate_A_on(
            G_bond0_eV,
            T_K=T_K,
            f_Hz=f_Hz,
            R=R,
            Kmax_Pa_sqrt_m=Kmax_Pa_sqrt_m,
            s_j_m=reference_patch_distance_m,
            r_contact_m=reference_contact_radius_m,
            cfg=cfg_template,
            n_phase=n_phase,
        )

    def action_off(G_off0_eV: float) -> float:
        return _integrate_A_off(
            G_off0_eV,
            T_K=T_K,
            f_Hz=f_Hz,
            R=R,
            Kmax_Pa_sqrt_m=Kmax_Pa_sqrt_m,
            s_j_m=reference_patch_distance_m,
            r_contact_m=reference_contact_radius_m,
            cfg=cfg_template,
            n_phase=n_phase,
        )

    G_bond0 = _monotone_solve_decreasing(action_on, float(A_on_ref))
    G_off0 = _monotone_solve_decreasing(action_off, float(A_off_ref))
    return replace(cfg_template, bond_barrier_eV=G_bond0, rupture_barrier_eV=G_off0)


def freeze_reference_action_preset(
    name: str,
    A_on_ref: float,
    A_off_ref: float,
    *,
    cfg_template: CrackRebondingControls,
    reference_patch_distance_m: float,
    reference_contact_radius_m: float,
    T_K: float = 300.0,
    f_Hz: float = 1000.0,
    R: float = -0.95,
    Kmax_Pa_sqrt_m: float = 18.0e6,
) -> dict[str, Any]:
    resolved = solve_reference_action_barriers(
        A_on_ref,
        A_off_ref,
        T_K=T_K,
        f_Hz=f_Hz,
        R=R,
        Kmax_Pa_sqrt_m=Kmax_Pa_sqrt_m,
        reference_patch_distance_m=reference_patch_distance_m,
        reference_contact_radius_m=reference_contact_radius_m,
        cfg_template=cfg_template,
    ).validate()
    canonical = json.dumps(_jsonable(asdict(resolved)), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "preset_name": name,
        "A_on_ref": float(A_on_ref),
        "A_off_ref": float(A_off_ref),
        "parameters": asdict(resolved),
        "sha256": digest,
    }


__all__ = [
    "KB_EV_PER_K",
    "EV_TO_J",
    "CONTACT_SEMANTICS_LABEL",
    "RebondModelLevel",
    "ContactModel",
    "FeedbackMode",
    "InitialPrecrackWakeMode",
    "CrackRebondingControls",
    "crack_rebonding_config_from_environment",
    "cooperative_hazard",
    "bond_formation_rate",
    "bond_rupture_rate",
    "depassivation_rate",
    "repassivation_rate",
    "contact_pressure",
    "opening_stress",
    "build_Q",
    "advance_markov",
    "build_phase_factors",
    "propagate",
    "build_augmented_Q",
    "transition_actions_and_fluxes",
    "strang_cycle_trajectory",
    "two_state_fixed_point",
    "two_state_iterate",
    "REFERENCE_ACTION_PRESETS",
    "solve_reference_action_barriers",
    "freeze_reference_action_preset",
]
