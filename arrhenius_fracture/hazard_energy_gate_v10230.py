"""Hazard-derived thermodynamic crack-extension gate for v10.2.30.

The Arrhenius cleavage clock remains the only event-initiation criterion.  Once
hazard action accumulates, the stochastic event reward is realized only to the
extent supported by positive configurational work.  No independent athermal
``Gc0`` is introduced.

For the selected cleavage direction,

    Gamma_haz(T, sigma, theta)
        = gamma_rel(theta) * m * DeltaG_cleave*(T, sigma) / b**2.

``DeltaG_cleave*`` is the same effective activation free energy returned by the
production cleavage hazard, ``m`` is the active cooperative-hit count, ``b`` is
the production Burgers vector, and ``gamma_rel`` is the existing cubic relative
plane factor.  The current directional-J probe supplies the trial elastic energy.
For fixed-DeltaK fatigue, the event-level release is scaled from the nonzero FEM
probe field by ``(K_event/K_probe)**2``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import EV_TO_J
from .fatigue_v1 import CycleHazardResult
from .persistent_site_cyclic_audited_v10229 import (
    AuditedPersistentSiteCyclicTipEngine,
)
from .stochastic_hazard_tip import StochasticHazardDiagnosticTipEngine
from . import stochastic_avalanche_tip as _avalanche

MODEL_ID = "v10.2.30_hazard_derived_energy_gated_extension"
SCHEMA = "v10.2.30_hazard_energy_gate_v1"


@dataclass(frozen=True)
class HazardEnergyGateContext:
    """Outer-mechanics state required to gate one front's crack reward."""

    J_probe_J_per_m2: float
    K_probe_Pa_sqrt_m: float
    K_event_Pa_sqrt_m: float
    gamma_rel: float
    loading_mode: str
    probe_source: str = "directional_J"

    def validate(self) -> "HazardEnergyGateContext":
        values = {
            "J_probe_J_per_m2": self.J_probe_J_per_m2,
            "K_probe_Pa_sqrt_m": self.K_probe_Pa_sqrt_m,
            "K_event_Pa_sqrt_m": self.K_event_Pa_sqrt_m,
            "gamma_rel": self.gamma_rel,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.J_probe_J_per_m2 < 0.0:
            raise ValueError("J_probe_J_per_m2 must be nonnegative")
        if self.K_probe_Pa_sqrt_m < 0.0 or self.K_event_Pa_sqrt_m < 0.0:
            raise ValueError("K probe and event values must be nonnegative")
        if self.gamma_rel <= 0.0:
            raise ValueError("gamma_rel must be positive")
        mode = str(self.loading_mode).strip().lower()
        if mode not in {"monotonic", "cyclic"}:
            raise ValueError("loading_mode must be monotonic or cyclic")
        return self


def canonical_gamma_rel(candidate: Any) -> float:
    """Return one positive relative plane factor for either selector contract."""
    if isinstance(candidate, dict):
        raw = candidate.get("gamma_rel", candidate.get("gamma", 1.0))
    else:
        raw = 1.0
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("selected cleavage direction has invalid relative plane factor")
    if isinstance(candidate, dict):
        candidate["gamma_rel"] = value
    return value


def hazard_K_from_event_K(K_event_Pa_sqrt_m: float, gamma_rel: float) -> float:
    """Apply the existing Griffith angular dependence to the hazard drive."""
    gamma = max(float(gamma_rel), 1.0e-300)
    return max(float(K_event_Pa_sqrt_m), 0.0) / math.sqrt(gamma)


def probe_to_event_energy_scale(
    K_event_Pa_sqrt_m: float,
    K_probe_Pa_sqrt_m: float,
) -> float:
    """Scale linear-elastic probe energy to the physical event amplitude."""
    event = max(float(K_event_Pa_sqrt_m), 0.0)
    probe = max(float(K_probe_Pa_sqrt_m), 0.0)
    if event <= 0.0:
        return 0.0
    if probe <= 0.0:
        raise RuntimeError(
            "nonzero event K cannot be energy-scaled from a zero FEM probe K"
        )
    ratio = event / probe
    scale = ratio * ratio
    if not math.isfinite(scale):
        raise RuntimeError("nonfinite probe-to-event elastic-energy scale")
    return scale


def hazard_dissipation_density_J_per_m2(
    engine,
    T_K: float,
    sigma_cleave_eff_Pa: float,
    gamma_rel: float,
) -> tuple[float, float]:
    """Return ``(Gamma_haz, DeltaG_cleave*)`` using the active hazard surface."""
    _lam, _raw, deltaG_J = engine.lambda_cleave(
        max(float(sigma_cleave_eff_Pa), 0.0),
        float(T_K),
    )
    deltaG = max(float(deltaG_J), 0.0)
    m_hits = max(float(engine.f.m_hits), 1.0)
    b = abs(float(engine.b))
    if not math.isfinite(b) or b <= 0.0:
        raise RuntimeError("hazard energy gate requires a positive Burgers vector")
    gamma = canonical_gamma_rel({"gamma_rel": gamma_rel})
    Gamma = gamma * m_hits * deltaG / (b * b)
    if not math.isfinite(Gamma) or Gamma < 0.0:
        raise RuntimeError("hazard-derived dissipation density is invalid")
    return Gamma, deltaG


def gate_fraction_from_context(
    engine,
    context: HazardEnergyGateContext,
    T_K: float,
    sigma_cleave_eff_Pa: float,
) -> dict[str, float | str]:
    """Return a differential thermodynamic gate for the current FEM state."""
    ctx = context.validate()
    scale = probe_to_event_energy_scale(
        ctx.K_event_Pa_sqrt_m,
        ctx.K_probe_Pa_sqrt_m,
    )
    J_event = max(float(ctx.J_probe_J_per_m2), 0.0) * scale
    Gamma, deltaG = hazard_dissipation_density_J_per_m2(
        engine,
        T_K,
        sigma_cleave_eff_Pa,
        ctx.gamma_rel,
    )
    if Gamma <= 0.0:
        fraction = 1.0 if J_event > 0.0 else 0.0
    else:
        fraction = min(max(J_event / Gamma, 0.0), 1.0)
    return {
        "schema": SCHEMA,
        "loading_mode": str(ctx.loading_mode),
        "probe_source": str(ctx.probe_source),
        "J_probe_J_per_m2": float(ctx.J_probe_J_per_m2),
        "K_probe_Pa_sqrt_m": float(ctx.K_probe_Pa_sqrt_m),
        "K_event_Pa_sqrt_m": float(ctx.K_event_Pa_sqrt_m),
        "probe_to_event_energy_scale": float(scale),
        "J_event_scaled_J_per_m2": float(J_event),
        "gamma_rel": float(ctx.gamma_rel),
        "cooperative_hit_count": float(engine.f.m_hits),
        "burgers_vector_m": abs(float(engine.b)),
        "DeltaG_cleave_eff_J": float(deltaG),
        "DeltaG_cleave_eff_eV": float(deltaG / EV_TO_J),
        "Gamma_haz_J_per_m2": float(Gamma),
        "gate_fraction": float(fraction),
        "sigma_cleave_eff_Pa": max(float(sigma_cleave_eff_Pa), 0.0),
    }


class HazardEnergyGatedPersistentSiteCyclicTipEngine(
    AuditedPersistentSiteCyclicTipEngine
):
    """Shared monotonic/cyclic persistent-site engine with differential energy gate."""

    hazard_energy_gate_v10230 = True

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        payload["hazard_energy_gate_v10230"] = {
            "schema": SCHEMA,
            "model_id": MODEL_ID,
            "event_initiation": "Arrhenius_first_passage_only",
            "absolute_athermal_Gc": False,
            "dissipation_density": (
                "gamma_rel*m*DeltaG_cleave_eff(T,sigma)/b^2"
            ),
            "anisotropic_hazard_drive": "K_event/sqrt(gamma_rel)",
            "fixed_DeltaK_energy_scaling": "(K_event/K_probe)^2",
            "gate_application": "differential_event_reward_before_MPZ_translation",
            "positive_signed_J_required": True,
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hazard_energy_gate_context: HazardEnergyGateContext | None = None
        self.hazard_energy_gate_event_advance_accum_m = 0.0
        self.hazard_energy_gate_last: dict[str, Any] = {}
        self.hazard_energy_gate_event_history: list[dict[str, Any]] = []

    def clone_split(self, daughter_fraction=0.5):
        child = super().clone_split(daughter_fraction)
        child.hazard_energy_gate_context = copy.deepcopy(
            self.hazard_energy_gate_context
        )
        child.hazard_energy_gate_event_advance_accum_m = float(
            self.hazard_energy_gate_event_advance_accum_m
        )
        child.hazard_energy_gate_last = copy.deepcopy(self.hazard_energy_gate_last)
        child.hazard_energy_gate_event_history = copy.deepcopy(
            self.hazard_energy_gate_event_history
        )
        return child

    def set_hazard_energy_gate_context(
        self,
        *,
        J_probe_J_per_m2: float,
        K_probe_Pa_sqrt_m: float,
        K_event_Pa_sqrt_m: float,
        gamma_rel: float,
        loading_mode: str,
        probe_source: str = "directional_J",
    ) -> None:
        self.hazard_energy_gate_context = HazardEnergyGateContext(
            J_probe_J_per_m2=float(J_probe_J_per_m2),
            K_probe_Pa_sqrt_m=float(K_probe_Pa_sqrt_m),
            K_event_Pa_sqrt_m=float(K_event_Pa_sqrt_m),
            gamma_rel=float(gamma_rel),
            loading_mode=str(loading_mode),
            probe_source=str(probe_source),
        ).validate()

    def _require_gate_context(self) -> HazardEnergyGateContext:
        if self.hazard_energy_gate_context is None:
            raise RuntimeError(
                "v10.2.30 requires an outer FEM hazard-energy context before "
                "monotonic or cyclic hazard integration"
            )
        return self.hazard_energy_gate_context.validate()

    def _current_gate(
        self,
        T_K: float,
        K_hazard_Pa_sqrt_m: float,
        stress_override: float | None,
    ) -> dict[str, Any]:
        sigma = (
            max(float(stress_override), 0.0)
            if stress_override is not None
            else float(self.sigma_tip(max(float(K_hazard_Pa_sqrt_m), 0.0)))
        )
        return gate_fraction_from_context(
            self,
            self._require_gate_context(),
            T_K,
            sigma,
        )

    def _integrate_coupled(
        self,
        K: float,
        T: float,
        dt: float,
        stress_override: float | None = None,
        lambda_override: float | None = None,
    ) -> dict[str, Any]:
        """Gate each differential crack reward while preserving hazard integration."""
        self._synchronize_driver_checkpoint_length()
        proposed_event_length = max(float(self.avalanche_event_advance_m), 0.0)
        proposed_factor = max(float(self.avalanche_event_length_factor), 0.0)
        gate = self._current_gate(T, K, stress_override)
        gated_checkpoint = proposed_event_length * float(gate["gate_fraction"])

        base = float(self.f.da)
        self.f.da = gated_checkpoint
        try:
            result = StochasticHazardDiagnosticTipEngine._integrate_coupled(
                self,
                K,
                T,
                dt,
                stress_override=stress_override,
                lambda_override=lambda_override,
            )
        finally:
            self.f.da = base

        accepted_step = max(float(result.get("da", 0.0)), 0.0)
        self.hazard_energy_gate_event_advance_accum_m += accepted_step
        fired = bool(result.get("fired", False))
        accepted_event = 0.0

        if fired:
            accepted_event = max(
                float(self.hazard_energy_gate_event_advance_accum_m), 0.0
            )
            # The stochastic-hazard integrator updates event-level counters with the
            # checkpoint active on the final differential step. Correct those
            # counters to the integrated accepted reward over the whole event.
            correction = accepted_event - gated_checkpoint
            self.a_adv += correction
            self.checkpoint_advance_total_m += correction

            accepted_factor = accepted_event / max(
                float(self.avalanche_base_checkpoint_m), 1.0e-300
            )
            completed_threshold = float(
                result.get("hazard_threshold_completed_action", 0.0)
            )
            event_row = {
                **gate,
                "event_index": int(self.hazard_event_index - 1),
                "threshold_action": completed_threshold,
                "proposed_event_advance_m": proposed_event_length,
                "proposed_event_length_factor": proposed_factor,
                "accepted_event_advance_m": accepted_event,
                "accepted_event_length_factor": accepted_factor,
                "rejected_event_advance_m": max(
                    proposed_event_length - accepted_event, 0.0
                ),
                "energy_available_for_proposed_event_J_per_m": (
                    float(gate["J_event_scaled_J_per_m2"])
                    * proposed_event_length
                ),
                "energy_dissipated_by_accepted_event_J_per_m": (
                    float(gate["Gamma_haz_J_per_m2"]) * accepted_event
                ),
                "differential_gate_integrated_over_event": True,
            }
            self.hazard_energy_gate_last = dict(event_row)
            self.hazard_energy_gate_event_history.append(dict(event_row))

            self.avalanche_last_completed_advance_m = accepted_event
            self.avalanche_last_completed_factor = accepted_factor
            self.avalanche_event_length_history.append(accepted_event)
            _avalanche._PENDING_GEOMETRY_EVENTS.append(
                {
                    "event_advance_m": accepted_event,
                    "proposed_event_advance_m": proposed_event_length,
                    "event_length_factor": accepted_factor,
                    "proposed_event_length_factor": proposed_factor,
                    "threshold_action": completed_threshold,
                    "hazard_seed": int(self.hazard_cfg.seed),
                    "hazard_event_index": int(self.hazard_event_index - 1),
                    "geometry_subsegment_fraction": float(
                        self.avalanche_cfg.geometry_subsegment_fraction
                    ),
                    "hazard_energy_gate": dict(event_row),
                }
            )
            self.hazard_energy_gate_event_advance_accum_m = 0.0
            self._set_current_event_length()

        result.update(
            {
                **gate,
                "hazard_energy_gate_model_id": MODEL_ID,
                "hazard_energy_gate_active": True,
                "hazard_energy_gate_proposed_checkpoint_m": proposed_event_length,
                "hazard_energy_gate_gated_checkpoint_m": gated_checkpoint,
                "hazard_energy_gate_accepted_step_m": accepted_step,
                "hazard_energy_gate_event_accumulated_m": float(
                    self.hazard_energy_gate_event_advance_accum_m
                ),
                "hazard_energy_gate_accepted_event_m": accepted_event,
                "avalanche_event_advance_m": accepted_event if fired else 0.0,
                "avalanche_event_length_factor": (
                    accepted_event
                    / max(float(self.avalanche_base_checkpoint_m), 1.0e-300)
                    if fired
                    else 0.0
                ),
                "avalanche_current_event_advance_m": float(
                    self.avalanche_event_advance_m
                ),
                "avalanche_current_event_length_factor": float(
                    self.avalanche_event_length_factor
                ),
            }
        )
        return result

    def preview_cycle_waveform(self, controller, waveform, T_K: float) -> CycleHazardResult:
        """Engine-native cyclic preview using the same differential gate."""
        phase = np.asarray(controller._phases(), dtype=float)
        if phase.size < 1:
            raise ValueError("cyclic preview requires at least one waveform phase")
        Kvals = np.asarray(waveform.K_phase(phase), dtype=float)
        dt_phase = float(waveform.period_s) / float(phase.size)
        trial = copy.deepcopy(self)

        mobile0 = float(trial.mpz.mobile_count)
        retained0 = float(trial.mpz.retained_count)
        emitted = trapped = released = escaped = 0.0
        peierls_progress = taylor_progress = 0.0
        cleavage_action = 0.0
        sig_samples: list[float] = []
        emit_weighted_sig = 0.0
        emit_weight = 0.0
        elapsed_s = 0.0

        for kval in Kvals:
            K = max(float(kval), 0.0)
            half = 0.5 * dt_phase
            sig0 = float(trial.sigma_tip(K))
            first = trial._plastic_half_step(half, T_K, sig0)
            sig_mid = float(trial.sigma_tip(K))
            lam_mid, _raw_mid, _Gc_mid = trial.lambda_cleave(sig_mid, T_K)
            lam_mid = max(float(lam_mid), 0.0) if math.isfinite(lam_mid) else 0.0

            remaining_action = max(1.0 - float(trial.B), 0.0)
            dB_unclipped = lam_mid * dt_phase
            dB = min(dB_unclipped, remaining_action)
            gate = trial._current_gate(T_K, K, sig_mid)
            proposed = max(float(trial.avalanche_event_advance_m), 0.0)
            da = proposed * float(gate["gate_fraction"]) * dB
            if da > 0.0:
                trial.mpz.advance(da)
            trial.B += dB
            trial.micro_advance_total_m += da
            trial.hazard_energy_gate_event_advance_accum_m += da

            sig1 = float(trial.sigma_tip(K))
            second = trial._plastic_half_step(half, T_K, sig1)
            trial.t += dt_phase
            elapsed_s += dt_phase
            cleavage_action += dB_unclipped
            sig_samples.extend((sig0, sig_mid, sig1))

            for update in (first, second):
                emitted += float(update.get("dN_emit", 0.0))
                trapped += float(update.get("dN_trapped", 0.0))
                released += float(update.get("dN_released", 0.0))
                escaped += float(update.get("dN_escaped", 0.0))
                peierls_progress += half * max(
                    float(update.get("peierls_rate_s", 0.0)), 0.0
                )
                taylor_progress += half * max(
                    float(update.get("taylor_completion_rate_s", 0.0)), 0.0
                )
                weight = max(float(update.get("dN_emit", 0.0)), 0.0)
                sigma_eff = float(
                    getattr(
                        trial.mpz,
                        "continuum_source_last_sigma_emit_effective_Pa",
                        sig_mid,
                    )
                )
                emit_weighted_sig += weight * max(sigma_eff, 0.0)
                emit_weight += weight

            if trial.B >= 1.0 - 1.0e-12:
                break

        elapsed_cycles = max(elapsed_s * float(waveform.frequency_Hz), 1.0e-300)
        scale = 1.0 / elapsed_cycles
        mobile_delta = abs(float(trial.mpz.mobile_count) - mobile0) * scale
        retained_delta = abs(float(trial.mpz.retained_count) - retained0) * scale
        emitted *= scale
        trapped *= scale
        released *= scale
        escaped *= scale
        peierls_progress *= scale
        taylor_progress *= scale
        cleavage_action *= scale

        stored_per_cycle = max(retained_delta, trapped)
        mobile_per_cycle = max(mobile_delta, emitted + released - trapped)
        avg_sigma = float(np.mean(sig_samples)) if sig_samples else 0.0
        max_sigma = float(np.max(sig_samples)) if sig_samples else 0.0
        avg_emit_sigma = (
            emit_weighted_sig / emit_weight if emit_weight > 0.0 else avg_sigma
        )
        storage_fraction = (
            min(max(stored_per_cycle / emitted, 0.0), 1.0)
            if emitted > 0.0
            else 0.0
        )

        return CycleHazardResult(
            mu_emit=max(emitted, 0.0),
            mu_peierls=max(peierls_progress, 0.0),
            mu_taylor=max(taylor_progress, 0.0),
            mu_escape=max(escaped, 0.0),
            mu_cleave=max(cleavage_action, 0.0),
            store_per_cycle=max(stored_per_cycle, 0.0),
            mobile_per_cycle=max(mobile_per_cycle, 0.0),
            escape_per_cycle=max(escaped, 0.0),
            peierls_per_cycle=max(peierls_progress, 0.0),
            taylor_per_cycle=max(taylor_progress, 0.0),
            avg_sigma_tip=avg_sigma,
            max_sigma_tip=max_sigma,
            avg_sigma_emit_eff=max(float(avg_emit_sigma), 0.0),
            storage_fraction=storage_fraction,
        )

    def _gate_diagnostics(self) -> dict[str, Any]:
        context = self.hazard_energy_gate_context
        return {
            "hazard_energy_gate_schema": SCHEMA,
            "hazard_energy_gate_model_id": MODEL_ID,
            "hazard_energy_gate_active": True,
            "hazard_energy_gate_context": (
                None if context is None else asdict(context)
            ),
            "hazard_energy_gate_event_accumulated_m": float(
                self.hazard_energy_gate_event_advance_accum_m
            ),
            "hazard_energy_gate_completed_event_count": len(
                self.hazard_energy_gate_event_history
            ),
            "hazard_energy_gate_last": dict(self.hazard_energy_gate_last),
        }

    def step(self, K, T, dt):
        result = super().step(K, T, dt)
        result.update(self._gate_diagnostics())
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(self._gate_diagnostics())
        return result

    def cycle_step_waveform(
        self, controller, waveform, T_K: float, requested_cycles=None, force_cycles=None
    ):
        result = super().cycle_step_waveform(
            controller,
            waveform,
            T_K,
            requested_cycles=requested_cycles,
            force_cycles=force_cycles,
        )
        result.update(self._gate_diagnostics())
        return result


__all__ = [
    "MODEL_ID",
    "SCHEMA",
    "HazardEnergyGateContext",
    "HazardEnergyGatedPersistentSiteCyclicTipEngine",
    "canonical_gamma_rel",
    "gate_fraction_from_context",
    "hazard_K_from_event_K",
    "hazard_dissipation_density_J_per_m2",
    "probe_to_event_energy_scale",
]
