"""v10.2 fatigue dispatch for the moving anisotropic process-zone engine.

The legacy sharp-front driver still owns cycle-block selection, cyclic FEM hooks,
spatial process-zone deposition, branching, and geometry commits.  This module
changes only the front-level fatigue calls: v10 moving-tip engines use their
native ``cycle_step_waveform`` method instead of the obsolete scalar-ledger
``FatigueCycleHazardController.cycle_step_front`` implementation.

The predictor uses the promoted material manifest, the current finite campaign
source budget, the current channel factors, and the current Taylor back stress.
No Paris law or direct da/dN relation is introduced.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .campaign_calibrated_tip import _campaign_backstress
from .fatigue_v1 import FatigueCycleHazardController


MODEL_ID = "v10.2.0_moving_mpz_fatigue_reintegration"


def is_v10_moving_engine(front: Any) -> bool:
    """Return True for the current moving-process-zone front engines."""
    return bool(
        getattr(front, "kinetic_tip_cell_active", False)
        and hasattr(front, "mpz")
        and callable(getattr(front, "cycle_step_waveform", None))
    )


def _system_array(value: Any, n_systems: int, default: float) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        array = np.full(n_systems, float(default), dtype=float)
    elif array.size < n_systems:
        array = np.pad(array, (0, n_systems - array.size), mode="edge")
    return array[:n_systems].astype(float, copy=False)


def predict_one_cycle_v1020(
    front: Any,
    waveform: Any,
    T_K: float,
    controller: FatigueCycleHazardController,
) -> SimpleNamespace:
    """Predict one cycle from the actual v10 MPZ state without mutating it.

    The FEM tensor probe supplies dimensionless channel factors at Kmax. Linear
    elasticity makes those factors the shape multipliers for the full waveform;
    the absolute opening stress is recomputed at every phase. The prediction is
    intentionally conservative for adaptive block sizing: every emitted event is
    treated as potentially mobile and retained. The committed moving-MPZ update
    then resolves transport, trapping, recovery, shielding, and source refresh.
    """
    phase = np.asarray(controller._phases(), dtype=float)
    if phase.size < 2:
        raise RuntimeError("fatigue quadrature requires at least two phase points")
    K_values = np.asarray(waveform.K_phase(phase), dtype=float)
    dt_phase = float(waveform.period_s) / float(phase.size)
    sigma_open = np.asarray(
        [max(float(front.sigma_tip(float(K))), 0.0) for K in K_values],
        dtype=float,
    )

    lambda_c = np.asarray(
        [max(float(front.lambda_cleave(float(sig), T_K)[0]), 0.0) for sig in sigma_open],
        dtype=float,
    )
    mu_cleave = float(np.sum(lambda_c) * dt_phase)

    mpz = front.mpz
    n_systems = int(getattr(mpz, "n_systems", 1))
    factors = _system_array(
        getattr(mpz, "_anisotropic_drive_factors", np.ones(n_systems)),
        n_systems,
        1.0,
    )
    factors = np.maximum(factors, 0.0)
    try:
        _rho, _tau_back, sigma_back = _campaign_backstress(mpz)
        sigma_back = _system_array(sigma_back, n_systems, 0.0)
    except Exception:
        sigma_back = np.zeros(n_systems, dtype=float)

    available = _system_array(
        getattr(mpz, "available_sites", np.ones(n_systems)),
        n_systems,
        0.0,
    )
    available = np.maximum(available, 0.0)

    emission_rate_phase = np.zeros((phase.size, n_systems), dtype=float)
    sigma_emit_phase = np.zeros_like(emission_rate_phase)
    for index, opening in enumerate(sigma_open):
        sigma_emit = np.maximum(factors * opening - sigma_back, 0.0)
        sigma_emit_phase[index] = sigma_emit
        emission_rate_phase[index] = np.asarray(
            [
                max(float(mpz.emission_rate_per_site(float(sig), T_K)), 0.0)
                for sig in sigma_emit
            ],
            dtype=float,
        )

    phase_event_rate = emission_rate_phase @ available
    mu_emit = float(np.sum(phase_event_rate) * dt_phase)
    total_weight = float(np.sum(emission_rate_phase * available[None, :]))
    avg_sigma_emit = (
        float(np.sum(sigma_emit_phase * emission_rate_phase * available[None, :]) / total_weight)
        if total_weight > 0.0
        else float(np.mean(sigma_emit_phase))
    )

    return SimpleNamespace(
        mu_emit=mu_emit,
        mu_peierls=0.0,
        mu_taylor=0.0,
        mu_escape=0.0,
        mu_cleave=mu_cleave,
        store_per_cycle=mu_emit,
        mobile_per_cycle=mu_emit,
        escape_per_cycle=0.0,
        peierls_per_cycle=0.0,
        taylor_per_cycle=0.0,
        storage_fraction=1.0 if mu_emit > 0.0 else 0.0,
        avg_sigma_tip=float(np.mean(sigma_open)),
        max_sigma_tip=float(np.max(sigma_open)),
        avg_sigma_emit_eff=avg_sigma_emit,
    )


def _state_snapshot(front: Any) -> dict[str, float]:
    mpz = front.mpz
    return {
        "mobile": float(getattr(mpz, "mobile_count", 0.0)),
        "retained": float(getattr(mpz, "retained_count", 0.0)),
        "emitted": float(getattr(mpz, "emitted_total", 0.0)),
        "escaped": float(getattr(mpz, "escaped_total", 0.0)),
        "recovered": float(getattr(mpz, "recovered_total", 0.0)),
    }


def _positive_delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return max(float(after[key]) - float(before[key]), 0.0)


def enrich_cycle_result_v1020(
    front: Any,
    waveform: Any,
    T_K: float,
    result: dict[str, Any],
    before: dict[str, float],
) -> dict[str, Any]:
    """Add legacy-driver aliases while retaining the native MPZ diagnostics."""
    out = dict(result)
    after = _state_snapshot(front)
    cycles = max(float(out.get("cycles", 0.0)), 0.0)
    d_emit = _positive_delta(after, before, "emitted")
    d_escape = _positive_delta(after, before, "escaped")
    d_recover = _positive_delta(after, before, "recovered")
    d_store = max(after["retained"] - before["retained"], 0.0)
    d_mobile = max(after["mobile"] - before["mobile"], 0.0)

    sigma_peak = max(float(front.sigma_tip(float(waveform.Kmax))), 0.0)
    cleavage = dict(front.cleavage_diagnostics(sigma_peak, T_K))
    out.update(cleavage)
    out.update(
        {
            "fatigue_reintegration_model_id": MODEL_ID,
            "fatigue_native_moving_mpz_dispatch": True,
            "fatigue_stochastic_engine_active": bool(
                getattr(front, "stochastic_hazard_threshold_active", False)
            ),
            "fatigue_stochastic_event_length_active": bool(
                getattr(front, "stochastic_avalanche_length_active", False)
            ),
            "sigma_tip": sigma_peak,
            "sigma_back": 0.0,
            "r_eff": float(front.r_eff()),
            "B": float(front.B),
            "N_em": float(front.N_em),
            "dB_block": float(out.get("dB", out.get("dB_block", 0.0))),
            "dN_emit_block": d_emit,
            "dN_store_block": d_store,
            "dN_mobile_block": d_mobile,
            "dN_escape_block": d_escape,
            "dN_recover_block": d_recover,
            "dN_peierls_block": float(out.get("dN_peierls_block", 0.0)),
            "dN_taylor_block": float(out.get("dN_taylor_block", 0.0)),
            "storage_fraction": d_store / d_emit if d_emit > 0.0 else 0.0,
            "mu_emit": d_emit / cycles if cycles > 0.0 else 0.0,
            "mu_escape": d_escape / cycles if cycles > 0.0 else 0.0,
            "mu_cleave_pred": float(out.get("mu_cleave_pred", 0.0)),
            "N_em_pre_renewal": float(out.get("N_em_pre_renewal", before["retained"])),
            "N_em_retained": float(front.N_em),
            "N_em_shed_to_wake": float(out.get("N_em_shed_to_wake", 0.0)),
            "r_eff_pre_renewal": float(out.get("r_eff_pre_renewal", front.r_eff())),
            "sigma_back_pre_renewal": float(out.get("sigma_back_pre_renewal", 0.0)),
            "dG_emb_pre_renewal_eV": float(out.get("dG_emb_pre_renewal_eV", 0.0)),
        }
    )

    fired = bool(out.get("fired", False))
    if fired:
        out["avalanche_event_advance_m"] = float(
            getattr(front, "avalanche_last_completed_advance_m", 0.0)
        )
        out["avalanche_event_length_factor"] = float(
            getattr(front, "avalanche_last_completed_factor", 0.0)
        )
    out["avalanche_current_event_advance_m"] = float(
        getattr(front, "avalanche_event_advance_m", 0.0)
    )
    out["avalanche_current_event_length_factor"] = float(
        getattr(front, "avalanche_event_length_factor", 1.0)
    )

    audit = getattr(type(front), "_audit_records", None)
    if isinstance(audit, list):
        audit.append(
            {
                "engine_id": int(getattr(front, "_engine_id", -1)),
                "fatigue_reintegration_model_id": MODEL_ID,
                "fatigue_native_moving_mpz_dispatch": True,
                "fatigue_cycles": cycles,
                "K_Pa_sqrt_m": float(waveform.Kmax),
                "DeltaK_Pa_sqrt_m": float(waveform.DeltaK),
                "R": float(waveform.R),
                "frequency_Hz": float(waveform.frequency_Hz),
                "fired": fired,
                "dB": float(out.get("dB_block", 0.0)),
                "dN_emit": d_emit,
                "active_mobile": after["mobile"],
                "active_retained": after["retained"],
                "micro_advance_total_m": float(
                    getattr(front, "micro_advance_total_m", 0.0)
                ),
                "avalanche_event_advance_m": float(
                    out.get("avalanche_event_advance_m", 0.0)
                ),
                "avalanche_event_length_factor": float(
                    out.get("avalanche_event_length_factor", 0.0)
                ),
                "hazard_threshold_next_action": float(
                    getattr(front, "hazard_threshold_action", 1.0)
                ),
            }
        )
    return out


def make_integrate_dispatch(original: Callable) -> Callable:
    def integrate(self, front, waveform, T_K):
        if is_v10_moving_engine(front):
            return predict_one_cycle_v1020(front, waveform, T_K, self)
        return original(self, front, waveform, T_K)

    integrate.__name__ = getattr(original, "__name__", "integrate_one_cycle")
    integrate.__doc__ = getattr(original, "__doc__", None)
    return integrate


def make_cycle_step_dispatch(original: Callable) -> Callable:
    def cycle_step(
        self,
        front,
        waveform,
        T_K,
        requested_cycles=None,
        force_cycles=None,
    ):
        if not is_v10_moving_engine(front):
            return original(
                self,
                front,
                waveform,
                T_K,
                requested_cycles=requested_cycles,
                force_cycles=force_cycles,
            )
        before = _state_snapshot(front)
        result = front.cycle_step_waveform(
            self,
            waveform,
            T_K,
            requested_cycles=requested_cycles,
            force_cycles=force_cycles,
        )
        return enrich_cycle_result_v1020(front, waveform, T_K, result, before)

    cycle_step.__name__ = getattr(original, "__name__", "cycle_step_front")
    cycle_step.__doc__ = getattr(original, "__doc__", None)
    return cycle_step


@contextmanager
def install_v1020_fatigue_dispatch():
    """Temporarily route fatigue controller calls into the current MPZ engine."""
    original_integrate = FatigueCycleHazardController.integrate_one_cycle
    original_step = FatigueCycleHazardController.cycle_step_front
    FatigueCycleHazardController.integrate_one_cycle = make_integrate_dispatch(
        original_integrate
    )
    FatigueCycleHazardController.cycle_step_front = make_cycle_step_dispatch(
        original_step
    )
    try:
        yield
    finally:
        FatigueCycleHazardController.integrate_one_cycle = original_integrate
        FatigueCycleHazardController.cycle_step_front = original_step


__all__ = [
    "MODEL_ID",
    "enrich_cycle_result_v1020",
    "install_v1020_fatigue_dispatch",
    "is_v10_moving_engine",
    "make_cycle_step_dispatch",
    "make_integrate_dispatch",
    "predict_one_cycle_v1020",
]
