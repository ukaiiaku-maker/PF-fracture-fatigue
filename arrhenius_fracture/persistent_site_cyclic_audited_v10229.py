"""Audit adapter for the v10.2.29 persistent-site cyclic engine."""
from __future__ import annotations

from .persistent_site_cyclic_v10229 import PersistentSiteCyclicTipEngine


def _add_persistent_fields(engine, result: dict) -> dict:
    sigma_back = float(
        getattr(engine.mpz, "continuum_source_last_sigma_back_Pa", 0.0)
    )
    aggregate_hazard = float(
        getattr(engine.mpz, "continuum_source_last_aggregate_hazard_s", 0.0)
    )
    geometry = dict(getattr(engine.mpz, "persistent_site_last_geometry", {}))
    result["sigma_back"] = sigma_back
    result["sigma_back_pre_renewal"] = sigma_back
    result["lambda_e"] = aggregate_hazard
    result["persistent_site_aggregate_hazard_s"] = aggregate_hazard
    result["persistent_site_multiplicity_per_system"] = float(
        geometry.get("multiplicity_per_system", 0.0)
    )
    result["persistent_site_front_width_m"] = float(
        geometry.get("front_width_m", 0.0)
    )
    result["persistent_site_source_area_m2"] = float(
        geometry.get("source_area_m2", 0.0)
    )
    result["persistent_tip_radius_m"] = float(
        geometry.get("tip_radius_m", engine.r_eff())
    )
    result["persistent_source_inventory_active"] = False
    result["persistent_source_refresh_active"] = False
    result["explicit_recovery_active"] = False
    return result


class AuditedPersistentSiteCyclicTipEngine(PersistentSiteCyclicTipEngine):
    """Expose the same persistent-source audit fields for monotonic and cyclic calls."""

    def step(self, K, T, dt):
        return _add_persistent_fields(self, super().step(K, T, dt))

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
        return _add_persistent_fields(self, result)


__all__ = ["AuditedPersistentSiteCyclicTipEngine"]
