"""Step-output audit adapter for the v10.2.21 persistent-site engine."""
from __future__ import annotations

from .persistent_site_source_v10221 import PersistentSiteStateResolvedTipEngine


class AuditedPersistentSiteStateResolvedTipEngine(
    PersistentSiteStateResolvedTipEngine
):
    """Expose the real source backstress and aggregate hazard in standard outputs."""

    def step(self, K, T, dt):
        result = super().step(K, T, dt)
        sigma_back = float(
            getattr(self.mpz, "continuum_source_last_sigma_back_Pa", 0.0)
        )
        aggregate_hazard = float(
            getattr(self.mpz, "continuum_source_last_aggregate_hazard_s", 0.0)
        )
        geometry = dict(getattr(self.mpz, "persistent_site_last_geometry", {}))
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
            geometry.get("tip_radius_m", self.r_eff())
        )
        result["persistent_source_inventory_active"] = False
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(
                {
                    "persistent_sigma_back_Pa": sigma_back,
                    "persistent_aggregate_emission_hazard_s": aggregate_hazard,
                    "persistent_site_multiplicity_per_system": result[
                        "persistent_site_multiplicity_per_system"
                    ],
                    "persistent_site_front_width_m": result[
                        "persistent_site_front_width_m"
                    ],
                    "persistent_site_source_area_m2": result[
                        "persistent_site_source_area_m2"
                    ],
                    "persistent_tip_radius_m": result["persistent_tip_radius_m"],
                    "persistent_source_inventory_active": False,
                }
            )
        return result


__all__ = ["AuditedPersistentSiteStateResolvedTipEngine"]
