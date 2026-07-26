from types import SimpleNamespace

from arrhenius_fracture.persistent_site_cyclic_audited_v10229 import (
    _add_persistent_fields,
)


class FakeEngine:
    def __init__(self):
        self.mpz = SimpleNamespace(
            continuum_source_last_sigma_back_Pa=2.0,
            continuum_source_last_aggregate_hazard_s=3.0,
            persistent_site_last_geometry={
                "multiplicity_per_system": 4.0,
                "front_width_m": 5.0,
                "source_area_m2": 6.0,
                "tip_radius_m": 7.0,
            },
        )

    def r_eff(self):
        return 8.0


def test_cyclic_audit_uses_real_persistent_source_fields():
    result = _add_persistent_fields(FakeEngine(), {})
    assert result["sigma_back"] == 2.0
    assert result["lambda_e"] == 3.0
    assert result["persistent_site_multiplicity_per_system"] == 4.0
    assert result["persistent_site_front_width_m"] == 5.0
    assert result["persistent_site_source_area_m2"] == 6.0
    assert result["persistent_tip_radius_m"] == 7.0
    assert result["persistent_source_inventory_active"] is False
    assert result["persistent_source_refresh_active"] is False
    assert result["explicit_recovery_active"] is False
