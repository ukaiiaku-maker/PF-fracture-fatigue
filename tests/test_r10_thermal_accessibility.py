import math

from arrhenius_fracture.material_manifest import ExpFloorBarrier
from arrhenius_fracture.thermal_accessibility_r10 import (
    maximum_floor_for_mark, opening_floor_bound,
)
from arrhenius_fracture.unified_fracture_material_v5 import (
    complete_row_bundle, material_bundle, material_manifest,
)


def audited_descendant_surface():
    # Exact committed R10J DBTT opening row.
    return ExpFloorBarrier(
        G00_eV=1.8934430984322184, gT_eV_per_K=0.0063046293771876335,
        sigc0_Pa=1.5290692945431272e9, sT_Pa_per_K=0.006060769372547189e9,
        alpha=0.24187734544293854, exponent=0.25,
        floor_fraction=0.48726421513082346, Tref_K=481.33,
        attempt_frequency_s=1.0e12,
    )


def test_uploaded_hot_censor_bounds_are_reproduced():
    surface = audited_descendant_surface()
    expected = {
        750.0: (1.747966755268016, 9.6469333323e-13, 1.5435093332e-8),
        1200.0: (3.130375883592746, 6.0373798552e-17, 9.6598077683e-13),
    }
    for temperature, (floor, rate, action) in expected.items():
        result = opening_floor_bound(
            surface, temperature_K=temperature, hits=3.0, tau_s=1.0e-6,
            exposure_s=16000.0, pending_mark=1.0,
        )
        assert math.isclose(result.evaluated_floor_eV, floor, rel_tol=2e-13)
        assert math.isclose(result.cooperative_rate_ceiling_s, rate, rel_tol=2e-10)
        assert math.isclose(result.action_ceiling, action, rel_tol=2e-10)
        assert result.necessarily_inaccessible


def test_unit_mark_floor_thresholds_are_reproduced():
    values = {
        temperature: maximum_floor_for_mark(
            temperature_K=temperature, attempt_frequency_s=1.0e12,
            hits=3.0, tau_s=1.0e-6, exposure_s=16000.0,
            pending_mark=1.0,
        )
        for temperature in (750.0, 1200.0)
    }
    assert math.isclose(values[750.0], 1.3604633379, rel_tol=2e-10)
    assert math.isclose(values[1200.0], 2.1767413406, rel_tol=2e-10)


def test_inapplicable_channel_is_not_declared_excluded():
    result = opening_floor_bound(
        audited_descendant_surface(), temperature_K=1200.0, hits=3.0,
        tau_s=1.0e-6, exposure_s=16000.0, pending_mark=1.0,
        applicable=False,
    )
    assert result.action_ceiling < result.pending_mark
    assert not result.necessarily_inaccessible


def test_complete_row_spatial_bundle_preserves_exact_fracture_surface():
    pinned = material_bundle("DBTT")
    rebound = complete_row_bundle(pinned.fracture_row)
    assert rebound.fracture_material_row_id == pinned.fracture_material_row_id
    assert rebound.void_kinetics_row_id == pinned.void_kinetics_row_id
    assert rebound.elastic_row_id == pinned.elastic_row_id
    assert material_manifest(rebound) == material_manifest(pinned)
