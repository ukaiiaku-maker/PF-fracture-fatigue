import pytest

from arrhenius_fracture.current_coupled_execution_r4 import (
    CurrentPersistentFractureAdapter, execute_current_coupled_r4,
)
from arrhenius_fracture.matched_persistent_execution_r3 import spatial_rows


def test_current_adapter_rejects_stale_complete_trial():
    adapter = CurrentPersistentFractureAdapter()
    row = spatial_rows()[0]
    kwargs = {
        "applied_K_Pa_sqrt_m": row["KJ_Pa_sqrtm"],
        "temperature_K": 900.0,
        "duration_s": row["dt_cur_s"],
    }
    first = adapter.preview(**kwargs)
    second = adapter.preview(**kwargs)
    adapter.commit(first)
    with pytest.raises(RuntimeError, match="stale"):
        adapter.commit(second)


def test_r4_entry_import_preserves_protected_continuum_alias():
    from arrhenius_fracture import continuum_source_tip
    from arrhenius_fracture.separated_source_tip import (
        SeparatedSourceKineticTipEngine,
    )
    from arrhenius_fracture.sharp_front_r4_current_law import (
        CurrentLawSpatialTipEngine,
    )

    assert CurrentLawSpatialTipEngine.__name__ == "CurrentLawSpatialTipEngine"
    assert continuum_source_tip.ContinuumSourceKineticTipEngine is (
        SeparatedSourceKineticTipEngine
    )


def test_current_backend_active_fixed_void_coupling(tmp_path):
    result = execute_current_coupled_r4(tmp_path / "r4")
    assert result["passed"]
    assert result["direct_adapter_void_off_parity"]["all_bitwise_equal"]
    assert result["direct_adapter_void_off_parity"]["event_bearing"]
    assert result["qualification"]["native_fracture_event_observed"]
    assert result["qualification"]["downstream_interval_with_cavity_owned"]
    assert result["qualification"]["coupled_restart_exact"]
    assert result["mechanics_binding"]["K_energy_equivalent_ratio"] != 1.0
    assert not result["qualification"]["moving_void_reduced_map_qualified"]
