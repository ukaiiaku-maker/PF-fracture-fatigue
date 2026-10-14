import numpy as np
import pytest

from arrhenius_fracture.bilinear_opening_fit import (
    bilinear_barrier_and_volume, fit_temperature_slice,
)
from arrhenius_fracture.material_manifest import ExpFloorBarrier


def _surface():
    return ExpFloorBarrier(
        G00_eV=2.0, gT_eV_per_K=2.0e-4,
        sigc0_Pa=5.0e9, sT_Pa_per_K=-1.0e6,
        alpha=0.7, exponent=1.4, floor_fraction=0.02,
        attempt_frequency_s=1.0e12,
    )


def test_bilinear_barrier_is_defined_only_below_instability():
    barrier, volume = bilinear_barrier_and_volume(
        np.array([0.0, 1.0e9]), 2.0, 4.0e9
    )
    np.testing.assert_allclose(barrier, [2.0, 1.125])
    assert np.all(volume > 0.0)
    with pytest.raises(ValueError, match="q<tc"):
        bilinear_barrier_and_volume(np.array([4.0e9]), 2.0, 4.0e9)


def test_fit_is_positive_deterministic_and_keeps_support_unidentified():
    first = fit_temperature_slice(_surface(), 800.0)
    second = fit_temperature_slice(_surface(), 800.0)
    assert first["B0_A_times_Gamma_eV"] > 0.0
    assert first["peak_traction_Pa"] > first["fit_window_stress_Pa"][1]
    assert first["B0_A_times_Gamma_eV"] == pytest.approx(
        second["B0_A_times_Gamma_eV"], rel=1.0e-12
    )
    assert first["peak_traction_Pa"] == pytest.approx(
        second["peak_traction_Pa"], rel=1.0e-12
    )
    assert first["activation_area_m2"] is None
    assert first["separation_work_J_m2"] is None
    assert first["final_separation_m"] is None
    assert first["multistart_count"] == 12
