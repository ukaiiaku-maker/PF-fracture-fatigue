"""Signed pile-up tip interaction: same-sign content shields, only mixed sign is forest."""
import os
from pathlib import Path

import numpy as np
import pytest

NATIVE = Path(os.environ.get("AST2018_NATIVE_SOURCE", "/nonexistent"))
pytestmark = pytest.mark.skipif(not NATIVE.exists(), reason="pinned native source checkout required")


def _states():
    from experimental_fits.ast2018.single_condition import Evaluator
    from experimental_fits.ast2018.single_condition_search import SOURCE, DATA
    from experimental_fits.ast2018.spatial import _mechanical_state_type
    ev = Evaluator(SOURCE, DATA, 'floor', protocol='measured_extension_stop_v2')
    make = lambda mode: _mechanical_state_type(ev.spatial.EmergentGNDState, ev.specimen, mode)(
        ev.material, ev.physics)
    return make("native"), make("signed_pileup"), ev


def test_same_sign_emissions_shield_but_are_not_forest():
    # Regression: emitted same-sign dislocations formed a Taylor forest whose
    # uncapped amplification released every captured line at once, while
    # mobile content never shielded the tip (K_shield ~1e-11 at 250 C).
    native, pileup, _ = _states()
    floor = pileup.forest_density_m2().copy()
    for state in (native, pileup):
        state.mobile_m2[0, 1, :5] = 1e14
    assert native.K_shield_MPa_sqrt_m() == 0.0
    assert np.all(native.forest_density_m2()[:, :5] > floor[:, :5])
    assert pileup.K_shield_MPa_sqrt_m() > 0.0
    assert np.allclose(pileup.signed_gnd_m2()[0, :5], 1e14)
    assert np.array_equal(pileup.forest_density_m2(), floor)


def test_only_mixed_sign_content_is_forest():
    native, pileup, _ = _states()
    floor = pileup.forest_density_m2().copy()
    pileup.mobile_m2[0, 1, :5] = 1e14
    pileup.retained_m2[0, 0, :5] = 4e13
    interaction = np.asarray(pileup.c.forest_interaction_matrix, float)
    mixed = np.zeros_like(floor); mixed[0, :5] = 8e13
    expected = np.maximum(pileup.c.rho_forest_floor_m2 + interaction @ mixed, 1.0)
    assert np.allclose(pileup.forest_density_m2(), expected)
    assert np.all(pileup.forest_density_m2()[:, :5] >= floor[:, :5])
    assert np.allclose(pileup.signed_gnd_m2()[0, :5], 6e13)
    # The Taylor amplification is not capped by the variant.
    assert not np.isfinite(pileup.c.taylor_phi_max) or pileup.c.taylor_phi_max == native.c.taylor_phi_max


def test_native_interaction_is_the_default_and_unknown_modes_fail():
    from experimental_fits.ast2018.spatial import _mechanical_state_type
    native, _, ev = _states()
    default = _mechanical_state_type(ev.spatial.EmergentGNDState, ev.specimen)(ev.material, ev.physics)
    for state in (native, default):
        state.mobile_m2[0, 1, :5] = 1e14
        state.retained_m2[0, 1, :3] = 2e13
    assert np.array_equal(default.signed_gnd_m2(), native.signed_gnd_m2())
    assert np.array_equal(default.forest_density_m2(), native.forest_density_m2())
    with pytest.raises(ValueError):
        _mechanical_state_type(ev.spatial.EmergentGNDState, ev.specimen, "capped")
