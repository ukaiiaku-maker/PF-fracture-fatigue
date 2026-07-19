from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.signed_kernel_family_v1026 import KernelState
from arrhenius_fracture.signed_kernel_family_v1029 import (
    StateResolvedSignedShieldingKernelFamily,
)
from arrhenius_fracture.state_resolved_signed_engine_v1029 import (
    StateResolvedSignedBurgersTipEngine,
)


def _family(*, boundary_policy="strict"):
    states = []
    for index, opening in enumerate((0.0, 0.5, 1.0)):
        states.append(
            KernelState(
                state_id=f"o{index}",
                coordinates=np.array([1.0, opening, 0.0]),
                active_I=np.array([[4.0e6 * opening]]),
                wake_I=np.zeros((1, 0)),
                active_II=np.zeros((1, 1)),
                wake_II=np.zeros((1, 0)),
                metadata={},
            )
        )
    family = StateResolvedSignedShieldingKernelFamily(
        states=states,
        active_x_m=np.array([0.5]),
        wake_x_m=np.zeros(0),
        activation_to_line_content=np.ones(1),
        source_capacity_bounds=np.array([[0.0, 10.0]]),
        fixed_kernel_assessment={"fixed_kernel_accepted": False},
        interpolation={
            "method": "inverse_distance",
            "neighbors": 2,
            "power": 2.0,
            "envelope_relative_tolerance": 1.0e-10,
        },
        metadata={"production_parameterization_allowed": False},
        source_path="synthetic.json",
    )
    family._opening_boundary_policy = {
        "policy": boundary_policy,
        "lower_boundary_validated": boundary_policy != "strict",
        "upper_boundary_validated": boundary_policy != "strict",
    }
    family._last_boundary_action = "none"
    family._validate_complete_grid()
    return family


def test_axis_specific_envelope_rejects_radius_and_strict_opening_extrapolation():
    family = _family()
    with pytest.raises(RuntimeError, match="r_eff_over_r0"):
        family.resolve(
            r_eff_over_r0=1.1,
            opening_strength_fraction=0.5,
            crack_extension_m=0.0,
        )
    with pytest.raises(RuntimeError, match="opening state"):
        family.resolve(
            r_eff_over_r0=1.0,
            opening_strength_fraction=1.1,
            crack_extension_m=0.0,
        )


def test_reviewed_opening_boundary_saturation_is_explicit():
    family = _family(boundary_policy="validated_boundary_saturation")
    family.resolve(
        r_eff_over_r0=1.0,
        opening_strength_fraction=1.1,
        crack_extension_m=0.0,
    )
    assert family._last_boundary_action == "upper_saturation"
    assert family._last_query[1] == pytest.approx(1.0)


def test_effective_opening_fixed_point_uses_signed_shielding_not_applied_K():
    family = _family()
    engine = object.__new__(StateResolvedSignedBurgersTipEngine)
    r_eff = (10.0e6 / 1.0e9) ** 2 / (2.0 * np.pi)
    engine.f = SimpleNamespace(r0=r_eff, sigma_cap=1.0e9)
    engine.r_eff = lambda: r_eff
    engine.micro_advance_total_m = 0.0
    engine._state_kernel_family = family
    engine._signed_last_state_coordinates = {
        "r_eff_over_r0": 1.0,
        "opening_strength_fraction": 1.0,
        "crack_extension_m": 0.0,
    }
    engine._signed_current_K_Pa_sqrt_m = 10.0e6
    engine._kernel_resolution_active = False
    engine._fixed_point_iterations = 0
    engine._fixed_point_residual = 0.0
    engine._fixed_point_converged = False
    engine._effective_K_tip_Pa_sqrt_m = 0.0
    engine._opening_sigma_uncapped_Pa = 0.0
    engine._opening_sigma_local_Pa = 0.0
    engine.mpz = SimpleNamespace(
        _signed_kernel=family,
        retained_positive=np.ones((1, 1)),
        retained_negative=np.zeros((1, 1)),
        mobile_positive=np.zeros((1, 1)),
        mobile_negative=np.zeros((1, 1)),
        wake_retained_positive=np.zeros((1, 0)),
        wake_retained_negative=np.zeros((1, 0)),
        wake_mobile_positive=np.zeros((1, 0)),
        wake_mobile_negative=np.zeros((1, 0)),
        advance_total_m=0.0,
        cfg=SimpleNamespace(mobile_shield_fraction=1.0, wake_shielding=False),
    )

    engine._resolve_state_kernel(10.0e6)
    opening = engine._signed_last_state_coordinates["opening_strength_fraction"]
    assert engine._fixed_point_converged
    assert engine._fixed_point_iterations > 1
    assert abs(engine._fixed_point_residual) <= 5.0e-8
    assert 0.5 < opening < 0.9
    assert engine._effective_K_tip_Pa_sqrt_m < 10.0e6
    assert engine.sigma_tip(10.0e6) / engine.f.sigma_cap == pytest.approx(
        opening, abs=5.0e-8
    )
