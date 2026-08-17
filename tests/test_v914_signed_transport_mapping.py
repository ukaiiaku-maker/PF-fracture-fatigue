from __future__ import annotations

import numpy as np

from scripts.v914_reversible_transport_utils import signed_transport_stress_fields


def _stress_fields(K: float, shield: float):
    return signed_transport_stress_fields(
        K,
        shield,
        tip_radius_m=1.0e-6,
        drive_factors=np.array([0.5, -0.5]),
        tau_gnd_Pa=np.zeros((2, 3)),
    )


def test_negative_applied_K_remains_negative_for_mobile_transport() -> None:
    fields = _stress_fields(-2.0, 0.5)
    assert fields["K_transport_MPa_sqrt_m"] == -2.5
    assert fields["sigma_applied_Pa"] < 0.0
    assert fields["sigma_transport_Pa"] < 0.0


def test_positive_K_can_reverse_transport_when_shielding_is_larger() -> None:
    fields = _stress_fields(0.2, 0.5)
    assert np.isclose(fields["K_transport_MPa_sqrt_m"], -0.3)
    tau = np.asarray(fields["tau_effective_Pa"])
    assert tau[0, 0] < 0.0
    assert tau[1, 0] > 0.0


def test_zero_shield_preserves_signed_applied_transport_field() -> None:
    fields = _stress_fields(1.25, 0.0)
    assert fields["K_transport_MPa_sqrt_m"] == 1.25
    np.testing.assert_allclose(
        fields["tau_transport_Pa"], fields["tau_applied_Pa"]
    )
