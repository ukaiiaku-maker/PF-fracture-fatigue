from __future__ import annotations

import numpy as np

from scripts.v914_reverse_drive_utils import (
    forward_projected_transport_stress,
    reverse_drive_mask,
    tensile_reference_signs,
)


def test_tensile_reference_signs_follow_drive_factors() -> None:
    np.testing.assert_array_equal(
        tensile_reference_signs(np.array([0.5, -0.25, 0.0])),
        np.array([1.0, -1.0, 1.0]),
    )


def test_opposite_burgers_velocity_is_not_automatically_reverse_drive() -> None:
    # Positive tensile K produces +tau on system 0 and -tau on system 1 when
    # their Schmid/projection factors have opposite signs.  Both are forward
    # relative to their own tensile reference directions.
    tau = np.array([[3.0, 2.0], [-4.0, -1.0]])
    drive = np.array([0.5, -0.5])
    projected = forward_projected_transport_stress(tau, drive)
    assert np.all(projected > 0.0)
    assert not np.any(reverse_drive_mask(tau, drive))


def test_true_stress_reversal_is_detected_per_system_and_bin() -> None:
    tau = np.array([[-1.0, 2.0], [-3.0, 4.0]])
    drive = np.array([0.5, -0.5])
    expected = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(reverse_drive_mask(tau, drive), expected)
