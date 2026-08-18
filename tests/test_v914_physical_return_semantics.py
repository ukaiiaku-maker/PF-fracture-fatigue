from __future__ import annotations

from scripts.v914_minimal_reversible_state_v4 import (
    physical_surface_return_qualifies,
)


def test_forward_drive_left_flux_does_not_cancel_blunting() -> None:
    assert not physical_surface_return_qualifies(
        q=1,
        emitted_q=1,
        reverse_drive_at_surface=False,
        returned_per_m=2.0,
    )


def test_nonemitted_population_never_cancels_blunting() -> None:
    assert not physical_surface_return_qualifies(
        q=0,
        emitted_q=1,
        reverse_drive_at_surface=True,
        returned_per_m=2.0,
    )


def test_zero_boundary_flux_does_not_cancel_blunting() -> None:
    assert not physical_surface_return_qualifies(
        q=1,
        emitted_q=1,
        reverse_drive_at_surface=True,
        returned_per_m=0.0,
    )


def test_true_reverse_return_of_emitted_population_qualifies() -> None:
    assert physical_surface_return_qualifies(
        q=1,
        emitted_q=1,
        reverse_drive_at_surface=True,
        returned_per_m=2.0,
    )
