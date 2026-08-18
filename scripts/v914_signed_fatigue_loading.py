"""v4-only extension of the qualified v9.14 fatigue waveform to signed R.

The original FatigueLoading contract accepts only 0 <= R < 1 because the
historical fatigue campaign was tension-tension.  Reversible-transport testing
needs controlled compression while preserving the exact same waveform,
frequency, phase convention, and Kmax/deltaK relation.

This subclass changes validation only.  All loading kinematics, including
K_at_phase(), are inherited unchanged from the qualified external v9.14 class.
Positive-R behavior is therefore exactly the historical implementation.
"""
from __future__ import annotations

from arrhenius_fracture import fatigue_v914 as base


class SignedFatigueLoading(base.FatigueLoading):
    """Qualified v9.14 fatigue waveform with -1 <= R < 1 validation.

    For R >= 0, delegate directly to the original validator.  For R < 0,
    validate every other field through the original implementation using a
    temporary R=0 probe, then restore the requested signed R.  object.__setattr__
    keeps this compatible with either mutable or frozen dataclass definitions.
    """

    def validate(self) -> None:
        R = float(self.R)
        if not (-1.0 <= R < 1.0):
            raise ValueError("signed-fatigue R must lie in [-1, 1)")
        if R >= 0.0:
            return super().validate()

        object.__setattr__(self, "R", 0.0)
        try:
            super().validate()
        finally:
            object.__setattr__(self, "R", R)


__all__ = ["SignedFatigueLoading"]
