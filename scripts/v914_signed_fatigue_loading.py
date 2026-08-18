"""v4-only extension of the qualified v9.14 fatigue waveform to signed R.

The original FatigueLoading contract accepts only 0 <= R < 1 because the
historical fatigue campaign was tension-tension. Reversible-transport testing
needs controlled compression while preserving the exact same waveform,
frequency, phase convention, and Kmax/deltaK relation.

This subclass changes validation only. All loading kinematics, including
K_at_phase(), are inherited unchanged from the qualified external v9.14 class.
Positive-R behavior is therefore exactly the historical implementation.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

_SCRIPTS = Path(__file__).resolve().parent
_DEFAULT_V914 = Path(
    os.environ.get(
        "V914_ROOT",
        "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
    )
)
for _path in (str(_SCRIPTS), str(_DEFAULT_V914)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(_DEFAULT_V914))
sys.path.insert(0, str(_SCRIPTS))

from arrhenius_fracture import fatigue_v914 as base


class SignedFatigueLoading(base.FatigueLoading):
    """Qualified v9.14 fatigue waveform with -1 <= R < 1 validation.

    For R >= 0, delegate directly to the original validator. For R < 0,
    validate every other field through the original implementation using a
    temporary R=0 probe, then restore the requested signed R. object.__setattr__
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
