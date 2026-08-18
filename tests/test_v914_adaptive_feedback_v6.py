from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V914 = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search"
)
for _path in (str(ROOT / "scripts"), str(V914)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(V914))
sys.path.insert(0, str(ROOT / "scripts"))

from v914_adaptive_feedback_v6 import (
    AdaptiveFeedbackControls,
    _relative_difference,
    adaptive_one_cycle,
    error_passes,
)


def test_relative_difference_uses_physical_floor():
    assert _relative_difference(0.0, 0.01, 0.1) == 0.1
    assert _relative_difference(1.0, 1.01, 0.1) < 0.01


def test_error_passes_separate_tip_and_hazard_tolerances():
    controls = AdaptiveFeedbackControls(
        state_rtol=0.02,
        tip_radius_rtol=0.001,
        hazard_rtol=0.005,
    )
    good = {
        "shielding": 0.01,
        "mobile": 0.01,
        "retained": 0.01,
        "returned": 0.01,
        "tip_radius": 0.0005,
        "hazard": 0.004,
    }
    assert error_passes(good, controls)
    bad = dict(good)
    bad["tip_radius"] = 0.002
    assert not error_passes(bad, controls)


class _ToyLoading:
    temperature_K = 300.0

    @staticmethod
    def K_at_phase(phase: float) -> float:
        return float(phase)


class _ToyState:
    """Scalar y' = y represented through the audit state interface."""

    def __init__(self):
        self.y = 1.0
        self.mobile_m2 = np.ones((1, 1, 1), dtype=float)
        self.retained_m2 = np.ones((1, 1, 1), dtype=float)

    @property
    def cell_area_m2(self) -> float:
        return 1.0

    def _sync(self):
        self.mobile_m2[...] = self.y
        self.retained_m2[...] = self.y

    def K_shield_MPa_sqrt_m(self) -> float:
        return float(self.y)

    def tip_radius_m(self) -> float:
        return 1.0e-6 * self.y

    def reversibility_diagnostics(self):
        return {
            "reversible_returned_source_slip_count": 0.1 * self.y,
            "reversible_physical_return_fraction_of_emitted": 0.01 * self.y,
            "reversible_reverse_mobile_exposure_fraction": 0.5,
        }

    def local_rates(self, K, T):
        del T
        return {
            "reversible_transport_K_signed_MPa_sqrt_m": float(K - self.y)
        }


def _toy_advance(state, loading, p0, p1):
    del loading
    dt = float(p1 - p0)
    # Forward Euler intentionally has a step-size error so step doubling must
    # refine.  The accepted fine path should approach exp(1).
    state.y = state.y + dt * state.y
    state._sync()
    return dt * state.y * 1.0e-6


def test_adaptive_one_cycle_refines_and_approaches_continuous_limit():
    controls = AdaptiveFeedbackControls(
        state_rtol=0.01,
        tip_radius_rtol=0.01,
        hazard_rtol=0.02,
        base_phase_intervals=1,
        max_refinement_depth=10,
        shielding_scale_floor_MPa_sqrt_m=0.01,
        line_content_scale_floor=0.01,
        returned_slip_scale_floor=0.001,
        hazard_scale_floor=1.0e-9,
    )
    final, hazard, telemetry = adaptive_one_cycle(
        _ToyState(),
        _ToyLoading(),
        controls,
        advance_phase_fn=_toy_advance,
    )
    assert telemetry["refined_intervals"] > 0
    assert telemetry["accepted_intervals"] > 1
    assert abs(final.y - math.e) / math.e < 0.03
    assert hazard > 0.0
    assert telemetry["minimum_accepted_phase_width"] < 1.0
