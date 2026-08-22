from __future__ import annotations

import copy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v914_adaptive_feedback_v6 import AdaptiveFeedbackControls
from v914_v7_adaptive_block_accelerator import AdaptiveBlockControls
from v914_v7_event_localizer import (
    _adaptive_interval,
    advance_v7_phase_span,
    localize_v7_action_in_phase_span,
)
from v914_v7_vhcf_event_engine import (
    VHCFRunControls,
    _minimum_dmd_candidate_stride,
)


class _Loading:
    temperature_K = 300.0


class _ToyState:
    def __init__(self):
        self.value = 0.0
        self.mobile_m2 = np.ones((1, 2, 2), dtype=float)
        self.retained_m2 = np.ones((1, 2, 2), dtype=float)
        self.returned_slip_m2 = np.zeros((1, 2, 2), dtype=float)
        self.accumulated_slip_m2 = np.ones((1, 2, 2), dtype=float)

    @property
    def cell_area_m2(self):
        return 1.0

    def K_shield_MPa_sqrt_m(self):
        return 0.0

    def tip_radius_m(self):
        return 2.0e-6

    def reversibility_diagnostics(self):
        return {
            "reversible_returned_source_slip_count": 0.0,
            "reversible_physical_return_fraction_of_emitted": 0.0,
        }


def _linear_phase_advance(state, loading, p0, p1):
    del loading
    width = float(p1) - float(p0)
    state.value += width
    state.mobile_m2 += width
    state.retained_m2 += 0.5 * width
    state.accumulated_slip_m2 += 2.0 * width
    return 2.0 * width


def _controls():
    return AdaptiveFeedbackControls(
        state_rtol=1.0e-10,
        tip_radius_rtol=1.0e-10,
        hazard_rtol=1.0e-10,
        base_phase_intervals=16,
        max_refinement_depth=8,
    )


def test_phase_span_matches_linear_integral_with_adaptive_fine_path():
    state, hazard = advance_v7_phase_span(
        _ToyState(),
        _Loading(),
        _controls(),
        0.125,
        0.875,
        advance_phase_fn=_linear_phase_advance,
    )
    assert hazard == pytest.approx(1.5)
    assert state.value == pytest.approx(0.75)


def test_phase_first_passage_localizes_without_cycle_iteration():
    state, phase, action = localize_v7_action_in_phase_span(
        _ToyState(),
        _Loading(),
        _controls(),
        start_phase=0.0,
        end_phase=1.0,
        required_action=1.25,
        phase_tolerance=1.0e-12,
        advance_phase_fn=_linear_phase_advance,
    )
    assert phase == pytest.approx(0.625, abs=2.0e-12)
    assert action >= 1.25
    assert state.value == pytest.approx(phase, abs=2.0e-12)


def test_post_event_sharp_transient_can_refine_six_levels_beyond_depth_30():
    boundary = 0.3671875 + 0.00390625 * math.sqrt(2.0) / 2.0

    def sharp_transition(state, loading, p0, p1):
        del loading
        width = float(p1) - float(p0)
        if float(p0) < boundary < float(p1):
            state.mobile_m2 += 6.4e10 * width
        return width

    controls = AdaptiveFeedbackControls(
        state_rtol=0.0025,
        tip_radius_rtol=0.001,
        hazard_rtol=0.01,
        base_phase_intervals=256,
        max_refinement_depth=35,
    )
    with pytest.raises(RuntimeError, match="adaptive refinement failed closed"):
        _adaptive_interval(
            _ToyState(),
            _Loading(),
            controls,
            0.3671875,
            0.37109375,
            advance_phase_fn=sharp_transition,
        )

    state, hazard = _adaptive_interval(
        _ToyState(),
        _Loading(),
        replace(controls, max_refinement_depth=36),
        0.3671875,
        0.37109375,
        advance_phase_fn=sharp_transition,
    )
    assert hazard == pytest.approx(0.00390625)
    assert np.all(np.isfinite(state.mobile_m2))


def test_vhcf_controls_accept_1e14_cycle_horizon():
    controls = VHCFRunControls(maximum_physical_cycles=10**14)
    controls.validate()
    assert controls.maximum_physical_cycles == 100_000_000_000_000


def test_block_controls_accept_power_of_two_beyond_1e14_horizon():
    controls = AdaptiveBlockControls(
        initial_block_stride=4,
        maximum_block_stride=1 << 47,
    )
    controls.validate()
    assert controls.maximum_block_stride > 10**14


def test_validated_dmd_map_may_subdivide_below_training_stride():
    block = AdaptiveBlockControls(initial_block_stride=32)
    run = VHCFRunControls(event_guard_stride=4)
    assert _minimum_dmd_candidate_stride(None, block, run) == 32
    assert _minimum_dmd_candidate_stride(object(), block, run) == 4
