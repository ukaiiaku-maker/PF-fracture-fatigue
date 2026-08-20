from __future__ import annotations

import copy
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
    advance_v7_phase_span,
    localize_v7_action_in_phase_span,
)
from v914_v7_vhcf_event_engine import VHCFRunControls


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
