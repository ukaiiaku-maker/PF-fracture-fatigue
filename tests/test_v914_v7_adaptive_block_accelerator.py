from __future__ import annotations

import copy
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

from v914_v7_adaptive_block_accelerator import (
    AdaptiveBlockControls,
    aggregate_log_bridge_hazard,
    evaluate_step_doubled_block,
    run_adaptive_block_multicycle,
)


class _Loading:
    frequency_Hz = 1.0


class _State:
    def __init__(self, level: float = 0.0):
        self.c = SimpleNamespace(n_systems=1, n_bins=2, mpz_length_m=2.0)
        self.dx = 1.0
        self.extension_m = 0.0
        self.time_s = float(level)
        self.mobile_m2 = np.full((1, 2, 2), 2.0 + level)
        self.retained_m2 = np.full((1, 2, 2), 1.0 + 0.5 * level)
        self.accumulated_slip_m2 = np.full((1, 2, 2), 3.0 + 2.0 * level)
        self.returned_slip_m2 = np.full((1, 2, 2), 0.1 * level)
        self.cumulative_source_activations = np.full(1, level)
        self.cumulative_line_content = np.full(1, 2.0 * level)
        self.cumulative_returned_mobile_per_m = np.full((1, 2), 0.2 * level)
        self.cumulative_escaped_mobile_per_m = np.full((1, 2), 0.05 * level)
        self.cumulative_cancelled_slip_line_content = np.full((1, 2), 0.1 * level)
        self.cumulative_transport_channel_time_s = 4.0 * level
        self.cumulative_reverse_channel_time_s = 2.0 * level
        self.cumulative_mobile_exposure_m2_s = 10.0 * level
        self.cumulative_reverse_mobile_exposure_m2_s = 3.0 * level

    @property
    def cell_area_m2(self):
        return 1.0

    def K_shield_MPa_sqrt_m(self):
        return 0.01 * float(np.sum(self.retained_m2))

    def tip_radius_m(self):
        net = np.maximum(self.accumulated_slip_m2 - self.returned_slip_m2, 0.0)
        return 1.0e-6 + 1.0e-9 * float(np.sum(net))

    def reversibility_diagnostics(self):
        accumulated = float(np.sum(self.accumulated_slip_m2))
        returned = float(np.sum(self.returned_slip_m2))
        return {
            "reversible_returned_source_slip_count": returned,
            "reversible_physical_return_fraction_of_emitted": returned / max(accumulated, 1.0),
            "reversible_raw_return_fraction_of_emitted": 0.2,
            "reversible_cumulative_source_slip_count": accumulated,
        }


def _linear_geometric_cycle_map(state, loading, controls):
    del controls
    out = copy.deepcopy(state)
    out.mobile_m2 += 1.0
    out.retained_m2 += 0.5
    out.accumulated_slip_m2 += 2.0
    out.returned_slip_m2 += 0.1
    out.cumulative_source_activations += 1.0
    out.cumulative_line_content += 2.0
    out.cumulative_returned_mobile_per_m += 0.2
    out.cumulative_escaped_mobile_per_m += 0.05
    out.cumulative_cancelled_slip_line_content += 0.1
    out.cumulative_transport_channel_time_s += 4.0
    out.cumulative_reverse_channel_time_s += 2.0
    out.cumulative_mobile_exposure_m2_s += 10.0
    out.cumulative_reverse_mobile_exposure_m2_s += 3.0
    out.time_s += 1.0 / loading.frequency_Hz
    hazard = math.exp(-0.05 * out.time_s)
    telemetry = {
        "cycle_map_id": "toy",
        "accepted_intervals": 1,
        "refined_intervals": 0,
        "maximum_depth_reached": 0,
        "minimum_accepted_phase_width": 1.0,
    }
    return out, hazard, telemetry


def test_aggregate_log_bridge_matches_geometric_sequence():
    assert aggregate_log_bridge_hazard(1.0, 0.25, 2) == pytest.approx(0.75)
    q = math.exp(-0.2)
    expected = sum(q**j for j in range(1, 9))
    assert aggregate_log_bridge_hazard(1.0, q**8, 8) == pytest.approx(expected)


def test_step_doubled_block_is_exact_for_linear_state_geometric_hazard():
    previous = _State(3.0)
    current = _State(4.0)
    trial = evaluate_step_doubled_block(
        previous,
        current,
        previous_anchor_cycle=3,
        current_anchor_cycle=4,
        current_anchor_hazard=math.exp(-0.05 * 4.0),
        block_stride=8,
        loading=_Loading(),
        cycle_controls=None,
        cycle_map_fn=_linear_geometric_cycle_map,
    )
    assert trial["end_cycle"] == 12
    assert trial["fine_end_state"].time_s == pytest.approx(12.0)
    assert trial["endpoint_state_error"]["maximum_relative_error"] < 1.0e-12
    assert trial["block_hazard_relative_error"] < 1.0e-12
    assert trial["maximum_projection_constraint_correction"] == pytest.approx(0.0)


def test_adaptive_block_growth_promotes_and_increases_stride():
    final, anchors, blocks, metadata = run_adaptive_block_multicycle(
        _State(0.0),
        _Loading(),
        None,
        40,
        block_controls=AdaptiveBlockControls(
            minimum_exact_cycles=4,
            readiness_relative_tolerance=1.0e-12,
            readiness_consecutive_passes=2,
            initial_block_stride=4,
            maximum_block_stride=32,
            block_state_rtol=1.0e-12,
            block_hazard_rtol=1.0e-12,
            max_projection_constraint_correction=0.1,
        ),
        cycle_map_fn=_linear_geometric_cycle_map,
    )
    assert metadata["promotion_cycle"] == 4
    assert metadata["accepted_block_count"] == 4
    assert metadata["rejected_block_count"] == 0
    assert metadata["maximum_accepted_stride"] >= 16
    assert final.time_s == pytest.approx(40.0)
    assert max(row["block_stride"] for row in blocks if row["accepted"]) >= 16
    assert metadata["actual_cycle_map_evaluation_speedup"] > 2.0
    assert anchors[-1]["cycle_index"] == 40


def test_adaptive_block_cumulative_hazard_matches_exact_geometric_sum():
    _, _, _, metadata = run_adaptive_block_multicycle(
        _State(0.0),
        _Loading(),
        None,
        40,
        block_controls=AdaptiveBlockControls(
            minimum_exact_cycles=4,
            readiness_relative_tolerance=1.0e-12,
            readiness_consecutive_passes=2,
            initial_block_stride=4,
            maximum_block_stride=32,
            block_state_rtol=1.0e-12,
            block_hazard_rtol=1.0e-12,
        ),
        cycle_map_fn=_linear_geometric_cycle_map,
    )
    expected = sum(math.exp(-0.05 * n) for n in range(1, 41))
    assert metadata["cumulative_hazard_action"] == pytest.approx(expected, rel=1.0e-12)
