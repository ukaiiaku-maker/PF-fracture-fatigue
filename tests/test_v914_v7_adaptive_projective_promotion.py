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

from v914_v7_adaptive_projective_accelerator import (
    AdaptivePromotionControls,
    readiness_prediction,
    run_adaptive_projective_multicycle,
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
        # Toy-state normalization used only by endpoint-observable tests.  The
        # production v7 state provides the physical MPZ cell area itself.
        return 1.0

    def K_shield_MPa_sqrt_m(self):
        return float(np.sum(self.retained_m2[:, 1, :] - self.retained_m2[:, 0, :]))

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


def _linear_cycle_map(state, loading, controls):
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
    telemetry = {
        "cycle_map_id": "toy",
        "accepted_intervals": 1,
        "refined_intervals": 0,
        "maximum_depth_reached": 0,
        "minimum_accepted_phase_width": 1.0,
    }
    return out, 1.0e-6 / (1.0 + out.time_s), telemetry


def test_readiness_prediction_is_exact_for_linear_endpoint_state():
    a = _State(1.0)
    b = _State(2.0)
    c = _State(3.0)
    r = readiness_prediction(a, b, c, frequency_Hz=1.0, cycle_index=3)
    assert r["maximum_relative_error"] == pytest.approx(0.0)
    assert r["projection_constraint_correction"] == pytest.approx(0.0)


def test_readiness_prediction_detects_nonlinear_retained_transient():
    a = _State(1.0)
    b = _State(2.0)
    c = _State(3.0)
    c.retained_m2 *= 0.2
    r = readiness_prediction(a, b, c, frequency_Hz=1.0, cycle_index=3)
    assert r["maximum_relative_error"] > 0.05


def test_adaptive_promotion_skips_only_after_consecutive_readiness_passes():
    final, records, telemetry, metadata = run_adaptive_projective_multicycle(
        _State(0.0),
        _Loading(),
        None,
        10,
        promotion_controls=AdaptivePromotionControls(
            minimum_exact_cycles=4,
            readiness_relative_tolerance=1.0e-12,
            readiness_consecutive_passes=2,
            block_stride=2,
            max_projection_constraint_correction=0.1,
        ),
        cycle_map_fn=_linear_cycle_map,
    )
    assert metadata["promotion_cycle"] == 4
    assert metadata["projected_cycle_count"] > 0
    assert metadata["resolved_cycle_count"] < 10
    assert any(row["resolution"] == "projected_skip" for row in records)
    assert len(telemetry) == 10
    assert final.time_s == pytest.approx(10.0)
