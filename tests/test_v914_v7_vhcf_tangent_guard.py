from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v914_v7_vhcf_tangent_guard import (
    GUARD_ID,
    evaluate_step_doubled_block_tangent_guarded,
    reset_tangent_probe_maps,
    tangent_probe_maps,
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
    return out, math.exp(-0.05 * out.time_s), {"cycle_map_id": "toy"}


def _relaxed_cycle_map(state, loading, controls):
    """Same early secant, but the true local rate shuts off after time 6."""
    del controls
    out = copy.deepcopy(state)
    active = 1.0 if float(out.time_s) < 6.0 else 0.0
    out.mobile_m2 += active
    out.retained_m2 += 0.5 * active
    out.accumulated_slip_m2 += 2.0 * active
    out.returned_slip_m2 += 0.1 * active
    out.cumulative_source_activations += active
    out.cumulative_line_content += 2.0 * active
    out.cumulative_returned_mobile_per_m += 0.2 * active
    out.cumulative_escaped_mobile_per_m += 0.05 * active
    out.cumulative_cancelled_slip_line_content += 0.1 * active
    out.cumulative_transport_channel_time_s += 4.0
    out.cumulative_reverse_channel_time_s += 2.0
    out.cumulative_mobile_exposure_m2_s += 10.0 * active
    out.cumulative_reverse_mobile_exposure_m2_s += 3.0 * active
    out.time_s += 1.0 / loading.frequency_Hz
    return out, math.exp(-0.05 * out.time_s), {"cycle_map_id": "toy_relaxed"}


def test_tangent_guard_is_exact_for_linear_endpoint_dynamics():
    reset_tangent_probe_maps()
    trial = evaluate_step_doubled_block_tangent_guarded(
        _State(3.0),
        _State(4.0),
        previous_anchor_cycle=3,
        current_anchor_cycle=4,
        current_anchor_hazard=math.exp(-0.2),
        block_stride=8,
        loading=_Loading(),
        cycle_controls=None,
        cycle_map_fn=_linear_cycle_map,
    )
    guard = trial["constitutive_tangent_guard"]
    assert guard["guard_id"] == GUARD_ID
    assert guard["maximum_active_rate_relative_error"] < 1.0e-12
    assert trial["endpoint_state_error"]["constitutive_tangent_maximum_relative_error"] < 1.0e-12
    assert tangent_probe_maps() == 2


def test_tangent_guard_rejects_secant_that_outruns_relaxed_exact_rate():
    reset_tangent_probe_maps()
    trial = evaluate_step_doubled_block_tangent_guarded(
        _State(3.0),
        _State(4.0),
        previous_anchor_cycle=3,
        current_anchor_cycle=4,
        current_anchor_hazard=math.exp(-0.2),
        block_stride=8,
        loading=_Loading(),
        cycle_controls=None,
        cycle_map_fn=_relaxed_cycle_map,
    )
    guard = trial["constitutive_tangent_guard"]
    assert guard["maximum_active_rate_relative_error"] > 0.5
    assert trial["endpoint_state_error"]["maximum_relative_error"] > 0.5
    assert tangent_probe_maps() == 2
