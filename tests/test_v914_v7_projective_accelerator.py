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

from v914_v7_multicycle_accelerator import run_exact_multicycle
from v914_v7_projective_accelerator import (
    ProjectiveAcceleratorControls,
    compare_projective_to_exact,
    log_bridge_hazards,
    run_projective_multicycle,
)
from v914_v7_projective_state import project_v7_state_secant


class _Loading:
    frequency_Hz = 1.0


class _ProjectiveState:
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
    hazard = math.exp(-0.2 * out.time_s)
    telemetry = {
        "cycle_map_id": "toy_shared_cycle_map",
        "accepted_intervals": 1,
        "refined_intervals": 0,
        "maximum_depth_reached": 0,
        "minimum_accepted_phase_width": 1.0,
    }
    return out, hazard, telemetry


def test_log_bridge_is_geometric_and_positive():
    values = log_bridge_hazards(1.0, 0.25, 1)
    assert values == pytest.approx([0.5])
    assert values[0] > 0.0


def test_projector_advances_complete_spatial_fields_and_time():
    a = _ProjectiveState(3.0)
    b = _ProjectiveState(4.0)
    p, diagnostics = project_v7_state_secant(
        a,
        b,
        anchor_gap_cycles=1,
        skip_cycles=1,
        frequency_Hz=1.0,
    )
    expected = _ProjectiveState(5.0)
    for name in (
        "mobile_m2",
        "retained_m2",
        "accumulated_slip_m2",
        "returned_slip_m2",
        "cumulative_source_activations",
        "cumulative_line_content",
    ):
        assert np.asarray(getattr(p, name)) == pytest.approx(np.asarray(getattr(expected, name)))
    assert p.time_s == pytest.approx(5.0)
    assert diagnostics["maximum_relative_constraint_correction"] == pytest.approx(0.0)


def test_projector_uses_log_secant_for_decaying_active_density():
    a = _ProjectiveState(0.0)
    b = _ProjectiveState(1.0)
    a.mobile_m2[...] = 4.0
    b.mobile_m2[...] = 1.0
    p, diagnostics = project_v7_state_secant(
        a,
        b,
        anchor_gap_cycles=1,
        skip_cycles=1,
        frequency_Hz=1.0,
    )
    assert np.all(p.mobile_m2 == pytest.approx(0.25))
    assert diagnostics["maximum_relative_constraint_correction"] == pytest.approx(0.0)
    assert diagnostics["active_predictor_departure_from_linear"]["mobile_m2"] > 0.0
    assert diagnostics["active_nonnegative_predictor"] == (
        "linear_growth_logarithmic_decay_secant"
    )


def test_projector_caps_returned_slip_by_accumulated_slip():
    a = _ProjectiveState(0.0)
    b = _ProjectiveState(1.0)
    a.returned_slip_m2[...] = 0.0
    b.returned_slip_m2[...] = 20.0
    p, diagnostics = project_v7_state_secant(
        a,
        b,
        anchor_gap_cycles=1,
        skip_cycles=1,
        frequency_Hz=1.0,
    )
    assert np.all(p.returned_slip_m2 <= p.accumulated_slip_m2)
    assert diagnostics["relative_constraint_corrections"]["returned_slip_pointwise_cap"] > 0.0


def test_projector_fails_closed_across_crack_extension():
    a = _ProjectiveState(0.0)
    b = _ProjectiveState(1.0)
    b.extension_m = 1.0e-6
    with pytest.raises(RuntimeError, match="crack extension"):
        project_v7_state_secant(
            a,
            b,
            anchor_gap_cycles=1,
            skip_cycles=1,
            frequency_Hz=1.0,
        )


def test_stride2_projective_path_matches_linear_state_geometric_hazard_toy_problem():
    initial = _ProjectiveState(0.0)
    exact_final, exact, _ = run_exact_multicycle(
        initial,
        _Loading(),
        None,
        12,
        cycle_map_fn=_linear_geometric_cycle_map,
    )
    accel_final, accel, telemetry, metadata = run_projective_multicycle(
        initial,
        _Loading(),
        None,
        12,
        accelerator_controls=ProjectiveAcceleratorControls(
            warmup_cycles=4,
            block_stride=2,
        ),
        cycle_map_fn=_linear_geometric_cycle_map,
    )
    assert metadata["projected_cycle_count"] == 4
    assert metadata["resolved_cycle_count"] == 8
    assert metadata["ideal_cycle_map_speedup"] == pytest.approx(1.5)
    assert len(telemetry) == 12
    for e, a in zip(exact, accel):
        for name in (
            "hazard_action",
            "mobile_line_content",
            "retained_line_content",
            "returned_source_slip",
            "tip_radius_m",
            "cumulative_source_slip",
        ):
            assert float(a[name]) == pytest.approx(float(e[name]), rel=1.0e-12, abs=1.0e-12)
    comparison = compare_projective_to_exact(
        exact_final,
        exact,
        accel_final,
        accel,
        warmup_cycles=4,
    )
    assert comparison["pass"] is True
    assert comparison["post_warmup_cumulative_hazard_relative_error"] < 1.0e-12


def test_stride_other_than_two_fails_closed_in_first_qualification_version():
    controls = ProjectiveAcceleratorControls(warmup_cycles=4, block_stride=4)
    with pytest.raises(ValueError, match="block_stride=2"):
        controls.validate()
