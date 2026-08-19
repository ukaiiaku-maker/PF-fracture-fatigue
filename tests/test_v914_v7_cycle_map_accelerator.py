from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v914_v7_cycle_map import intrinsic_phase_sample
from v914_v7_multicycle_accelerator import (
    affine_next_summary,
    run_accelerator_anchor_path,
    run_exact_multicycle,
)


class _Loading:
    temperature_K = 300.0

    @staticmethod
    def K_at_phase(phase):
        return 2.0 * float(phase) - 1.0


class _State:
    def __init__(self, value=1.0):
        self.value = float(value)
        self.mobile_m2 = np.array([[[2.0, 1.0]], [[1.0, 3.0]]]) * self.value
        self.retained_m2 = np.array([[[0.5, 0.2]], [[0.4, 0.1]]]) * self.value
        self.x = np.array([0.25e-6, 0.75e-6])

    @property
    def cell_area_m2(self):
        return 1.0

    def emission_drive_factors(self):
        return np.array([1.0, -1.0])

    def K_shield_MPa_sqrt_m(self):
        return 0.2 * self.value

    def tip_radius_m(self):
        return 2.0e-6

    def signed_gnd_m2(self):
        return np.array([[1.0, -2.0], [-3.0, 4.0]]) * self.value

    def reversibility_diagnostics(self):
        return {
            "reversible_returned_source_slip_count": 0.1 * self.value,
            "reversible_physical_return_fraction_of_emitted": 0.01 * self.value,
            "reversible_raw_return_fraction_of_emitted": 0.2,
            "reversible_cumulative_source_slip_count": 2.0 * self.value,
            "reversible_reverse_mobile_exposure_fraction": 0.3,
        }

    def local_rates(self, K, T):
        del T
        applied = np.array([[[-2.0e9, -2.0e9]], [[1.0e9, 1.0e9]]]).reshape(2, 2)
        gnd = np.array([[0.5e9, -1.0e9], [0.2e9, 0.3e9]])
        eff = applied + gnd
        return {
            "reversible_transport_K_signed_MPa_sqrt_m": np.asarray(float(K)),
            "reversible_tau_transport_external_Pa": applied,
            "tau_gnd_Pa": gnd,
            "reversible_tau_transport_eff_Pa": eff,
        }


def test_intrinsic_shear_sample_reconstructs_applied_plus_gnd():
    sample = intrinsic_phase_sample(_State(), _Loading(), 0.25, 3)
    assert sample["tau_reconstruction_error_Pa"] == pytest.approx(0.0)
    assert sample["tau_eff_projected_GPa"] == pytest.approx(
        sample["tau_applied_projected_GPa"] + sample["tau_gnd_projected_GPa"]
    )
    assert sample["shear_system"] in (0, 1)
    assert sample["shear_bin"] in (0, 1)


def _toy_cycle_map(state, loading, controls):
    del loading, controls
    state = _State(state.value + 1.0)
    telemetry = {
        "cycle_map_id": "toy",
        "accepted_intervals": 1,
        "refined_intervals": 0,
        "maximum_depth_reached": 0,
        "minimum_accepted_phase_width": 1.0,
    }
    return state, state.value * 1.0e-6, telemetry


def test_exact_and_anchor_stride_one_share_cycle_map_semantics():
    exact_state, exact, _ = run_exact_multicycle(
        _State(), _Loading(), None, 4, cycle_map_fn=_toy_cycle_map
    )
    anchor_state, anchor, _ = run_accelerator_anchor_path(
        _State(), _Loading(), None, 4, anchor_stride=1, cycle_map_fn=_toy_cycle_map
    )
    assert exact_state.value == anchor_state.value
    assert exact == anchor


def test_accelerator_cycle_skipping_fails_closed_until_qualified():
    with pytest.raises(NotImplementedError):
        run_accelerator_anchor_path(
            _State(), _Loading(), None, 4, anchor_stride=2, cycle_map_fn=_toy_cycle_map
        )


def test_affine_prediction_uses_only_intercycle_endpoint_deltas():
    a = {
        "cycle_index": 1,
        "hazard_action": 1.0,
        "shielding_MPa_sqrt_m": 2.0,
        "mobile_line_content": 3.0,
        "retained_line_content": 4.0,
        "returned_source_slip": 5.0,
        "physical_return_fraction": 0.1,
        "tip_radius_m": 6.0,
        "cumulative_source_slip": 7.0,
        "raw_return_fraction": 0.2,
    }
    b = dict(a)
    b.update({"cycle_index": 2, "hazard_action": 1.5, "tip_radius_m": 7.0})
    p = affine_next_summary(a, b)
    assert p["cycle_index"] == 3
    assert p["hazard_action"] == pytest.approx(2.0)
    assert p["tip_radius_m"] == pytest.approx(8.0)
