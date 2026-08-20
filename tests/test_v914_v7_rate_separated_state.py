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

from v914_v7_rate_separated_state import (
    apply_ledger_delta,
    capture_ledgers,
    ledger_delta,
    restore_active_state,
    serialize_active_state,
)
from v914_v7_rate_separated_dmd import evaluate_rate_separated_dmd_block


class State:
    def __init__(self):
        self.c = SimpleNamespace(n_systems=1, n_bins=2, mpz_length_m=2.0)
        self.dx = 1.0
        self.extension_m = 0.0
        self.mobile_m2 = np.full((1, 2, 2), 3.0)
        self.retained_m2 = np.full((1, 2, 2), 4.0)
        self.accumulated_slip_m2 = np.full((1, 2, 2), 10.0)
        self.returned_slip_m2 = np.full((1, 2, 2), 2.0)
        self.cumulative_source_activations = np.array([5.0])
        self.time_s = 0.0


class Loading:
    frequency_Hz = 10.0


def linear_cycle_map(state, loading, controls):
    del loading, controls
    end = copy.deepcopy(state)
    end.mobile_m2 += 0.25
    end.retained_m2 += 0.5
    end.accumulated_slip_m2 += 2.0
    end.returned_slip_m2 += 0.5
    end.cumulative_source_activations += 3.0
    end.time_s += 0.1
    return end, 0.125, {}


def test_reduced_state_contains_net_blunting_not_raw_flux_ledgers():
    state = State()
    snapshot = serialize_active_state(state)
    assert [field.name for field in snapshot.fields] == [
        "mobile_m2", "retained_m2", "net_slip_m2"
    ]
    net_field = snapshot.fields[-1]
    assert np.all(snapshot.vector[net_field.start:net_field.stop] == 8.0)


def test_gross_return_ledger_and_projected_net_reconstruct_accumulated_slip():
    state = State()
    snapshot = serialize_active_state(state)
    projected = snapshot.vector.copy()
    net_field = snapshot.fields[-1]
    projected[net_field.start:net_field.stop] = 6.0
    apply_ledger_delta(state, {"returned_slip_m2": np.full((1, 2, 2), 3.0)})
    restore_active_state(state, snapshot, projected)
    assert np.all(state.returned_slip_m2 == 5.0)
    assert np.all(state.accumulated_slip_m2 == 11.0)
    assert np.all(state.accumulated_slip_m2 - state.returned_slip_m2 == 6.0)


def test_ledger_delta_is_independent_of_active_state_vector():
    before_state = State()
    after_state = copy.deepcopy(before_state)
    after_state.accumulated_slip_m2 += 7.0
    after_state.returned_slip_m2 += 2.0
    after_state.cumulative_source_activations += 4.0
    delta = ledger_delta(capture_ledgers(before_state), capture_ledgers(after_state))
    assert np.all(delta["accumulated_slip_m2"] == 7.0)
    assert np.all(delta["returned_slip_m2"] == 2.0)
    assert delta["cumulative_source_activations"] == pytest.approx(np.array([4.0]))


def test_rate_separated_dmd_projects_net_state_and_integrates_gross_ledgers(monkeypatch):
    monkeypatch.setenv("V914_V7_DMD_BURST_CYCLES", "6")
    result = evaluate_rate_separated_dmd_block(
        State(), block_stride=64, loading=Loading(), cycle_controls=None,
        state_rtol=1.0e-9, hazard_rtol=1.0e-9,
        cycle_map_fn=linear_cycle_map,
    )
    assert result["numerical_pass"] is True
    end = result["fine_end_state"]
    assert np.all(end.mobile_m2 == pytest.approx(3.0 + 64 * 0.25))
    assert np.all(end.returned_slip_m2 == pytest.approx(2.0 + 64 * 0.5))
    assert np.all(
        end.accumulated_slip_m2 - end.returned_slip_m2
        == pytest.approx(8.0 + 64 * 1.5)
    )
    assert end.cumulative_source_activations == pytest.approx(np.array([5.0 + 64 * 3.0]))
    assert result["fine_block_hazard_action"] == pytest.approx(64 * 0.125)
