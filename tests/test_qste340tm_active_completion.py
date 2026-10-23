from pathlib import Path

import numpy as np

from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.paris_slope_r9 import _NativePhaseController
from arrhenius_fracture.persistent_site_poincare_v10230 import one_cycle_map
from scripts.complete_qste340tm_active_calibration import (
    CANDIDATES,
    FREQUENCY_HZ,
    PARENT_CANDIDATE,
    TEMPERATURE_K,
    _identity,
    _initialize_engine,
    load_parent_binding,
)


ROOT = Path(__file__).resolve().parents[1]
ALT = ROOT / "integration_evidence/r10_alternative_gstar_joint_search/alternative_gstar_registry.json"
PARENT = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/"
    "v3-fatigue-informed-joint-barrier-search/analysis_outputs/"
    "v3_fatigue_informed_joint_barrier_search/v3_complete_candidate_registry.json"
)


def _parent():
    return load_parent_binding(ALT, PARENT, candidate_id=PARENT_CANDIDATE)


def test_state_only_acceleration_has_distinct_complete_identity():
    plastic = CANDIDATES["sustained_active"]
    production = _identity(
        "sustained_active", plastic, 8, 4, 8, FREQUENCY_HZ,
        "index_rank", False,
    )
    accelerated = _identity(
        "sustained_active", plastic, 8, 4, 8, FREQUENCY_HZ,
        "index_rank", True,
    )
    assert production["sha256"] != accelerated["sha256"]
    assert accelerated["payload"]["load_substep_cycle_partition"].startswith("nearest_integer")
    assert accelerated["payload"]["accepted_cycle_roundoff_policy"].startswith("snap_to_integer")


def test_state_only_opening_observer_changes_hazard_not_native_active_state():
    parent = _parent()
    plastic = CANDIDATES["sustained_active"]
    production = _initialize_engine(parent, plastic, state_only_acceleration=False)
    accelerated = _initialize_engine(parent, plastic, state_only_acceleration=True)
    waveform = FatigueWaveform(
        Kmax=10.0e6/(1.0-0.1), R=0.1, frequency_Hz=FREQUENCY_HZ,
    )
    full = one_cycle_map(
        production, _NativePhaseController(8), waveform, TEMPERATURE_K,
    )
    state_only = one_cycle_map(
        accelerated, _NativePhaseController(8), waveform, TEMPERATURE_K,
    )
    np.testing.assert_array_equal(full.state_end.vector, state_only.state_end.vector)
    assert full.ledger_delta_per_cycle == state_only.ledger_delta_per_cycle
    assert full.plastic_totals == state_only.plastic_totals
    assert full.hazard_action_per_cycle > 0.0
    assert state_only.hazard_action_per_cycle == 0.0
