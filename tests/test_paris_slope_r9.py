from pathlib import Path

import numpy as np

from arrhenius_fracture.archived_fatigue_material_r9 import (
    EXPECTED_OPTIMIZED_SHA256, EXPECTED_PARENT_SHA256,
    build_archived_current_engine, load_archived_material_binding,
)
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.paris_slope_r9 import (
    configure_stochastic_engine, mean_cycle_inputs, qualify_mean_cycle_overlap,
)


ARCHIVE = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_worktrees/v3-fatigue-informed-joint-barrier-search"
)
OPTIMIZED = ARCHIVE/"analysis_outputs/v3_1_coupled_slope_continuation/v3_1_optimized_candidate_registry.json"
PARENT = ARCHIVE/"analysis_outputs/v3_fatigue_informed_joint_barrier_search/v3_complete_candidate_registry.json"


def _binding():
    return load_archived_material_binding(OPTIMIZED, PARENT)


def test_exact_v31_row_binds_to_current_backend_without_historical_engine():
    binding = _binding()
    assert binding.provenance["candidate_sha256"] == EXPECTED_OPTIMIZED_SHA256
    assert binding.provenance["complete_v3_row_sha256"] == EXPECTED_PARENT_SHA256
    assert binding.provenance["historical_engine_loaded"] is False
    engine = build_archived_current_engine(binding, kernel_family_path=R4_KERNEL_FAMILY)
    assert engine.manifest.candidate_id == binding.optimized_row["candidate_id"]
    assert engine.f.sigma_cap == 30.0e9
    assert engine.f.m_hits == binding.optimized_row["opening_surface"]["hits"]
    assert engine.mpz._persistent_site_cfg == binding.persistent_sites


def test_archived_opening_reference_retains_near_four_anchor_secants():
    engine = build_archived_current_engine(_binding(), kernel_family_path=R4_KERNEL_FAMILY)
    loads = np.asarray([4.781762498950186, 8.598738351065925, 15.462474735549586])
    actions = []
    for value in loads:
        waveform = FatigueWaveform(Kmax=value*1.0e6, R=0.1, frequency_Hz=1000.0)
        actions.append(mean_cycle_inputs(engine, waveform, 300.0, 64)["hazard_action_per_cycle"])
    slopes = np.diff(np.log(actions))/np.diff(np.log(loads))
    assert np.all((slopes >= 3.4) & (slopes <= 4.6))


def test_mean_block_overlap_is_qualified_on_current_state():
    engine = configure_stochastic_engine(_binding(), kernel_family_path=R4_KERNEL_FAMILY, seed=1720)
    waveform = FatigueWaveform(Kmax=15.462474735549586e6, R=0.1, frequency_Hz=1000.0)
    overlap = qualify_mean_cycle_overlap(engine, waveform, 300.0, 64)
    assert overlap["passed"] is True
    assert overlap["hazard_relative_error"] < 2.0e-3
    assert overlap["state_relative_max_error"] < 2.0e-3


def test_high_load_overlap_fails_closed_to_exact_phase_execution():
    engine = configure_stochastic_engine(_binding(), kernel_family_path=R4_KERNEL_FAMILY, seed=1720)
    waveform = FatigueWaveform(Kmax=50.0e6, R=0.1, frequency_Hz=1000.0)
    overlap = qualify_mean_cycle_overlap(engine, waveform, 300.0, 64)
    assert overlap["passed"] is False
    assert overlap["state_relative_max_error"] > 2.0e-3
