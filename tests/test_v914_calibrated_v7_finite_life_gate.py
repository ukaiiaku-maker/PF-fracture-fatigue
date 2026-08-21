from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
V914 = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search"
)
for _path in (str(ROOT / "scripts"), str(V914)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(V914))
sys.path.insert(0, str(ROOT / "scripts"))

from arrhenius_fracture import fatigue_v914 as base  # noqa: E402
from arrhenius_fracture.emergent_gnd_campaign_v913 import (  # noqa: E402
    candidate_from_registry_row,
)
from arrhenius_fracture.endurance_knee_v914 import physics_for_row  # noqa: E402
from run_v914_v7_vhcf_event_to_event import load_common  # noqa: E402
from v914_adaptive_feedback_v6 import AdaptiveFeedbackControls  # noqa: E402
from v914_intrinsic_reverse_glide_v7 import (  # noqa: E402
    IntrinsicReverseGlideNoReturnState,
)
from v914_signed_fatigue_loading import SignedFatigueLoading  # noqa: E402
from v914_v7_adaptive_block_accelerator import AdaptiveBlockControls  # noqa: E402
from v914_v7_vhcf_event_engine import (  # noqa: E402
    VHCFRunControls,
    _v7_state_from_checkpoint,
    run_v7_vhcf_event_to_event,
)


REGISTRY = V914 / "runtime_inputs/v914/fracture_equivalence_registry.csv"
DERIVED = (
    ROOT
    / "artifacts/v914_calibrated_v7_transport_extension/"
    "calibrated_v7_transport_extension.json"
)


def _calibrated_candidate_and_physics(n_bins: int = 128):
    rows = list(csv.DictReader(REGISTRY.open()))
    row = next(r for r in rows if r["candidate_id"] == "v913_zeroD_sobol_0129902")
    candidate = candidate_from_registry_row(row)
    physics = physics_for_row(replace(load_common(DERIVED), n_bins=n_bins), row)
    return candidate, physics


def test_v7_checkpoint_restores_transport_and_return_state_fields():
    candidate, physics = _calibrated_candidate_and_physics(n_bins=16)
    state = IntrinsicReverseGlideNoReturnState(candidate, physics)
    state.mobile_m2[0, 0, 0] = 17.0
    state.returned_slip_m2[1, 1, 1] = 2.5
    state.cumulative_gross_source_slip_m2 = state.accumulated_slip_m2.copy()
    state.cumulative_gross_source_slip_m2[0, 0, 0] = 4.25
    saved = base._state_to_checkpoint(state)
    extension = state.reversible_checkpoint_payload()

    restored = _v7_state_from_checkpoint(
        IntrinsicReverseGlideNoReturnState, candidate, physics, saved, extension
    )
    restored.restore_reversible_checkpoint_payload(extension)

    assert base._state_to_checkpoint(restored) == saved


def test_calibrated_transport_first_passage_resolves_past_depth22_failure(tmp_path):
    rows = list(csv.DictReader(REGISTRY.open()))
    row = next(r for r in rows if r["candidate_id"] == "v913_zeroD_sobol_0129902")
    candidate = candidate_from_registry_row(row)
    physics = physics_for_row(replace(load_common(DERIVED), n_bins=128), row)
    loading = SignedFatigueLoading(
        14.608375898315288,
        R=0.1,
        frequency_Hz=1000.0,
        temperature_K=300.0,
        phase_steps=32,
    )
    controls = AdaptiveFeedbackControls(
        state_rtol=0.0025,
        tip_radius_rtol=0.001,
        hazard_rtol=0.01,
        base_phase_intervals=32,
        max_refinement_depth=26,
    )
    result = run_v7_vhcf_event_to_event(
        candidate,
        physics,
        loading,
        seed=1720,
        cycle_controls=controls,
        block_controls=AdaptiveBlockControls(),
        run_controls=VHCFRunControls(
            maximum_physical_cycles=1,
            maximum_cycle_map_evaluations=8,
        ),
        numerics=replace(base.FatigueNumerics(), target_extension_m=2.0e-6),
        checkpoint_path=tmp_path / "checkpoint.json",
        contract_metadata={"configuration_label": "CAL-TRANS-IRREV"},
        state_class=IntrinsicReverseGlideNoReturnState,
    )

    assert result["status"] == "growth_target_reached"
    assert result["event_count"] == 1
    event = result["events"][0]
    assert event["cumulative_cycles"] == pytest.approx(0.4302902708843135, abs=2e-12)
    assert event["tip_radius_m"] == pytest.approx(1.0e-6, rel=2e-8)
    assert event["reversible_physical_return_fraction_of_emitted"] == 0.0


def test_calibrated_short_life_event_burst_resolves_past_depth26_failure(tmp_path):
    rows = list(csv.DictReader(REGISTRY.open()))
    row = next(r for r in rows if r["candidate_id"] == "v913_zeroD_sobol_0129902")
    candidate = candidate_from_registry_row(row)
    physics = physics_for_row(replace(load_common(DERIVED), n_bins=128), row)
    loading = SignedFatigueLoading(
        15.24352267650291,
        R=0.1,
        frequency_Hz=1000.0,
        temperature_K=300.0,
        phase_steps=32,
    )
    controls = AdaptiveFeedbackControls(
        state_rtol=0.0025,
        tip_radius_rtol=0.001,
        hazard_rtol=0.01,
        base_phase_intervals=32,
        max_refinement_depth=30,
    )
    result = run_v7_vhcf_event_to_event(
        candidate,
        physics,
        loading,
        seed=1720,
        cycle_controls=controls,
        block_controls=AdaptiveBlockControls(),
        run_controls=VHCFRunControls(
            maximum_physical_cycles=1,
            maximum_cycle_map_evaluations=8,
        ),
        numerics=replace(base.FatigueNumerics(), target_extension_m=15.0e-6),
        checkpoint_path=tmp_path / "checkpoint.json",
        contract_metadata={"configuration_label": "CAL-TRANS-IRREV"},
        state_class=IntrinsicReverseGlideNoReturnState,
    )

    assert result["status"] == "growth_target_reached"
    assert result["event_count"] == 5
    assert result["cumulative_extension_m"] == pytest.approx(
        16.036529362652427e-6
    )
    assert result["events"][-1]["cumulative_cycles"] == pytest.approx(
        0.4446618760730985, abs=2e-12
    )
