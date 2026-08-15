import csv
from dataclasses import fields
from pathlib import Path
import sys

import pytest


V914 = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search")
if not V914.exists():
    pytest.skip("authoritative v9.14 repository unavailable", allow_module_level=True)
sys.path.insert(0, str(V914))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

# Pytest may have imported the local v10 package while collecting an earlier
# module.  Extend that package's search path so the immutable external v9.14
# submodules remain importable independent of collection order.
import arrhenius_fracture
external_package = str(V914 / "arrhenius_fracture")
if external_package not in arrhenius_fracture.__path__:
    arrhenius_fracture.__path__.append(external_package)

from arrhenius_fracture import fatigue_v914 as base
from arrhenius_fracture.emergent_gnd_campaign_v913 import candidate_from_registry_row
from arrhenius_fracture.endurance_knee_v914 import physics_for_row
from run_v1032_explicit_lcf import load_physics
from v1032_explicit_cycle_lcf import run_explicit_cycle_fatigue


REGISTRY = V914 / "runs/v914_endurance_knee_global_300K_stageA_analysis/stageB_registry.csv"
PHYSICS = V914 / "mpz_v9_13_v10222_transfer_common_physics.json"


def inputs(phase_steps=32):
    with REGISTRY.open() as stream:
        row = next(r for r in csv.DictReader(stream) if r["candidate_id"] == "v914_endurance_knee_0658")
    candidate = candidate_from_registry_row(row)
    physics = physics_for_row(load_physics(PHYSICS), row)
    loading = base.FatigueLoading(115.74582064678523, phase_steps=phase_steps)
    numerics = base.FatigueNumerics(maximum_cycles=10, target_extension_m=10e-6)
    return candidate, physics, loading, numerics


def run(phase_steps=32, **kwargs):
    candidate, physics, loading, numerics = inputs(phase_steps)
    return run_explicit_cycle_fatigue(candidate, physics, loading, seed=1720,
                                      numerics=numerics, maximum_physical_cycles=10, **kwargs)


def test_multiple_events_continue_within_same_physical_cycle():
    result = run()
    assert result["status"] == "growth_target_reached"
    assert len(result["events"]) >= 2
    assert result["events"][1]["cycle_index"] == result["events"][0]["cycle_index"]
    assert result["events"][1]["cycle_phase"] > result["events"][0]["cycle_phase"]


def test_phase_32_and_64_are_converged_for_bounded_lcf_case():
    a, b = run(32), run(64)
    assert abs(a["final_cycles"] / b["final_cycles"] - 1) < 1e-3
    assert a["final_extension_m"] == b["final_extension_m"]


def test_midcycle_checkpoint_restart_is_bitwise_event_reproducible(tmp_path):
    uninterrupted = run()
    checkpoint = tmp_path / "checkpoint.json"
    paused = run(checkpoint_path=checkpoint, pause_after_phase_advances=3)
    assert paused["status"] == "diagnostic_pause"
    candidate, physics, loading, numerics = inputs(32)
    resumed = run_explicit_cycle_fatigue(
        candidate, physics, loading, seed=1720, numerics=numerics,
        maximum_physical_cycles=10, checkpoint_path=checkpoint,
        restart_from=checkpoint,
    )
    for key in ("final_cycles", "final_extension_m", "events", "rng_state",
                "current_threshold_action", "current_hazard_action"):
        assert resumed[key] == uninterrupted[key]
