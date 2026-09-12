from dataclasses import replace
import inspect
import math

import pytest

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import (
    CrackBranchState,
    CrackNetworkState,
    ROOT_BRANCH_ID,
)
from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState,
    tungsten_cleavage_candidates,
)
from arrhenius_fracture.sharp_front import make_emergent_config
from arrhenius_fracture.voiding_production_v5 import (
    build_production_void_state,
    capture_sharp_front_engine,
    directional_sharp_front_rates,
    downstream_front_transaction,
    fresh_sharp_front_engine,
    restore_sharp_front_engine,
    sharp_front_constitutive_response,
)


def _material():
    return make_emergent_config().material


def _owned_engine(*, N_em=37.0, B=0.125, W_emit=2.5e-9, K_prev=2.0e6):
    engine = fresh_sharp_front_engine(_material())
    engine.N_em = N_em
    engine.B = B
    engine.W_emit = W_emit
    engine.K_prev = K_prev
    return engine


def test_a_root_child_engine_equivalence():
    root = _owned_engine()
    child = restore_sharp_front_engine(_material(), capture_sharp_front_engine(root))
    root_competition = DirectionalCompetitionState.initialize(
        tungsten_cleavage_candidates(theta_deg=0.0), global_hazard_seed=3621,
    )
    child_competition = DirectionalCompetitionState.initialize(
        tungsten_cleavage_candidates(theta_deg=0.0), global_hazard_seed=3621,
    )
    assert root_competition.hazard_states == child_competition.hazard_states

    inputs = dict(K_Pa_sqrt_m=5.0e6, temperature_K=900.0, dt_s=2.0e-9)
    root_response = sharp_front_constitutive_response(root, **inputs)
    child_response = sharp_front_constitutive_response(child, **inputs)
    assert child_response == root_response
    assert capture_sharp_front_engine(child) == capture_sharp_front_engine(root)


def test_b_canonical_radius_sensitivity_uses_N_em_ledger():
    baseline = fresh_sharp_front_engine(_material())
    baseline_radius = 2.0 * baseline.f.r0
    responses = []
    for factor in (0.75, 1.0, 1.25):
        engine = fresh_sharp_front_engine(_material())
        engine.N_em = (
            factor * baseline_radius - engine.f.r0
        ) / (engine.f.c_blunt * engine.b)
        response = sharp_front_constitutive_response(
            engine, K_Pa_sqrt_m=1.0e6, temperature_K=900.0, dt_s=1.0e-9,
        )
        assert response["r_eff_m"] == pytest.approx(factor * baseline_radius)
        assert response["sigma_tip_Pa"] == pytest.approx(
            1.0e6 / math.sqrt(2.0 * math.pi * response["r_eff_m"])
        )
        direct_cleave = engine.lambda_cleave(response["sigma_tip_Pa"], 900.0)
        direct_emit = engine.lambda_emit(response["sigma_tip_Pa"], 900.0)
        assert response["cleavage_barrier_J"] == direct_cleave[2]
        assert response["cleavage_rate_s"] == direct_cleave[0]
        assert response["emission_barrier_J"] == direct_emit[2]
        assert response["emission_rate_s"] == direct_emit[0]
        responses.append(response)
    assert responses[0]["sigma_tip_Pa"] > responses[1]["sigma_tip_Pa"] > responses[2]["sigma_tip_Pa"]
    assert len({row["cleavage_barrier_J"] for row in responses}) == 3
    assert len({row["cleavage_rate_s"] for row in responses}) == 3


def test_c_reciprocal_void_radius_control_has_no_constitutive_input():
    engine = _owned_engine()
    inputs = dict(K_Pa_sqrt_m=5.0e6, temperature_K=900.0, dt_s=2.0e-9)
    controls = {
        R_void: sharp_front_constitutive_response(
            restore_sharp_front_engine(_material(), capture_sharp_front_engine(engine)),
            **inputs,
        )
        for R_void in (2.5e-5, 5.0e-5, 1.0e-4)
    }
    assert controls[2.5e-5] == controls[5.0e-5] == controls[1.0e-4]
    assert "R_void" not in inspect.signature(sharp_front_constitutive_response).parameters


def test_d_cavity_and_child_stages_have_distinct_sources():
    transaction = inspect.getsource(downstream_front_transaction)
    child_rates = inspect.getsource(directional_sharp_front_rates)
    assert "cavity_boundary_tensor" in transaction
    assert "source_kind = \"cavity_surface\"" in transaction
    assert "sharp_front_load_provider" in transaction
    assert "established_directional_J_K_provider" in transaction
    assert "restore_sharp_front_engine" in child_rates
    assert "sharp_front_constitutive_response" in child_rates
    assert "cavity_boundary_tensor" not in child_rates
    assert '"r_tip_m"' not in transaction


def test_e_child_canonical_state_checkpoint_round_trip(tmp_path):
    state, _ = build_production_void_state()
    root = replace(state.crack_network.branch(ROOT_BRANCH_ID), status="terminated")
    child_id = "void-front-1"
    child = CrackBranchState(
        child_id, ROOT_BRANCH_ID, 1, 1, (root.tip,), (0.0,),
        local_state={"front_engine_state_owner": child_id},
    )
    network = CrackNetworkState(
        (root, child), primary_branch_id=ROOT_BRANCH_ID,
        geometry_generation=state.crack_network.geometry_generation + 1,
        branching_enabled=True,
    )
    engine = _owned_engine(N_em=81.0, B=0.375, W_emit=7.5e-9, K_prev=3.0e6)
    event_identity = ("candidate#event:0000000000000001",)
    payload = {**capture_sharp_front_engine(engine), "last_event_identity": event_identity}
    child_state = replace(
        state, crack_network=network,
        tip_process_state={"active_branch_id": child_id, "by_branch": {child_id: payload}},
    )
    checkpoint = tmp_path / "child-before-continuation.json"
    write_checkpoint(child_state, checkpoint)
    restored = restore_checkpoint(checkpoint)
    restored_payload = restored.tip_process_state["by_branch"][child_id]
    assert restored_payload == payload
    restored_engine = restore_sharp_front_engine(restored.material, restored_payload)
    inputs = dict(K_Pa_sqrt_m=4.0e6, temperature_K=900.0, dt_s=3.0e-9)
    assert sharp_front_constitutive_response(restored_engine, **inputs) == sharp_front_constitutive_response(
        engine, **inputs
    )
    assert restored_payload["last_event_identity"] == event_identity
    assert "r_tip_m" not in restored_payload
    assert restored_engine.r_eff() == pytest.approx(
        restored_engine.f.r0 + restored_engine.f.c_blunt * restored_engine.b * restored_engine.N_em
    )
