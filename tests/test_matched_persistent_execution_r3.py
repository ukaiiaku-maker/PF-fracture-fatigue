import pytest

from arrhenius_fracture.matched_persistent_execution_r3 import (
    AVAILABLE_STOCHASTIC_EVENT_LAW_ID,
    REDUCED_EVENT_LAW_ID,
    archived_spatial_stress_contract,
    build_matched_reduced_engine,
    event_contract,
    run_recorded_operator_replay,
    spatial_event_contract,
    spatial_rows,
    stress_contract,
)


def test_post_install_event_consumer_wins_over_stale_wrapper_declaration():
    contract = spatial_event_contract()
    assert contract["law_id"] == REDUCED_EVENT_LAW_ID
    assert contract["observed_engine_hazard_mode"] == "deterministic"
    assert contract["observed_engine_reward_mode"] == "fixed"
    assert contract["wrapper_declared_hazard_mode"] == "exponential"
    assert contract["wrapper_matches_post_install_consumer"] is False
    assert contract["available_but_not_selected_stochastic_law_id"] == (
        AVAILABLE_STOCHASTIC_EVENT_LAW_ID
    )


def test_exact_spatial_strength_source_is_explicit_and_not_a_K_shield_cap():
    engine = build_matched_reduced_engine()
    first, last = spatial_rows()[0], spatial_rows()[-1]
    virgin = stress_contract(engine, first["KJ_Pa_sqrtm"])
    high = stress_contract(engine, 100.0e6)
    archived = archived_spatial_stress_contract(last)
    assert virgin["local_strength_Pa"] == 30.0e9
    assert virgin["local_strength_source"] == "explicit_spatial_driver_cohesive_ceiling"
    assert virgin["local_strength_limiter_active"] is False
    assert high["local_strength_limiter_active"] is True
    assert high["cleavage_stress_supplied_Pa"] == 30.0e9
    assert archived["stress_override"] is None
    assert archived["local_strength_source"] == "run_args.sigma_cap_GPa"


def test_recorded_operator_replay_matches_before_post_event_taylor_divergence():
    engine = build_matched_reduced_engine()
    rows = run_recorded_operator_replay(engine)
    assert event_contract(engine)["law_id"] == REDUCED_EVENT_LAW_ID
    assert [row["reduced_step"]["fired"] for row in rows] == [False, True, True]
    for row in rows[:2]:
        spatial = row["recorded_spatial"]["sigma_tip_Pa"]
        reduced = row["reduced_step"]["sigma_tip"]
        assert reduced == pytest.approx(spatial, rel=3.0e-4)
    # This is an equation-level replay, not a claim that the R2 Taylor consumer
    # is the historical consumer.  Divergence after the second commit is kept.
    assert rows[-1]["reduced_step"]["N_em"] != pytest.approx(
        rows[-1]["recorded_spatial"]["N_em"], rel=0.1
    )
