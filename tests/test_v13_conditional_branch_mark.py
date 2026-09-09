from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
from scipy.special import gammainc

from arrhenius_fracture.conditional_branch_mark_v13 import (
    ParentCleavageEvent, BranchOpportunityState, BranchMarkParameters, CompanionChannel,
    BranchMark, conditional_probabilities, branch_uniform, draw_branch_mark,
    apply_conditional_branch_mark,
)
from arrhenius_fracture.marked_topology_trial_v13 import trial_conditional_pair
from arrhenius_fracture.branch_cluster_v11 import create_unresolved_branch_cluster
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.directional_competition_v11 import construct_action_proposals
from arrhenius_fracture.topology_transaction_v11 import execute_topology_trial, apply_sharp_wake_trial_geometry
from test_topology_transaction_v11 import fem_state, arm, equilibrate_to


def parent():
    return ParentCleavageEvent(("root", "daughter0"), "owner0", "primary", "primary#event1", 1,
                              1.0, 1e-5, "accepted1", "fullstate", "cleavage_rng", "process_rng")


def channel(rate=1.0, m=1.0, candidate="companion"):
    return CompanionChannel(candidate, rate, m, 1e-19, 1000.0, 0.0, 50e-6, "fullstate")


@pytest.mark.parametrize("rate,m,tau", [(1e-6, 1, 1), (1e6, 1, 1), (3, 2, .2), (3, 2.5, .2), (0, 3, 1)])
def test_one_companion_multihit_exact(rate, m, tau):
    p = conditional_probabilities([channel(rate, m)], BranchMarkParameters(True, tau))
    assert p.companions[0][1] == float(gammainc(m, rate*tau))
    assert p.single + p.companions[0][1] == pytest.approx(1, abs=1e-15)


@pytest.mark.parametrize("rates", [(1e-6, 1e-6), (2, 3), (1e5, 1e5), (1e-8, 1e4), (0, 2)])
def test_exponential_competing_hazards_and_conserved_intensity(rates):
    tau = .1
    p = conditional_probabilities([channel(rates[0], candidate="a"), channel(rates[1], candidate="b")], BranchMarkParameters(True, tau))
    total = sum(rates)
    assert p.single == pytest.approx(math.exp(-total*tau), abs=1e-14)
    for (_, value), rate in zip(p.companions, rates):
        assert value == pytest.approx(rate/total * -math.expm1(-total*tau), abs=1e-10)
    s, b = p.partition_cleavage_intensity(98765.4)
    assert s + sum(v for _, v in b) == pytest.approx(98765.4, rel=2e-15)


def test_baseline_raw_rate_and_continuous_overlap_no_owner_switch():
    c = channel(1e5)
    assert c.arrival_rate(BranchMarkParameters(True, 1)) == c.raw_arrival_rate_per_s
    p = BranchMarkParameters(True, 1, overlap_barrier_J=1e-20)
    values = [replace(c, junction_distance_m=d).arrival_rate(p) for d in (0, 49.999e-6, 50e-6, 50.001e-6, 100e-6)]
    assert values == sorted(values)
    assert values[3]/values[1] < 1.0001


def test_branch_rng_counter_identity_independence_and_replay():
    op = BranchOpportunityState(parent())
    u, identity = branch_uniform(op, "companion", 71)
    assert 0 < u < 1
    assert (u, identity) == branch_uniform(op, "companion", 71)
    alternatives = [branch_uniform(op, "other", 71), branch_uniform(op, "companion", 72),
        branch_uniform(replace(op, opportunity_ordinal=1), "companion", 71),
        branch_uniform(replace(op, parent=replace(parent(), front_lineage=("root", "daughter1"))), "companion", 71),
        branch_uniform(replace(op, parent=replace(parent(), front_lineage=("root", "daughter0", "grandchild"))), "companion", 71),
        branch_uniform(replace(op, parent=replace(parent(), primary_event_ordinal=2)), "companion", 71)]
    assert len({identity, *(x[1] for x in alternatives)}) == 7


@pytest.mark.parametrize("has_parent,enabled", [(False, True), (False, False), (True, False)])
def test_disabled_or_precleavage_never_evaluates_channels_or_rng(monkeypatch, has_parent, enabled):
    def forbidden(*args, **kwargs):
        pytest.fail("branch work before accepted cleavage or while disabled")
    monkeypatch.setattr("arrhenius_fracture.conditional_branch_mark_v13.branch_uniform", forbidden)
    state = object()
    result = apply_conditional_branch_mark(parent=parent() if has_parent else None,
        baseline_single_state=state, parameters=BranchMarkParameters(enabled, 1),
        channels=forbidden, exact_pair_trial=forbidden)
    assert result.state is state
    assert result.mark.branch_rng_identities == ()


def test_stale_self_duplicate_and_inadmissible_companions_fail_closed():
    op, p = BranchOpportunityState(parent()), BranchMarkParameters(True, 1)
    for channels in ([replace(channel(), state_sha256="old")], [channel(candidate="primary")], [channel(), channel()]):
        with pytest.raises(ValueError):
            draw_branch_mark(op, channels, p)
    result = draw_branch_mark(op, [replace(channel(), geometrically_admissible=False)], p)
    assert result.companion_id is None and not result.branch_rng_identities


def test_no_companion_and_rejected_companion_preserve_exact_single():
    state = object()
    for rate, disposition in [(0, "SINGLE_NO_COMPANION"), (1e9, "SINGLE_COMPANION_REJECTED")]:
        result = apply_conditional_branch_mark(parent=parent(), baseline_single_state=state,
            parameters=BranchMarkParameters(True, 1), channels=lambda: [channel(rate)],
            exact_pair_trial=lambda *_: SimpleNamespace(accepted=False, rejection_reason="exact_energy_gate"))
        assert result.state is state and result.disposition == disposition
        assert result.mark.baseline_physical_time_increment_s == 0


def test_multicompany_multihit_draws_match_partition():
    channels = [channel(4, 2, "a"), channel(1, 3, "b")]
    params = BranchMarkParameters(True, .5, branch_seed=71)
    p = conditional_probabilities(channels, params)
    counts = {None: 0, "a": 0, "b": 0}
    n = 10000
    for ordinal in range(n):
        counts[draw_branch_mark(BranchOpportunityState(parent(), ordinal), channels, params).companion_id] += 1
    for key, probability in [(None, p.single), *p.companions]:
        assert abs(counts[key]/n-probability) < 6*math.sqrt(probability*(1-probability)/n) + 1/n


def test_scientific_hash_is_independent_of_aliases_and_mapping_order():
    import numpy as np
    from scripts.v13_value_fingerprint import physical_state_fingerprint as fp
    arr = np.array([1.0, 2.0])
    assert fp({"a": arr, "b": arr, "set": {"x", "y"}}) == fp({"set": {"y", "x"}, "b": arr.copy(), "a": arr.copy()})
    with pytest.raises(TypeError, match="callable"):
        fp({"method": lambda: None})


@pytest.mark.parametrize("pair_energy,accepted", [(7.0, True), (9.9, False)])
def test_existing_exact_pair_trial_without_companion_clock_consumption(pair_energy, accepted):
    pre = fem_state()
    singles = [p for p in construct_action_proposals(pre.competition.hazard_states, correlation_interval_s=0) if p.action_type == "one_arm"]
    proposal = singles[0]
    primary = proposal.member_candidate_ids[0]
    companion = next(c.candidate_id for c in pre.competition.candidates if c.candidate_id != primary)
    single_arm = arm(primary, end=(1.4, -.3), dissipation=1.0)
    geometry_single = lambda s, arms: apply_sharp_wake_trial_geometry(s, arms, kill_radius_m=.2)
    single = execute_topology_trial(pre, proposal, [single_arm], apply_trial_geometry=geometry_single, equilibrate_fixed_load=equilibrate_to(8))
    assert single.accepted
    # Mimic already completed process renewal; pair may not overwrite it.
    baseline = replace(single.state, tip_process_state={"renewal_count": 1},
        energy_ledgers={**single.state.energy_ledgers, "emission_work": 4.0})
    p = replace(parent(), primary_candidate_id=primary, primary_event_id=proposal.member_event_ids[0], primary_event_ordinal=proposal.member_event_ordinals[0])
    network, cluster = create_unresolved_branch_cluster(pre.crack_network, parent_branch_id=ROOT_BRANCH_ID,
        candidate_ids=(primary, companion), event_index=5,
        shared_process_state={"tip_state": baseline.tip_process_state}, conserved_ledgers={"emission_work": 4})
    arms = (arm(primary, cluster.arm_branch_ids[0], end=single_arm.end_xy_m, dissipation=1),
            arm(companion, cluster.arm_branch_ids[1], end=(1.4, .3), dissipation=1))
    def geometry(s, arms):
        return apply_sharp_wake_trial_geometry(replace(s, crack_network=network,
            junction_process_state={"cluster": cluster}), arms, kill_radius_m=.2)
    result = trial_conditional_pair(pre_event_state=pre, baseline_single_state=baseline, parent=p,
        mark=BranchMark(companion, 1e-12, ()), arms=arms, apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_to(pair_energy))
    assert result.accepted == accepted
    assert result.state.competition is baseline.competition
    assert result.state.tip_process_state == baseline.tip_process_state
    assert result.state.event_counters == baseline.event_counters
    assert result.state.energy_ledgers["emission_work"] == 4
    assert result.state.rng_state == baseline.rng_state
    if accepted:
        assert len(result.state.crack_network.active_tip_ids) == 2
        assert result.state.competition.consumed_event_ids == (p.primary_event_id,)
        assert result.state.energy_ledgers["topology_release_J_per_m"] == 3
    else:
        assert result.state is baseline
        assert result.rejection_reason == "insufficient_whole_topology_energy_release"
