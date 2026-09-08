from fractions import Fraction
import pickle
import pytest
from arrhenius_fracture.canonical_kinetic_time_v1 import (
    AcceptedTime, exact, integrated_constant_rate, integrate_piecewise_linear,
)


@pytest.mark.parametrize('parts', [1, 2, 4, 8, 16])
def test_lossless_ticks_and_subticks_partition_restart(parts):
    duration = 1.234567891e-9
    clock = AcceptedTime()
    for _ in range(parts): clock = pickle.loads(pickle.dumps(clock.advance(duration/parts)))
    assert clock == AcceptedTime.from_seconds(duration)
    assert clock.seconds_exact() == exact(duration)


@pytest.mark.parametrize('parts', [1, 2, 4, 8, 16])
def test_exact_hazard_integral_not_repeated_float_addition(parts):
    duration, rate, initial = 1.234567891e-9, 137294671.33, .1234
    clock = AcceptedTime()
    for _ in range(parts): clock = clock.advance(duration/parts)
    assert integrated_constant_rate(initial, rate, clock.seconds_exact()) == integrated_constant_rate(initial, rate, duration)


@pytest.mark.parametrize('parts', [1, 2, 4, 8, 16])
def test_variable_rate_uses_identical_physical_anchors(parts):
    anchors = [(0., 0.), (.25, 7.), (.75, 2.), (1., 5.)]
    partitioned = sum((integrate_piecewise_linear(anchors, Fraction(i, parts), Fraction(i+1, parts))
                       for i in range(parts)), Fraction(0))
    assert partitioned == integrate_piecewise_linear(anchors, 0, 1)


def test_nonzero_subtick_is_not_dropped():
    assert AcceptedTime().advance(1e-30).seconds_exact() == exact(1e-30)
    assert AcceptedTime().advance(1e-30).ticks == 0
    with pytest.raises(ValueError): AcceptedTime().advance(-1.)


@pytest.mark.parametrize('parts', [1, 2, 4, 8, 16])
def test_real_site_clocks_preserve_full_partition_state(parts):
    from arrhenius_fracture.voiding_v5 import HazardClock, VoidSite, ProductionVoidState, VoidPhase, advance_site
    site = VoidSite('s', (0., 0.), VoidPhase.EMBRYO, 2, 2, 1.,
        HazardClock(1., 1.), HazardClock(.123, 1.), HazardClock(.07, 2.))
    state = ProductionVoidState((site,))
    rates = {'birth_s': 0., 'stabilization_s': 7.33, 'healing_s': .713}
    total = site.stabilization.crossing_time(rates['stabilization_s'])
    direct, _ = advance_site(state, 's', total, rates=rates)
    current = state
    for i in range(parts):
        dt = total/parts if i+1 < parts else current.sites[0].stabilization.crossing_time(rates['stabilization_s'])
        current, _ = advance_site(current, 's', dt, rates=rates)
        current = pickle.loads(pickle.dumps(current))
    assert current == direct


@pytest.mark.parametrize('parts', [1, 2, 4, 8, 16])
def test_actual_subgrid_growth_inventory_and_generation_are_partition_exact(parts):
    import math
    from arrhenius_fracture.voiding_v5 import Cavity2D, ProductionVoidState, VoidPhase, update_cavity_growth
    r = 2.5e-5; area = math.pi*r*r
    cavity = Cavity2D('v', 's', (0., 0.), r, area, area, VoidPhase.STABLE_SUBGRID_VOID)
    state = ProductionVoidState((), (cavity,), available_defect_inventory_area_m2=1e-8-area,
                               consumed_defect_inventory_area_m2=area)
    def advance(state, dt):
        return update_cavity_growth(state, 'v', rates={'series_limited_growth_s': 137.213},
                                   dt_s=dt, radial_growth_scale_m=1e-8)
    direct = advance(state, .731)
    current = state
    for _ in range(parts): current = pickle.loads(pickle.dumps(advance(current, .731/parts)))
    assert current == direct
    assert current.cavities[0].geometry_generation == 0
    assert advance(current, 0.) == current


@pytest.mark.parametrize('parts', [1, 2, 4, 8, 16])
def test_actual_directional_event_and_anchor_state_are_partition_exact(parts):
    from arrhenius_fracture.canonical_directional_interval_v1 import advance
    from arrhenius_fracture.directional_competition_v11 import DirectionalHazardState
    hazards = (DirectionalHazardState.stochastic('c', threshold_seed=3621),)
    rates = [{'rate_s': 137.213}]; clock = AcceptedTime(); anchors = {}
    direct, events, direct_anchor, direct_dt = advance(hazards, rates, clock=clock, anchors=anchors, source_signature='fixed')
    total = float(direct_dt)
    for index in range(parts):
        hazards, events, anchors, dt = advance(hazards, rates, clock=clock, anchors=anchors,
            source_signature='fixed', maximum_duration_s=total/parts if index+1 < parts else None)
        clock = clock.advance(dt)
    assert hazards == direct and anchors == direct_anchor
    assert clock.seconds_exact() == direct_dt


@pytest.mark.parametrize('parts',[1,2,4,8,16])
def test_growth_target_localization_uses_existing_exact_radius_integral(parts):
    import math
    from arrhenius_fracture.voiding_v5 import Cavity2D,ProductionVoidState,VoidPhase,update_cavity_growth,growth_time_to_radius_exact
    r=2.5e-5;area=math.pi*r*r
    cavity=Cavity2D('v','s',(0.,0.),r,area,area,VoidPhase.STABLE_SUBGRID_VOID)
    before=ProductionVoidState((),(cavity,),available_defect_inventory_area_m2=1e-7-area,consumed_defect_inventory_area_m2=area)
    kwargs={'rates':{'series_limited_growth_s':137.213},'radial_growth_scale_m':1e-8}
    total=growth_time_to_radius_exact(before,'v',5e-5,**kwargs)
    direct=update_cavity_growth(before,'v',dt_s=total,**kwargs);state=before;elapsed=Fraction(0)
    for index in range(parts):
        duration=total/parts if index+1<parts else growth_time_to_radius_exact(state,'v',5e-5,**kwargs)
        state=pickle.loads(pickle.dumps(update_cavity_growth(state,'v',dt_s=duration,**kwargs)))
        elapsed+=duration
    assert state==direct and elapsed==total and state.cavities[0].radius_m==5e-5
