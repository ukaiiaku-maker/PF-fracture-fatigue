"""Regression for sub-unit persistent-site activation roots."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from experimental_fits.ast2018.config import AstAssumptions
from experimental_fits.ast2018.data import AstData
from experimental_fits.ast2018.implicit_source import (
    mechanically_blocked_at_source_bound, solve_backstress_limited_activations)
from experimental_fits.ast2018.native import historical_start, load_native
from experimental_fits.ast2018.objective import material_from_normalized
from experimental_fits.ast2018.reduced import predict_reduced


SOURCE=Path(os.environ.get('AST2018_NATIVE_SOURCE','/nonexistent'))
ARCHIVE=Path(__file__).resolve().parents[1]/'Ast2018_tungsten_joint_fit_data.zip'


def test_tiny_emission_count_scales_with_duration_without_activation_floor():
    def rate(stress):
        return 1e-20 if stress>0 else 0.0

    common=dict(multiplicity=1e5,drive_stress_Pa=1e9,
                rho_initial_m2=0.0,rho_increment_per_activation_m2=1e15,
                backstress_prefactor_Pa_sqrt_m2=1e5,rate_function=rate)
    long=solve_backstress_limited_activations(dt_s=0.01,**common)
    short=solve_backstress_limited_activations(dt_s=0.005,**common)
    assert long==pytest.approx(1e-17,rel=1e-9)
    assert short==pytest.approx(5e-18,rel=1e-9)
    assert long==pytest.approx(2*short,rel=1e-10)


def test_resolved_root_satisfies_backward_euler_and_blocking_bound():
    M=1e3;dt=.01;drive=1e9;rho_per=1e12;kback=1e2
    rate=lambda stress: 1e-3*stress/drive
    value=solve_backstress_limited_activations(
        multiplicity=M,dt_s=dt,drive_stress_Pa=drive,
        rho_initial_m2=0.,rho_increment_per_activation_m2=rho_per,
        backstress_prefactor_Pa_sqrt_m2=kback,rate_function=rate)
    upper=(drive/kback)**2/rho_per
    stress=drive-kback*(rho_per*value)**.5
    assert 0<value<upper
    assert value==pytest.approx(M*rate(stress)*dt,rel=1e-9)


def test_root_at_the_blocking_jump_returns_the_exact_bound():
    # Regression: Brent stopped ~1e-10 below the bound, the blocked check
    # (2 ulp) missed it and ~30% of Ast joint-fit runs failed the residual.
    drive=3e8;rho0=1e12;rho_per=1e15;kback=2e2
    upper=((drive/kback)**2-rho0)/rho_per
    # Weakly stress-dependent rate whose explicit count overshoots the bound.
    rate=lambda stress: 1.13*upper*(1+.01*stress/drive) if stress>0 else 0.0
    common=dict(multiplicity=1.,drive_stress_Pa=drive,rho_initial_m2=rho0,
                rho_increment_per_activation_m2=rho_per,
                backstress_prefactor_Pa_sqrt_m2=kback,rate_function=rate)
    value=solve_backstress_limited_activations(dt_s=1.,**common)
    assert value==upper
    assert mechanically_blocked_at_source_bound(activations=value,
        drive_stress_Pa=drive,rho_initial_m2=rho0,
        rho_increment_per_activation_m2=rho_per,backstress_prefactor_Pa_sqrt_m2=kback)
    # A continuous root below the bound is not snapped to it.
    inner=solve_backstress_limited_activations(dt_s=.5,**common)
    stress=drive-kback*(rho0+rho_per*inner)**.5
    assert 0<inner<upper and inner==pytest.approx(.5*rate(stress),rel=1e-9)


def test_near_balanced_blocked_root_is_recognized_despite_cancellation():
    # Regression (Ast 250 C joint fit): drive and backstress balanced to
    # ~1e-7, so the numerical zero-stress point sat 1.4e-9 off the analytic
    # bound and the source root residual check failed.
    kw=dict(multiplicity=11288.091592943678,dt_s=0.05,
            drive_stress_Pa=1440787029.259271,rho_initial_m2=373507014650143.2,
            rho_increment_per_activation_m2=7764160386034.437,
            backstress_prefactor_Pa_sqrt_m2=74.55046716073821)
    value=solve_backstress_limited_activations(
        rate_function=lambda stress: 2.102e-2 if stress>0 else 0.0,**kw)
    upper=((kw['drive_stress_Pa']/kw['backstress_prefactor_Pa_sqrt_m2'])**2-
           kw['rho_initial_m2'])/kw['rho_increment_per_activation_m2']
    assert value==upper
    assert mechanically_blocked_at_source_bound(activations=value,
        **{k:v for k,v in kw.items() if k not in ('multiplicity','dt_s')})


@pytest.mark.skipif(not SOURCE.exists(),reason='pinned native source checkout required')
def test_warm_reduced_loading_has_no_step_count_emission_floor(tmp_path):
    with ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if name.startswith('Ast2018_joint_fit_data/data/') and name.endswith('.csv'):
                archive.extract(name,tmp_path)
    data=AstData.load(tmp_path/'Ast2018_joint_fit_data')
    condition=data.condition('Tp150C_rate0p2')
    mods,_=load_native(SOURCE)
    import arrhenius_fracture.emergent_gnd_state_v913 as spatial_state
    assert mods[0].solve_backstress_limited_activations is solve_backstress_limited_activations
    assert spatial_state.solve_backstress_limited_activations is solve_backstress_limited_activations
    base,physics0,_=historical_start(SOURCE,mods[2],mods[3].CommonPhysics)
    x=[0.9360271402317094,0.7313207018385744,0.26777511779276647,
       0.9713184818971736,0.1557893682735712,0.3641617284550316,
       0.07392292711550338]
    candidate=material_from_normalized(base,x)
    original=AstAssumptions()
    outputs=[]
    for step in (0.05,0.025):
        assumptions=replace(original,nominal_K_step_MPa_sqrt_m=step)
        specimen=assumptions.specimen(condition)
        physics=assumptions.physics(physics0,specimen)
        result=predict_reduced(mods[0],candidate,physics,specimen,
            float(condition['temperature_K']),0.2,assumptions,
            deterministic_threshold=True,required_u_m=.8e-6,
            required_extension_m=0.)
        assert result.status=='complete',result.numerical_failure
        assert result.events==[]
        outputs.append(result.history[-1])
    for point in outputs:
        assert point.plastic_u_m<1e-20
        # The condition's E=404 GPa gives 8.38799675588, not the rounded
        # historical 8.4. Compare with the elastic identity at the actual
        # specimen and imposed displacement, retaining the strict tolerance.
        expected=specimen.current_K_MPa_sqrt_m(
            specimen.equilibrium_force(point.imposed_u_m,0.,specimen.a0_m),
            specimen.a0_m)
        assert point.local_drive_K_MPa_sqrt_m==pytest.approx(expected,abs=1e-8)
    assert outputs[0].plastic_u_m==pytest.approx(outputs[1].plastic_u_m,abs=1e-20)


@pytest.mark.skipif(not SOURCE.exists(),reason='pinned native source checkout required')
def test_active_warm_plastic_displacement_is_step_stable(tmp_path):
    with ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if name.startswith('Ast2018_joint_fit_data/data/') and name.endswith('.csv'):
                archive.extract(name,tmp_path)
    condition=AstData.load(tmp_path/'Ast2018_joint_fit_data').condition('Tp150C_rate0p2')
    mods,_=load_native(SOURCE)
    base,physics0,_=historical_start(SOURCE,mods[2],mods[3].CommonPhysics)
    x=[0.9360271402317094,0.7313207018385744,0.26777511779276647,
       .15,.5,0.3641617284550316,0.07392292711550338]
    # This explicitly active fixture has G0_emit=0.92 eV and zero emission
    # temperature slope. The old (.4,.55) fixture had G0_emit=2.12 eV and
    # produced only ~0.009 activations over this test; it could not exercise
    # the >0.4 um plastic branch asserted below.
    candidate=material_from_normalized(base,x)
    values=[]
    for step in (.05,.025):
        assumptions=replace(AstAssumptions(),nominal_K_step_MPa_sqrt_m=step)
        specimen=assumptions.specimen(condition)
        physics=assumptions.physics(physics0,specimen)
        result=predict_reduced(mods[0],candidate,physics,specimen,
            float(condition['temperature_K']),.2,assumptions,
            deterministic_threshold=True,required_u_m=.8e-6,
            required_extension_m=0.)
        assert result.status=='complete',result.numerical_failure
        assert result.events==[]
        assert sum(result.final_state['cumulative_activations'])>100
        values.append(result.history[-1].plastic_u_m)
    assert values[0]>4e-7
    assert values[0]==pytest.approx(values[1],abs=1e-10)
