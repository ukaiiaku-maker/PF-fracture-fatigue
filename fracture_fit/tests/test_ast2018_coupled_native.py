"""Targeted native integration regressions; set AST2018_NATIVE_SOURCE to run."""
import os
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from experimental_fits.ast2018.config import AstAssumptions
from experimental_fits.ast2018.data import AstData
from experimental_fits.ast2018.mechanics import observe_like_ast
from experimental_fits.ast2018.native import PIN,historical_start,load_native
from experimental_fits.ast2018.objective import material_from_normalized,normalized_start
from experimental_fits.ast2018.reduced import predict_reduced
from experimental_fits.ast2018.spatial import predict_spatial


SOURCE=Path(os.environ.get("AST2018_NATIVE_SOURCE","/nonexistent"))
ARCHIVE=Path(__file__).resolve().parents[1]/"Ast2018_tungsten_joint_fit_data.zip"


@pytest.mark.skipif(not SOURCE.exists(),reason="pinned native source checkout required")
def test_strong_plastic_feedback_scalar_root_and_onset_support(tmp_path):
    with ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if name.startswith("Ast2018_joint_fit_data/data/") and name.endswith(".csv"):
                archive.extract(name,tmp_path)
    data=AstData.load(tmp_path/"Ast2018_joint_fit_data")
    mods,receipt=load_native(SOURCE)
    assert receipt["commit"]==PIN
    candidate,base,_=historical_start(SOURCE,mods[2],mods[3].CommonPhysics)
    candidate=replace(candidate,peierls=replace(candidate.peierls,H0_eV=.5))
    condition=data.condition("Tp21C_rate0p2")
    assumptions=AstAssumptions(target_extension_m=.25e-6,
        nominal_K_step_MPa_sqrt_m=.05,localization_bisections=12)
    specimen=assumptions.specimen(condition)
    physics=assumptions.physics(base,specimen)
    result=predict_reduced(mods[0],candidate,physics,specimen,294.15,.2,
                           assumptions,deterministic_threshold=True)
    assert result.status=="complete",result.numerical_failure
    assert len(result.events)==13
    assert result.history[-1].plastic_u_m>0
    for point in result.history:
        reconstructed=(specimen.elastic_compliance(point.a_m)*point.force_N+
                       point.plastic_u_m)
        assert point.imposed_u_m==pytest.approx(reconstructed,abs=1e-10)
    observed=observe_like_ast(result.history,specimen)
    assert observed[-1].da_m>=.2e-6


@pytest.mark.skipif(not SOURCE.exists(),reason="pinned native source checkout required")
def test_spatial_confirmation_integrates_small_source_work(tmp_path):
    with ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if name.startswith("Ast2018_joint_fit_data/data/") and name.endswith(".csv"):
                archive.extract(name,tmp_path)
    data=AstData.load(tmp_path/"Ast2018_joint_fit_data")
    mods,_=load_native(SOURCE)
    candidate,base,_=historical_start(SOURCE,mods[2],mods[3].CommonPhysics)
    candidate=replace(candidate,peierls=replace(candidate.peierls,H0_eV=.5))
    condition=data.condition("Tp21C_rate0p2")
    assumptions=AstAssumptions(target_extension_m=.02e-6)
    specimen=assumptions.specimen(condition)
    physics=assumptions.physics(base,specimen)
    result=predict_spatial(mods[1],candidate,physics,specimen,294.15,.2,
                           assumptions,deterministic_threshold=True)
    assert result.status=="complete",result.numerical_failure
    assert result.events[0]["cumulative_external_plastic_work_J_per_m"]>0
    assert result.history[-1].plastic_u_m>0


@pytest.mark.skipif(not SOURCE.exists(),reason="pinned native source checkout required")
def test_active_fit_mapping_preserves_fixed_source_and_shape():
    mods,_=load_native(SOURCE)
    base,_,_=historical_start(SOURCE,mods[2],mods[3].CommonPhysics)
    fitted=material_from_normalized(base,normalized_start(base))
    assert fitted.cleavage.G00_eV==pytest.approx(base.cleavage.G00_eV)
    assert fitted.emission.G00_eV==pytest.approx(base.emission.G00_eV)
    assert fitted.cleavage.exp_a==base.cleavage.exp_a
    assert fitted.emission.exp_n==base.emission.exp_n
    assert fitted.cleavage.floor_fraction==base.cleavage.floor_fraction
    assert fitted.emission.floor_fraction==base.emission.floor_fraction
    assert fitted.peierls.H0_eV==pytest.approx(base.peierls.H0_eV)
    assert fitted.peierls.exp_a==base.peierls.exp_a
    assert fitted.taylor==base.taylor
    assert fitted.rho_source0_m2==base.rho_source0_m2
    assert fitted.c_blunt==base.c_blunt
