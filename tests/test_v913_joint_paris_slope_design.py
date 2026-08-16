import pandas as pd

from scripts.design_v913_joint_paris_slope_candidates import apply_variant, historical_bounds
from scripts.analyze_v913_joint_paris_slope_physics import exp_floor_from_deltaK


def parent() -> pd.Series:
    return pd.Series({"candidate_id":"p","Tref_K":500.0,"cleave_G00_eV":3.0,"cleave_gT_eV_per_K":.001,
        "cleave_sigc0_GPa":7.0,"cleave_sT_GPa_per_K":-.001,"cleave_exp_a":1.0,"cleave_exp_n":2.0,
        "cleave_floor_frac":.05,"peierls_H0_eV":2.0})


def broad_bounds() -> pd.DataFrame:
    fields=["Tref_K","cleave_G00_eV","cleave_gT_eV_per_K","cleave_sigc0_GPa","cleave_sT_GPa_per_K",
            "cleave_exp_a","cleave_exp_n","cleave_floor_frac","peierls_H0_eV"]
    return pd.DataFrame({"parameter":fields,"historical_min":[-1e9]*len(fields),"historical_max":[1e9]*len(fields),"robust_scale":[1.0]*len(fields)}).set_index("parameter")


def test_P1_changes_slope_but_preserves_barrier_at_design_load():
    row=parent(); bounds=broad_bounds(); reference=20.0
    changed=apply_variant(row,"P1",1.0,1.0,reference,bounds)
    old=exp_floor_from_deltaK(row,1.04*reference,300.0,"cleave")
    new=exp_floor_from_deltaK(changed,1.04*reference,300.0,"cleave")
    assert abs(float(old["G_eV"])-float(new["G_eV"]))<1e-12
    assert float(old["dG_dK_eV_per_MPa_sqrt_m"])!=float(new["dG_dK_eV_per_MPa_sqrt_m"])


def test_P3_preserves_entire_300K_surface_but_changes_thermal_derivative():
    row=parent(); changed=apply_variant(row,"P3",1.0,1.0,20.0,broad_bounds())
    for k in (5.0,15.0,30.0):
        old=exp_floor_from_deltaK(row,k,300.0,"cleave"); new=exp_floor_from_deltaK(changed,k,300.0,"cleave")
        assert abs(float(old["G_eV"])-float(new["G_eV"]))<1e-12
        assert abs(float(old["dG_dK_eV_per_MPa_sqrt_m"])-float(new["dG_dK_eV_per_MPa_sqrt_m"]))<1e-12
    assert float(old["dG_dT_eV_per_K"])!=float(new["dG_dT_eV_per_K"])
