import math
from pathlib import Path
import pytest
from arrhenius_fracture.virtual_ct_v10230 import (PRIMARY_CT,SENSITIVITY_CT,ct_geometry_factor,ct_k_from_load_pa_sqrt_m,
 ct_load_from_k_pa_sqrt_m,projected_macro_crack_m,nominal_ranges,energy_equivalent_driver_k_pa_sqrt_m)

ROOT=Path(__file__).parents[1]
def test_ct_geometry_factor_reference_values():
 assert ct_geometry_factor(.45)==pytest.approx(8.33958568327015,rel=1e-14)
 assert ct_geometry_factor(.5)==pytest.approx(9.659078631008239,rel=1e-14)
 assert ct_geometry_factor(.6)==pytest.approx(13.654145726628231,rel=1e-14)
def test_ct_load_K_inversion_and_units():
 k=18e6;p=ct_load_from_k_pa_sqrt_m(k,PRIMARY_CT,PRIMARY_CT.initial_crack_m)
 assert ct_k_from_load_pa_sqrt_m(p,PRIMARY_CT,PRIMARY_CT.initial_crack_m)==pytest.approx(k,rel=1e-14)
 assert p>0
def test_projected_macro_crack_and_no_radius_argument():
 assert projected_macro_crack_m(PRIMARY_CT,100e-6)==pytest.approx(5.1e-3)
 assert "radius" not in ct_k_from_load_pa_sqrt_m.__annotations__
def test_constant_load_preserves_R_and_increases_K():
 p=ct_load_from_k_pa_sqrt_m(18e6,PRIMARY_CT,PRIMARY_CT.initial_crack_m);a=projected_macro_crack_m(PRIMARY_CT,100e-6)
 kp=ct_k_from_load_pa_sqrt_m(p,PRIMARY_CT,a);km=ct_k_from_load_pa_sqrt_m(-.95*p,PRIMARY_CT,a)
 assert km/kp==pytest.approx(-.95);assert kp>18e6
def test_nominal_K_control_adjusts_load_downward():
 p0=ct_load_from_k_pa_sqrt_m(18e6,PRIMARY_CT,PRIMARY_CT.initial_crack_m);p1=ct_load_from_k_pa_sqrt_m(18e6,PRIMARY_CT,projected_macro_crack_m(PRIMARY_CT,100e-6))
 assert p1<p0
def test_negative_R_full_and_tensile_ranges():
 q=nominal_ranges(18e6,-.95);assert q["deltaK_full_Pa_sqrt_m"]==pytest.approx(35.1e6);assert q["deltaK_tensile_Pa_sqrt_m"]==pytest.approx(18e6)
def test_J_equivalence_and_scale_sensitivity():
 assert energy_equivalent_driver_k_pa_sqrt_m(18e6,400e9,400e9)==18e6
 assert energy_equivalent_driver_k_pa_sqrt_m(18e6,400e9,100e9)==pytest.approx(9e6)
def test_three_row_controller_contracts_are_fresh_bounded_and_exact():
 text=(ROOT/"scripts/complete_v10_2_30_R_nominal_deltaK_study.py").read_text()
 assert 'OPTIONS=("A_NATIVE"' in text;assert '"resume":False' in text;assert 'V10230_HIGH_CYCLE_EXPLICIT_ONLY' in text
 assert '1<=a.workers<=3' in text;assert 'len(jobs)==60' not in text
def test_final_verifier_is_fail_closed():
 text=(ROOT/"scripts/verify_v10_2_30_R_nominal_deltaK_study.py").read_text()
 for phrase in ["resumed trajectory admitted","unqualified acceleration admitted","tip radius","fewer than three points","Markdown/JSON decision mismatch"]:assert phrase in text
