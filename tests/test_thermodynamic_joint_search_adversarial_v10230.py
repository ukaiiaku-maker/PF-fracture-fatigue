"""Adversarial contract tests for the analytical joint-barrier search."""
import numpy as np
import pytest

from scripts.thermodynamic_joint_barrier_v10230 import Basis,ThermodynamicSurface
from scripts.verify_thermodynamic_joint_search_v10230 import (
    classify_adversarial,independent_surface_audit,validate_candidate,
    validate_entropy_accounting,validate_guard,validate_renewal,
    validate_search_inputs,
)

def surface(amplitude=1.,entropy=0.):
    return ThermodynamicSurface('cleavage',1e13,.1,(amplitude,),(Basis(1.,1e9,2.),),0.,(entropy,))

@pytest.mark.parametrize('target',['v9.13 target curve','v9.14 TARGET','historical DBTT','historical Peak-T','canonical fracture curve'])
def test_historical_or_canonical_target_rejected(target):
    with pytest.raises(ValueError):validate_search_inputs(target)

def test_canonical_material_row_in_optimizer_rejected():
    with pytest.raises(ValueError):validate_search_inputs('canonical fracture material row')

def test_unrecorded_parent_surface_change_rejected():
    stress=np.linspace(1e9,2e9,5)
    with pytest.raises(ValueError):validate_guard(surface(),surface(1.02),stress)

def test_guard_fatigue_window_tolerance_rejected():
    with pytest.raises(ValueError):validate_guard(surface(),surface(1.01),np.array([1e8]))

def test_independent_entropy_volume_violation_rejected():
    class Broken(type(surface())):
        def activation_volume_m3(self,stress_Pa,temperature_K):return np.zeros_like(np.asarray(stress_Pa,float))+1e-25
    s=surface();broken=Broken(**s.__dict__)
    with pytest.raises(ValueError):independent_surface_audit(broken,np.linspace(1e6,2e9,20),600)

def test_double_entropy_prefactor_rejected():
    with pytest.raises(ValueError):validate_entropy_accounting(True)

def test_generic_renewal_rejected():
    with pytest.raises(ValueError):validate_renewal(3.,1e-6)

@pytest.mark.parametrize('bad',[
    ThermodynamicSurface('cleavage',1e13,-.2,(.1,),(Basis(1.,1e9,2.),),0.,(0.,)),
    ThermodynamicSurface('cleavage',1e13,.1,(-.5,),(Basis(1.,1e9,2.),),0.,(0.,)),
])
def test_negative_or_stress_increasing_cleavage_rejected(bad):
    with pytest.raises(ValueError):independent_surface_audit(bad,np.linspace(1e6,3e9,30),300)

def test_renewal_plateau_cannot_be_weak_t():
    with pytest.raises(ValueError):classify_adversarial({'renewal_ceiling_plateau':True,'label':'WEAK_T'})

def test_saturated_sign_change_cannot_be_peak_t():
    with pytest.raises(ValueError):classify_adversarial({'saturated_sign_change':True,'label':'PEAK_T'})

def test_f0_only_cannot_be_full_state_dbtt():
    with pytest.raises(ValueError):classify_adversarial({'label':'DBTT_LIKE'},tier='F0')

def test_normalization_cannot_hide_absolute_scale_failure():
    with pytest.raises(ValueError):classify_adversarial({'absolute_scale_failed':True})

def test_low_emission_entropy_cannot_be_preferred():
    with pytest.raises(ValueError):classify_adversarial({'emission_entropy_active_kB':12},preferred_entropy=True)

@pytest.mark.parametrize('key',['peierls_stress','taylor_factor','source_density','blunting_rate','attempt_frequency_s'])
def test_protected_fatigue_coordinates_cannot_change(key):
    with pytest.raises(ValueError):validate_candidate({key:1.},{key:2.})

def test_external_canonical_cannot_change_retained_set():
    with pytest.raises(ValueError):classify_adversarial({},external_selected=True)
