import numpy as np
from arrhenius_fracture.swept_slip_ledger_v10230 import SweptSlipLedger


def test_immobile_density_and_exit_do_not_create_or_delete_slip():
    q=SweptSlipLedger.zeros(1,2); rho=np.ones((1,2,2))*3.
    assert np.array_equal(q.advance_motion(rho,np.zeros_like(rho),2.),np.zeros((1,2)))
    q.advance_motion(rho,np.ones_like(rho),2.); saved=q.plastic_shear.copy()
    rho[:]=0. # exit/annihilation changes residents, not compatible deformation
    q.advance_motion(rho,np.ones_like(rho),2.)
    assert np.array_equal(q.plastic_shear,saved)


def test_reverse_motion_reverses_increment_and_opposite_signs_can_add():
    q=SweptSlipLedger.zeros(1,1); rho=np.ones((1,2,1))
    add=q.advance_motion(rho,np.array([[[-1.],[1.]]]),1.,burgers_signs=(-1.,1.))
    assert add[0,0]==2.
    reverse=q.advance_motion(rho,np.array([[[1.],[-1.]]]),1.,burgers_signs=(-1.,1.))
    assert reverse[0,0]==-2. and q.plastic_shear[0,0]==0.


def test_native_orowan_two_sign_populations_add():
    q=SweptSlipLedger.zeros(1,1); rho=np.array([[[2.],[3.]]]); velocity=np.array([[4.]])
    q.advance_orowan_rate(rho,velocity,.5,2.)
    assert np.isclose(q.plastic_shear[0,0],20.)


def test_finite_nucleation_patch_is_booked_once_only_when_area_is_given():
    q=SweptSlipLedger.zeros(1,1)
    assert q.nucleate(0,1,4.,2.,None,8.)==0.
    assert q.nucleate(0,1,4.,2.,3.,8.)==3.
    assert q.plastic_shear[0,0]==3. and q.nucleation_patch_shear[0,0]==3.


def test_beta_p_retains_rotation_information_beyond_symmetric_strain():
    q=SweptSlipLedger.zeros(1,1); q.plastic_shear[0,0]=2.
    beta=q.beta_p([[1.,0.]],[[0.,1.]])
    ep=q.engineering_eigenstrain([[1.,0.]],[[0.,1.]])
    assert beta[0,0,1]==2. and beta[0,1,0]==0.
    assert np.array_equal(ep[:,0],[0.,0.,2.])
