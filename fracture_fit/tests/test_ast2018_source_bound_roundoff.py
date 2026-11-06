"""The source root's exact mechanical bound is a blocked complementarity state."""
from math import ulp

from experimental_fits.ast2018.implicit_source import mechanically_blocked_at_source_bound


def test_one_ulp_positive_stress_at_exact_blocking_bound_is_blocked():
    drive=90786546.89262038
    rho0=5.66098208e11
    rho_per=9.17956677e12
    kback=72.50557130573048
    upper=((drive/kback)**2-rho0)/rho_per
    assert mechanically_blocked_at_source_bound(
        activations=upper,drive_stress_Pa=drive,rho_initial_m2=rho0,
        rho_increment_per_activation_m2=rho_per,
        backstress_prefactor_Pa_sqrt_m2=kback)


def test_interior_or_far_stress_cannot_use_blocking_exception():
    args=dict(drive_stress_Pa=1e8,rho_initial_m2=1e11,
              rho_increment_per_activation_m2=1e12,
              backstress_prefactor_Pa_sqrt_m2=100.)
    upper=((args["drive_stress_Pa"]/args["backstress_prefactor_Pa_sqrt_m2"])**2-
           args["rho_initial_m2"])/args["rho_increment_per_activation_m2"]
    assert not mechanically_blocked_at_source_bound(activations=upper-10*ulp(upper),**args)
    assert not mechanically_blocked_at_source_bound(activations=0.5*upper,**args)
