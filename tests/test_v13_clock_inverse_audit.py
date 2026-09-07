import math
import pytest
from scipy.special import gammainc
from scripts.audit_v13_inherited_clocks import balance, inverse_scales


def test_rate_balance_has_no_amplitude_gate():
    assert balance(1.,1.) == 1
    assert balance(0.,1.) == 0
    assert balance(0.,0.) is None
    assert balance(1e250,2e250) == pytest.approx(8/9)


@pytest.mark.parametrize('order',[1,2,3])
def test_inverse_requirements_reproduce_probabilities_without_sampling(order):
    rate,tau,T=6455.047009795081,1e-6,1000
    for row in inverse_scales(rate,tau,T,order):
        p=row['target_probability']
        assert gammainc(order,rate*row['required_tau_B_s']) == pytest.approx(p)
        assert gammainc(order,rate*tau*row['raw_arrival_multiplicity_factor']) == pytest.approx(p)
        increased=rate*math.exp(row['barrier_reduction_eV']/(8.617333262145e-5*T))
        assert gammainc(order,increased*tau) == pytest.approx(p)
        base=float(gammainc(order,rate*tau))
        assert -math.expm1(row['independent_complete_site_multiplicity']*math.log1p(-base)) == pytest.approx(p)
