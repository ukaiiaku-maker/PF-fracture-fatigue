"""Barrier plots must use effective stresses from the confirmed 1-D state."""
import pytest

from experimental_fits.ast2018.barriers import _stress_domains


def test_effective_mechanism_stress_domains_are_distinct():
    event={"cleavage_effective_stress_Pa":3.0,
           "emission_effective_stress_min_Pa":1.0,
           "emission_effective_stress_max_Pa":2.0,
           "peierls_effective_stress_min_Pa":4.0,
           "peierls_effective_stress_max_Pa":5.0,
           "taylor_effective_stress_min_Pa":6.0,
           "taylor_effective_stress_max_Pa":7.0}
    assert _stress_domains([("condition",300.0,event)])=={
        "cleavage":(3.0,3.0),"emission":(1.0,2.0),
        "Peierls":(4.0,5.0),"Taylor":(6.0,7.0)}
    event["taylor_effective_stress_max_Pa"]=-1.0
    with pytest.raises(RuntimeError,match="Taylor"):
        _stress_domains([("condition",300.0,event)])

