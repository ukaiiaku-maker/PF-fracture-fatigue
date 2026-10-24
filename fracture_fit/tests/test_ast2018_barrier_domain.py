"""Barrier plots must use effective stresses from the confirmed 1-D state."""
import pytest
import numpy as np
from types import SimpleNamespace
import json

from experimental_fits.ast2018 import barriers
from experimental_fits.ast2018.barriers import _stress_domains
from experimental_fits.ast2018.spatial import _event_rate_audit


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


def test_reverse_signed_transport_stress_counts_in_visited_barrier_domain():
    class State:
        mobile_m2=np.ones((1,2,1))
        retained_m2=np.ones((1,2,1))
        def local_rates(self,*args):
            return {"tau_eff_Pa":np.array([[-2.0]]),
                    "emission_effective_stress_Pa":np.array([1.0]),
                    "emission_rate_per_site_s":np.array([1.0]),
                    "encounter_s":np.array([[1.0]]),
                    "taylor_completion_s":np.array([[1.0]]),
                    "peierls_velocity_m_s":np.array([[-1.0]])}
        def forest_density_m2(self):return np.ones((1,1))
        def K_shield_MPa_sqrt_m(self):return 0.0
        def tip_radius_m(self):return 1.0
        def source_geometry(self):return {"multiplicity_per_system":1.0}
        def cleavage_rate_s(self,*args):return 1.0
    candidate=SimpleNamespace(peierls=SimpleNamespace(stress_fraction=1.0),
                              taylor=SimpleNamespace(stress_fraction=1.0))
    physics=SimpleNamespace(jump_fraction_of_forest_spacing=1.0,
                            b_m=1.0,taylor_phi_max=float("inf"))
    rates=_event_rate_audit(State(),candidate,physics,1.0,300.0)
    assert rates["peierls_effective_stress_max_Pa"]==2.0
    assert rates["taylor_effective_stress_max_Pa"]==1.0


def test_terminal_ligament_marker_is_not_a_committed_barrier_event(tmp_path,monkeypatch):
    event={"cleavage_effective_stress_Pa":3.0,
           "emission_effective_stress_min_Pa":1.0,
           "emission_effective_stress_max_Pa":2.0,
           "peierls_effective_stress_min_Pa":4.0,
           "peierls_effective_stress_max_Pa":5.0,
           "taylor_effective_stress_min_Pa":6.0,
           "taylor_effective_stress_max_Pa":7.0}
    folder=tmp_path/"confirmation_table_primary/spatial/case"
    folder.mkdir(parents=True)
    (folder/"events.json").write_text(json.dumps([event,
        {"terminal_ligament_failure":True,"K_pre_MPa_sqrt_m":10.0}]))
    monkeypatch.setattr(barriers.AstData,"load",lambda _:SimpleNamespace(
        condition=lambda _:{"temperature_K":300.0}))
    assert barriers._visited_events(tmp_path,"TABLE_PRIMARY")==[
        ("case",300.0,event)]

