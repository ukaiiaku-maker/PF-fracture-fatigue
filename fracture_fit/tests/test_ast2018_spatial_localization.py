from types import SimpleNamespace

import numpy as np

from experimental_fits.ast2018.config import AstAssumptions
from experimental_fits.ast2018.mechanics import Specimen
from experimental_fits.ast2018 import spatial as spatial_adapter
from experimental_fits.ast2018.spatial import predict_spatial


def test_spatial_first_passage_bisection_does_not_force_full_translation(monkeypatch):
    moves=[]
    monkeypatch.setattr(spatial_adapter,"_event_rate_audit",lambda *args:{})

    class State:
        def __init__(self, _candidate, _physics):
            self.time_s=0.0
            self.external_plastic_work_J_per_m=0.0
            self.cumulative_source_activations=np.zeros(2)

        def cleavage_rate_s(self, _K, _T):
            return 1.0

        def translate_tip(self, da_m):
            moves.append(da_m)

        def K_shield_MPa_sqrt_m(self):
            return 0.0

        def source_geometry(self):
            return {"tip_radius_m":1e-7,"front_width_m":4e-6,
                    "multiplicity_per_system":1.0}

        def backstress_state(self):
            return None,None,np.zeros(2)

    spatial=SimpleNamespace(EmergentGNDState=State,
        _state_advance_is_resolved=lambda *args,**kwargs:False)
    specimen=Specimen(4e-6,6e-6,12.5e-6,2.25e-6,411e9,.28,1e-4)
    assumptions=AstAssumptions(target_extension_m=20e-9,
                               nominal_K_step_MPa_sqrt_m=.05,
                               localization_bisections=10,
                               maximum_load_steps=10)
    result=predict_spatial(spatial,None,None,specimen,294.15,.01,assumptions,
                           deterministic_threshold=True)
    assert result.status=="complete",result.numerical_failure
    assert len(result.events)==1
    assert any(0<da<.999*assumptions.event_advance_m for da in moves)
