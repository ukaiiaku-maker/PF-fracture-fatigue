import math
from types import SimpleNamespace
import numpy as np
from arrhenius_fracture.cohesive_arrhenius import CohesiveThermalLaw
from arrhenius_fracture.coupled_opening_reduction_r10 import (
    bilinear_accessibility_coordinates, coupled_cycle_reduction,
    coupled_monotonic_ramp, score_recurrence,
)
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative
from arrhenius_fracture.czm_high_cycle_adapter_r10 import (
    CZMOwnedPersistentEngine, _integrate_czm, _lambda_czm, install_czm_owner,
)
from scripts.analyze_r10_coupled_opening_search import native_disposition


def test_inactive_bilinear_reference_separates_cut_loss_and_discrete_screen():
    thermal=CohesiveThermalLaw(1.895736714987287,1.0332740562155006e10,
                              0.,300.)
    got=bilinear_accessibility_coordinates(thermal,300.,2.8159376426256364e19,1e-6,8)
    assert math.isclose(got["static_completed_branch_K_MPa_sqrt_m"],25.900339647524365,rel_tol=2e-15)
    assert math.isclose(got["occupied_bonded_branch_loss_K_MPa_sqrt_m"],27.65295454858545,rel_tol=2e-15)
    assert math.isclose(got["discrete_waveform_Kmax_for_branch_loss_MPa_sqrt_m"],28.63378214589751,rel_tol=2e-15)
    assert got["static_completed_branch_K_MPa_sqrt_m"] < got["occupied_bonded_branch_loss_K_MPa_sqrt_m"]


def test_subcycle_first_passage_is_not_a_developed_rate():
    result = {
        "unsupported_crossover_encountered": False,
        "stop_category": "OBSERVATION_PATH_ENDPOINT_DURING_CASCADE",
        "events": [
            {"cycle": 9.0e-6, "extension_m": 20e-6},
            {"cycle": 2.9e-5, "extension_m": 60e-6},
        ],
    }
    assert native_disposition(result) == "TARGET_TRUNCATED_SAME_CYCLE_CASCADE"


def test_unsupported_mixed_nonadvance_remains_unresolved():
    result = {
        "unsupported_crossover_encountered": True,
        "stop_category": "SUPPORTED_CYCLIC_SHAKEDOWN_NO_ADVANCE",
        "events": [],
    }
    assert native_disposition(result) == "UNSUPPORTED_CROSSOVER_UNRESOLVED"


def test_bounded_two_increment_bursts_can_supply_recurrent_rate():
    events=[]
    for cycle in range(40):
        for index in range(2):
            events.append({"cycle":cycle+0.25,"extension_m":1e-6*(2*cycle+index+1)})
    got=score_recurrence(events,120e-6)
    assert got["outcome"] == "FINITE_RECURRENT_GROWTH"
    assert got["observation_target_hit"] is False
    assert math.isclose(got["developed_rate_m_per_cycle"],2e-6,rel_tol=1e-12)
    assert got["developed_window"]["duration_cycles"] == 20.0


class _RemapMPZ:
    def __init__(self):
        for name in ("mobile_positive","mobile_negative","retained_positive","retained_negative",
                     "accumulated_slip_positive","accumulated_slip_negative","returned_slip_positive",
                     "returned_slip_negative","wake_mobile_positive","wake_mobile_negative",
                     "wake_retained_positive","wake_retained_negative","wake_slip_positive","wake_slip_negative"):
            setattr(self,name,np.zeros(1))
        self.mobile_count=0.;self.retained_count=0.;self._reversible_transport_installed=False
    def advance(self,patch): return {"advanced_m":patch}


class _RemapEngine:
    def __init__(self,successor_drive):
        self.mpz=_RemapMPZ();self.t=0.;self.n_adv=0;self.a_adv=0.;self.checkpoint_advance_total_m=0.
        self.W_emit=0.;self.K_prev=0.;self.successor_drive=successor_drive
        self.plastic_calls=0
    def sigma_tip(self,K): return 2. if self.n_adv==0 else self.successor_drive
    def _plastic_half_step(self,dt,T,sigma): self.plastic_calls+=1;return {}
    def r_eff(self): return 1e-6
    def K_shield(self): return 0.


def _mechanical_probe(alternative,imposed,kenv,thermal,T,prior,phase_duration_s=None):
    complete=imposed>=1.
    return {"state":"SEPARATION_COMPLETED" if complete else "BONDED_STABLE",
            "connected_state":{"branch":"SEPARATED" if complete else "BONDED","delta_m":0.},
            "kinetic_validity":"NOT_APPLICABLE_ATHERMAL"}


def test_monotonic_frozen_reference_advances_time_without_evolving_plasticity(monkeypatch):
    monkeypatch.setattr("arrhenius_fracture.coupled_opening_reduction_r10.evaluate_opening",
                        _mechanical_probe)
    alt=OpeningAlternative("cohesive_athermal","bilinear","quasistatic_mechanical")
    evolving=_RemapEngine(.5)
    full=coupled_monotonic_ramp(evolving,alt,object(),1.,300.,Kdot_MPa_sqrt_m_s=1.,
                                dK_MPa_sqrt_m=.5,Kmax_MPa_sqrt_m=1.)
    frozen=_RemapEngine(.5)
    reference=coupled_monotonic_ramp(frozen,alt,object(),1.,300.,Kdot_MPa_sqrt_m_s=1.,
                                     dK_MPa_sqrt_m=.5,Kmax_MPa_sqrt_m=1.,
                                     plastic_feedback=False)
    assert evolving.plastic_calls == 2
    assert frozen.plastic_calls == 0
    assert math.isclose(evolving.t,frozen.t)
    assert full["plastic_feedback_reference"] == "EVOLVING_FULL_STATE"
    assert reference["plastic_feedback_reference"] == "FROZEN_INITIAL_PLASTIC_STATE"


def test_successor_opening_drive_is_recomputed_after_relaxing_remap(monkeypatch):
    monkeypatch.setattr("arrhenius_fracture.coupled_opening_reduction_r10.evaluate_opening",_mechanical_probe)
    alt=OpeningAlternative("cohesive_athermal","bilinear","quasistatic_mechanical")
    got=coupled_cycle_reduction(_RemapEngine(.5),alt,object(),1.,1e-6,cycles=1,phases=1,
                                patch_m=1e-6,observation_limit_m=5e-6)
    assert len(got["events"]) == 1
    assert got["observation_target_hit"] is False


def test_successor_opening_drive_is_recomputed_when_remap_amplifies(monkeypatch):
    monkeypatch.setattr("arrhenius_fracture.coupled_opening_reduction_r10.evaluate_opening",_mechanical_probe)
    alt=OpeningAlternative("cohesive_athermal","bilinear","quasistatic_mechanical")
    got=coupled_cycle_reduction(_RemapEngine(3.),alt,object(),1.,1e-6,cycles=1,phases=1,
                                patch_m=1e-6,observation_limit_m=3e-6)
    assert len(got["events"]) == 3
    assert got["events"][1]["accepted_opening_drive_Pa"] == 3.
    assert got["outcome"] == "TARGET_TRUNCATED_SINGLE_EXCURSION_CASCADE"


def test_mixed_unit_action_is_a_screen_not_successor_commitment(monkeypatch):
    def high_rate(*args,**kwargs):
        return {"state":"BONDED_METASTABLE","connected_state":{"branch":"METASTABLE","delta_m":0.},
                "kinetic_validity":"QUALIFIED_HIGH_BARRIER","elementary_rate_s-1":2500.}
    monkeypatch.setattr("arrhenius_fracture.coupled_opening_reduction_r10.evaluate_opening",high_rate)
    alt=OpeningAlternative("cohesive_mixed","bilinear","time_dependent_first_passage",1e-20,1e12)
    got=coupled_cycle_reduction(_RemapEngine(.5),alt,object(),1.,1e-6,cycles=4,phases=1,
                                patch_m=1e-6,observation_limit_m=5e-6)
    assert got["events"] == []
    assert got["mixed_action_crossing"]["action"] == 2.5
    assert got["outcome"] == "SCREENED_MIXED_ACTION_CROSSING_REQUIRES_NATIVE_LOCALIZATION"
    assert got["developed_rate_m_per_cycle"] is None


def test_unsupported_prefix_vetoes_an_otherwise_recurrent_window():
    events=[{"cycle":float(i),"extension_m":2e-6*(i+1)} for i in range(40)]
    got=score_recurrence(events,120e-6,unsupported=True)
    assert got["outcome"] == "UNSUPPORTED_MIXED_CROSSOVER"
    assert got["developed_rate_m_per_cycle"] is None


def test_athermal_completion_is_a_deterministic_phase_event(monkeypatch):
    engine=_RemapEngine(.5)
    engine._r10_czm_unsupported=False
    engine._r10_czm_pending_event=None
    engine.hazard_threshold_action=.37
    engine.hazard_action_current=.11
    engine.B=.2
    monkeypatch.setattr(
        "arrhenius_fracture.czm_high_cycle_adapter_r10._opening",
        lambda *args,**kwargs: {
            "state":"SEPARATION_COMPLETED", "barrier_J":None,
            "kinetic_validity":"NOT_APPLICABLE_ATHERMAL",
        },
    )
    got=_integrate_czm(engine,1.,300.,2.5e-5,stress_override=2.)
    assert got["fired"] is True
    assert got["dt_consumed"] == 2.5e-5
    assert got["physical_hazard_action_step"] == 0.
    assert engine.hazard_threshold_action == .37
    assert engine._r10_czm_pending_event["stochastic_threshold"] is False


def test_frozen_czm_reference_retains_opening_clock_without_plastic_step(monkeypatch):
    engine=_RemapEngine(.5)
    engine._r10_czm_unsupported=False
    engine._r10_czm_pending_event=None
    engine._r10_czm_plastic_feedback=False
    engine._r10_czm_alternative=SimpleNamespace(opening_mode="cohesive_mixed")
    engine.hazard_threshold_action=10.
    engine.hazard_action_current=0.;engine.B=0.
    monkeypatch.setattr(
        "arrhenius_fracture.czm_high_cycle_adapter_r10._opening",
        lambda *args,**kwargs: {
            "state":"BONDED_METASTABLE", "barrier_J":1e-20,
            "kinetic_validity":"QUALIFIED_HIGH_BARRIER",
            "elementary_rate_s-1":4., "connected_state":{"branch":"METASTABLE","delta_m":0.},
        },
    )
    got=_integrate_czm(engine,1.,300.,.5,stress_override=2.)
    assert got["fired"] is False
    assert got["physical_hazard_action_step"] == 2.
    assert engine.plastic_calls == 0
    assert engine.t == .5


def test_accepted_cycle_map_carries_connected_branch_state(monkeypatch):
    engine=SimpleNamespace(_r10_czm_branch={"branch":"BONDED","delta_m":0.})
    monkeypatch.setattr(
        "arrhenius_fracture.czm_high_cycle_adapter_r10._opening",
        lambda *args,**kwargs: {
            "state":"BONDED_METASTABLE",
            "connected_state":{"branch":"METASTABLE","delta_m":2e-10},
            "kinetic_validity":"QUALIFIED_HIGH_BARRIER",
            "elementary_rate_s-1":4., "barrier_J":1e-20,
        },
    )
    engine._r10_czm_alternative=SimpleNamespace(opening_mode="cohesive_mixed")
    rate,raw,barrier=_lambda_czm(engine,2.,300.)
    assert (rate,raw,barrier) == (4.,4.,1e-20)
    assert engine._r10_czm_branch == {"branch":"METASTABLE","delta_m":2e-10}


def test_mixed_phase_duration_tracks_the_requested_frequency():
    engine=object.__new__(CZMOwnedPersistentEngine)
    install_czm_owner(engine,SimpleNamespace(),object(),2.,20,7,frequency_Hz=10.)
    assert engine._r10_czm_phase_duration_s == .005
