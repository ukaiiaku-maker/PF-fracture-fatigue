import math
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.hazard_energy_event_gate_v10230 import (
    OBSERVER,
    audit_payload,
    continuum_gate_diagnostics,
    hazard_resistance_J_per_m2,
    reset_runtime_state,
    wrap_cleave_direction_competition,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_v10230 import (
    HazardEnergyGatedPersistentSiteCyclicTipEngine,
)
from arrhenius_fracture.sharp_front_v10_2_27 import VALID_OPTIONS


def test_hazard_resistance_uses_only_active_hazard_quantities():
    barrier = 0.8 * 1.602176634e-19
    b = 2.74e-10
    value = hazard_resistance_J_per_m2(
        barrier_J=barrier,
        cooperative_hits=3.0,
        burgers_vector_m=b,
        gamma_relative=1.25,
    )
    assert math.isclose(value, 1.25 * 3.0 * barrier / b**2)
    payload = audit_payload()
    assert payload["athermal_Gc_active"] is False
    assert payload["generic_FractureResistanceConfig_used"] is False
    assert "DeltaG_cleave_effective" in payload["hazard_resistance_expression"]


def test_continuum_gate_uses_same_effective_barrier_and_current_K():
    class Engine:
        b = 2.0e-10
        f = SimpleNamespace(m_hits=2.0)

        @staticmethod
        def sigma_tip(K):
            return K

        @staticmethod
        def lambda_cleave(stress, temperature):
            return 1.0, 1.0, 2.0e-19

    reset_runtime_state()
    OBSERVER.snapshot = {"mat": SimpleNamespace(Eprime=4.0e11)}
    OBSERVER.direction = {
        "direction": np.array([1.0, 0.0]),
        "gamma_relative": 1.5,
    }
    result = continuum_gate_diagnostics(Engine(), 20.0e6, 300.0)
    expected_resistance = 1.5 * 2.0 * 2.0e-19 / (2.0e-10) ** 2
    assert math.isclose(result["hazard_resistance_J_per_m2"], expected_resistance)
    assert math.isclose(result["continuum_driving_J_per_m2"], 1000.0)
    assert result["energy_gate_continuum_open"] is True


def test_direction_observer_preserves_existing_relative_anisotropy():
    reset_runtime_state()

    def original(*args, **kwargs):
        candidate = {
            "t": np.array([0.6, 0.8]),
            "gamma": 1.37,
            "name": "cleave",
            "angle_deg": 53.130102,
        }
        return [candidate], [candidate]

    wrapped = wrap_cleave_direction_competition(original)
    selected, _ = wrapped(np.eye(2), 30.0, np.array([1.0, 0.0]))
    assert selected
    assert math.isclose(OBSERVER.direction["gamma_relative"], 1.37)
    assert OBSERVER.direction["source"] == "continuous_cubic_competition"


def test_transactional_commit_moves_mpz_and_geometry_state_by_same_length():
    class MPZ:
        def __init__(self):
            self.distances = []

        def advance(self, distance):
            self.distances.append(distance)
            return {
                "wake_mobile": 2.0,
                "wake_retained": 3.0,
                "source_sites_refreshed": 4.0,
            }

    class Dummy:
        _audit_records = [{"engine_id": 7}]

        def __init__(self):
            self._engine_id = 7
            self._energy_gate_pending = {
                "proposal_m": 5.0e-6,
                "proposal_factor": 1.0,
            }
            self.mpz = MPZ()
            self.micro_advance_total_m = 0.0
            self.a_adv = 0.0
            self.checkpoint_advance_total_m = 0.0
            self.avalanche_base_checkpoint_m = 5.0e-6
            self.avalanche_last_completed_advance_m = 0.0
            self.avalanche_last_completed_factor = 0.0
            self.avalanche_event_length_history = []
            self.energy_gate_committed_event_count = 0
            self.energy_gate_committed_path_m = 0.0
            self.N_em = 11.0

    dummy = Dummy()
    info = {"kinetic_dt_consumed_s": 2.0}
    gate = {
        "energy_admissible_event_length_m": 3.0e-6,
        "arrest_reason": "hazard_derived_energy_arrest",
        "hazard_resistance_J_per_m2": 6.0,
        "orientation_gamma_relative": 1.2,
    }
    HazardEnergyGatedPersistentSiteCyclicTipEngine.commit_energy_gated_event(
        dummy, 3.0e-6, gate, info
    )
    assert dummy.mpz.distances == [3.0e-6]
    assert dummy.micro_advance_total_m == 3.0e-6
    assert dummy.a_adv == 3.0e-6
    assert dummy.checkpoint_advance_total_m == 3.0e-6
    assert info["avalanche_event_advance_m"] == 3.0e-6
    assert info["stochastic_event_proposed_advance_m"] == 5.0e-6
    assert info["N_em_shed_to_wake"] == 5.0
    assert dummy._energy_gate_pending is None


def test_four_canonical_parameterizations_are_preserved():
    assert list(VALID_OPTIONS) == [
        "v913_paper_peak01_0242980_persistent_sites",
        "v913_paper_dbtt01_0202500_persistent_sites",
        "v913_paper_weakT01_0129902_persistent_sites",
        "v913_paper_ceramic01_0077080_persistent_sites",
    ]
