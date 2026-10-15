import numpy as np

from arrhenius_fracture.history_reset_checkpoint_v1 import (
    RESET_FIELDS, transform_history_reset_checkpoint,
)
from arrhenius_fracture.run_state_checkpoint_v10230 import (
    load_combined_checkpoint, write_combined_checkpoint,
)


def _checkpoint(root):
    fields = []
    for index, name in enumerate(sorted(RESET_FIELDS)):
        fields.append({
            "owner": "mpz", "name": name, "shape": [1],
            "start": index, "stop": index + 1, "floor": 0.0,
        })
    outer = {
        "case": {"da_phys_m": 1.0},
        "driver": {"Uapp_accepted": 2.0, "Ftop_prev": 3.0},
        "geometry": {
            "crack_tip_m": [0.0, 0.0], "front_paths": [[[0.0, 0.0]]],
            "committed_event_count": 0, "kinetic_event_index": 0,
            "front_inventory": [{
                "xy": [0.0, 0.0], "fwd": [1.0, 0.0],
                "last_plane": {"t": [1.0, 0.0], "n": [0.0, 1.0]},
                "win_plane": {"t": [1.0, 0.0], "n": [0.0, 1.0]},
            }],
        },
        "history": {"hist": {"Ftop": [3.0], "W_emit": [4.0]}},
    }
    kinetic = {
        "fields": fields, "ledgers": {"mpz.recovered_total": 7.0},
        "geometry_signature": [0, 0.0, 0.0, 0.0],
        "stochastic": {
            "hazard_threshold_history": [], "hazard_event_index": 0,
            "avalanche_base_checkpoint_m": 1.0,
            "avalanche_event_length_factor": 1.0,
            "avalanche_event_advance_m": 1.0,
            "hazard_action_current": 0.0,
            "hazard_threshold_action": 1.0, "B": 0.0,
        },
        "high_cycle_cache": {"stale": True},
    }
    write_combined_checkpoint(
        root, outer=outer, arrays={"displacement": np.array([5.0, 6.0])},
        kinetic=kinetic, kinetic_vector=np.arange(1, len(fields) + 1, dtype=float),
    )


def test_history_reset_is_new_readable_generation_and_preserves_physical_ledgers(
    tmp_path, monkeypatch,
):
    source, destination = tmp_path / "source", tmp_path / "reset"
    _checkpoint(source)
    source_outer, source_kinetic, source_arrays = load_combined_checkpoint(source)

    def equilibrium(outer, arrays):
        assert outer["driver"]["Uapp_accepted"] == 2.0
        return np.array([8.0, 9.0]), {
            "converged": True, "residual_norm_N_per_m": 0.0,
            "displacement_correction_norm_m": 1.0,
            "reaction_N_per_m": 11.0, "native_mechanics": True,
            "opening_m": 2.0,
        }

    monkeypatch.setattr(
        "arrhenius_fracture.history_reset_checkpoint_v1._reequilibrate", equilibrium,
    )
    receipt = transform_history_reset_checkpoint(source, destination)
    outer, kinetic, arrays = load_combined_checkpoint(destination)
    assert np.count_nonzero(arrays["kinetic_active_vector"]) == 0
    assert np.array_equal(arrays["displacement"], [8.0, 9.0])
    assert kinetic["ledgers"] == source_kinetic["ledgers"]
    assert kinetic["high_cycle_cache"] == {}
    assert outer["driver"]["Ftop_prev"] == 11.0
    assert outer["history"]["hist"]["Ftop"][-1] == 11.0
    assert receipt["removed_population_total"] > 0.0
    # The source generation is immutable.
    _, _, source_after = load_combined_checkpoint(source)
    assert np.array_equal(
        source_after["kinetic_active_vector"], source_arrays["kinetic_active_vector"]
    )
    assert source_outer["driver"]["Ftop_prev"] == 3.0
