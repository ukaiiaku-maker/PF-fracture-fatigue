from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.topology_transaction_v11 import (
    EQUILIBRIUM_OBSERVABLE_SCHEMA,
    EquilibriumObservables,
    EquilibriumObservablesUnavailable,
    _measure_accepted_equilibrium,
    require_equilibrium_observables,
)


def test_manufactured_equilibrium_records_direct_reactions_and_energy_identity():
    mesh = SimpleNamespace(ndof=8, area_e=np.asarray((1.0,)))
    boundary = SimpleNamespace(
        top_nodes=np.asarray((2, 3)), bot_nodes=np.asarray((0, 1)),
        left_bot=0, right_bot=1,
    )
    state = SimpleNamespace(
        mesh=mesh, boundary=boundary,
        displacement=np.asarray((0.0, -0.1, 0.0, -0.1, 0.0, 0.1, 0.0, 0.1)),
        ep_gp=np.zeros((3, 1)),
    )
    residual = np.asarray((0.0, -5.0, 0.0, -5.0, 0.0, 5.0, 0.0, 5.0))
    sigma = np.zeros((3, 1))
    ledger = _measure_accepted_equilibrium(state, residual, sigma, 1.0)
    assert ledger["equilibrium_observable_schema"] == EQUILIBRIUM_OBSERVABLE_SCHEMA
    assert ledger["latest_top_reaction_N_per_m"] == 10.0
    assert ledger["latest_bottom_reaction_N_per_m"] == -10.0
    assert ledger["latest_applied_opening_m"] == pytest.approx(0.2)
    assert ledger["latest_compliance_m2_per_N"] == pytest.approx(0.02)
    assert ledger["latest_external_work_J_per_m"] == pytest.approx(1.0)
    assert ledger["latest_energy_reaction_identity"] == pytest.approx(0.0)


def test_omitted_or_zero_reaction_ledger_fails_closed_instead_of_becoming_zero():
    with pytest.raises(EquilibriumObservablesUnavailable, match="no certified"):
        require_equilibrium_observables(SimpleNamespace(energy_ledgers={}))
    complete = {
        "equilibrium_observable_schema": EQUILIBRIUM_OBSERVABLE_SCHEMA,
        "latest_top_reaction_N_per_m": 0.0,
        "latest_bottom_reaction_N_per_m": 0.0,
        "latest_reaction_N_per_m": 0.0,
        "latest_applied_opening_m": 4.0e-7,
        "latest_compliance_m2_per_N": 1.0,
        "latest_external_work_J_per_m": 0.0,
        "latest_stored_recoverable_energy_J_per_m": 0.02,
        "latest_plastic_eigenstrain_half_work_J_per_m": 0.0,
        "latest_energy_identity_reference_J_per_m": 0.0,
        "latest_residual_l2_N_per_m": 0.0,
        "latest_free_dof_residual_l2_N_per_m": 0.0,
        "latest_constrained_reaction_l2_N_per_m": 0.0,
        "latest_top_bottom_reaction_balance": 0.0,
        "latest_energy_reaction_identity": 0.0,
    }
    with pytest.raises(EquilibriumObservablesUnavailable, match="physical floor"):
        require_equilibrium_observables(SimpleNamespace(energy_ledgers=complete))


def test_accepted_dbtt_state_has_physical_reaction_compliance_and_energy():
    from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
    from arrhenius_fracture.voiding_production_v5 import build_production_void_state, observables

    state, _ = build_production_void_state(bundle=material_bundle("DBTT"), enabled=True)
    row = observables(state, "V6_ACCEPTED_STATE_EQUILIBRIUM")
    assert row["reaction_N_per_m"] != 0.0
    assert row["top_reaction_N_per_m"] == row["reaction_N_per_m"]
    assert row["bottom_reaction_N_per_m"] != 0.0
    assert np.isfinite(row["compliance_m2_per_N"])
    assert row["compliance_m2_per_N"] > 0.0
    assert row["stored_recoverable_energy_J_per_m"] > 0.0
    assert row["top_bottom_reaction_balance"] <= 0.03
    assert row["energy_reaction_identity"] <= 0.01
    assert isinstance(state.equilibrium_observables, EquilibriumObservables)
    assert state.equilibrium_observables.mean_reaction_magnitude_N_per_m > 0.0
    assert state.equilibrium_observables.source_state_fingerprint
    assert state.equilibrium_observables.mesh_fingerprint
    assert state.equilibrium_observables.material_fingerprint
    missing = replace(state, energy_ledgers={}, equilibrium_observables=None)
    with pytest.raises(EquilibriumObservablesUnavailable, match="no certified"):
        observables(missing, "V6_MISSING_LEDGER")


def test_typed_equilibrium_observations_survive_checkpoint_restart(tmp_path):
    from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
    from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
    from arrhenius_fracture.voiding_production_v5 import build_production_void_state

    state, _ = build_production_void_state(bundle=material_bundle("DBTT"), enabled=True)
    path = tmp_path / "accepted.json"
    manifest = write_checkpoint(state, path)
    restored = restore_checkpoint(path)
    assert manifest["equilibrium_observables"] == state.equilibrium_observables.to_dict()
    assert restored.equilibrium_observables == state.equilibrium_observables
    assert dict(require_equilibrium_observables(restored)) == dict(require_equilibrium_observables(state))
