from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.multifront_field_atlas_v12 import (
    CLAIM_LABEL, SparseAcceptedStateObserver, export_snapshot,
)
from scripts.run_pf_current_source_multifront_field_atlas_v12 import (
    CASES, FAMILY_PHYSICS, FAMILY_SHA256, MECHANICAL_FINGERPRINT, ROWS,
    common_arguments, registry_row, terminal_label,
)


def test_eight_case_registry_is_exact_and_hash_bound():
    assert CASES == (
        "Peak_300K", "Peak_1000K", "DBTT_300K", "DBTT_1000K",
        "weakT_300K", "weakT_1000K", "ceramic_300K", "ceramic_1000K",
    )
    expected = {
        "v913_zeroD_sobol_0242980": "e7bac08adfe18342f40168265f467749bdc4857477aad7cdd26086ddd3db8017",
        "v913_zeroD_sobol_0202500": "6d2d454e3c79e2171b8547c895a0ee2d42fa4897af90354efe906e7b27d044d4",
        "oneD_v2_focused_weak_T_0016": "275fc39b3df6a9c70c16076807893a080b1d0b61c882d6ec142e0bd6329bdcd9",
        "oneD_v2_focused_ceramic_like_0018": "302e4fa9ff684c81280f25e676f756010b39671fcf2cdbe469b2df37466e4a2d",
    }
    for canonical, alias in ROWS.values():
        row, digest = registry_row(canonical)
        assert row["option_key"] == alias
        assert digest == expected[canonical]
    assert len(FAMILY_SHA256) == len(FAMILY_PHYSICS) == len(MECHANICAL_FINGERPRINT) == 64


def test_physical_arguments_have_no_birth_limit_and_pin_requested_conditions(tmp_path):
    alias = ROWS["Peak"][1]
    values = common_arguments(alias, 300, tmp_path / "family.json", tmp_path / "out")
    joined = " ".join(values)
    assert "--v12-stop-after-total-births" not in joined
    assert "--crystal-theta-deg 40" in joined
    assert "--dU 2e-7" in joined and "--dt 8.4" in joined
    assert "--da-phys 5e-6" in joined
    assert "--bulk-plasticity-mode tip_only" in joined
    assert "--directional-j-mode root_signed" in joined
    assert "--active-shielding" in values and "--signed-active-shielding" in values
    assert "--mobile-shield-fraction 0" in joined and "--no-wake-shielding" in values
    assert "--crack-backend sharp_wake" in joined


def test_terminal_labels_preserve_policy_and_fail_closed_identity():
    assert terminal_label("maximum_network_forward_reach_target_reached") == "V12_FIELD_ATLAS_TARGET_1000UM_REACHED"
    assert terminal_label("configured_front_resource_limit_reached") == "V12_FIELD_ATLAS_STOPPED_POLICY_BOUND_AT_SIX_ACTIVE_FRONTS"
    assert terminal_label("owner_local_kernel_coordinate_outside_qualified_family_domain") == "V12_FIELD_ATLAS_STOPPED_OWNER_KERNEL_ENVELOPE"
    assert terminal_label("opening_scale_not_resolved_above_probe_uncertainty") == (
        "V12_FIELD_ATLAS_STOPPED_FAIL_CLOSED_OPENING_SCALE_NOT_RESOLVED_ABOVE_PROBE_UNCERTAINTY"
    )


def test_portable_export_contains_exact_fe_and_derived_fields(tmp_path, monkeypatch):
    from arrhenius_fracture import multifront_field_atlas_v12 as atlas
    monkeypatch.setattr(atlas, "_owner_arrays", lambda runtime: ({}, {"archived_reservoirs": {}}))
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]], dtype=int)
    mesh = SimpleNamespace(
        nodes=nodes, elems=elems, ne=1,
        B_e=np.array([[[-1, 0, 1, 0, 0, 0], [0, -1, 0, 0, 0, 1], [-1, -1, 0, 1, 1, 0]]], dtype=float),
        element_damage_gp=None,
    )
    displacement = np.array([0.0, 0.0, 1e-6, 0.0, 0.0, 2e-6])
    state = SimpleNamespace(
        mesh=mesh, displacement=displacement, damage=np.array([1.0, 0.0, 0.0]),
        ep_gp=np.array([[3e-3], [4e-3], [1e-3]]), rho_gp=np.array([5e12]),
        material=SimpleNamespace(nu=0.3),
    )
    network = CrackNetworkState.one_tip(((0.0, 0.0), (0.5e-3, 0.0)))
    runtime = SimpleNamespace(
        accepted_state_id="accepted", stress_field_state_id="stress",
        registry_fingerprint="registry", topology_fingerprint="topology",
        crack_network=network, cumulative_branch_births=0,
        active_front_ids=network.active_tip_ids, owner_by_front={}, junctions={},
    )
    sigma = np.array([[10.0], [2.0], [3.0]])
    metadata = export_snapshot(
        tmp_path, case="Peak_300K", runtime=runtime, accepted_fem_state=state,
        accepted_stress_field=sigma, physical_time_s=0.0, accepted_opening_m=0.0,
        step_count=0, mechanics_source_identity="test", selection_reasons=("initial_accepted_state",),
    )
    arrays = dict(np.load(metadata["files"]["fields_npz"], allow_pickle=False))
    assert np.array_equal(arrays["nodes_m"], nodes)
    assert np.array_equal(arrays["elements"], elems)
    assert np.array_equal(arrays["sigma_xx_Pa"], sigma[0])
    assert np.allclose(arrays["displacement_magnitude_m"], [0, 1e-6, 2e-6])
    assert np.allclose(
        arrays["equivalent_plastic_strain"],
        np.sqrt((2 / 3) * (3e-3**2 + 4e-3**2 + 2 * 1e-3**2)),
    )
    assert Path(metadata["files"]["fields_vtu"]).is_file()
    assert Path(metadata["files"]["crack_network_vtp"]).is_file()
    assert metadata["claim_label"] == CLAIM_LABEL


def test_sparse_observer_records_birth_and_owner_change_after_accept(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "arrhenius_fracture.multifront_field_atlas_v12.export_snapshot",
        lambda *args, **kwargs: calls.append(kwargs) or {},
    )
    network = CrackNetworkState.one_tip(((0.0, 0.0), (0.5e-3, 0.0)))
    region = SimpleNamespace(
        member_front_ids=frozenset(network.active_tip_ids), unresolved_junction_ids=frozenset(),
        process_engine_id="engine", source_reservoir_id=None,
    )
    runtime = SimpleNamespace(
        crack_network=network, cumulative_branch_births=1,
        owner_by_front={network.active_tip_ids[0]: "owner"}, process_regions={"owner": region},
    )
    def writer(*args, **kwargs):
        (tmp_path / "latest.pkl").write_bytes(b"accepted")
        return {"ok": True}
    observer = SparseAcceptedStateObserver(
        case="Peak_300K", snapshot_root=tmp_path / "snapshots",
        checkpoint_path=tmp_path / "latest.pkl", original_writer=writer,
        previous_births=0, previous_owner_signature="different",
    )
    result = observer(
        runtime, tmp_path / "latest.pkl", accepted_fem_state=object(),
        accepted_stress_field=np.ones((3,1)), physical_time_s=1.0,
        accepted_opening_m=2e-7, step_count=1, mechanics_source_identity="test",
    )
    assert result == {"ok": True}
    assert len(calls) == 1
    assert set(calls[0]["selection_reasons"]) == {
        "accepted_binary_branch_birth_1", "process_owner_handoff_or_partition",
    }
