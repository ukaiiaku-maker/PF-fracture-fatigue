import json

import pytest

from arrhenius_fracture.fem_backend_adapter import inspect_fem_backend, require_fem_backend
from arrhenius_fracture.shared_execution import (
    build_execution_plan, construct_and_attest, execute_shared,
)
from arrhenius_fracture.shared_execution_worker import _multifront_restart_flag
from arrhenius_fracture.shared_options import (
    CavityMode, CrackPathMode, KineticsStrategy, LoadingMode, PRESETS,
    PlasticityMode, SharedOptions, SourceLifecycle, SourceLocation,
)
from arrhenius_fracture.branch_cluster_v11 import BranchClusterState
from arrhenius_fracture.stateful_multifront_production_v12 import _json_compatible_state


def by_role(plan):
    return {item.role: item for item in plan.bindings}


def test_loading_role_is_not_discarded_by_bulk_scheduler_selection():
    options = SharedOptions(
        loading=LoadingMode.EXPLICIT_CYCLIC,
        source_location=SourceLocation.BULK_UNIFORM,
        source_lifecycle=SourceLifecycle.FINITE_INVENTORY,
        plasticity=PlasticityMode.FULL_FIELD_EVOLVING_RHO,
        kinetics=KineticsStrategy.REVERSIBLE_SIGNED,
    )
    plan = build_execution_plan(options)
    roles = by_role(plan)
    assert roles["loading"].attribute == "FatigueWaveform"
    assert roles["source_location"].attribute == "accepted_bulk_load_step"
    assert not plan.executable
    assert "no qualified shared native executor" in plan.limitation


def test_cavity_does_not_imply_branch_permission_or_current_source():
    options = SharedOptions(cavity=CavityMode.FULL_ORDER_V5)
    plan = build_execution_plan(options)
    roles = by_role(plan)
    assert roles["crack_path"].requested == "single_front"
    assert roles["source_lifecycle"].requested == "persistent_reusable"
    assert roles["cavity"].attribute == "build_production_void_state"


def test_fixed_range_branch_is_preserved_but_explicitly_pending():
    options = SharedOptions(
        loading=LoadingMode.FIXED_DELTA_K,
        crack_path=CrackPathMode.BRANCHING_MULTIFRONT,
        source_lifecycle=SourceLifecycle.CURRENT_SOURCE,
        kinetics=KineticsStrategy.REVERSIBLE_SIGNED,
    )
    plan = build_execution_plan(options)
    assert by_role(plan)["loading"].attribute == "install_fixed_deltaK_waveform"
    assert by_role(plan)["crack_path"].requested == "branching_multifront"
    assert not plan.executable


def test_construct_attestation_measures_loaded_files_and_seed(monkeypatch):
    options = PRESETS["pf_monotonic_tip_persistent"]
    monkeypatch.setenv("CLEAVAGE_HAZARD_SEED", str(options.random_seed))
    plan = build_execution_plan(options)
    executor, receipt = construct_and_attest(plan, options)
    assert callable(executor)
    assert receipt["status"] == "constructed"
    assert receipt["execution_verified"] is False
    assert receipt["seed"]["hazard"] == options.random_seed
    assert receipt["observed_bindings"]
    assert all(len(row["module_sha256"]) == 64 for row in receipt["observed_bindings"])
    assert receipt["source"]["commit"]


def test_monotonic_reversible_uses_effective_native_executor():
    options = SharedOptions(kinetics=KineticsStrategy.REVERSIBLE_SIGNED)
    plan = build_execution_plan(options)
    assert plan.executable
    assert plan.executor_module.endswith("sharp_front_v10_2_27_reversible_audited")
    assert by_role(plan)["kinetics"].attribute == "install_reversible_transport"


def test_finite_inventory_maps_to_native_finite_site_driver():
    options = SharedOptions(source_lifecycle=SourceLifecycle.FINITE_INVENTORY)
    plan = build_execution_plan(options)
    assert plan.executable
    assert plan.executor_module == "arrhenius_fracture.sharp_front_v10_1"
    assert by_role(plan)["source_lifecycle"].module.endswith("sharp_front_v10_1")


def test_operator_consistent_preset_maps_to_developed_native_campaign():
    plan = build_execution_plan(PRESETS["pf_operator_consistent_cyclic"])
    assert plan.executable
    assert plan.executor_module.endswith("shared_operator_consistent_adapter")
    assert plan.restart_family == "v3_6_event_checkpoint_write_only"


@pytest.mark.parametrize("name", [
    "pf_operator_consistent_cyclic",
    "pf_monotonic_bulk",
    "pf_monotonic_tip_plus_bulk",
])
def test_public_restart_is_rejected_for_write_only_or_unsupported_paths(name, tmp_path):
    restart = tmp_path / "would-be-restart"
    restart.mkdir()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="not supported by this public execution path"):
        execute_shared(PRESETS[name], [], output, restart_from=restart)
    assert not output.exists()


def test_hold_dwell_maps_to_native_ramp_then_hold_fatigue_entry():
    plan = build_execution_plan(PRESETS["pf_hold_dwell_tip_persistent"])
    assert plan.executable
    assert plan.executor_module.endswith("sharp_front_v10_2_30_energy_gated_fatigue")
    assert by_role(plan)["loading"].consumer.startswith("native ramp-then-held")


def test_combined_tip_and_bulk_maps_to_native_v104_full_field_entry():
    plan = build_execution_plan(PRESETS["pf_monotonic_tip_plus_bulk"])
    assert plan.executable
    assert plan.executor_module.endswith("sharp_front_v10_4_bulk_peierls_taylor_audited")
    assert by_role(plan)["source_location"].requested == "tip_plus_bulk"
    assert by_role(plan)["plasticity"].requested == "tip_plus_full_field"


def test_pure_bulk_maps_to_same_native_full_field_entry_with_distinct_restart_scope():
    plan = build_execution_plan(PRESETS["pf_monotonic_bulk"])
    assert plan.executable
    assert plan.executor_module.endswith("sharp_front_v10_4_bulk_peierls_taylor_audited")
    assert plan.restart_family == "v10_4_pure_bulk_full_field_no_public_restart"
    assert by_role(plan)["source_location"].requested == "bulk_uniform"


@pytest.mark.parametrize("name", [
    "pf_cavity_v5_single_front", "pf_cavity_v5_disabled_control",
])
def test_v5_cavity_and_matching_disabled_control_share_native_adapter(name):
    plan = build_execution_plan(PRESETS[name])
    assert plan.executable
    assert plan.executor_module.endswith("shared_v5_cavity_adapter")
    assert plan.restart_family == "v11_native_full_state_v5_physical_pair"
    assert by_role(plan)["crack_path"].requested == "single_front"


def test_multifront_restart_dispatches_v11_without_discarding_owners(tmp_path):
    checkpoint = tmp_path / "accepted.json"
    checkpoint.write_text(json.dumps({"schema": "v11.production-branch-network-checkpoint/2"}))
    assert _multifront_restart_flag(checkpoint) == "--v11-restart-checkpoint"
    assert _multifront_restart_flag(tmp_path / "latest.v12.pkl") == "--v12-restart-checkpoint"


def test_fem_adapter_verifies_real_pinned_checkout_and_rejects_missing(tmp_path):
    receipt = inspect_fem_backend()
    assert receipt.available
    assert receipt.observed_commit == receipt.expected_commit
    assert receipt.clean is True
    assert len(receipt.entry_sha256) == 64
    assert receipt.execution_verified is False
    with pytest.raises(RuntimeError, match="backend unavailable"):
        require_fem_backend(tmp_path)


def test_plan_is_json_serializable():
    payload = build_execution_plan(PRESETS["pf_cyclic_tip_persistent"]).to_dict()
    assert json.loads(json.dumps(payload))["executable"] is True


def test_v12_wake_audit_serializes_typed_branch_cluster_state():
    # Historical V11 restarts legitimately carry this typed object in their
    # junction registry; a V12 transaction record must preserve it as data.
    cluster = BranchClusterState(
        cluster_id="j1", junction_xy_m=(0.0, 0.0), parent_branch_id="b0",
        arm_branch_ids=("b1", "b2"), shared_process_state={"clock": 0.5},
        conserved_ledgers={},
    )
    payload = _json_compatible_state({"branch_clusters": (cluster,)})
    assert json.loads(json.dumps(payload))["branch_clusters"][0]["cluster_id"] == "j1"
