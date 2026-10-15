import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arrhenius_fracture.fem_backend_adapter import (
    inspect_fem_backend, inspect_generic_fem_backend, require_fem_backend,
)
from arrhenius_fracture.shared_execution import (
    build_execution_plan, construct_and_attest, execute_shared,
)
from arrhenius_fracture.shared_execution_worker import _multifront_restart_flag
from arrhenius_fracture.shared_options import (
    CavityMode, CrackPathMode, DiagnosticIntervention, KineticsStrategy,
    LoadingMode, PRESETS, PlasticityMode, SharedOptions, SourceLifecycle, SourceLocation,
    TransportStrategy,
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


def test_channel_transport_is_an_independent_executable_binding():
    scalar = build_execution_plan(PRESETS["pf_monotonic_tip_persistent"])
    channel = build_execution_plan(PRESETS["pf_monotonic_tip_persistent_channel"])
    assert channel.executable
    assert channel.executor_module == scalar.executor_module
    assert by_role(channel)["kinetics"].requested == "native_signed"
    assert by_role(channel)["transport"].requested == "channel_resolved"
    assert by_role(channel)["transport"].attribute == "_signed_evolve_channel"
    assert channel.options_sha256 != scalar.options_sha256


@pytest.mark.parametrize("intervention", [
    DiagnosticIntervention.RETAINED_SHIELDING_OFF,
    DiagnosticIntervention.SOURCE_BACKSTRESS_OFF,
    DiagnosticIntervention.BLUNTING_FIXED,
    DiagnosticIntervention.TRANSPORT_RETENTION_OFF,
])
def test_populated_state_diagnostic_maps_to_dedicated_native_adapter(intervention):
    plan = build_execution_plan(SharedOptions(diagnostic_intervention=intervention))
    assert plan.executable
    assert plan.executor_module.endswith("shared_diagnostic_intervention_adapter")
    assert plan.restart_family == "single_front_diagnostic_outer_checkpoint"
    assert by_role(plan)["diagnostic_intervention"].requested == intervention.value


def test_history_reset_maps_to_persisted_reequilibrium_adapter():
    plan = build_execution_plan(SharedOptions(
        diagnostic_intervention=DiagnosticIntervention.HISTORY_RESET_REEQUILIBRATED,
    ))
    assert plan.executable
    assert plan.executor_module.endswith("shared_diagnostic_intervention_adapter")
    assert plan.restart_family == "single_front_diagnostic_outer_checkpoint"


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
    assert plan.restart_family == "v3_6_material_frame_native"


def test_branch_cavity_preset_maps_to_combined_native_adapter():
    plan = build_execution_plan(PRESETS["pf_branching_void_v5"])
    assert plan.executable
    assert plan.executor_module.endswith("shared_branch_cavity_adapter")
    assert plan.restart_family == "v11_branch_cavity_compatibility"
    roles = by_role(plan)
    assert roles["crack_path"].requested == "branching_multifront"
    assert roles["cavity"].requested == "full_order_v5"
    assert roles["crack_path"].module.endswith("crack_network_v11")


def test_branch_cavity_and_cavity_suppressed_routes_remain_distinct():
    enabled = build_execution_plan(PRESETS["pf_branching_void_v5"])
    suppressed = build_execution_plan(PRESETS["pf_branching_current"])
    assert enabled.executor_module.endswith("shared_branch_cavity_adapter")
    assert suppressed.executor_module.endswith(
        "sharp_front_current_source_multifront_physical_v12"
    )
    assert enabled.options_sha256 != suppressed.options_sha256


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
    assert plan.restart_family == "v10_4_tip_plus_bulk_combined_outer"


def test_pure_bulk_maps_to_same_native_full_field_entry_with_distinct_restart_scope():
    plan = build_execution_plan(PRESETS["pf_monotonic_bulk"])
    assert plan.executable
    assert plan.executor_module.endswith("sharp_front_v10_4_bulk_peierls_taylor_audited")
    assert plan.restart_family == "v10_4_pure_bulk_combined_outer"
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


def test_generic_fem_route_is_distinct_and_accepts_preserved_untracked_evidence():
    reference = build_execution_plan(PRESETS["fem_czm_theta0"])
    generic = build_execution_plan(PRESETS["fem_czm_generic_current"])
    receipt = inspect_generic_fem_backend()
    assert reference.executor_module.endswith("theta0_pf_parity")
    assert reference.restart_family == "external_fem_native"
    assert generic.executor_module.endswith("generic_current_pf_companion")
    assert generic.restart_family == "external_fem_generic_current"
    assert generic.options_sha256 != reference.options_sha256
    assert receipt.available
    assert receipt.clean is True
    assert receipt.observed_commit == receipt.expected_commit


def test_external_fem_worker_request_contains_executor_and_compact_output(
    tmp_path, monkeypatch,
):
    class Backend:
        root = str(tmp_path)
        interpreter = "/qualified/fem/python"

        def to_dict(self):
            return {
                "root": self.root,
                "interpreter": self.interpreter,
                "expected_commit": "293491157063484bb10df6adc479f847a6a08ba6",
                "expected_branch": "v10.0.5.18.4.0-theta0-pf-parity",
            }

    monkeypatch.setattr(
        "arrhenius_fracture.fem_backend_adapter.require_fem_backend",
        lambda: Backend(),
    )

    def fake_run(command, **kwargs):
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text())
        assert request["executor_module"].endswith(
            "mode_i_first_passage_v10_0_5_18_4_0_theta0_pf_parity"
        )
        assert request["executor_attribute"] == "main"
        assert request["native_args"][-3:] == [
            "--save-snapshots", "0", "--no-plots",
        ]
        output = Path(request["output_dir"])
        (output / "shared_runtime_attestation.post.json").write_text(
            json.dumps({"status": "executed", "execution_verified": True}) + "\n"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "arrhenius_fracture.shared_execution.subprocess.run", fake_run,
    )
    result = execute_shared(
        PRESETS["fem_czm_theta0"], ["--target-crack-extension-um", "10"],
        tmp_path / "out",
    )
    assert result["execution_verified"] is True


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
