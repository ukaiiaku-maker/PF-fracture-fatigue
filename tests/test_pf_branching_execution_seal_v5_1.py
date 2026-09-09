import json
from pathlib import Path
import pytest

from scripts.build_verify_pf_branching_v5_1_archive import (
    build_archive, rewrite_archive_for_negative_test, verify_archive,
)
from scripts.run_pf_current_source_branching_corrected_pair_v5_1 import (
    EXPECTED_HASHES, case_plan, identity_audit, sealed_environment,
)


def test_sealed_environment_pins_full_campaign_and_excludes_unqualified_stop(tmp_path):
    environment = sealed_environment(
        execution_worktree=tmp_path / "source",
        family=tmp_path / "family.json",
        mechanical=tmp_path / "mechanical.json",
        cache=tmp_path / "case" / "cache",
    )
    required = {
        "PARAMETER_CAMPAIGN": "1",
        "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_HAZARD_SEED": "3621",
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5",
        "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
        "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION": "0.1",
        "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar",
        "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1",
        "ANISOTROPIC_EMISSION_ENABLED": "1",
        "KERNEL_STRICT_FAMILY_OVERRIDE": "1",
        "PERSISTENT_SOURCE_MIN_WIDTH_UM": "0",
        "ONED_V2_TP_STATE_DIAGNOSTICS": "events",
        "PYTHONUNBUFFERED": "1",
    }
    assert all(environment[key] == value for key, value in required.items())
    assert "PF_QUALIFIED_DAUGHTER_STOP_UM" not in environment
    assert environment["MECHANICAL_CONFIG_SHA256"] == EXPECTED_HASHES["mechanical"]


def test_case_plan_has_hard_ceiling_own_checkpoint_and_fresh_cache(tmp_path):
    family = tmp_path / "family.json"
    mechanical = tmp_path / "mechanical.json"
    checkpoint = tmp_path / "step1.json"
    family.write_text("family")
    mechanical.write_text("mechanical")
    checkpoint.write_text("checkpoint")
    Path(str(checkpoint) + ".state.pkl").write_bytes(b"state")
    plan = case_plan(
        role="control", maximum_fronts=1, checkpoint=checkpoint,
        outroot=tmp_path / "absent-output", execution_worktree=tmp_path,
        family=family, mechanical=mechanical,
    )
    command = plan["command"]
    assert command[command.index("--maximum-fronts") + 1] == "1"
    assert command[command.index("--v11-restart-checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--target-crack-extension-um") + 1] == "300"
    assert plan["qualified_daughter_early_stop_enabled"] is False
    assert Path(plan["destination_cache_root"]).parent == Path(plan["output_directory"])


def test_identity_audit_fails_on_any_physical_difference():
    common = {
        "role": "control", "maximum_fronts": 1, "manifest": "a",
        "state_file": "a.state", "manifest_sha256": "a", "state_sha256": "a",
        "manifest_state_sha256": "a", "topology_policy_value": False,
        "physical_time_s": 0.0, "field": "same",
    }
    enabled = {
        **common, "role": "enabled", "maximum_fronts": 2,
        "manifest": "b", "state_file": "b.state", "manifest_sha256": "b",
        "state_sha256": "b", "manifest_state_sha256": "b",
        "topology_policy_value": True,
    }
    assert identity_audit(common, enabled)["qualification"] == "PASS"
    with pytest.raises(RuntimeError, match="field"):
        identity_audit(common, {**enabled, "field": "different"})


def _minimal_archive_source(root: Path):
    required = [
        "PF_CURRENT_SOURCE_BRANCHING_EXECUTION_SEAL_V5_1.md",
        "pf_branching_v5_execution_source.patch",
        "pf_branching_v5_record_delta.patch",
        "pf_branching_v5_1_archive_presence_audit.json",
        "pf_branching_v5_1_tensor_conditioning_audit.csv",
        "pf_branching_qualified_daughter_stop_audit.json",
        "run_pf_current_source_branching_corrected_pair_v5_1.py",
        "immutable_source/pf_branching_v5_execution_source_725709f.tar",
    ]
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    family = root / "pinned_inputs/theta40_signed_kernel_family.json"
    mechanical = root / "pinned_inputs/theta40_mechanical_configuration.json"
    family.parent.mkdir(parents=True, exist_ok=True)
    family.write_text("family")
    mechanical.write_text("mechanical")
    snapshot = root / "source_snapshot_execution/example.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("pass\n")
    import hashlib
    family_hash = hashlib.sha256(family.read_bytes()).hexdigest()
    mechanical_hash = hashlib.sha256(mechanical.read_bytes()).hexdigest()
    (root / "pf_branching_matched_pair_launch_preflight.json").write_text(json.dumps({
        "cases": [{
            "signed_kernel_family_sha256": family_hash,
            "mechanical_configuration_sha256": mechanical_hash,
        }],
    }))
    (root / "pf_branching_v5_1_source_provenance.json").write_text(json.dumps({
        "source_snapshot_sha256": {"source_snapshot_execution/example.py": "unused"},
    }))


def test_archive_verifier_passes_complete_and_fails_removed_or_modified(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _minimal_archive_source(source)
    archive = tmp_path / "complete.zip"
    assert build_archive(source, archive)["result"] == "PASS"
    assert verify_archive(archive)["all_hashes_match"] is True

    removed = tmp_path / "removed.zip"
    rewrite_archive_for_negative_test(
        archive, removed, remove="pf_branching_v5_execution_source.patch",
    )
    with pytest.raises(RuntimeError, match="presence mismatch"):
        verify_archive(removed)

    modified = tmp_path / "modified.zip"
    rewrite_archive_for_negative_test(
        archive, modified, modify="pf_branching_v5_execution_source.patch",
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_archive(modified)
