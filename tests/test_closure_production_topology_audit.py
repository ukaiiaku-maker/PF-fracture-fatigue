import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import qualify_voiding_v5_closure_production_topology as audit


def manifest(root):
    values = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file() and p.name != "sha256_manifest.json"}
    (root / "sha256_manifest.json").write_text(json.dumps(values, sort_keys=True))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "production"
    (root / "checkpoints").mkdir(parents=True)
    sha = "a" * 40
    (root / "transfer_manifest.json").write_text(json.dumps({
        "schema": "v12.production-source-transfer/2", "executed_code_sha": sha,
        "physical_registry": [list(level) for level in audit.LEVELS]}))
    for n, r in audit.LEVELS:
        prefix = f"production_{n}_{r}"
        for stage in ("pre", "connected"):
            (root / "checkpoints" / f"{prefix}_{stage}.json").write_text(stage)
        (root / f"{prefix}.json").write_text(json.dumps({"case_id": f"production:{n}:{r}",
            "executed_code_sha": sha, "connection_executed": True,
            "initial_state_fingerprint": prefix + "_pre",
            "connected_state_fingerprint": prefix + "_connected"}))
    manifest(root)
    restored = []

    def restore(path):
        restored.append(path)
        return SimpleNamespace(identity=path.stem)

    monkeypatch.setattr(audit, "restore_checkpoint", restore)
    monkeypatch.setattr(audit, "fingerprint", lambda state: state.identity)
    monkeypatch.setattr(audit, "auditor_identity", lambda: "b" * 40)
    monkeypatch.setattr(audit, "stagewise_topology", lambda state: {
        "passed": True, "topology_source_fingerprint": state.identity,
        "independent_cut": {"intact_cross_graph_path_exists": False, "insufficient_seed_segment_ids": []},
        "actual_cavity_cycle": {"passed": True}, "actual_component_incidence": {"passed": True}})
    monkeypatch.setattr(audit, "conservation", lambda state, initial: {
        "passed": True, "active_tip_ids": [], "support_tip_ids": []})
    return root, restored


def test_four_independent_cases_preserve_exact_provenance_and_fingerprints(bundle):
    root, restored = bundle
    result = audit.audit_bundle(root)
    assert result["passed"] and len(restored) == 8
    assert result["producer_code_sha"] == "a" * 40
    assert result["audit_code_sha"] == "b" * 40
    assert result["classification"] == "DERIVED_EXISTING_CHECKPOINTS_NOT_NEW_PHYSICAL_EXECUTIONS"
    assert [row["case_id"] for row in result["rows"]] == [f"production:{n}:{r}" for n, r in audit.LEVELS]
    assert all(row["passed"] and row["pre_unchanged"] and row["connected_unchanged"] for row in result["rows"])
    assert all(len(row["certificate_fingerprints"]) == 3 for row in result["rows"])


@pytest.mark.parametrize("defect", ["tamper", "extra", "missing", "symlink"])
def test_complete_manifest_is_verified_before_any_checkpoint_restore(bundle, defect):
    root, restored = bundle
    target = root / "checkpoints/production_32_12_pre.json"
    if defect == "tamper":
        target.write_text("tampered")
    elif defect == "extra":
        (root / "unexpected.json").write_text("extra")
    elif defect == "missing":
        target.unlink()
    else:
        (root / "alias").symlink_to(target)
    with pytest.raises(ValueError):
        audit.audit_bundle(root)
    assert not restored


@pytest.mark.parametrize("predicate", ["topology", "accounting", "incidence", "mutation", "exception"])
def test_failed_predicates_are_retained_and_do_not_skip_other_cases(bundle, monkeypatch, predicate):
    root, restored = bundle
    original = audit.stagewise_topology

    def topology(state):
        result = original(state)
        if state.identity == "production_32_12_connected":
            if predicate == "exception":
                raise RuntimeError("independent certificate failed")
            if predicate == "mutation":
                state.identity += ":changed"
            if predicate == "topology":
                result["passed"] = False
            if predicate == "incidence":
                result["actual_component_incidence"]["passed"] = False
        return result

    monkeypatch.setattr(audit, "stagewise_topology", topology)
    if predicate == "accounting":
        monkeypatch.setattr(audit, "conservation", lambda state, pre: {
            "passed": state.identity != "production_32_12_connected"})
    result = audit.audit_bundle(root)
    assert not result["passed"] and not result["rows"][0]["passed"]
    assert all(row["passed"] for row in result["rows"][1:])
    assert len(restored) == 8


def test_claimed_checkpoint_identity_cannot_replace_actual_fingerprint(bundle):
    root, restored = bundle
    path = root / "production_32_12.json"
    row = json.loads(path.read_text())
    row["connected_state_fingerprint"] = "invented"
    path.write_text(json.dumps(row))
    manifest(root)
    result = audit.audit_bundle(root)
    assert not result["passed"]
    assert "fingerprint mismatch" in result["rows"][0]["exception"]["message"]


def test_cli_writes_failed_diagnostic_and_returns_failure(bundle, tmp_path, monkeypatch):
    root, _ = bundle
    monkeypatch.setattr(audit, "conservation", lambda state, pre: {"passed": False})
    output = tmp_path / "derived.json"
    assert audit.main([str(root), str(output)]) == 1
    assert not json.loads(output.read_text())["passed"]
    with pytest.raises(ValueError, match="overwrite"):
        audit.main([str(root), str(output)])


def test_cli_cannot_write_inside_source_bundle(bundle):
    root, restored = bundle
    with pytest.raises(ValueError, match="outside"):
        audit.main([str(root), str(root / "derived.json")])
    assert not restored
