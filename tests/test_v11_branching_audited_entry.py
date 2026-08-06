import json

import pytest

from arrhenius_fracture.live_topology_kernel_v11 import PROVIDER_ID
from arrhenius_fracture.sharp_front_v11_branching_audited import (
    AUDIT_NAME, _write_failure, main, validate_audited_arguments,
)


def test_audited_entry_requires_explicit_mechanistic_branching():
    with pytest.raises(SystemExit, match="requires --mechanistic-branching"):
        validate_audited_arguments(("--mode", "2d"))


@pytest.mark.parametrize("flag", [
    "--fatigue-cycles", "--full-field-plasticity", "--absolute-directional-j",
    "--clone-split", "--mpz-partition", "--topology-interpolation",
])
def test_audited_entry_rejects_forbidden_physics(flag):
    with pytest.raises(SystemExit, match="rejects"):
        validate_audited_arguments(("--mechanistic-branching", flag))


def test_audited_entry_rejects_conflicting_forced_modes():
    with pytest.raises(SystemExit, match="forces --mode=2d"):
        validate_audited_arguments(("--mechanistic-branching", "--mode", "1d"))
    with pytest.raises(SystemExit, match="maximum-fronts=2"):
        validate_audited_arguments(("--mechanistic-branching", "--maximum-fronts", "3"))


def test_audit_only_writes_complete_fail_closed_policy(tmp_path):
    assert main([
        "--mechanistic-branching", "--mode", "2d", "--crack-backend", "sharp_wake",
        "--maximum-fronts", "2", "--out", str(tmp_path), "--audit-only",
        "--temperatures=700", "--theta-deg=45", "--hazard-seed=3621",
        "--material-option=v913_paper_weakT01_0129902_persistent_sites",
    ]) == 0
    payload = json.loads((tmp_path / AUDIT_NAME).read_text())
    assert payload["model_id"].startswith("v11.mechanistic_branching")
    assert payload["policy"]["maximum_fronts"] == 2
    assert payload["policy"]["branch_process_zone_mode"] == "shared_unresolved_cluster"
    assert payload["mechanics_provider_sequence"][-1] == PROVIDER_ID
    assert isinstance(payload["dirty_tree"], bool)


def test_failure_capture_is_atomic_and_preserves_full_traceback(tmp_path):
    try:
        raise RuntimeError("mechanics stopped at exact topology")
    except RuntimeError as error:
        _write_failure(tmp_path, error)
    summary = json.loads((tmp_path / "failure_summary.json").read_text())
    assert summary["exception_class"] == "RuntimeError"
    assert summary["exception_message"] == "mechanics stopped at exact topology"
    traceback_text = (tmp_path / "failure_traceback.txt").read_text()
    assert "Traceback (most recent call last)" in traceback_text
    assert "RuntimeError: mechanics stopped at exact topology" in traceback_text
    assert not list(tmp_path.glob("*.tmp"))
