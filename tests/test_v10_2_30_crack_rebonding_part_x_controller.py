"""PX2.5 item 7: physical controller preflight qualification.

Exercises the real controller (scripts/part_x_physical_controller.py)
against the actual PX2 screen_job_registry.csv, in --preflight mode (a
handful of blocks, not the real 12-event/60um screen budget) and against
a SCRATCH run root (never the real runs/crack_rebonding_part_x_v1/) so
this qualification never creates a real physical result path -- mission
section 2's gate (no physical result directory before PX2 lands) stays
intact; this is controller-machinery qualification, not science.

The "controller never launches a BLOCKED_* row" tests use a SYNTHETIC
fixture registry (a small, self-contained CSV this file writes itself),
not the real developed_job_registry.csv -- that file's authorized/blocked
mix is expected to keep evolving as later PX stages resolve more gates
(PX3's adaptive selection legitimately authorized D1/D2/D5/D6 for PX4;
D3/D7 remain blocked pending further work), so pinning this controller-
machinery test to "the real registry currently has zero authorized rows"
would make it fail every time real mission progress is made, for a reason
that has nothing to do with the controller's own gating logic. The
synthetic fixture isolates exactly what this test is meant to prove: the
controller launches ONLY rows whose status literally starts with
"AUTHORIZED_", regardless of how many BLOCKED_*/alias rows sit alongside
them.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import part_x_physical_controller as controller  # noqa: E402


REGISTRY_PATH = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1" / "screen_job_registry.csv"
DEVELOPED_REGISTRY_PATH = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1" / "developed_job_registry.csv"

_SYNTHETIC_FIELDNAMES = [
    "protocol", "row_name", "config_hash", "material_row_hash", "Kmax_Pa_sqrt_m", "R",
    "frequency_Hz", "minimum_load_hold_s", "chemistry_factor", "K_rebond_max_target_Pa_sqrt_m",
    "seed", "integrator_mode", "physical_producer_sha", "canonical_job_key", "cohesion",
    "note", "alias_of_protocol", "alias_of_row_name", "status",
]


def _write_all_blocked_registry(path: Path) -> None:
    """A small, self-contained registry every one of whose rows is
    fail-closed (BLOCKED_*/alias, never AUTHORIZED_*) -- exercises the
    controller's own gating logic in isolation from any real mission
    registry's current, evolving authorization state."""
    rows = [
        {
            "protocol": "SYNTH1", "row_name": "SYNTH_ROW", "config_hash": "deadbeef",
            "material_row_hash": "deadbeef", "Kmax_Pa_sqrt_m": "18000000.0", "R": "-0.5",
            "frequency_Hz": "1000.0", "minimum_load_hold_s": "0.0", "chemistry_factor": "1.0",
            "K_rebond_max_target_Pa_sqrt_m": "900000.0", "seed": "1720", "integrator_mode": "explicit",
            "physical_producer_sha": "0" * 40, "canonical_job_key": f"synth_key_{i}", "cohesion": "finite",
            "note": "", "alias_of_protocol": "", "alias_of_row_name": "", "status": status,
        }
        for i, status in enumerate([
            "BLOCKED_PENDING_PX3_COMPLETION", "BLOCKED_PENDING_PX3_FREQUENCY_GATE",
            "BLOCKED_PENDING_STAGE1_DECISION", "ALIAS_OF_EXISTING_JOB",
        ])
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SYNTHETIC_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_registry_only_screen_rows_are_authorized():
    jobs = controller._load_registry(REGISTRY_PATH)
    authorized = [j for j in jobs if j["status"].startswith("AUTHORIZED_")]
    assert len(authorized) > 0
    assert all(j["status"] == "AUTHORIZED_PX3" for j in authorized)


def test_developed_registry_status_vocabulary_is_recognized():
    """Every developed_job_registry.csv row must be one of the known,
    fail-closed-by-default statuses -- this evolves as later PX stages
    resolve more gates (unlike the removed "zero authorized rows"
    invariant, which broke the moment PX3's adaptive selection correctly
    authorized any row), so this only pins the VOCABULARY, not a count."""
    jobs = controller._load_registry(DEVELOPED_REGISTRY_PATH)
    statuses = {j["status"] for j in jobs}
    assert statuses <= {
        "BLOCKED_PENDING_PX3_COMPLETION", "BLOCKED_PENDING_PX3_FREQUENCY_GATE",
        "BLOCKED_PENDING_PX3_DWELL_GATE", "BLOCKED_PENDING_PX3_PASSIVATION_GATE",
        "BLOCKED_PENDING_PX3_PERSISTENT_DISTINCTION", "BLOCKED_PENDING_PX3_COHESIVE_NONLINEARITY_GATE",
        "BLOCKED_PENDING_STAGE1_DECISION", "BLOCKED_PROTOCOL_MISMATCH",
        "BLOCKED_DWELL_GATE_NOT_SATISFIED", "ALIAS_OF_EXISTING_JOB",
        "AUTHORIZED_PX4",
    }


def test_controller_never_launches_a_blocked_row(tmp_path):
    """Point the controller at an all-BLOCKED_*/alias synthetic registry
    and confirm it launches nothing."""
    synthetic_registry = tmp_path / "synthetic_all_blocked_registry.csv"
    _write_all_blocked_registry(synthetic_registry)
    scratch_root = tmp_path / "scratch_runs"
    summary = controller.run(
        registry_path=synthetic_registry, preflight=True, max_jobs=None, allow_drift=True,
        run_root=scratch_root, require_clean=False, enforce_head=False,
    )
    assert summary["launched"] == 0
    assert summary["total_authorized_in_registry"] == 0


def test_controller_preflight_launches_authorized_screen_jobs(tmp_path):
    scratch_root = tmp_path / "scratch_runs"
    summary = controller.run(
        registry_path=REGISTRY_PATH, preflight=True, max_jobs=2, allow_drift=True,
        run_root=scratch_root, require_clean=False, enforce_head=False,
    )
    assert summary["launched"] == 2
    assert summary["skipped_duplicate"] == 0

    state = json.loads((scratch_root / "controller_state.json").read_text())
    assert len(state["jobs"]) == 2
    for key, record in state["jobs"].items():
        assert record["status"] == "COMPLETE_PREFLIGHT"
        result_dir = Path(record["result_dir"])
        assert result_dir.exists()
        result_path = result_dir / "result.json"
        assert result_path.exists()
        result = json.loads(result_path.read_text())
        assert result["preflight"] is True
        assert result["blocks_run"] == 5
        # Virgin path: exactly the files this job created, nothing carried
        # over from a prior (nonexistent) run.
        assert (result_dir / "job_status.json").exists()


def test_controller_rejects_relaunch_into_a_nonvirgin_path(tmp_path):
    scratch_root = tmp_path / "scratch_runs"
    controller.run(
        registry_path=REGISTRY_PATH, preflight=True, max_jobs=1, allow_drift=True,
        run_root=scratch_root, require_clean=False, enforce_head=False,
    )
    # Running again with the SAME scratch root and the same first job must
    # skip it as already-complete (canonical-key dedup), not error and not
    # silently overwrite.
    summary = controller.run(
        registry_path=REGISTRY_PATH, preflight=True, max_jobs=1, allow_drift=True,
        run_root=scratch_root, require_clean=False, enforce_head=False,
    )
    assert summary["launched"] == 0
    assert summary["skipped_duplicate"] == 1


def test_require_clean_worktree_check_raises_when_dirty(tmp_path, monkeypatch):
    def _fake_dirty_status(*args, **kwargs):
        class _R:
            stdout = " M some_file.py\n"
        return _R()

    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **k: (
        _fake_dirty_status() if a[0][:2] == ["git", "status"] else
        pytest.fail("unexpected subprocess.run call in this monkeypatch")
    ))
    with pytest.raises(RuntimeError, match="not clean"):
        controller._require_clean_worktree()
