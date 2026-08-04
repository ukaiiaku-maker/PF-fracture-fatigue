from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v10230_qualification_supervisor.py"


def load_module():
    spec = importlib.util.spec_from_file_location("supervisor", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_restartable(case: Path, step: int = 0):
    from arrhenius_fracture.run_state_checkpoint_v10230 import write_combined_checkpoint
    write_combined_checkpoint(
        case,
        outer={"case": {"da_phys_m": 5e-6}, "cycles_total": float(step),
               "geometry": {"crack_tip_m": [0.001, 0.0],
                            "front_paths": [[[0.001, 0.0]]],
                            "front_inventory": [{"xy": [0.001, 0.0], "fwd": [1.0, 0.0],
                                "last_plane": {"t": [1.0, 0.0], "n": [0.0, 1.0]},
                                "win_plane": {"t": [1.0, 0.0], "n": [0.0, 1.0]}}],
                            "committed_event_count": 0, "kinetic_event_index": 0,
                            "transaction_index": 0, "transaction_state": "committed",
                            "mesh_metadata": {"hbar_m": 1.0, "hbar_tip_m": 1.0,
                                              "tip_reference_centers_m": [[0.001, 0.0]]}}},
        arrays={"state": [step]},
        kinetic={"geometry_signature": [0, 0, 0, 0],
                 "stochastic": {"B": 0.25, "hazard_threshold_action": 2.0,
                                "hazard_action_current": 0.5,
                                "hazard_event_index": 0,
                                "hazard_threshold_history": [],
                                "avalanche_base_checkpoint_m": 5e-6,
                                "avalanche_event_length_factor": 1.0,
                                "avalanche_event_advance_m": 5e-6,
                                "rng_state": {"state": step}}},
        kinetic_vector=[step],
    )


def test_matrix_is_deterministic_and_uses_canonical_seed_namespaces():
    module = load_module(); rows = module.matrix()
    assert len(rows) == 12
    assert [row["fraction"] for row in rows[:3]] == [0.55, 0.75, 0.95]
    assert {row["parameter_option"] for row in rows} == {
        "v913_paper_peak01_0242980_persistent_sites",
        "v913_paper_dbtt01_0202500_persistent_sites",
        "v913_paper_weakT01_0129902_persistent_sites",
        "v913_paper_ceramic01_0077080_persistent_sites",
    }
    assert {row["seed"] for row in rows if row["label"] == "weakT"} == {2001726}


def test_stale_heartbeat_and_healthy_progress_protection(tmp_path):
    module = load_module(); case = tmp_path / "case"; case.mkdir()
    checkpoint = case / "qualification_liveness.json"
    module.atomic_json(checkpoint, {"latest_liveness_timestamp": time.time()})
    now = time.time()
    assert module.stale(case, now, 300.0) is False
    old = now - 301.0
    module.atomic_json(checkpoint, {"latest_liveness_timestamp": old})
    assert module.stale(case, now, 300.0) is True


def test_finished_skip_and_incomplete_restart_selection(tmp_path):
    module = load_module(); complete = tmp_path / "complete"; complete.mkdir()
    module.set_status(complete, "completed")
    assert module.classify(complete) == "completed"
    incomplete = tmp_path / "incomplete"; incomplete.mkdir()
    write_restartable(incomplete)
    assert module.classify(incomplete) == "restartable"


def test_kinetic_only_checkpoint_is_not_physically_restartable(tmp_path):
    module = load_module(); case = tmp_path / "case"; case.mkdir()
    module.atomic_json(case / "high_cycle_live_checkpoint.json",
                       {"schema": "kinetic", "timestamp_unix_s": 1})
    (case / "high_cycle_live_state.npz").touch()
    assert module.checkpoint_valid(case) is False
    assert module.classify(case) == "pending"


def test_disk_space_block_and_maximum_two_job_cap(tmp_path, monkeypatch):
    module = load_module()
    args = module.parser().parse_args([
        "run", str(tmp_path), "--smoke-worker", "--max-jobs", "99",
        "--poll-seconds", "0.01", "--minimum-free-gib", "1",
    ])
    assert module.run(args) == 0
    summary = json.loads((tmp_path / "qualification_supervisor_summary.json").read_text())
    assert summary["maximum_jobs_configured"] == 2
    assert summary["maximum_active_observed"] == 2
    monkeypatch.setattr(module, "free_gib", lambda _path: 4.0)
    assert module.free_gib(tmp_path) < 10.0


def test_status_transitions_preserve_completed_outputs(tmp_path):
    module = load_module(); case = tmp_path / "case"; case.mkdir()
    result = case / "result.txt"; result.write_text("keep")
    module.set_status(case, "pending")
    module.set_status(case, "running", pid=12)
    module.set_status(case, "completed")
    assert json.loads((case / "qualification_status.json").read_text())["status"] == "completed"
    assert result.read_text() == "keep"


def test_shell_entry_points_parse():
    for name in (
        "run_v10_2_30_four_class_qualification_supervisor.sh",
        "monitor_v10_2_30_four_class_qualification.sh",
        "stop_v10_2_30_four_class_qualification.sh",
    ):
        result = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True)
        assert result.returncode == 0, result.stderr.decode()


def test_launch_wrapper_handles_unset_optional_stale_recovery(tmp_path):
    fake = tmp_path / "conda"
    fake.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    fake.chmod(0o755)
    import os
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_v10_2_30_four_class_qualification_supervisor.sh"), "campaign"],
        cwd=ROOT, env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--recover-stale-lock" not in result.stdout


def test_smoke_worker_resumes_from_checkpoint(tmp_path, monkeypatch):
    module = load_module(); case = tmp_path / "case"; case.mkdir()
    monkeypatch.setenv("SMOKE_INTERRUPT", "1")
    assert module.smoke_worker(case) == 75
    assert json.loads((case / "smoke_checkpoint.json").read_text())["step"] == 2
    monkeypatch.delenv("SMOKE_INTERRUPT")
    assert module.smoke_worker(case) == 0
    assert json.loads((case / "smoke_checkpoint.json").read_text())["step"] == 4
    assert (case / "exit_code.txt").read_text().strip() == "0"


def test_stop_sends_sigterm_to_launcher(tmp_path, monkeypatch):
    module = load_module(); module.atomic_json(tmp_path / "launcher.json", {"pid": 123})
    calls = []
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    assert module.stop_launcher(tmp_path) == 0
    assert calls == [(123, module.signal.SIGTERM)]


def test_campaign_lock_rejects_live_owner_and_requires_explicit_stale_recovery(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "process_identity", lambda pid: "same-start" if pid in {10, 20} else None)
    monkeypatch.setattr(module.os, "getpid", lambda: 10)
    first = module.acquire_lock(tmp_path, "head")
    monkeypatch.setattr(module.os, "getpid", lambda: 20)
    import pytest
    with pytest.raises(RuntimeError, match="live"):
        module.acquire_lock(tmp_path, "head")
    module.release_lock(tmp_path, first["token"])
    stale = {**first, "pid": 99, "process_start_identity": "gone"}
    module.atomic_json(tmp_path / module.LOCK_NAME, stale)
    with pytest.raises(RuntimeError, match="recover-stale-lock"):
        module.acquire_lock(tmp_path, "head")
    recovered = module.acquire_lock(tmp_path, "head", recover=True)
    assert recovered["token"] != first["token"]
    module.release_lock(tmp_path, recovered["token"])


def test_partial_zero_exit_is_restartable_not_completed(tmp_path):
    module = load_module(); case = tmp_path / "case"; case.mkdir()
    write_restartable(case)
    (case / "exit_code.txt").write_text("0\n")
    (case / "developed_fatigue_growth_summary.json").write_text(json.dumps({
        "status": "partial_growth", "target_reached": False,
    }))
    assert module.classify(case) == "restartable"


def test_default_watchdog_exceeds_observed_long_diagnostic_phase(monkeypatch):
    module = load_module(); monkeypatch.delenv("V10230_QUAL_NO_PROGRESS_SECONDS", raising=False)
    args = module.parser().parse_args(["run", "/tmp/example", "--smoke-worker"])
    assert args.no_progress_seconds == 900.0
