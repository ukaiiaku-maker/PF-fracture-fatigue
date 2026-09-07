"""PX5: part_x_run_one_job.py's STATIC_SHIELD_COHESIONS routing -- a
"ceiling_static"/"orbit_matched_static" cohesion job must run with
rebonding_cfg=None and a static_shield_control step function (K_b=0.0
before the first accepted event, the frozen K_b_static value from every
event after), never through the Markov P/C/B kinetics path a "finite"/
"zero" cohesion job uses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
import part_x_run_one_job as runner  # noqa: E402


def test_static_shield_cohesions_are_exactly_ceiling_and_orbit_matched():
    assert runner.STATIC_SHIELD_COHESIONS == {"ceiling_static", "orbit_matched_static"}


def test_max_wall_seconds_override_absent_uses_pilot_default(tmp_path):
    """PX5 discovered two Kmax=12/ceiling_static conditions genuinely need
    more than the pilot's own 1800s default wall-clock budget to reach 30
    events (censor_reason='wall_time_budget_exhausted_mid_event' at only
    5 events) -- an optional per-job max_wall_seconds_override column lets
    those two jobs alone use a larger budget, with zero effect on every
    job that omits or leaves it blank."""
    job_without = {"max_wall_seconds_override": ""}
    job_missing = {}
    for job in (job_without, job_missing):
        resolved = float(job["max_wall_seconds_override"]) if job.get("max_wall_seconds_override") else pilot.MAX_WALL_SECONDS_PER_TRAJECTORY
        assert resolved == pilot.MAX_WALL_SECONDS_PER_TRAJECTORY
    job_with = {"max_wall_seconds_override": "21600.0"}
    resolved = float(job_with["max_wall_seconds_override"]) if job_with.get("max_wall_seconds_override") else pilot.MAX_WALL_SECONDS_PER_TRAJECTORY
    assert resolved == 21600.0


def test_run_one_job_static_shield_end_to_end(tmp_path, monkeypatch):
    """End-to-end: a D2 job with cohesion='ceiling_static' must produce a
    trajectory whose K_b sequence is [0.0 once, then the static value] --
    the same step-function contract the pre-Part-X static_shield_
    attribution study already qualified -- shrunk to 2 events for speed."""
    import csv
    with open(REPO_ROOT / "artifacts/crack_rebonding_part_x_v1/kinetic_regime_registry.json") as f:
        registry = json.load(f)["rows"]
    config_hash = registry["COMPETING_REVERSIBLE"]["config_hash"]

    job = {
        "protocol": "D2", "row_name": "COMPETING_REVERSIBLE", "config_hash": config_hash,
        "material_row_hash": "irrelevant_not_checked_by_static_path",
        "Kmax_Pa_sqrt_m": "18000000.0", "R": "-0.5", "frequency_Hz": "1000.0",
        "minimum_load_hold_s": "0.0", "chemistry_factor": "1.0",
        "K_rebond_max_target_Pa_sqrt_m": "900000.0", "seed": "1720", "integrator_mode": "explicit",
        "physical_producer_sha": "test_not_enforced", "canonical_job_key": "test_key_static_shield_routing",
        "cohesion": "ceiling_static", "note": "", "alias_of_protocol": "", "alias_of_row_name": "",
        "status": "AUTHORIZED_PX5",
    }

    orig_events, orig_extension, orig_cycles = (
        runner.DEVELOPED_MAX_ACCEPTED_EVENTS, runner.DEVELOPED_MAX_PROJECTED_EXTENSION_m, runner.DEVELOPED_MAX_CUMULATIVE_CYCLES,
    )
    runner.DEVELOPED_MAX_ACCEPTED_EVENTS = 2
    runner.DEVELOPED_MAX_PROJECTED_EXTENSION_m = 1.0e300
    runner.DEVELOPED_MAX_CUMULATIVE_CYCLES = float("inf")
    try:
        result_dir = tmp_path / "result"
        job_json = tmp_path / "job.json"
        job_json.write_text(json.dumps(job))
        old_argv = sys.argv
        sys.argv = ["part_x_run_one_job.py", "--job-json", str(job_json), "--result-dir", str(result_dir)]
        try:
            rc = runner.main()
        finally:
            sys.argv = old_argv
    finally:
        (runner.DEVELOPED_MAX_ACCEPTED_EVENTS, runner.DEVELOPED_MAX_PROJECTED_EXTENSION_m,
         runner.DEVELOPED_MAX_CUMULATIVE_CYCLES) = (orig_events, orig_extension, orig_cycles)

    assert rc == 0
    result = json.loads((result_dir / "result.json").read_text())
    assert result["schema"] == "v10230_part_x_developed_job_result_v1"
    events = result["trajectory"]["events"]
    assert len(events) == 2
    k_b_sequence = [e["K_b_applied_Pa_sqrt_m"] for e in events]
    assert k_b_sequence == [0.0, 900000.0]
    assert result["actual_config_hash"].startswith("STATIC_SHIELD_NO_KINETICS:")
