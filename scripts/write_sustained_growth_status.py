#!/usr/bin/env python3
"""Publish a compact, inspectable status for the one managed candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT/"runs/managed_sustained_growth_v1_20260920"
OUT = JOB/"output"
SOURCE = JOB/"source_v9"
ONED = Path("/Volumes/Data/Data/Nanopillar_calculation/voiding/oneD-v3-stateful-voiding-reduction")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    progress = json.loads((OUT/"progress.json").read_text())
    worker = json.loads((OUT/"active_worker.json").read_text())
    comparison = json.loads((ONED/"analysis_outputs/oneD_v3_sustained_growth_enriched_v3/"
                              "two_site_history_comparison.json").read_text())
    payload = {
        "schema": "managed-sustained-growth-status/1",
        "classification": "PROSPECTIVE_UNCALIBRATED_PARAMETER_HYPOTHESIS",
        "worker_live": "stopped_wall_time_s" not in worker,
        "worker_pid": worker["pid"],
        "detached_session": "sustained_growth_v1",
        "immutable_source": str(SOURCE),
        "immutable_source_manifest_sha256": sha(SOURCE/"snapshot_manifest.json"),
        "source_revision_note": (
            "v1-v8 are preserved controller-qualification/runtime snapshots; v9 "
            "adds source/action/growth-aware adaptivity and rejection recovery. "
            "No physical row changed."),
        "declared_inputs": str(OUT/"declared_inputs.json"),
        "accepted_time_s": progress["accepted_time_s"],
        "remaining_horizon_s": 10000.0-float(progress["accepted_time_s"]),
        "latest_restart": str(OUT/progress["latest_restart"]),
        "latest_restart_state_file": progress["latest_restart_state_file"],
        "latest_restart_direct_restored_equal": progress.get(
            "latest_restart_direct_restored_equal"),
        "accepted_state_fingerprint": progress.get(
            "latest_restart_accepted_state_fingerprint"),
        "adaptive_step_s": progress["adaptive_step_s"],
        "phase": progress["phase"],
        "oneD_prediction": str(ONED/"analysis_outputs/oneD_v3_sustained_growth_enriched_v3/"
                                "prediction.json"),
        "continued_two_site_transfer": comparison,
        "adaptive_functional_qualification": str(
            JOB/"adaptive_functional_qualification.json"),
        "plots": {
            "executed": str(ONED/"analysis_outputs/oneD_v3_sustained_growth_candidate/"
                            "executed_oneD_twoD_overlap.png"),
            "continued_two_site": str(
                ONED/"analysis_outputs/oneD_v3_sustained_growth_enriched_v3/"
                "continued_two_site_history.png"),
            "scout_only": str(ONED/"analysis_outputs/oneD_v3_sustained_growth_candidate/"
                              "scout_projection_not_executed.png"),
        },
        "resume_command": (
            "cd "+str(SOURCE)+" && SUSTAINED_GROWTH_RUN_OUT="+str(OUT)+" "
            "PYTHON_BIN=/opt/homebrew/Caskroom/miniconda/base/envs/"
            "arrhenius-sharp-front-v10-codex/bin/python MAX_WALL_SECONDS=900 "
            "MAX_SCHEDULER_TRANSACTIONS_PER_CHECKPOINT=64 "
            "MPLCONFIGDIR=/private/tmp/mpl-sustained-growth "
            "/bin/sh scripts/run_managed_sustained_growth_loop.sh"),
        "fatigue_interface": (
            "PRESERVED_NOT_EXECUTED; next coverage is one room-temperature "
            "cycle-resolved void-on/void-off mechanics observation with the existing "
            "signed waveform, wake, vacancy, source, and plastic state carried across the cycle."),
    }
    (JOB/"status.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(JOB/"status.json")


if __name__ == "__main__":
    main()
