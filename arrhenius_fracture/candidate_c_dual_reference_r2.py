"""Exact-source receipt for the recovered Candidate-C dual birth adapter.

The donor result is a bounded no-birth result.  This module deliberately does
not turn its nonzero clocks into an event or promote the transferred direct
row to a calibrated 300 K material law.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


MODEL_ID = "r2.recovered_candidate_c_dual_nucleation/1"
DONOR = Path("/Volumes/Data/Data/Nanopillar_calculation/voiding/fatigue-induced-voiding-v1")
COMMIT = "f8cda646b50294d5fee8eb917cb93910b59f4c85"
TREE = "4918243e7b9fce65331922dedb7fc10a49dc6211"
FILES = (
    "scripts/v914_fatigue_voiding_state_v1.py",
    "tests/test_v914_fatigue_voiding_state_v1.py",
    "CANDIDATE_C_DUAL_NUCLEATION_REPORT.md",
)


def _git(*args: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(DONOR), *args), check=True, capture_output=True,
    ).stdout


def _blob(path: str) -> bytes:
    return _git("show", f"{COMMIT}:{path}")


def recovered_candidate_c_receipt() -> dict[str, Any]:
    if _git("rev-parse", f"{COMMIT}^{{tree}}").decode().strip() != TREE:
        raise RuntimeError("Candidate-C donor tree identity changed")
    sources = {
        path: {
            "sha256": hashlib.sha256(_blob(path)).hexdigest(),
            "bytes": len(_blob(path)),
        }
        for path in FILES
    }
    report = _blob(FILES[-1]).decode()
    required = (
        "Vacancy/diffusion birth",
        "Direct dislocation-interaction birth",
        "Shared embryo owner",
        "REFERENCE_UNCALIBRATED",
        "no event",
    )
    missing = [text for text in required if text not in report]
    if missing:
        raise RuntimeError(f"Candidate-C report lost required contracts: {missing}")
    dirty = bool(_git("status", "--porcelain").strip())
    return {
        "schema": "r2-candidate-c-dual-source-receipt/1",
        "model_id": MODEL_ID,
        "repository": str(DONOR),
        "commit": COMMIT,
        "tree": TREE,
        "execution_source": "clean blobs at exact commit; donor worktree not imported",
        "source_worktree_dirty_ignored": dirty,
        "sources": sources,
        "birth_pathways": {
            "vacancy_diffusion": {
                "owner": "independent two-hit clock and mobile-vacancy inventory",
                "terminal_action_1p02": 5.08824260e-30,
                "event": False,
            },
            "direct_dislocation_interaction": {
                "owner": "independent three-hit clock and direct free-volume inventory",
                "terminal_action_1p02": 2.10715867548e-4,
                "event": False,
                "row_calibration": "REFERENCE_UNCALIBRATED_AT_300_K",
            },
            "cavity_surface_emission": {
                "role": "separate post-birth growth process",
                "active_in_candidate_c_result": False,
            },
        },
        "shared_embryo_owner": "first completed birth mechanism; later commits blocked",
        "scope": "300 K Candidate C at declared 1.05x and 1.02x amplitudes only",
        "scientific_result": "both corrected clocks nonzero; no hit, embryo, or cavity",
        "forced_event": False,
        "matched_enabled_suppressed_result": True,
        "restart_result": "lower enabled terminal exact excluding restart_count",
        "limits": [
            "transferred direct row is uncalibrated at 300 K",
            "no post-birth lifecycle was physically eligible in this material run",
            "this source receipt does not rerun the historical long trajectories",
        ],
    }


def write_candidate_c_receipt(path: str | Path) -> dict[str, Any]:
    payload = recovered_candidate_c_receipt()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


__all__ = ["MODEL_ID", "recovered_candidate_c_receipt", "write_candidate_c_receipt"]
