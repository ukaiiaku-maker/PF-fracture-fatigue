"""Atomic, opt-in forensic records for production crack-path selection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SCHEMA = "v10.2.30_path_selection_forensics_v1"
FILENAME = "path_selection_forensics_v10230.json"


def enabled() -> bool:
    return os.environ.get("V10230_PATH_SELECTION_FORENSICS", "0").lower() in {
        "1", "true", "yes", "on",
    }


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def candidate_record(candidate: dict[str, Any], index: int, tip: np.ndarray,
                     proposal_m: float | None = None) -> dict[str, Any]:
    direction = np.asarray(candidate["t"], dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-300)
    identity = {
        "name": str(candidate.get("name", "unknown")),
        "family": str(candidate.get("family", "unknown")),
        "angle_deg": float(candidate.get("angle_deg", np.nan)),
        "direction": direction.tolist(),
    }
    stable_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"),
                   allow_nan=True).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "enumeration_index": int(index),
        "stable_candidate_id": stable_id,
        **identity,
        "normal": np.asarray(candidate.get("n", [np.nan, np.nan]), float).tolist(),
        "sigma_nn": float(candidate.get("sigma_nn", np.nan)),
        "gamma_relative": float(candidate.get("gamma_rel", candidate.get("gamma", np.nan))),
        "overdrive": float(candidate.get("overdrive", np.nan)),
        "score": float(candidate.get("overdrive", np.nan)),
        "admissible": True,
        "rejection_reason": None,
        "projected_endpoint": (
            np.asarray(tip, float) + direction * float(proposal_m or 1.0)
        ).tolist(),
    }


def _generation(root: Path) -> str | None:
    try:
        return str(json.loads((root / "run_state_checkpoint.json").read_text())["generation"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def record_selection(*, outroot: str | Path, phase: str, step: int, cycles: float,
                     front: dict[str, Any], sigma2: np.ndarray, all_candidates: list,
                     selected: list, mesh, damage: np.ndarray, displacement: np.ndarray,
                     ep_gp: np.ndarray, rho_gp: np.ndarray, proposed_length_m: float | None,
                     threshold: float | None, hazard_action: float | None) -> None:
    if not enabled():
        return
    root = Path(outroot).resolve()
    rows = [candidate_record(candidate, index, np.asarray(front["xy"], float), proposed_length_m)
            for index, candidate in enumerate(all_candidates)]
    selected_rows = [candidate_record(candidate, index, np.asarray(front["xy"], float), proposed_length_m)
                     for index, candidate in enumerate(selected)]
    winner = selected_rows[0] if selected_rows else None
    runner = rows[1] if len(rows) > 1 else None
    record = {
        "phase": str(phase), "step": int(step), "event_index": int(front["eng"].n_adv + 1),
        "cycles": float(cycles), "git_head": os.environ.get("EXPECTED_HEAD", ""),
        "seed": int(os.environ.get("CLEAVAGE_HAZARD_SEED", "0")),
        "execution": "restored" if os.environ.get("V10230_RESTART_CHECKPOINT_DIR") else "continuous",
        "combined_checkpoint_generation": _generation(root),
        "threshold_action": threshold, "physical_hazard_action": hazard_action,
        "proposed_stochastic_event_length_m": proposed_length_m,
        "front": {
            "id": int(front.get("id", 0)), "xy": np.asarray(front["xy"], float).tolist(),
            "fwd": np.asarray(front["fwd"], float).tolist(),
            "path": [np.asarray(point, float).tolist() for point in front.get("path", [])],
            "last_plane": _safe(front.get("last_plane")),
            "win_plane": _safe(front.get("win_plane")),
        },
        "selector_input": {"near_tip_stress_tensor_Pa": np.asarray(sigma2, float).tolist()},
        "hashes": {
            "mesh_nodes": array_hash(mesh.nodes), "mesh_elems": array_hash(mesh.elems),
            "damage": array_hash(damage), "displacement": array_hash(displacement),
            "ep_gp": array_hash(ep_gp), "rho_gp": array_hash(rho_gp),
        },
        "candidate_order_before_sort": [row["stable_candidate_id"] for row in rows],
        "candidate_order_after_sort": [row["stable_candidate_id"] for row in rows],
        "candidates": rows, "selected_candidates": selected_rows,
        "selected_stable_candidate_id": None if winner is None else winner["stable_candidate_id"],
        "selected_direction": None if winner is None else winner["direction"],
        "winning_score": None if winner is None else winner["score"],
        "runner_up_score": None if runner is None else runner["score"],
        "score_separation": None if winner is None or runner is None
        else float(winner["score"] - runner["score"]),
        "tie_break": "stable first maximum in fixed ascending alpha grid; no RNG",
        "rng_consumed": False,
    }
    path = root / FILENAME
    payload = {"schema": SCHEMA, "records": []}
    try:
        existing = json.loads(path.read_text())
        if existing.get("schema") == SCHEMA:
            payload = existing
    except (OSError, ValueError, TypeError):
        pass
    payload["records"].append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


__all__ = ["FILENAME", "SCHEMA", "array_hash", "candidate_record", "enabled",
           "record_selection"]
