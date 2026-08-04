"""Atomic cross-layer restart checkpoints for the v10.2.30 2-D driver.

The manifest is the commit record.  A generation is invisible to readers until
all outer-driver arrays, metadata, and the matching kinetic snapshot have been
written and hashed, and the manifest has been atomically replaced.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np


MODEL_ID = "v10.2.30_combined_outer_kinetic_run_state_v1"
MANIFEST = "run_state_checkpoint.json"


class RestartCheckpointError(RuntimeError):
    pass


_FRONT_VECTOR_KEYS = {
    "xy", "fwd", "t_win", "t_trial", "birth_xy", "direction",
}
_PLANE_VECTOR_KEYS = {"t", "n"}
_TRANSIENT_FRONT_KEYS = {
    "info", "fatigue_pred_trial", "fatigue_wave_trial", "cands_trial",
    "win_trial", "t_trial",
}


def serialize_front_state(front: dict[str, Any]) -> dict[str, Any]:
    """Serialize every persistent front field except the engine object itself."""
    return {
        key: _safe(value) for key, value in front.items()
        if key != "eng" and key not in _TRANSIENT_FRONT_KEYS
    }


def restore_front_state(target: dict[str, Any], recorded: dict[str, Any]) -> None:
    def restore(value, key=""):
        if isinstance(value, dict):
            return {k: restore(v, k) for k, v in value.items()}
        if isinstance(value, list):
            if key in _FRONT_VECTOR_KEYS or key in _PLANE_VECTOR_KEYS:
                return np.asarray(value, dtype=float)
            if key == "path":
                return [np.asarray(point, dtype=float) for point in value]
            return [restore(item) for item in value]
        return value
    engine = target.get("eng")
    target.clear()
    target.update({key: restore(value, key) for key, value in recorded.items()})
    target["eng"] = engine


def _safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"checkpoint value is not JSON serializable: {type(value)!r}")


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_combined_checkpoint(
    root: str | Path,
    *,
    outer: dict[str, Any],
    arrays: dict[str, np.ndarray],
    kinetic: dict[str, Any],
    kinetic_vector: np.ndarray,
) -> dict[str, Any]:
    """Commit one complete restart generation without exposing partial state."""
    root = Path(root).resolve()
    generations = root / "run_state_generations"
    generations.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=generations))
    try:
        outer_payload = {"schema": MODEL_ID, **_safe(outer)}
        kinetic_payload = _safe(kinetic)
        (temporary / "outer.json").write_text(
            json.dumps(outer_payload, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "kinetic.json").write_text(
            json.dumps(kinetic_payload, indent=2, sort_keys=True) + "\n"
        )
        np.savez_compressed(
            temporary / "state.npz",
            kinetic_active_vector=np.asarray(kinetic_vector, dtype=float),
            **{str(k): np.asarray(v) for k, v in arrays.items()},
        )
        files = {
            name: _digest(temporary / name)
            for name in ("outer.json", "kinetic.json", "state.npz")
        }
        generation_id = hashlib.sha256(
            "".join(files.values()).encode("ascii")
        ).hexdigest()[:24]
        final = generations / generation_id
        if final.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, final)
        manifest = {
            "schema": MODEL_ID,
            "generation": generation_id,
            "created_unix_s": time.time(),
            "files": files,
        }
        _atomic_json(root / MANIFEST, manifest)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_combined_checkpoint(root: str | Path) -> tuple[dict, dict, dict[str, np.ndarray]]:
    root = Path(root).resolve()
    try:
        manifest = json.loads((root / MANIFEST).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RestartCheckpointError("missing or corrupt combined checkpoint manifest") from exc
    if manifest.get("schema") != MODEL_ID:
        raise RestartCheckpointError("incompatible combined checkpoint schema")
    generation = str(manifest.get("generation", ""))
    if not generation or "/" in generation or generation.startswith("."):
        raise RestartCheckpointError("invalid checkpoint generation")
    directory = root / "run_state_generations" / generation
    for name in ("outer.json", "kinetic.json", "state.npz"):
        path = directory / name
        if not path.is_file() or _digest(path) != manifest.get("files", {}).get(name):
            raise RestartCheckpointError(f"corrupt or incomplete checkpoint file: {name}")
    try:
        outer = json.loads((directory / "outer.json").read_text())
        kinetic = json.loads((directory / "kinetic.json").read_text())
        with np.load(directory / "state.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RestartCheckpointError("checkpoint generation cannot be decoded") from exc
    if outer.get("schema") != MODEL_ID:
        raise RestartCheckpointError("outer checkpoint schema mismatch")
    return outer, kinetic, arrays


def validate_compatibility(outer: dict, expected: dict[str, Any]) -> None:
    recorded = outer.get("case", {})
    mismatches = {
        key: (recorded.get(key), value)
        for key, value in expected.items()
        if recorded.get(key) != value
    }
    if mismatches:
        raise RestartCheckpointError(f"checkpoint case incompatibility: {mismatches}")


def validate_cross_layer(outer: dict, kinetic: dict) -> None:
    geometry = outer.get("geometry", {})
    stochastic = kinetic.get("stochastic", {})
    paths = geometry.get("front_paths", [])
    event_count = int(geometry.get("committed_event_count", -1))
    history = stochastic.get("hazard_threshold_history", [])
    event_index = int(stochastic.get("hazard_event_index", -1))
    recorded_event_index = int(geometry.get("kinetic_event_index", -1))
    if event_count < 0 or event_index != recorded_event_index:
        raise RestartCheckpointError("kinetic event index differs across checkpoint layers")
    if len(history) < event_index:
        raise RestartCheckpointError("kinetic event history is shorter than event index")
    if not paths:
        raise RestartCheckpointError("checkpoint contains no committed crack path")
    tip = np.asarray(geometry.get("crack_tip_m", []), dtype=float)
    path_tip = np.asarray(paths[0][-1] if paths[0] else [], dtype=float)
    if tip.shape != (2,) or path_tip.shape != (2,) or not np.array_equal(tip, path_tip):
        raise RestartCheckpointError("driver crack tip differs from committed path tip")
    inventory = geometry.get("front_inventory", [])
    if len(inventory) != len(paths) or not inventory:
        raise RestartCheckpointError("front inventory differs from committed paths")
    primary = inventory[0]
    if not np.array_equal(np.asarray(primary.get("xy", []), float), tip):
        raise RestartCheckpointError("primary front position differs from crack tip")
    heading = np.asarray(primary.get("fwd", []), float)
    last_plane = primary.get("last_plane") or {}
    win_plane = primary.get("win_plane") or {}
    for label, plane in (("last_plane", last_plane), ("win_plane", win_plane)):
        tangent = np.asarray(plane.get("t", []), float)
        normal = np.asarray(plane.get("n", []), float)
        if tangent.shape != (2,) or normal.shape != (2,):
            raise RestartCheckpointError(f"front {label} lacks directional vectors")
    if heading.shape != (2,) or not np.isfinite(heading).all():
        raise RestartCheckpointError("front heading is invalid")
    da = float(outer.get("case", {}).get("da_phys_m", 0.0))
    signature = kinetic.get("geometry_signature", [])
    if da <= 0.0 or len(signature) < 4:
        raise RestartCheckpointError("checkpoint lacks final physical geometry scale")
    base = float(stochastic.get("avalanche_base_checkpoint_m", da))
    if base != da:
        raise RestartCheckpointError("avalanche checkpoint base differs from da_phys")
    factor = float(stochastic.get("avalanche_event_length_factor", 0.0))
    proposal = float(stochastic.get("avalanche_event_advance_m", 0.0))
    if factor <= 0.0 or proposal <= 0.0 or not np.isclose(
        proposal, base * factor, rtol=1e-14, atol=1e-18
    ):
        raise RestartCheckpointError("current event proposal differs from checkpoint base")
    action = float(stochastic.get("hazard_action_current", 0.0))
    threshold = float(stochastic.get("hazard_threshold_action", 0.0))
    B = float(stochastic.get("B", 0.0))
    expected_B = action / threshold if threshold > 0.0 else 0.0
    if not np.isclose(B, expected_B, rtol=1e-12, atol=1e-14):
        raise RestartCheckpointError("B is inconsistent with physical hazard action")


__all__ = [
    "MANIFEST", "MODEL_ID", "RestartCheckpointError",
    "load_combined_checkpoint", "validate_compatibility",
    "restore_front_state", "serialize_front_state", "validate_cross_layer",
    "write_combined_checkpoint",
]
