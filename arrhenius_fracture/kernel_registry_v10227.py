"""Registry, integrity, and locking helpers for v10.2.27 kernel resolution."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from .kernel_configuration_v10227 import canonical_json_bytes
from .signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
    SCHEMA as FAMILY_SCHEMA,
)

REGISTRY_SCHEMA = "v10.2.27_kernel_registry_v1"
LOCAL_REGISTRY_SCHEMA = "v10.2.27_local_kernel_cache_registry_v1"

_VOLATILE_KEYS = {
    "archive",
    "capture_root",
    "coverage_audit",
    "engine_config",
    "family",
    "family_out",
    "load_invariance_report",
    "load_invariance_root",
    "mechanics_normalization",
    "normalization",
    "out",
    "response",
    "responses",
    "snapshot",
    "snapshot_root",
    "source_path",
}


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physics_payload(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _VOLATILE_KEYS:
                continue
            if lowered.endswith("_path") or lowered.endswith("_sha256"):
                continue
            if lowered.endswith("_root") and lowered not in {"square_root"}:
                continue
            output[str(key)] = _physics_payload(item)
        return output
    if isinstance(value, list):
        return [_physics_payload(item) for item in value]
    return value


def family_physics_fingerprint(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text())
    return hashlib.sha256(canonical_json_bytes(_physics_payload(payload))).hexdigest()


def family_extension_range_um(path: str | Path) -> tuple[float, float]:
    payload = json.loads(Path(path).read_text())
    values = sorted(
        {1.0e6 * float(row["crack_extension_m"]) for row in payload.get("states", [])}
    )
    if len(values) < 2:
        raise ValueError("signed-kernel family must contain at least two extensions")
    return values[0], values[-1]


def validate_family(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_physics_fingerprint: str | None = None,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    file_sha = sha256_file(source)
    if expected_file_sha256 is not None and file_sha != expected_file_sha256:
        raise ValueError(f"kernel family SHA mismatch: {file_sha} != {expected_file_sha256}")
    family = ActiveOnlySigned2DShieldingKernelFamily.from_json(source)
    if family.metadata.get("schema") != FAMILY_SCHEMA:
        raise ValueError("kernel family schema mismatch")
    if family.metadata.get("production_parameterization_allowed") is not True:
        raise ValueError("kernel family is not production-authorized")
    if family.metadata.get("active_kernel_mechanically_measured") is not True:
        raise ValueError("active kernel is not mechanically measured")
    if family.metadata.get("wake_shielding_supported") is not False:
        raise ValueError("wake shielding is unexpectedly supported")
    if any(
        np.any(np.abs(state.wake_I) > 0.0)
        or np.any(np.abs(state.wake_II) > 0.0)
        for state in family.states
    ):
        raise ValueError("active-only family contains nonzero wake kernels")
    physics = family_physics_fingerprint(source)
    if expected_physics_fingerprint is not None and physics != expected_physics_fingerprint:
        raise ValueError(
            "kernel family physics fingerprint mismatch: "
            f"{physics} != {expected_physics_fingerprint}"
        )
    minimum_um, maximum_um = family_extension_range_um(source)
    return {
        "family": str(source),
        "file_sha256": file_sha,
        "physics_fingerprint": physics,
        "minimum_extension_um": minimum_um,
        "maximum_extension_um": maximum_um,
        "state_count": len(family.states),
        "schema": FAMILY_SCHEMA,
    }


def load_registry(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {"schema": REGISTRY_SCHEMA, "entries": [], "recipes": []}
    payload = json.loads(source.read_text())
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"kernel registry schema mismatch: {source}")
    if not isinstance(payload.get("entries", []), list) or not isinstance(
        payload.get("recipes", []), list
    ):
        raise ValueError(f"invalid kernel registry structure: {source}")
    return payload


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def select_entry(
    registry: Mapping[str, Any],
    *,
    configuration_fingerprint: str,
    required_max_extension_um: float,
    repo_root: Path,
) -> tuple[dict[str, Any], Path] | None:
    candidates = []
    for raw in registry.get("entries", []):
        if raw.get("configuration_fingerprint") != configuration_fingerprint:
            continue
        maximum = float(raw.get("maximum_extension_um", -1.0))
        if maximum + 1.0e-9 < float(required_max_extension_um):
            continue
        path = resolve_repo_path(repo_root, raw["family"])
        candidates.append((maximum, dict(raw), path))
    for _, entry, path in sorted(candidates, key=lambda item: item[0]):
        try:
            validate_family(
                path,
                expected_file_sha256=entry.get("family_sha256"),
                expected_physics_fingerprint=entry.get("physics_fingerprint"),
            )
        except (FileNotFoundError, ValueError):
            continue
        return entry, path
    return None


def recipe_matches(recipe: Mapping[str, Any], configuration: Mapping[str, Any]) -> bool:
    match = dict(recipe.get("match", {}))
    for key, expected in match.items():
        actual = configuration.get(key)
        if isinstance(expected, float):
            try:
                if abs(float(actual) - expected) > 1.0e-10:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected:
            return False
    return True


def select_recipe(
    registry: Mapping[str, Any], configuration: Mapping[str, Any]
) -> dict[str, Any] | None:
    matches = [
        dict(row)
        for row in registry.get("recipes", [])
        if recipe_matches(row, configuration)
    ]
    if len(matches) > 1:
        matches.sort(key=lambda row: len(row.get("match", {})), reverse=True)
        if len(matches[0].get("match", {})) == len(matches[1].get("match", {})):
            raise ValueError("multiple equally specific kernel build recipes match")
    return matches[0] if matches else None


def update_local_registry(path: str | Path, entry: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        payload = json.loads(destination.read_text())
    else:
        payload = {"schema": LOCAL_REGISTRY_SCHEMA, "entries": []}
    if payload.get("schema") != LOCAL_REGISTRY_SCHEMA:
        raise ValueError(f"local kernel registry schema mismatch: {destination}")
    rows = [
        row
        for row in payload.get("entries", [])
        if not (
            row.get("configuration_fingerprint")
            == entry.get("configuration_fingerprint")
            and row.get("family") == entry.get("family")
        )
    ]
    rows.append(copy.deepcopy(dict(entry)))
    payload["entries"] = rows
    with tempfile.NamedTemporaryFile(
        "w", dir=destination.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(destination)


@contextmanager
def kernel_lock(path: str | Path) -> Iterator[None]:
    lock_path = Path(path).expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+")
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover
            fcntl = None
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if "fcntl" in locals() and fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


__all__ = [
    "REGISTRY_SCHEMA",
    "LOCAL_REGISTRY_SCHEMA",
    "sha256_file",
    "family_physics_fingerprint",
    "family_extension_range_um",
    "validate_family",
    "load_registry",
    "resolve_repo_path",
    "select_entry",
    "select_recipe",
    "update_local_registry",
    "kernel_lock",
]
