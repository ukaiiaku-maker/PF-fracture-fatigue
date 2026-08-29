"""Pure crack-network state for the v11 compatibility increment.

This module deliberately does not import or call a hazard, fatigue, geometry,
energy, or random-number engine.  It gives the unchanged v10.2.30 single-front
trajectory a validated one-tip network representation without entering the
production call path.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping


SCHEMA = "v11.crack-network/1"
ROOT_BRANCH_ID = "b00000000"
VALID_STATUSES = frozenset({"active", "arrested", "merged", "terminated"})

Point = tuple[float, float]


def _point(value: Iterable[float]) -> Point:
    values = tuple(float(component) for component in value)
    if len(values) != 2 or not all(math.isfinite(component) for component in values):
        raise ValueError("crack-network points must contain two finite coordinates")
    return values[0], values[1]


def _canonical_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-safe, key-sorted copy or fail on non-serializable state."""
    result = json.loads(json.dumps(dict(value or {}), sort_keys=True, allow_nan=False))
    if not isinstance(result, dict):  # defensive; Mapping should already ensure it
        raise ValueError("branch local state must be a JSON object")
    return result


@dataclass(frozen=True)
class CrackBranchState:
    branch_id: str
    parent_branch_id: str | None
    generation: int
    initiation_event: int
    path: tuple[Point, ...]
    orientation_history_rad: tuple[float, ...]
    status: str = "active"
    local_state: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_id", str(self.branch_id))
        parent = None if self.parent_branch_id is None else str(self.parent_branch_id)
        object.__setattr__(self, "parent_branch_id", parent)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "initiation_event", int(self.initiation_event))
        object.__setattr__(self, "path", tuple(_point(point) for point in self.path))
        orientations = tuple(float(value) for value in self.orientation_history_rad)
        object.__setattr__(self, "orientation_history_rad", orientations)
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "local_state", _canonical_mapping(self.local_state))
        self.validate()

    @property
    def root(self) -> Point:
        return self.path[0]

    @property
    def tip(self) -> Point:
        return self.path[-1]

    @property
    def physical_path_length_m(self) -> float:
        return math.fsum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(self.path, self.path[1:])
        )

    @property
    def projected_extension_m(self) -> float:
        return self.tip[0] - self.root[0]

    @property
    def current_orientation_rad(self) -> float:
        return self.orientation_history_rad[-1]

    def validate(self) -> None:
        if not self.branch_id:
            raise ValueError("branch_id must not be empty")
        if self.generation < 0 or self.initiation_event < 0:
            raise ValueError("generation and initiation_event must be nonnegative")
        if len(self.path) < 1:
            raise ValueError("a branch path must contain at least its root point")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid branch status: {self.status}")
        expected_orientations = max(len(self.path) - 1, 1)
        if len(self.orientation_history_rad) != expected_orientations:
            raise ValueError(
                "orientation history must contain one value per segment "
                "(or one initial orientation for a root-only path)"
            )
        if not all(math.isfinite(value) for value in self.orientation_history_rad):
            raise ValueError("orientation history must be finite")
        for index, (a, b) in enumerate(zip(self.path, self.path[1:])):
            if a == b:
                raise ValueError(f"zero-length segment at index {index}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "generation": self.generation,
            "initiation_event": self.initiation_event,
            "path_m": [list(point) for point in self.path],
            "orientation_history_rad": list(self.orientation_history_rad),
            "status": self.status,
            "physical_path_length_m": self.physical_path_length_m,
            "projected_extension_m": self.projected_extension_m,
            "local_state": _canonical_mapping(self.local_state),
        }


@dataclass(frozen=True)
class CrackNetworkState:
    branches: tuple[CrackBranchState, ...]
    primary_branch_id: str = ROOT_BRANCH_ID
    geometry_generation: int = 0
    branching_enabled: bool = False

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.branches, key=lambda branch: branch.branch_id))
        object.__setattr__(self, "branches", ordered)
        object.__setattr__(self, "primary_branch_id", str(self.primary_branch_id))
        object.__setattr__(self, "geometry_generation", int(self.geometry_generation))
        object.__setattr__(self, "branching_enabled", bool(self.branching_enabled))
        self.validate()

    @classmethod
    def one_tip(
        cls,
        path_m: Iterable[Iterable[float]],
        *,
        initial_orientation_rad: float | None = None,
        local_state: Mapping[str, Any] | None = None,
    ) -> "CrackNetworkState":
        path = tuple(_point(point) for point in path_m)
        if not path:
            raise ValueError("one-tip compatibility path must not be empty")
        segment_orientations = tuple(
            math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(path, path[1:])
        )
        if segment_orientations:
            orientations = segment_orientations
        elif initial_orientation_rad is not None:
            orientations = (float(initial_orientation_rad),)
        else:
            raise ValueError("a root-only path requires initial_orientation_rad")
        root = CrackBranchState(
            branch_id=ROOT_BRANCH_ID,
            parent_branch_id=None,
            generation=0,
            initiation_event=0,
            path=path,
            orientation_history_rad=orientations,
            local_state=local_state,
        )
        return cls(branches=(root,), branching_enabled=False)

    @property
    def active_tip_ids(self) -> tuple[str, ...]:
        return tuple(
            branch.branch_id for branch in self.branches if branch.status == "active"
        )

    @property
    def total_physical_crack_length_m(self) -> float:
        return math.fsum(branch.physical_path_length_m for branch in self.branches)

    @property
    def primary_projected_extension_m(self) -> float:
        return self.branch(self.primary_branch_id).projected_extension_m

    def branch(self, branch_id: str) -> CrackBranchState:
        for branch in self.branches:
            if branch.branch_id == branch_id:
                return branch
        raise KeyError(branch_id)

    def validate(self) -> None:
        if not self.branches:
            raise ValueError("crack network must contain at least one branch")
        if self.geometry_generation < 0:
            raise ValueError("geometry_generation must be nonnegative")
        ids = tuple(branch.branch_id for branch in self.branches)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate branch identifiers")
        if self.primary_branch_id not in ids:
            raise ValueError("primary branch is missing")
        by_id = {branch.branch_id: branch for branch in self.branches}
        roots = [branch for branch in self.branches if branch.parent_branch_id is None]
        if len(roots) != 1:
            raise ValueError("crack network must contain exactly one root branch")
        for branch in self.branches:
            parent_id = branch.parent_branch_id
            if parent_id is None:
                if branch.generation != 0:
                    raise ValueError("root branch generation must be zero")
                continue
            if parent_id not in by_id:
                raise ValueError(f"missing parent branch: {parent_id}")
            parent = by_id[parent_id]
            if branch.generation != parent.generation + 1:
                raise ValueError("child generation must equal parent generation plus one")
            seen = {branch.branch_id}
            ancestor = parent
            while ancestor.parent_branch_id is not None:
                if ancestor.branch_id in seen:
                    raise ValueError("cycle in crack-network topology")
                seen.add(ancestor.branch_id)
                ancestor = by_id.get(ancestor.parent_branch_id)  # type: ignore[assignment]
                if ancestor is None:
                    raise ValueError("broken ancestor link")
        if not self.branching_enabled and len(self.branches) != 1:
            raise ValueError("branching-disabled network must contain exactly one branch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "branching_enabled": self.branching_enabled,
            "primary_branch_id": self.primary_branch_id,
            "geometry_generation": self.geometry_generation,
            "active_tip_ids": list(self.active_tip_ids),
            "total_physical_crack_length_m": self.total_physical_crack_length_m,
            "primary_projected_extension_m": self.primary_projected_extension_m,
            "branches": [branch.to_dict() for branch in self.branches],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CrackNetworkState":
        if payload.get("schema") != SCHEMA:
            raise ValueError("unsupported crack-network schema")
        branches = tuple(
            CrackBranchState(
                branch_id=item["branch_id"],
                parent_branch_id=item.get("parent_branch_id"),
                generation=item["generation"],
                initiation_event=item["initiation_event"],
                path=tuple(item["path_m"]),
                orientation_history_rad=tuple(item["orientation_history_rad"]),
                status=item["status"],
                local_state=item.get("local_state", {}),
            )
            for item in payload.get("branches", [])
        )
        result = cls(
            branches=branches,
            primary_branch_id=payload["primary_branch_id"],
            geometry_generation=payload["geometry_generation"],
            branching_enabled=payload["branching_enabled"],
        )
        # Derived fields are validation sentinels, not authoritative state.
        expected_active = list(result.active_tip_ids)
        if payload.get("active_tip_ids") != expected_active:
            raise ValueError("active-tip ordering or membership mismatch")
        for key, actual in (
            ("total_physical_crack_length_m", result.total_physical_crack_length_m),
            ("primary_projected_extension_m", result.primary_projected_extension_m),
        ):
            if not math.isclose(float(payload.get(key, math.nan)), actual, rel_tol=1e-13, abs_tol=1e-18):
                raise ValueError(f"derived crack-network accounting mismatch: {key}")
        return result


__all__ = [
    "CrackBranchState",
    "CrackNetworkState",
    "ROOT_BRANCH_ID",
    "SCHEMA",
    "VALID_STATUSES",
]
