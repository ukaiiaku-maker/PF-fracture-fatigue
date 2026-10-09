"""Chronological, provenance-preserving fracture observation history.

The schema separates crack coordinates from quantities observed there. Legacy
``KJ_Pa_sqrtm`` is retained as ``K_J_Pa_sqrt_m`` rather than relabelled as
mode-I or effective K.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
import csv
import json
import math
from pathlib import Path
from typing import Iterable

SCHEMA = "fracture-observation-history/2"

class ObservationStage(str, Enum):
    UNKNOWN = "unknown"
    PRE_EVENT = "pre_event"
    PROPOSED = "proposed"
    COMMITTED = "committed"
    REEQUILIBRATED = "reequilibrated"

class LoadDirection(str, Enum):
    UNKNOWN = "unknown"
    LOADING = "loading"
    UNLOADING = "unloading"
    HOLD = "hold"

class ValidityStatus(str, Enum):
    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"

class LengthCoordinate(str, Enum):
    EXTENSION = "extension"
    ABSOLUTE_PROJECTED = "absolute_projected"

@dataclass(frozen=True)
class FractureObservation:
    sequence: int
    time_s: float
    cycle: float
    segment_id: str
    stage: ObservationStage
    front_id: str
    event_id: str
    load_direction: LoadDirection
    pre_length_m: float
    committed_length_m: float
    K_I_Pa_sqrt_m: float = math.nan
    K_II_Pa_sqrt_m: float = math.nan
    K_effective_Pa_sqrt_m: float = math.nan
    K_J_Pa_sqrt_m: float = math.nan
    K_probe_Pa_sqrt_m: float = math.nan
    K_applied_Pa_sqrt_m: float = math.nan
    J_J_per_m2: float = math.nan
    G_J_per_m2: float = math.nan
    contour_valid: bool | None = None
    observer_valid: bool | None = None
    validity_status: ValidityStatus = ValidityStatus.UNKNOWN
    validity_reason: str = "status_not_recorded"
    contour_id: str = ""
    observer_id: str = ""
    initial_length_m: float = 0.0
    length_kind: str = "projected_extension"
    path_arclength_m: float = math.nan
    topology_generation: int = 0
    aggregate_event_count: int = 0
    source_file: str = ""
    source_row: int = -1
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        for name, kind in (("stage", ObservationStage), ("load_direction", LoadDirection),
                           ("validity_status", ValidityStatus)):
            value = getattr(self, name)
            if not isinstance(value, kind):
                object.__setattr__(self, name, kind(value))
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported observation schema: {self.schema!r}")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.committed_length_m + 1e-15 < self.pre_length_m:
            raise ValueError("committed length cannot precede pre-event length")
        inferred = (ValidityStatus.INVALID
                    if self.contour_valid is False or self.observer_valid is False
                    else ValidityStatus.VALID
                    if self.contour_valid is True and self.observer_valid is True
                    else ValidityStatus.UNKNOWN)
        if self.validity_status is ValidityStatus.UNKNOWN and inferred is not ValidityStatus.UNKNOWN:
            object.__setattr__(self, "validity_status", inferred)
        elif self.validity_status is ValidityStatus.VALID and inferred is ValidityStatus.INVALID:
            raise ValueError("validity_status conflicts with an invalid observer/contour flag")

    @property
    def stage_extension_m(self) -> float:
        return (self.pre_length_m if self.stage in {ObservationStage.PRE_EVENT, ObservationStage.PROPOSED}
                else self.committed_length_m)

    @property
    def stage_length_m(self) -> float:
        """Compatibility alias for the projected extension coordinate."""
        return self.stage_extension_m

    def length_m(self, coordinate: LengthCoordinate | str) -> float:
        value = self.stage_extension_m
        return value + self.initial_length_m if LengthCoordinate(coordinate) is LengthCoordinate.ABSOLUTE_PROJECTED else value

    def to_dict(self) -> dict:
        payload = asdict(self)
        for name in ("stage", "load_direction", "validity_status"):
            payload[name] = getattr(self, name).value
        for key, value in tuple(payload.items()):
            if isinstance(value, float) and not math.isfinite(value):
                payload[key] = None
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "FractureObservation":
        values = dict(payload)
        if values.get("schema") == "fracture-observation-history/1":
            values["schema"] = SCHEMA
            values.setdefault("validity_reason", "migrated_schema_1_flags")
        defaults = {"K_J_Pa_sqrt_m": None, "K_probe_Pa_sqrt_m": None,
                    "K_applied_Pa_sqrt_m": None,
                    "validity_status": "unknown", "validity_reason": "status_not_recorded",
                    "observer_id": "", "initial_length_m": 0.0,
                    "length_kind": "projected_extension", "path_arclength_m": None,
                    "topology_generation": 0, "aggregate_event_count": 0}
        for name, value in defaults.items():
            values.setdefault(name, value)
        int_fields = {"sequence", "source_row", "topology_generation", "aggregate_event_count"}
        float_fields = {"time_s", "cycle", "pre_length_m", "committed_length_m",
                        "K_I_Pa_sqrt_m", "K_II_Pa_sqrt_m", "K_effective_Pa_sqrt_m",
                        "K_J_Pa_sqrt_m", "K_probe_Pa_sqrt_m", "K_applied_Pa_sqrt_m", "J_J_per_m2",
                        "G_J_per_m2", "initial_length_m", "path_arclength_m"}
        for name in int_fields:
            values[name] = int(values[name])
        for name in float_fields:
            value = values.get(name)
            values[name] = math.nan if value in (None, "") else float(value)
        for name in ("contour_valid", "observer_valid"):
            value = values.get(name)
            values[name] = (None if value in (None, "", "unknown") else value
                            if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes"})
        return cls(**values)

@dataclass(frozen=True)
class LengthQueryResult:
    schema: str
    target_length_m: float
    coordinate: LengthCoordinate
    status: str
    exact: tuple[FractureObservation, ...]
    crossing: tuple[FractureObservation, ...]
    lower: FractureObservation | None
    upper: FractureObservation | None
    interpolation_performed: bool = False

    def to_dict(self) -> dict:
        return {"schema": self.schema, "target_length_m": self.target_length_m,
                "coordinate": self.coordinate.value, "status": self.status,
                "exact": [row.to_dict() for row in self.exact],
                "crossing": [row.to_dict() for row in self.crossing],
                "lower": None if self.lower is None else self.lower.to_dict(),
                "upper": None if self.upper is None else self.upper.to_dict(),
                "interpolation_performed": self.interpolation_performed}

class ObservationHistory:
    def __init__(self, rows: Iterable[FractureObservation]):
        self.rows = tuple(sorted(rows, key=lambda row: row.sequence))
        sequences = [row.sequence for row in self.rows]
        if len(sequences) != len(set(sequences)):
            raise ValueError("observation sequence identifiers must be unique")

    def filtered(self, *, front_id: str | None = None, segment_id: str | None = None,
                 stage: ObservationStage | str | None = None,
                 validity: ValidityStatus | str | None = None,
                 valid_only: bool | None = None) -> tuple[FractureObservation, ...]:
        selected = self.rows
        if front_id is not None:
            selected = tuple(row for row in selected if row.front_id == front_id)
        if segment_id is not None:
            selected = tuple(row for row in selected if row.segment_id == segment_id)
        if stage is not None:
            selected = tuple(row for row in selected if row.stage is ObservationStage(stage))
        if valid_only is True:
            validity = ValidityStatus.VALID
        if validity is not None:
            selected = tuple(row for row in selected if row.validity_status is ValidityStatus(validity))
        return selected

    def at_length(self, target_length_m: float, *, front_id: str | None = None,
                  segment_id: str | None = None,
                  stage: ObservationStage | str | None = None,
                  coordinate: LengthCoordinate | str = LengthCoordinate.EXTENSION,
                  validity: ValidityStatus | str | None = None,
                  valid_only: bool | None = None, atol_m: float = 1e-12) -> LengthQueryResult:
        """Query without blending selectors or interpolating discrete jumps."""
        target = float(target_length_m)
        coordinate = LengthCoordinate(coordinate)
        all_selected = self.filtered(front_id=front_id, segment_id=segment_id, stage=stage)
        rows = self.filtered(front_id=front_id, segment_id=segment_id, stage=stage,
                             validity=validity, valid_only=valid_only)
        position = lambda row: row.length_m(coordinate)
        exact = tuple(row for row in rows if abs(position(row) - target) <= atol_m)
        offset = lambda row: row.initial_length_m if coordinate is LengthCoordinate.ABSOLUTE_PROJECTED else 0.0
        crossing = tuple(row for row in rows
                         if row.committed_length_m - row.pre_length_m > atol_m
                         and row.pre_length_m + offset(row) - atol_m <= target
                         <= row.committed_length_m + offset(row) + atol_m)
        below = [row for row in rows if position(row) < target - atol_m]
        above = [row for row in rows if position(row) > target + atol_m]
        lower = max(below, key=lambda row: (position(row), row.sequence), default=None)
        upper = min(above, key=lambda row: (position(row), row.sequence), default=None)
        unqualified_here = any(abs(position(row) - target) <= atol_m for row in all_selected)
        if exact:
            status = "exact_repeated" if len(exact) > 1 else "exact"
        elif crossing:
            status = "discrete_jump_bracket"
        elif unqualified_here and (validity is not None or valid_only):
            status = "invalid_or_unqualified_at_length"
        elif lower is not None and upper is not None:
            status = "between_observations_no_interpolation"
        else:
            status = "out_of_range"
        return LengthQueryResult("fracture-length-query/2", target, coordinate, status,
                                 exact, crossing, lower, upper)

    def write_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w") as stream:
            for row in self.rows:
                stream.write(json.dumps(row.to_dict(), sort_keys=True, allow_nan=False) + "\n")

    def write_csv(self, path: str | Path) -> None:
        names = [field.name for field in fields(FractureObservation)]
        with Path(path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(row.to_dict() for row in self.rows)

    @classmethod
    def load(cls, path: str | Path) -> "ObservationHistory":
        path = Path(path)
        if path.suffix.lower() == ".jsonl":
            with path.open() as stream:
                rows = [FractureObservation.from_dict(json.loads(line)) for line in stream if line.strip()]
        else:
            with path.open(newline="") as stream:
                rows = [FractureObservation.from_dict(row) for row in csv.DictReader(stream)]
        return cls(rows)

def _number(row: dict[str, str], name: str, default: float = math.nan) -> float:
    value = row.get(name, "")
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default

def from_legacy_steps_csv(path: str | Path, *, front_id: str = "primary",
                          segment_id: str = "legacy", initial_length_m: float = 0.0) -> ObservationHistory:
    """Adapt every accepted legacy step without inventing absent semantics."""
    path = Path(path)
    with path.open(newline="") as stream:
        source = list(csv.DictReader(stream))
    rows: list[FractureObservation] = []
    previous_u = 0.0
    previous_direction: LoadDirection | None = None
    segment_number = -1
    for sequence, (source_row, row) in enumerate(enumerate(source, start=2)):
        u = _number(row, "Uapp_m")
        if math.isfinite(u):
            delta = u - previous_u
            scale = max(abs(u), abs(previous_u), 1.0e-30)
            direction = (LoadDirection.HOLD if abs(delta) <= 1.0e-14 * scale else
                         LoadDirection.LOADING if delta > 0.0 else LoadDirection.UNLOADING)
            previous_u = u
        else:
            direction = LoadDirection.UNKNOWN
        if direction != previous_direction:
            segment_number += 1
            previous_direction = direction
        n_fire = max(0, int(round(_number(row, "n_fire", 0.0))))
        committed = _number(row, "crack_extension_m", 0.0)
        increment = max(0.0, _number(row, "da_block_m", 0.0))
        rows.append(FractureObservation(
            sequence=sequence, time_s=_number(row, "time_s"),
            cycle=_number(row, "fatigue_cycles", _number(row, "cycle", _number(row, "cycles"))),
            segment_id=f"{segment_id}-{segment_number:03d}-{direction.value}",
            stage=ObservationStage.UNKNOWN, front_id=front_id,
            event_id=str(row.get("step", sequence)) if n_fire else "",
            load_direction=direction, pre_length_m=committed - increment if n_fire else committed,
            committed_length_m=committed, K_J_Pa_sqrt_m=_number(row, "KJ_Pa_sqrtm"),
            K_probe_Pa_sqrt_m=_number(row, "KJ_probe_Pa_sqrtm"),
            J_J_per_m2=_number(row, "J_effective_direct_J_per_m2"),
            validity_status=ValidityStatus.UNKNOWN,
            validity_reason="legacy_steps_has_no_explicit_contour_or_observer_status",
            contour_id=str(row.get("J_contour_id", "")), observer_id="legacy_KJ",
            initial_length_m=float(initial_length_m), aggregate_event_count=n_fire,
            source_file=str(path), source_row=source_row))
    return ObservationHistory(rows)

__all__ = ["SCHEMA", "ObservationStage", "LoadDirection", "ValidityStatus",
           "LengthCoordinate", "FractureObservation", "LengthQueryResult",
           "ObservationHistory", "from_legacy_steps_csv"]
