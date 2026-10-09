"""Chronological, stage-aware fracture observer history.

Repeated crack lengths are records, not duplicates: they can belong to
different load segments, stages, contours, fronts, or unloading/reloading
passes.  Discrete geometry jumps retain both pre- and committed lengths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Iterator


SCHEMA = "fracture-observation-history/1"


class ObservationStage(str, Enum):
    PRE_EVENT = "pre_event"
    PROPOSED = "proposed"
    COMMITTED = "committed"
    REEQUILIBRATED = "reequilibrated"


class LoadDirection(str, Enum):
    LOADING = "loading"
    UNLOADING = "unloading"
    HOLD = "hold"


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
    K_I_Pa_sqrt_m: float
    K_II_Pa_sqrt_m: float = math.nan
    K_effective_Pa_sqrt_m: float = math.nan
    J_J_per_m2: float = math.nan
    G_J_per_m2: float = math.nan
    contour_valid: bool = True
    observer_valid: bool = True
    contour_id: str = ""
    source_file: str = ""
    source_row: int = -1
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ObservationStage):
            object.__setattr__(self, "stage", ObservationStage(self.stage))
        if not isinstance(self.load_direction, LoadDirection):
            object.__setattr__(self, "load_direction", LoadDirection(self.load_direction))
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported observation schema: {self.schema!r}")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.committed_length_m + 1e-15 < self.pre_length_m:
            raise ValueError("committed length cannot precede pre-event length")

    @property
    def stage_length_m(self) -> float:
        if self.stage in {ObservationStage.PRE_EVENT, ObservationStage.PROPOSED}:
            return self.pre_length_m
        return self.committed_length_m

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        payload["load_direction"] = self.load_direction.value
        for key, value in tuple(payload.items()):
            if isinstance(value, float) and not math.isfinite(value):
                payload[key] = None
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "FractureObservation":
        values = dict(payload)
        int_fields = {"sequence", "source_row"}
        float_fields = {
            "time_s", "cycle", "pre_length_m", "committed_length_m",
            "K_I_Pa_sqrt_m", "K_II_Pa_sqrt_m", "K_effective_Pa_sqrt_m",
            "J_J_per_m2", "G_J_per_m2",
        }
        bool_fields = {"contour_valid", "observer_valid"}
        for name in int_fields:
            values[name] = int(values[name])
        for name in float_fields:
            value = values[name]
            values[name] = math.nan if value in (None, "") else float(value)
        for name in bool_fields:
            value = values[name]
            values[name] = value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes"}
        return cls(**values)


@dataclass(frozen=True)
class LengthQueryResult:
    schema: str
    target_length_m: float
    status: str
    exact: tuple[FractureObservation, ...]
    crossing: tuple[FractureObservation, ...]
    lower: FractureObservation | None
    upper: FractureObservation | None
    interpolation_performed: bool = False

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "target_length_m": self.target_length_m,
            "status": self.status,
            "exact": [row.to_dict() for row in self.exact],
            "crossing": [row.to_dict() for row in self.crossing],
            "lower": None if self.lower is None else self.lower.to_dict(),
            "upper": None if self.upper is None else self.upper.to_dict(),
            "interpolation_performed": self.interpolation_performed,
        }


class ObservationHistory:
    def __init__(self, rows: Iterable[FractureObservation]):
        self.rows = tuple(sorted(rows, key=lambda row: row.sequence))
        sequences = [row.sequence for row in self.rows]
        if len(sequences) != len(set(sequences)):
            raise ValueError("observation sequence identifiers must be unique")

    def filtered(self, *, front_id: str | None = None, segment_id: str | None = None,
                 stage: ObservationStage | str | None = None,
                 valid_only: bool = True) -> tuple[FractureObservation, ...]:
        selected = self.rows
        if front_id is not None:
            selected = tuple(row for row in selected if row.front_id == front_id)
        if segment_id is not None:
            selected = tuple(row for row in selected if row.segment_id == segment_id)
        if stage is not None:
            stage = ObservationStage(stage)
            selected = tuple(row for row in selected if row.stage is stage)
        if valid_only:
            selected = tuple(row for row in selected if row.observer_valid and row.contour_valid)
        return selected

    def at_length(self, target_length_m: float, *, front_id: str | None = None,
                  segment_id: str | None = None,
                  stage: ObservationStage | str | None = None,
                  valid_only: bool = True, atol_m: float = 1e-12) -> LengthQueryResult:
        """Query without blending load stages or interpolating discrete jumps."""
        target = float(target_length_m)
        rows = self.filtered(front_id=front_id, segment_id=segment_id, stage=stage,
                             valid_only=valid_only)
        exact = tuple(row for row in rows if abs(row.stage_length_m - target) <= atol_m)
        crossing = tuple(
            row for row in rows
            if row.committed_length_m - row.pre_length_m > atol_m
            and row.pre_length_m - atol_m <= target <= row.committed_length_m + atol_m
        )
        below = [row for row in rows if row.stage_length_m < target - atol_m]
        above = [row for row in rows if row.stage_length_m > target + atol_m]
        lower = max(below, key=lambda row: (row.stage_length_m, row.sequence), default=None)
        upper = min(above, key=lambda row: (row.stage_length_m, row.sequence), default=None)
        if exact:
            status = "exact_repeated" if len(exact) > 1 else "exact"
        elif crossing:
            status = "discrete_jump_bracket"
        elif lower is not None and upper is not None:
            status = "between_observations_no_interpolation"
        else:
            status = "out_of_range"
        return LengthQueryResult(
            schema="fracture-length-query/1", target_length_m=target, status=status,
            exact=exact, crossing=crossing, lower=lower, upper=upper,
        )

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


def from_legacy_steps_csv(path: str | Path, *, front_id: str = "primary",
                          segment_id: str = "monotonic-0") -> ObservationHistory:
    """Adapt v10.2.27 step rows while retaining each event's jump interval."""
    path = Path(path)
    with path.open(newline="") as stream:
        source = list(csv.DictReader(stream))
    rows: list[FractureObservation] = []
    for source_row, row in enumerate(source, start=2):
        n_fire = float(row.get("n_fire", "0") or 0.0)
        if n_fire <= 0.0:
            continue
        committed = float(row["crack_extension_m"])
        increment = float(row.get("da_block_m", "0") or 0.0)
        pre = committed - increment
        K = float(row["KJ_Pa_sqrtm"])
        J = float(row.get("J_effective_direct_J_per_m2", "nan") or "nan")
        rows.append(FractureObservation(
            sequence=len(rows), time_s=float(row.get("time_s", "nan") or "nan"),
            cycle=float(row.get("cycle", row.get("cycles", "nan")) or "nan"),
            segment_id=segment_id, stage=ObservationStage.COMMITTED,
            front_id=front_id, event_id=str(row.get("step", len(rows))),
            load_direction=LoadDirection.LOADING, pre_length_m=pre,
            committed_length_m=committed, K_I_Pa_sqrt_m=K,
            K_effective_Pa_sqrt_m=K, J_J_per_m2=J,
            contour_valid=math.isfinite(K), observer_valid=math.isfinite(K),
            contour_id=str(row.get("J_contour_id", "legacy_steps")),
            source_file=str(path), source_row=source_row,
        ))
    return ObservationHistory(rows)


__all__ = [
    "SCHEMA", "ObservationStage", "LoadDirection", "FractureObservation",
    "LengthQueryResult", "ObservationHistory", "from_legacy_steps_csv",
]
