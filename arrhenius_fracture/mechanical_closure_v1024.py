"""Mechanical closure calibrated from cap-free 2-D tensor-probe histories.

The constitutive state is shared exactly with v10.2.2/v10.2.3.  This module
replaces the temporary constant anisotropic factors used by the monotonic
reduced runner with a candidate-independent interpolation of factors measured
by the production 2-D tensor probe.

Only geometric/mechanical coordinates are used: applied K and continuous crack
progress within the current checkpoint.  Material parameters are never inputs
to the closure.  Extrapolation is explicit and audited; promoted candidates
must still pass cap-free 2-D endpoint validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODEL_ID = "v10.2.4_cap_free_2d_tensor_drive_atlas"


@dataclass(frozen=True)
class AtlasSample:
    K_MPa_sqrt_m: float
    progress: float
    factor_0: float
    factor_1: float
    source_trace: str = ""


@dataclass
class ClosureEvaluation:
    factors: tuple[float, float]
    normalized_distance: float
    outside_support: bool
    neighbor_count: int


class TensorDriveAtlas:
    """Inverse-distance interpolation of production 2-D tensor factors."""

    def __init__(self, samples: Iterable[AtlasSample], neighbors: int = 12):
        self.samples = list(samples)
        if len(self.samples) < 2:
            raise ValueError("tensor-drive atlas requires at least two samples")
        self.neighbors = max(1, min(int(neighbors), len(self.samples)))
        self._x = np.asarray(
            [[row.K_MPa_sqrt_m, row.progress] for row in self.samples], dtype=float
        )
        self._y = np.asarray(
            [[row.factor_0, row.factor_1] for row in self.samples], dtype=float
        )
        if np.any(~np.isfinite(self._x)) or np.any(~np.isfinite(self._y)):
            raise ValueError("atlas contains non-finite values")
        if np.any(self._y < 0.0):
            raise ValueError("tensor drive factors must be nonnegative")
        self._xmin = np.min(self._x, axis=0)
        self._xmax = np.max(self._x, axis=0)
        span = self._xmax - self._xmin
        self._scale = np.where(span > 1.0e-12, span, 1.0)
        self.evaluation_count = 0
        self.outside_support_count = 0
        self.maximum_normalized_distance = 0.0

    @classmethod
    def from_csv(cls, path: str | Path, neighbors: int = 12) -> "TensorDriveAtlas":
        path = Path(path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        samples = [
            AtlasSample(
                K_MPa_sqrt_m=float(row["K_MPa_sqrt_m"]),
                progress=float(row["progress"]),
                factor_0=float(row["drive_factor_0"]),
                factor_1=float(row["drive_factor_1"]),
                source_trace=str(row.get("source_trace", "")),
            )
            for row in rows
        ]
        return cls(samples, neighbors=neighbors)

    def evaluate(self, K_MPa_sqrt_m: float, progress: float) -> ClosureEvaluation:
        query = np.asarray(
            [max(float(K_MPa_sqrt_m), 0.0), float(np.clip(progress, 0.0, 1.0))],
            dtype=float,
        )
        outside = bool(np.any(query < self._xmin) or np.any(query > self._xmax))
        distance = np.linalg.norm((self._x - query[None, :]) / self._scale, axis=1)
        order = np.argsort(distance)[: self.neighbors]
        selected = distance[order]
        if selected[0] <= 1.0e-14:
            value = self._y[order[0]].copy()
        else:
            weight = 1.0 / np.maximum(selected, 1.0e-12) ** 2
            weight /= np.sum(weight)
            value = weight @ self._y[order]
        nearest = float(selected[0])
        self.evaluation_count += 1
        self.outside_support_count += int(outside)
        self.maximum_normalized_distance = max(
            self.maximum_normalized_distance, nearest
        )
        return ClosureEvaluation(
            factors=(max(float(value[0]), 0.0), max(float(value[1]), 0.0)),
            normalized_distance=nearest,
            outside_support=outside,
            neighbor_count=int(len(order)),
        )

    def audit(self) -> dict[str, Any]:
        return {
            "schema": MODEL_ID,
            "sample_count": len(self.samples),
            "neighbors": self.neighbors,
            "K_support_MPa_sqrt_m": [float(self._xmin[0]), float(self._xmax[0])],
            "progress_support": [float(self._xmin[1]), float(self._xmax[1])],
            "evaluation_count": int(self.evaluation_count),
            "outside_support_count": int(self.outside_support_count),
            "outside_support_fraction": (
                float(self.outside_support_count / self.evaluation_count)
                if self.evaluation_count
                else 0.0
            ),
            "maximum_nearest_normalized_distance": float(
                self.maximum_normalized_distance
            ),
            "material_parameters_used_by_closure": False,
            "shielding_cap_used": False,
            "promoted_candidates_require_2d_validation": True,
        }


def _read_schedule(path: Path, checkpoint_da_m: float) -> list[AtlasSample]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples: list[AtlasSample] = []
    for row in rows:
        K = float(row["K_Pa_sqrt_m"]) / 1.0e6
        advance = row.get("expected_micro_advance_total_m", "")
        if advance in {None, "", "nan", "NaN"}:
            progress = 0.0
        else:
            progress = float(advance) / max(float(checkpoint_da_m), 1.0e-30)
        samples.append(
            AtlasSample(
                K_MPa_sqrt_m=K,
                progress=float(np.clip(progress, 0.0, 1.0)),
                factor_0=float(row["drive_factor_0"]),
                factor_1=float(row["drive_factor_1"]),
                source_trace=str(path),
            )
        )
    return samples


def build_atlas_from_trace_roots(
    trace_roots: Iterable[str | Path],
    output_csv: str | Path,
    *,
    checkpoint_da_m: float = 5.0e-6,
    thin_stride: int = 1,
) -> dict[str, Any]:
    """Collect v10.2.3 replay schedules into one mechanical atlas."""
    all_samples: list[AtlasSample] = []
    sources: list[str] = []
    stride = max(int(thin_stride), 1)
    for raw_root in trace_roots:
        root = Path(raw_root)
        candidates = [
            root / "v10_2_3_2d_replay_schedule.csv",
            root / "two_d" / "v10_2_3_2d_replay_schedule.csv",
        ]
        path = next((value for value in candidates if value.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"no v10.2.3 replay schedule found below {root}"
            )
        rows = _read_schedule(path, checkpoint_da_m)[::stride]
        if not rows:
            raise ValueError(f"trace contains no atlas samples: {path}")
        all_samples.extend(rows)
        sources.append(str(path.resolve()))

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        fieldnames = [
            "K_MPa_sqrt_m",
            "progress",
            "drive_factor_0",
            "drive_factor_1",
            "source_trace",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_samples:
            writer.writerow(
                {
                    "K_MPa_sqrt_m": f"{row.K_MPa_sqrt_m:.17g}",
                    "progress": f"{row.progress:.17g}",
                    "drive_factor_0": f"{row.factor_0:.17g}",
                    "drive_factor_1": f"{row.factor_1:.17g}",
                    "source_trace": row.source_trace,
                }
            )

    atlas = TensorDriveAtlas(all_samples)
    payload = {
        "schema": MODEL_ID,
        "output_csv": str(output.resolve()),
        "trace_sources": sources,
        "checkpoint_da_m": float(checkpoint_da_m),
        "thin_stride": stride,
        **atlas.audit(),
    }
    Path(str(output) + ".json").write_text(json.dumps(payload, indent=2))
    return payload


__all__ = [
    "MODEL_ID",
    "AtlasSample",
    "ClosureEvaluation",
    "TensorDriveAtlas",
    "build_atlas_from_trace_roots",
]
