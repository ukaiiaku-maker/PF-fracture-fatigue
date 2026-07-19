"""v10.2.13 physical FEM capture semantics.

Snapshot requests target cumulative crack-path extension only.  Opening and the
analytical ``r_eff`` are recorded diagnostics, not matching coordinates.  A
production snapshot is rejected unless the FEM process zone has at least the
requested number of tip elements; trajectory discovery remains available on a
coarser mesh because it does not create kernel data.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .anisotropic_emission_v10174 import OBSERVER as DRIVE_OBSERVER
from .physical_fem_capture_v10212 import (
    CaptureRequest,
    PhysicalFEMCapture as _BaseCapture,
)
from .physical_fem_capture_trace_v10212 import PhysicalFEMCapture as _TraceCapture

MODEL_ID = "v10.2.13_extension_only_physical_fem_capture"
TRACE_FIELDS = (
    "trace_index",
    "temperature_K",
    "K_applied_Pa_sqrt_m",
    "observed_analytical_r_eff_over_r0",
    "kernel_radius_compatibility_coordinate",
    "opening_strength_fraction",
    "cumulative_crack_path_extension_m",
    "projected_crack_extension_m",
    "projected_x_extension_m",
    "crack_extension_m",
    "mechanics_serial",
    "drive_serial",
)


def load_extension_capture_requests(path: str | Path) -> list[CaptureRequest]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("extension capture request table is empty")
    extension_name = (
        "cumulative_crack_path_extension_m"
        if "cumulative_crack_path_extension_m" in rows[0]
        else "crack_extension_m"
    )
    required = {
        "state_id",
        "temperature_K",
        extension_name,
        "extension_tolerance_m",
        "interaction_ell_m",
    }
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"extension capture table is missing columns {missing}")
    result = []
    seen = set()
    for row in rows:
        state_id = str(row["state_id"]).strip()
        if not state_id or state_id in seen:
            raise ValueError(f"invalid or duplicate state_id {state_id!r}")
        seen.add(state_id)
        request = CaptureRequest(
            state_id=state_id,
            temperature_K=float(row["temperature_K"]),
            r_eff_over_r0=1.0,
            opening_strength_fraction=0.5,
            crack_extension_m=float(row[extension_name]),
            r_tolerance=1.0e30,
            opening_tolerance=1.0,
            extension_tolerance_m=float(row["extension_tolerance_m"]),
            interaction_ell_m=float(row["interaction_ell_m"]),
        ).validate()
        result.append(request)
    return result


class PhysicalFEMCapture(_TraceCapture):
    """Reachable-state trace plus extension-only snapshot matching."""

    def __init__(
        self,
        requests: list[CaptureRequest],
        outroot: str | Path,
        *,
        minimum_elements_per_process_zone: float = 3.0,
    ):
        super().__init__(requests, outroot)
        minimum = float(minimum_elements_per_process_zone)
        if not math.isfinite(minimum) or minimum < 1.0:
            raise ValueError("minimum process-zone resolution must be at least one")
        self.minimum_elements_per_process_zone = minimum
        self._initial_tip_by_temperature: dict[float, np.ndarray] = {}
        self._initial_direction_by_temperature: dict[float, np.ndarray] = {}
        self.mesh_gate_checks: list[dict[str, Any]] = []

    def _matching_request(self, temperature: float, coordinates: dict[str, float]):
        candidates = []
        extension = float(coordinates["crack_extension_m"])
        for request in self.pending:
            if not math.isclose(
                float(temperature), request.temperature_K, rel_tol=0.0, abs_tol=1.0e-8
            ):
                continue
            error = abs(extension - request.crack_extension_m)
            if error <= request.extension_tolerance_m:
                candidates.append((error, request))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _trace_coordinates(self, engine, K: float, T: float) -> None:
        if self.latest_assembly is None or self.latest_boundary is None:
            return
        drive = DRIVE_OBSERVER.latest_drive
        if not isinstance(drive, dict) or not bool(drive.get("reliable", False)):
            return
        if int(drive.get("mechanics_serial", -1)) != int(DRIVE_OBSERVER.mechanics_serial):
            return
        r0 = max(float(engine.f.r0), 1.0e-30)
        r_eff = max(float(engine.r_eff()), r0)
        sigma_cap = float(engine.f.sigma_cap)
        if sigma_cap <= 0.0:
            return
        sigma_local = float(engine.sigma_tip(K))
        opening = min(max(sigma_local / sigma_cap, 0.0), 1.0)
        path_extension = max(
            float(getattr(engine, "micro_advance_total_m", 0.0)),
            float(getattr(engine.mpz, "advance_total_m", 0.0)),
            0.0,
        )
        temperature = float(T)
        tip = np.asarray(drive["tip_xy_m"], dtype=float).reshape(2)
        direction = np.asarray(drive["front_direction"], dtype=float).reshape(2)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
        if temperature not in self._initial_tip_by_temperature:
            self._initial_tip_by_temperature[temperature] = tip.copy()
            self._initial_direction_by_temperature[temperature] = direction.copy()
        delta = tip - self._initial_tip_by_temperature[temperature]
        projected = float(delta @ self._initial_direction_by_temperature[temperature])
        row = {
            "trace_index": len(self.coordinate_trace),
            "temperature_K": temperature,
            "K_applied_Pa_sqrt_m": float(K),
            "observed_analytical_r_eff_over_r0": float(r_eff / r0),
            "kernel_radius_compatibility_coordinate": 1.0,
            "opening_strength_fraction": float(opening),
            "cumulative_crack_path_extension_m": float(path_extension),
            "projected_crack_extension_m": projected,
            "projected_x_extension_m": float(delta[0]),
            "crack_extension_m": float(path_extension),
            "mechanics_serial": int(DRIVE_OBSERVER.mechanics_serial),
            "drive_serial": int(drive.get("drive_serial", -1)),
        }
        numeric = (
            "temperature_K",
            "K_applied_Pa_sqrt_m",
            "observed_analytical_r_eff_over_r0",
            "opening_strength_fraction",
            "cumulative_crack_path_extension_m",
            "projected_crack_extension_m",
            "projected_x_extension_m",
        )
        if any(not math.isfinite(float(row[name])) for name in numeric):
            raise RuntimeError("non-finite coordinate in v10.2.13 reachable-state trace")
        self.coordinate_trace.append(row)

    def _write_trace(self) -> Path:
        if not self.coordinate_trace:
            raise RuntimeError(
                "physical atlas discovery captured no reliable FEM/tensor-probe states"
            )
        path = self.outroot / "reachable_physical_state_trace.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            writer.writerows(self.coordinate_trace)
        audit = {
            "schema": MODEL_ID,
            "trace": str(path),
            "records": len(self.coordinate_trace),
            "kernel_radius_axis_policy": "disabled_constant_compatibility",
            "opening_axis_policy": "validation_only_until_load_invariance_passes",
            "production_kernel_candidate_axis": "cumulative_crack_path_extension_m",
            "legacy_crack_extension_m_alias": "cumulative_crack_path_extension_m",
            "projected_extension_recorded_separately": True,
            "purpose": "select frozen crack geometries for load-invariance validation",
        }
        (self.outroot / "reachable_physical_state_trace.json").write_text(
            json.dumps(audit, indent=2)
        )
        return path

    def before_engine_step(self, engine, K: float, T: float) -> None:
        self._trace_coordinates(engine, K, T)
        if self.pending and self.latest_assembly is not None:
            r0 = max(float(engine.f.r0), 1.0e-30)
            r_eff = max(float(engine.r_eff()), r0)
            sigma_cap = float(engine.f.sigma_cap)
            coordinates = {
                "r_eff_over_r0": r_eff / r0,
                "opening_strength_fraction": min(
                    max(float(engine.sigma_tip(K)) / max(sigma_cap, 1.0e-30), 0.0),
                    1.0,
                ),
                "crack_extension_m": max(
                    float(getattr(engine, "micro_advance_total_m", 0.0)),
                    float(getattr(engine.mpz, "advance_total_m", 0.0)),
                    0.0,
                ),
            }
            request = self._matching_request(float(T), coordinates)
            if request is not None:
                h_tip = float(self.latest_assembly["mesh"].hbar_tip)
                process_zone = float(engine.mpz.length_m)
                elements = process_zone / max(h_tip, 1.0e-30)
                check = {
                    "state_id": request.state_id,
                    "temperature_K": float(T),
                    "hbar_tip_m": h_tip,
                    "process_zone_length_m": process_zone,
                    "elements_per_process_zone": elements,
                    "minimum_required": self.minimum_elements_per_process_zone,
                    "passed": elements >= self.minimum_elements_per_process_zone,
                }
                self.mesh_gate_checks.append(check)
                if not check["passed"]:
                    raise RuntimeError(
                        "production signed-kernel snapshot is under-resolved: "
                        f"L_pz/h_tip={elements:.6g} < "
                        f"{self.minimum_elements_per_process_zone:.6g}"
                    )
        _BaseCapture.before_engine_step(self, engine, K, T)

    def finalize(self, *, require_complete: bool = True) -> dict[str, Any]:
        trace_path = self._write_trace()
        payload = _BaseCapture.finalize(self, require_complete=require_complete)
        payload.update(
            {
                "schema": MODEL_ID,
                "reachable_state_trace": str(trace_path),
                "reachable_state_trace_records": len(self.coordinate_trace),
                "snapshot_matching_coordinates": [
                    "temperature_K",
                    "cumulative_crack_path_extension_m",
                ],
                "opening_is_snapshot_matching_coordinate": False,
                "analytical_r_eff_is_snapshot_matching_coordinate": False,
                "minimum_elements_per_process_zone": (
                    self.minimum_elements_per_process_zone
                ),
                "mesh_gate_checks": self.mesh_gate_checks,
            }
        )
        (self.outroot / "capture_complete.json").write_text(
            json.dumps(payload, indent=2)
        )
        return payload


__all__ = [
    "MODEL_ID",
    "TRACE_FIELDS",
    "PhysicalFEMCapture",
    "load_extension_capture_requests",
]
