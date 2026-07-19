"""Reachable-state trace extension for the v10.2.12 physical FEM capture."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .anisotropic_emission_v10174 import OBSERVER as DRIVE_OBSERVER
from .physical_fem_capture_v10212 import PhysicalFEMCapture as _BaseCapture

MODEL_ID = "v10.2.12_reachable_physical_state_trace"
TRACE_FIELDS = (
    "trace_index",
    "temperature_K",
    "K_applied_Pa_sqrt_m",
    "observed_analytical_r_eff_over_r0",
    "kernel_radius_compatibility_coordinate",
    "opening_strength_fraction",
    "crack_extension_m",
    "mechanics_serial",
    "drive_serial",
)


class PhysicalFEMCapture(_BaseCapture):
    """Base snapshot capture plus an auditable trajectory of reachable states."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinate_trace: list[dict[str, Any]] = []

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
        extension = max(
            float(getattr(engine, "micro_advance_total_m", 0.0)),
            float(getattr(engine.mpz, "advance_total_m", 0.0)),
            0.0,
        )
        row = {
            "trace_index": len(self.coordinate_trace),
            "temperature_K": float(T),
            "K_applied_Pa_sqrt_m": float(K),
            "observed_analytical_r_eff_over_r0": float(r_eff / r0),
            "kernel_radius_compatibility_coordinate": 1.0,
            "opening_strength_fraction": float(opening),
            "crack_extension_m": float(extension),
            "mechanics_serial": int(DRIVE_OBSERVER.mechanics_serial),
            "drive_serial": int(drive.get("drive_serial", -1)),
        }
        if any(not math.isfinite(float(row[name])) for name in (
            "temperature_K",
            "K_applied_Pa_sqrt_m",
            "observed_analytical_r_eff_over_r0",
            "opening_strength_fraction",
            "crack_extension_m",
        )):
            raise RuntimeError("non-finite coordinate in reachable-state trace")
        self.coordinate_trace.append(row)

    def before_engine_step(self, engine, K: float, T: float) -> None:
        self._trace_coordinates(engine, K, T)
        super().before_engine_step(engine, K, T)

    def _write_trace(self) -> Path:
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
            "observed_analytical_r_eff_retained": True,
            "active_kernel_design_coordinates": [
                "opening_strength_fraction",
                "crack_extension_m",
            ],
            "purpose": "design physical capture requests from reachable production trajectories",
        }
        (self.outroot / "reachable_physical_state_trace.json").write_text(
            json.dumps(audit, indent=2)
        )
        return path

    def finalize(self, *, require_complete: bool = True) -> dict[str, Any]:
        trace_path = self._write_trace()
        payload = super().finalize(require_complete=require_complete)
        payload.update(
            {
                "reachable_state_trace": str(trace_path),
                "reachable_state_trace_records": len(self.coordinate_trace),
                "kernel_radius_axis_policy": "disabled_constant_compatibility",
            }
        )
        (self.outroot / "capture_complete.json").write_text(json.dumps(payload, indent=2))
        return payload


__all__ = ["MODEL_ID", "TRACE_FIELDS", "PhysicalFEMCapture"]
