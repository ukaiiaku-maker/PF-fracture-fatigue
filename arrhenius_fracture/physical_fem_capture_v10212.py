"""Capture accepted 2-D FEM equilibria for v10.2.12 kernel generation.

The capture hooks observe the existing production solve; they do not implement a
second mechanics path.  A state is copied immediately before the front engine
advances its kinetic state, after the current tensor probe has been constructed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .anisotropic_emission_v10174 import OBSERVER as DRIVE_OBSERVER
from .physical_fem_snapshot_v10212 import SnapshotMetadata, save_snapshot

MODEL_ID = "v10.2.12_live_production_fem_state_capture"


@dataclass(frozen=True)
class CaptureRequest:
    state_id: str
    temperature_K: float
    r_eff_over_r0: float
    opening_strength_fraction: float
    crack_extension_m: float
    r_tolerance: float
    opening_tolerance: float
    extension_tolerance_m: float
    interaction_ell_m: float

    def validate(self) -> "CaptureRequest":
        if not str(self.state_id).strip():
            raise ValueError("state_id must be nonempty")
        for name in (
            "temperature_K",
            "r_eff_over_r0",
            "opening_strength_fraction",
            "crack_extension_m",
            "r_tolerance",
            "opening_tolerance",
            "extension_tolerance_m",
            "interaction_ell_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.temperature_K <= 0.0 or self.r_eff_over_r0 <= 0.0:
            raise ValueError("temperature and r_eff_over_r0 must be positive")
        if not 0.0 <= self.opening_strength_fraction <= 1.0:
            raise ValueError("opening target must lie in [0,1]")
        if self.crack_extension_m < 0.0:
            raise ValueError("crack extension must be nonnegative")
        if min(self.r_tolerance, self.opening_tolerance, self.extension_tolerance_m) < 0.0:
            raise ValueError("capture tolerances must be nonnegative")
        if self.interaction_ell_m <= 0.0:
            raise ValueError("interaction_ell_m must be positive")
        return self


def load_capture_requests(path: str | Path) -> list[CaptureRequest]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("capture request table is empty")
    required = {
        "state_id",
        "temperature_K",
        "r_eff_over_r0",
        "opening_strength_fraction",
        "crack_extension_m",
        "r_tolerance",
        "opening_tolerance",
        "extension_tolerance_m",
        "interaction_ell_m",
    }
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"capture request table is missing columns {missing}")
    result = []
    seen = set()
    for row in rows:
        request = CaptureRequest(
            state_id=str(row["state_id"]).strip(),
            temperature_K=float(row["temperature_K"]),
            r_eff_over_r0=float(row["r_eff_over_r0"]),
            opening_strength_fraction=float(row["opening_strength_fraction"]),
            crack_extension_m=float(row["crack_extension_m"]),
            r_tolerance=float(row["r_tolerance"]),
            opening_tolerance=float(row["opening_tolerance"]),
            extension_tolerance_m=float(row["extension_tolerance_m"]),
            interaction_ell_m=float(row["interaction_ell_m"]),
        ).validate()
        if request.state_id in seen:
            raise ValueError(f"duplicate state_id {request.state_id!r}")
        seen.add(request.state_id)
        result.append(request)
    return result


def _public_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    result = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if isinstance(item, (bool, int, float, str)) or item is None:
            result[name] = item
    return result


def _engine_payload(engine) -> dict[str, Any]:
    manifest = getattr(engine, "manifest", None)
    return {
        "schema": MODEL_ID,
        "front_config": _public_mapping(engine.f),
        "mpz_config": _public_mapping(engine.mpz.cfg),
        "tip_config": _public_mapping(engine.tip_cfg),
        "anisotropic_config": _public_mapping(engine.anisotropic_cfg),
        "campaign_config": {
            "backstress_scale": float(engine.mpz._campaign_backstress_scale),
            "refresh_scale": float(engine.mpz._campaign_refresh_scale),
        },
        "G_Pa": float(engine.G),
        "poisson": float(engine.nu),
        "b_m": float(engine.b),
        "material_manifest": manifest.as_dict() if manifest is not None else {},
        "transport_mode": str(getattr(engine.mpz, "_signed_transport_mode", getattr(engine.mpz, "_anisotropic_transport_mode", "validated_scalar"))),
        "capture_loading_path": "mechanics_only_shielding_disabled",
        "local_strength_sigma_cap_is_not_Kshield_cap": True,
        "constitutive_K_shield_cap_applied": False,
    }


class PhysicalFEMCapture:
    def __init__(self, requests: list[CaptureRequest], outroot: str | Path):
        self.requests = [request.validate() for request in requests]
        self.outroot = Path(outroot)
        if self.outroot.exists():
            raise FileExistsError(f"refusing to overwrite {self.outroot}")
        self.outroot.mkdir(parents=True)
        self.captured: dict[str, dict[str, Any]] = {}
        self.latest_assembly: dict[str, Any] | None = None
        self.latest_boundary = None
        self.latest_Uy_top = 0.0
        self.latest_Uy_bot = 0.0
        self.solve_serial = 0
        self.assembly_serial = 0
        self.attempts = 0

    @property
    def pending(self) -> list[CaptureRequest]:
        return [request for request in self.requests if request.state_id not in self.captured]

    def wrap_assemble_factory(self, inherited_factory: Callable) -> Callable:
        def factory(original: Callable) -> Callable:
            inherited = inherited_factory(original)

            def wrapped(*args, **kwargs):
                result = inherited(*args, **kwargs)
                try:
                    cohesive = kwargs.get("cohesive_network")
                    if cohesive is None and len(args) > 9:
                        cohesive = args[9]
                    self.latest_assembly = {
                        "mesh": args[0],
                        "u": np.asarray(args[1], dtype=float).copy(),
                        "ep_gp": np.asarray(args[2], dtype=float).copy(),
                        "rho_gp": np.asarray(args[3], dtype=float).copy(),
                        "d": np.asarray(args[4], dtype=float).copy(),
                        "D": np.asarray(args[5], dtype=float).copy(),
                        "mat": args[6],
                        "cohesive_network": cohesive,
                        "sigma_gp": np.asarray(result[2], dtype=float).copy(),
                    }
                    self.assembly_serial += 1
                except Exception:
                    self.latest_assembly = None
                return result

            wrapped.__name__ = getattr(original, "__name__", "assemble_mechanics")
            return wrapped

        return factory

    def wrap_solve_dirichlet(self, original: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            self.latest_boundary = args[3]
            self.latest_Uy_top = float(args[4])
            self.latest_Uy_bot = float(args[5])
            self.solve_serial += 1
            return result

        wrapped.__name__ = getattr(original, "__name__", "solve_dirichlet")
        return wrapped

    def _matching_request(self, temperature: float, coordinates: dict[str, float]):
        candidates = []
        for request in self.pending:
            if not math.isclose(float(temperature), request.temperature_K, rel_tol=0.0, abs_tol=1.0e-8):
                continue
            dr = abs(coordinates["r_eff_over_r0"] - request.r_eff_over_r0)
            do = abs(coordinates["opening_strength_fraction"] - request.opening_strength_fraction)
            de = abs(coordinates["crack_extension_m"] - request.crack_extension_m)
            if dr <= request.r_tolerance and do <= request.opening_tolerance and de <= request.extension_tolerance_m:
                score = (
                    dr / max(request.r_tolerance, 1.0e-30)
                    + do / max(request.opening_tolerance, 1.0e-30)
                    + de / max(request.extension_tolerance_m, 1.0e-30)
                )
                candidates.append((score, request))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def before_engine_step(self, engine, K: float, T: float) -> None:
        self.attempts += 1
        if not self.pending or self.latest_assembly is None or self.latest_boundary is None:
            return
        drive = DRIVE_OBSERVER.latest_drive
        if not isinstance(drive, dict) or not bool(drive.get("reliable", False)):
            return
        if int(drive.get("mechanics_serial", -1)) != int(DRIVE_OBSERVER.mechanics_serial):
            return
        sigma_local = float(engine.sigma_tip(K))
        r0 = max(float(engine.f.r0), 1.0e-30)
        r_eff = max(float(engine.r_eff()), r0)
        sigma_cap = float(engine.f.sigma_cap)
        if sigma_cap <= 0.0:
            raise RuntimeError("physical state capture requires the local strength sigma_cap")
        coordinates = {
            "r_eff_over_r0": r_eff / r0,
            "opening_strength_fraction": min(max(sigma_local / sigma_cap, 0.0), 1.0),
            "crack_extension_m": max(
                float(getattr(engine, "micro_advance_total_m", 0.0)),
                float(getattr(engine.mpz, "advance_total_m", 0.0)),
                0.0,
            ),
        }
        request = self._matching_request(float(T), coordinates)
        if request is None:
            return
        assembly = self.latest_assembly
        cohesive = assembly.get("cohesive_network")
        if cohesive is not None:
            raise RuntimeError(
                "cohesive-network state is not serializable in v10.2.12; "
                "use the sharp-front PF backend for atlas collection"
            )
        directions = tuple(tuple(row) for row in drive["trace_directions"])
        normals = tuple(tuple(row) for row in drive["trace_normals"])
        tip_xy = tuple(float(value) for value in drive["tip_xy_m"])
        front_direction = tuple(float(value) for value in drive["front_direction"])
        material = assembly["mat"]
        metadata = SnapshotMetadata(
            state_id=request.state_id,
            r_eff_over_r0=float(coordinates["r_eff_over_r0"]),
            opening_strength_fraction=float(coordinates["opening_strength_fraction"]),
            crack_extension_m=float(coordinates["crack_extension_m"]),
            temperature_K=float(T),
            Uy_top_m=float(self.latest_Uy_top),
            Uy_bot_m=float(self.latest_Uy_bot),
            crack_tip_xy_m=tip_xy,
            crack_direction=front_direction,
            interaction_ell_m=float(request.interaction_ell_m),
            exclude_radius_m=max(float(assembly["mesh"].hbar_tip), 0.0),
            active_x_m=tuple(float(value) for value in engine.mpz.x),
            wake_x_m=tuple(float(value) for value in engine.mpz.wake_x),
            channel_directions=directions,
            channel_normals=normals,
            material={
                "E": float(material.E),
                "nu": float(material.nu),
                "b": float(material.b),
                "Tm": float(material.Tm),
            },
            engine_config=_engine_payload(engine),
            fem_tip_geometry_blunted=False,
            r_eff_is_analytical_tip_state=True,
            cohesive_network_present=False,
        )
        root = self.outroot / request.state_id
        payload = save_snapshot(
            root,
            metadata=metadata,
            mesh=assembly["mesh"],
            boundary=self.latest_boundary,
            u=assembly["u"],
            ep_gp=assembly["ep_gp"],
            rho_gp=assembly["rho_gp"],
            d=assembly["d"],
            D=assembly["D"],
        )
        self.captured[request.state_id] = {
            "requested": asdict(request),
            "actual": coordinates,
            "snapshot": str(root),
            "assembly_serial": self.assembly_serial,
            "solve_serial": self.solve_serial,
            "drive_serial": int(drive.get("drive_serial", -1)),
            "payload": payload,
        }

    def wrap_engine_step(self, original: Callable) -> Callable:
        def wrapped(engine, K, T, dt):
            self.before_engine_step(engine, K, T)
            return original(engine, K, T, dt)

        wrapped.__name__ = getattr(original, "__name__", "step")
        return wrapped

    def finalize(self, *, require_complete: bool = True) -> dict[str, Any]:
        payload = {
            "schema": MODEL_ID,
            "requested_states": len(self.requests),
            "captured_states": len(self.captured),
            "pending_state_ids": [request.state_id for request in self.pending],
            "capture_attempts": self.attempts,
            "states": self.captured,
            "same_production_fem_path_observed": True,
            "production_state_mutated": False,
            "shielding_disabled_during_mechanics_collection": True,
            "fem_tip_geometry_blunted": False,
            "r_eff_axis_is_analytical_tip_state": True,
            "parameterization_authorized": False,
        }
        (self.outroot / "capture_complete.json").write_text(json.dumps(payload, indent=2))
        if require_complete and self.pending:
            raise RuntimeError(
                "physical FEM capture did not reach requested states: "
                + ", ".join(request.state_id for request in self.pending)
            )
        return payload


__all__ = [
    "MODEL_ID",
    "CaptureRequest",
    "PhysicalFEMCapture",
    "load_capture_requests",
]
