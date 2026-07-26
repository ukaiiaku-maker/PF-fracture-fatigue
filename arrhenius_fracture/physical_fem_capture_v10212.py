"""Capture accepted 2-D FEM equilibria for active signed-kernel generation.

The capture hooks observe the production solve.  v10.2.27 can optionally clone a
matched trajectory state onto a separate endpoint-resolved measurement mesh.  The
clone is re-equilibrated with plasticity and kinetics frozen; the production
front engine and moving process zone are never advanced or mutated by capture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .anisotropic_emission_v10174 import OBSERVER as DRIVE_OBSERVER
from .frozen_measurement_reconstruction_v10227 import (
    FrozenMeasurementMeshConfig,
    reconstruct_frozen_measurement_state,
)
from .physical_fem_snapshot_v10212 import SnapshotMetadata, save_snapshot

MODEL_ID = "v10.2.27_live_production_state_capture_with_frozen_measurement_clone"


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
            "temperature_K", "r_eff_over_r0", "opening_strength_fraction",
            "crack_extension_m", "r_tolerance", "opening_tolerance",
            "extension_tolerance_m", "interaction_ell_m",
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
        "state_id", "temperature_K", "r_eff_over_r0",
        "opening_strength_fraction", "crack_extension_m", "r_tolerance",
        "opening_tolerance", "extension_tolerance_m", "interaction_ell_m",
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
        "active_kernel_supported": True,
        "wake_kernel_supported": False,
        "production_moving_process_zone_physics_preserved": True,
    }


def _coerce_crack_path(drive: dict[str, Any], tip_xy: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    raw = drive.get("crack_path_xy_m", ())
    try:
        path = tuple(tuple(float(v) for v in row) for row in raw)
    except Exception:
        path = ()
    if path and len(path) >= 2:
        end = np.asarray(path[-1], dtype=float)
        tip = np.asarray(tip_xy, dtype=float)
        if np.linalg.norm(end - tip) <= max(1.0e-9, 1.0e-6 * np.linalg.norm(tip)):
            return path
    return ()


def _digest_array(hasher, name: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.tobytes())


def _engine_kinetic_state_digest(engine) -> str:
    """Hash the production state that capture is forbidden to mutate."""
    hasher = hashlib.sha256()
    for name in (
        "B", "N_em", "W_emit", "t", "a_adv", "n_adv",
        "micro_advance_total_m", "checkpoint_advance_total_m",
        "packet_count_mean_total", "packet_variance_total_m2",
    ):
        hasher.update(name.encode("utf-8"))
        hasher.update(repr(getattr(engine, name, None)).encode("utf-8"))
    mpz = engine.mpz
    for name in (
        "mobile", "retained", "accumulated_slip", "available_sites",
        "wake_mobile", "wake_retained", "wake_slip",
    ):
        if hasattr(mpz, name):
            _digest_array(hasher, f"mpz.{name}", getattr(mpz, name))
    for name in (
        "advance_total_m", "wake_discarded_mobile_total",
        "wake_discarded_retained_total", "wake_discarded_slip_total",
        "escaped_total", "recovered_total",
    ):
        hasher.update(f"mpz.{name}".encode("utf-8"))
        hasher.update(repr(getattr(mpz, name, None)).encode("utf-8"))
    return hasher.hexdigest()


class PhysicalFEMCapture:
    def __init__(
        self,
        requests: list[CaptureRequest],
        outroot: str | Path,
        *,
        measurement_mesh_config: FrozenMeasurementMeshConfig | None = None,
    ):
        self.requests = [request.validate() for request in requests]
        self.outroot = Path(outroot)
        if self.outroot.exists():
            raise FileExistsError(f"refusing to overwrite {self.outroot}")
        self.outroot.mkdir(parents=True)
        self.measurement_mesh_config = (
            measurement_mesh_config.validate()
            if measurement_mesh_config is not None
            else None
        )
        self.captured: dict[str, dict[str, Any]] = {}
        self.latest_assembly: dict[str, Any] | None = None
        self.latest_boundary = None
        self.latest_Uy_top = 0.0
        self.latest_Uy_bot = 0.0
        self.latest_solved_u: np.ndarray | None = None
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
                        "u_input": np.asarray(args[1], dtype=float).copy(),
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
            self.latest_solved_u = np.asarray(result[0], dtype=float).copy()
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

    def before_engine_step(self, engine, K: float, T: float) -> dict[str, Any] | None:
        self.attempts += 1
        if (
            not self.pending
            or self.latest_assembly is None
            or self.latest_boundary is None
            or self.latest_solved_u is None
        ):
            return None
        drive = DRIVE_OBSERVER.latest_drive
        if not isinstance(drive, dict) or not bool(drive.get("reliable", False)):
            return None
        if int(drive.get("mechanics_serial", -1)) != int(DRIVE_OBSERVER.mechanics_serial):
            return None
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
            return None
        assembly = self.latest_assembly
        cohesive = assembly.get("cohesive_network")
        if cohesive is not None:
            raise RuntimeError(
                "cohesive-network state is not serializable; use the sharp-front PF "
                "backend for atlas collection"
            )
        directions = tuple(tuple(row) for row in drive["trace_directions"])
        normals = tuple(tuple(row) for row in drive["trace_normals"])
        tip_xy = tuple(float(value) for value in drive["tip_xy_m"])
        front_direction = tuple(float(value) for value in drive["front_direction"])
        crack_path = _coerce_crack_path(drive, tip_xy)
        material = assembly["mat"]

        kinetic_digest_before = _engine_kinetic_state_digest(engine)
        if self.measurement_mesh_config is None:
            state = {
                "mesh": assembly["mesh"],
                "boundary": self.latest_boundary,
                "u": self.latest_solved_u,
                "ep_gp": assembly["ep_gp"],
                "rho_gp": assembly["rho_gp"],
                "d": assembly["d"],
                "D": assembly["D"],
                "audit": {
                    "trajectory_state_cloned": False,
                    "production_state_mutated": False,
                    "plasticity_frozen": True,
                    "kinetics_not_advanced": True,
                    "moving_process_zone_not_advanced": True,
                    "endpoint_mesh_reconstructed": False,
                    "endpoint_mesh_re_equilibrated": False,
                    "trajectory_mesh_hbar_tip_m": float(assembly["mesh"].hbar_tip),
                    "measurement_mesh_hbar_tip_m": float(assembly["mesh"].hbar_tip),
                },
            }
        else:
            state = reconstruct_frozen_measurement_state(
                source_mesh=assembly["mesh"],
                source_boundary=self.latest_boundary,
                source_u=self.latest_solved_u,
                source_ep_gp=assembly["ep_gp"],
                source_rho_gp=assembly["rho_gp"],
                source_d=assembly["d"],
                D=assembly["D"],
                material=material,
                Uy_top_m=self.latest_Uy_top,
                Uy_bot_m=self.latest_Uy_bot,
                crack_tip_xy_m=tip_xy,
                crack_path_xy_m=crack_path,
                config=self.measurement_mesh_config,
            )
        kinetic_digest_after = _engine_kinetic_state_digest(engine)
        if kinetic_digest_after != kinetic_digest_before:
            raise RuntimeError(
                "capture-only endpoint reconstruction mutated the production moving-tip "
                "or process-zone kinetic state"
            )

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
            exclude_radius_m=max(float(state["mesh"].hbar_tip), 0.0),
            active_x_m=tuple(float(value) for value in engine.mpz.x),
            wake_x_m=tuple(float(value) for value in engine.mpz.wake_x),
            channel_directions=directions,
            channel_normals=normals,
            material={
                "E": float(material.E), "nu": float(material.nu),
                "b": float(material.b), "Tm": float(material.Tm),
            },
            engine_config=_engine_payload(engine),
            fem_tip_geometry_blunted=False,
            r_eff_is_analytical_tip_state=True,
            cohesive_network_present=False,
            crack_path_xy_m=crack_path,
            displacement_state="post_dirichlet_equilibrium",
            active_kernel_supported=True,
            wake_kernel_supported=False,
        )
        root = self.outroot / request.state_id
        payload = save_snapshot(
            root,
            metadata=metadata,
            mesh=state["mesh"],
            boundary=state["boundary"],
            u=state["u"],
            ep_gp=state["ep_gp"],
            rho_gp=state["rho_gp"],
            d=state["d"],
            D=state["D"],
        )
        provenance = {
            **dict(state["audit"]),
            "production_engine_state_sha256_before": kinetic_digest_before,
            "production_engine_state_sha256_after": kinetic_digest_after,
            "production_engine_state_bitwise_unchanged": True,
            "production_fractional_moving_frame_preserved": True,
            "production_mobile_kinetic_solver_preserved": True,
            "measurement_reconstruction_called_engine_step": False,
            "measurement_reconstruction_called_mpz_evolve": False,
            "measurement_reconstruction_called_mpz_advance": False,
        }
        payload.update(provenance)
        (root / "snapshot.json").write_text(json.dumps(payload, indent=2))
        record = {
            "requested": asdict(request),
            "actual": coordinates,
            "snapshot": str(root),
            "assembly_serial": self.assembly_serial,
            "solve_serial": self.solve_serial,
            "drive_serial": int(drive.get("drive_serial", -1)),
            "payload": payload,
            "measurement_provenance": provenance,
            "post_dirichlet_equilibrium_displacement_saved": True,
            "crack_path_serialized": bool(crack_path),
            "active_kernel_supported": True,
            "wake_kernel_supported": False,
        }
        self.captured[request.state_id] = record
        return record

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
            "measurement_mesh_config": (
                None if self.measurement_mesh_config is None
                else self.measurement_mesh_config.as_dict()
            ),
            "production_moving_process_zone_physics_preserved": True,
            "capture_reconstruction_is_measurement_only": True,
            "post_dirichlet_equilibrium_displacement_saved": True,
            "active_kernel_supported": True,
            "wake_kernel_supported": False,
            "production_parameterization_allowed": False,
        }
        (self.outroot / "capture_manifest.json").write_text(json.dumps(payload, indent=2))
        if require_complete and self.pending:
            raise RuntimeError(
                "physical FEM state capture is incomplete; pending="
                + ",".join(request.state_id for request in self.pending)
            )
        return payload


__all__ = [
    "MODEL_ID",
    "CaptureRequest",
    "PhysicalFEMCapture",
    "load_capture_requests",
]
