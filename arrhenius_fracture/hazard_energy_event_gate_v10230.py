"""Hazard-derived energy gate for stochastic fatigue crack-event rewards.

The gate is not an independent fracture criterion. Cleavage first passage remains
the only event trigger. Once the event fires, the stochastic reward is truncated
to the largest crack extension whose fixed-load elastic-energy release can pay the
dissipation derived from the same effective cleavage free-energy barrier,

    Gamma_hazard = gamma_rel * m_hits * DeltaG_cleave_eff / b**2.

No athermal surface energy, toughness floor, Paris law, or fitted fracture-energy
constant is introduced.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable
import weakref

import numpy as np

from .crack_backend import CrackAdvanceResult
from .fem import elastic_energy_densities, solve_dirichlet
from .fixed_deltaK_v1021 import fixed_deltaK_audit_payload
from . import stochastic_avalanche_tip as _avalanche_tip


MODEL_ID = "v10.2.30_hazard_derived_energy_gated_event_reward"


@dataclass
class HazardEnergyGateConfig:
    """Numerical controls only; no material resistance is defined here."""

    enabled: bool = True
    trial_fraction: float = 0.1
    bisection_iterations: int = 24
    relative_energy_tolerance: float = 1.0e-8
    absolute_energy_tolerance_J_per_m: float = 1.0e-12
    direction_match_cosine: float = 0.95

    def validate(self) -> "HazardEnergyGateConfig":
        self.enabled = bool(self.enabled)
        self.trial_fraction = float(self.trial_fraction)
        if not (0.0 < self.trial_fraction <= 1.0):
            raise ValueError("trial_fraction must lie in (0, 1]")
        self.bisection_iterations = max(int(self.bisection_iterations), 1)
        self.relative_energy_tolerance = max(
            float(self.relative_energy_tolerance), 0.0
        )
        self.absolute_energy_tolerance_J_per_m = max(
            float(self.absolute_energy_tolerance_J_per_m), 0.0
        )
        self.direction_match_cosine = min(
            max(float(self.direction_match_cosine), 0.0), 1.0
        )
        return self


def config_from_environment(
    default_trial_fraction: float = 0.1,
) -> HazardEnergyGateConfig:
    raw_enabled = os.environ.get("V10230_ENERGY_GATE_ENABLED", "1")
    return HazardEnergyGateConfig(
        enabled=raw_enabled.strip().lower() not in {"0", "false", "no", "off"},
        trial_fraction=float(
            os.environ.get(
                "V10230_ENERGY_GATE_TRIAL_FRACTION",
                os.environ.get(
                    "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION",
                    str(default_trial_fraction),
                ),
            )
        ),
        bisection_iterations=int(
            os.environ.get("V10230_ENERGY_GATE_BISECTION_ITERATIONS", "24")
        ),
        relative_energy_tolerance=float(
            os.environ.get("V10230_ENERGY_GATE_RELATIVE_TOL", "1e-8")
        ),
        absolute_energy_tolerance_J_per_m=float(
            os.environ.get("V10230_ENERGY_GATE_ABSOLUTE_TOL_J_PER_M", "1e-12")
        ),
        direction_match_cosine=float(
            os.environ.get("V10230_ENERGY_GATE_DIRECTION_MATCH_COSINE", "0.95")
        ),
    ).validate()


class _Observer:
    def __init__(self) -> None:
        self.original_assemble: Callable | None = None
        self.snapshot: dict[str, Any] | None = None
        self.direction: dict[str, Any] | None = None
        self.mechanics_serial = 0
        self.direction_serial = 0
        self.engine_registry: dict[int, weakref.ReferenceType] = {}
        self.event_records: list[dict[str, Any]] = []
        self.latest_probe_K_Pa_sqrt_m: float | None = None
        self.config = config_from_environment()
        self.installed = False


OBSERVER = _Observer()
_LAST_BACKEND = None


def reset_runtime_state(config: HazardEnergyGateConfig | None = None) -> None:
    OBSERVER.snapshot = None
    OBSERVER.direction = None
    OBSERVER.mechanics_serial = 0
    OBSERVER.direction_serial = 0
    OBSERVER.engine_registry = {}
    OBSERVER.event_records = []
    OBSERVER.latest_probe_K_Pa_sqrt_m = None
    if config is not None:
        OBSERVER.config = copy.deepcopy(config).validate()


def register_engine(engine) -> None:
    OBSERVER.engine_registry[int(getattr(engine, "_engine_id", id(engine)))] = weakref.ref(
        engine
    )


def set_latest_probe_K(K_Pa_sqrt_m: float | None) -> None:
    if K_Pa_sqrt_m is None:
        OBSERVER.latest_probe_K_Pa_sqrt_m = None
        return
    value = max(float(K_Pa_sqrt_m), 0.0)
    OBSERVER.latest_probe_K_Pa_sqrt_m = value


def _engine_from_id(engine_id: int):
    ref = OBSERVER.engine_registry.get(int(engine_id))
    return None if ref is None else ref()


def attach_pending_event_info(engine_id: int, result: dict[str, Any]) -> None:
    for descriptor in reversed(_avalanche_tip._PENDING_GEOMETRY_EVENTS):
        if int(descriptor.get("energy_gate_engine_id", -1)) == int(engine_id):
            descriptor["energy_gate_result_ref"] = result
            return


def _stored_energy(
    mesh,
    u: np.ndarray,
    ep_gp: np.ndarray,
    sigma_gp: np.ndarray,
    D: np.ndarray,
) -> float:
    stored, _ = elastic_energy_densities(mesh, u, ep_gp, sigma_gp, D)
    return float(np.sum(stored * mesh.area_e))


def wrap_assemble_mechanics(original: Callable) -> Callable:
    """Capture the latest accepted mechanics state for fixed-load event trials."""

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            mesh, u, ep_gp, rho_gp, damage, D, mat = args[:7]
            cohesive = kwargs.get("cohesive_network")
            if cohesive is None and len(args) >= 9:
                cohesive = args[8]
            sigma_gp = np.asarray(result[2], dtype=float)
            OBSERVER.snapshot = {
                "mesh": mesh,
                "u": np.asarray(u, dtype=float).copy(),
                "ep_gp": np.asarray(ep_gp, dtype=float).copy(),
                "rho_gp": np.asarray(rho_gp, dtype=float).copy(),
                "damage": np.asarray(damage, dtype=float).copy(),
                "D": np.asarray(D, dtype=float).copy(),
                "mat": mat,
                "cohesive_network": cohesive,
                "sigma_gp": sigma_gp.copy(),
                "stored_energy_J_per_m": _stored_energy(
                    mesh,
                    np.asarray(u, dtype=float),
                    np.asarray(ep_gp, dtype=float),
                    sigma_gp,
                    np.asarray(D, dtype=float),
                ),
            }
            OBSERVER.mechanics_serial += 1
        except Exception as exc:
            OBSERVER.snapshot = {
                "capture_error": f"{type(exc).__name__}: {exc}",
                "mechanics_serial": OBSERVER.mechanics_serial,
            }
        return result

    wrapped.__name__ = getattr(original, "__name__", "assemble_mechanics")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    return wrapped


def _capture_direction(candidate: dict[str, Any] | None, source: str, **metadata) -> None:
    if not candidate:
        return
    direction = np.asarray(candidate.get("t"), dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        return
    gamma = float(candidate.get("gamma_rel", candidate.get("gamma", 1.0)))
    OBSERVER.direction_serial += 1
    OBSERVER.direction = {
        "source": source,
        "direction": (direction / norm).copy(),
        "gamma_relative": max(gamma, 1.0e-12),
        "plane_name": str(candidate.get("name", "unknown")),
        "angle_deg": float(candidate.get("angle_deg", math.nan)),
        "direction_serial": OBSERVER.direction_serial,
        **metadata,
    }


def wrap_cleave_direction_competition(original: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        selected, all_candidates = original(*args, **kwargs)
        if selected:
            _capture_direction(
                selected[0],
                "continuous_cubic_competition",
                theta_deg=float(args[1] if len(args) > 1 else kwargs.get("theta_deg", 0.0)),
                gamma_aniso=float(kwargs.get("gamma_aniso", 0.3)),
            )
        return selected, all_candidates

    wrapped.__name__ = getattr(original, "__name__", "cleave_direction_competition")
    return wrapped


def wrap_cleavage_branch_candidates(original: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        selected = original(*args, **kwargs)
        if selected:
            _capture_direction(selected[0], "discrete_cleavage_planes")
        return selected

    wrapped.__name__ = getattr(original, "__name__", "cleavage_branch_candidates")
    return wrapped


def current_direction_gamma(direction: np.ndarray | None = None) -> tuple[float, dict[str, Any]]:
    record = dict(OBSERVER.direction or {})
    gamma = max(float(record.get("gamma_relative", 1.0)), 1.0e-12)
    if direction is not None and "direction" in record:
        requested = np.asarray(direction, dtype=float).reshape(2)
        requested /= max(float(np.linalg.norm(requested)), 1.0e-300)
        captured = np.asarray(record["direction"], dtype=float).reshape(2)
        match = abs(float(requested @ captured))
        record["direction_match_cosine"] = match
        if match < OBSERVER.config.direction_match_cosine:
            record["direction_match_warning"] = True
    return gamma, record


def hazard_resistance_J_per_m2(
    *,
    barrier_J: float,
    cooperative_hits: float,
    burgers_vector_m: float,
    gamma_relative: float,
) -> float:
    """Convert the active hazard barrier into an incremental cleavage work."""

    barrier = max(float(barrier_J), 0.0)
    hits = max(float(cooperative_hits), 1.0)
    b = max(abs(float(burgers_vector_m)), 1.0e-300)
    gamma = max(float(gamma_relative), 1.0e-12)
    return gamma * hits * barrier / (b * b)


def continuum_gate_diagnostics(
    engine,
    K_Pa_sqrt_m: float,
    temperature_K: float,
    *,
    stress_override_Pa: float | None = None,
) -> dict[str, Any]:
    """Check infinitesimal energetic admissibility from the active hazard surface."""

    K = max(float(K_Pa_sqrt_m), 0.0)
    T = float(temperature_K)
    stress = (
        max(float(stress_override_Pa), 0.0)
        if stress_override_Pa is not None
        else max(float(engine.sigma_tip(K)), 0.0)
    )
    _, _, barrier_J = engine.lambda_cleave(stress, T)
    gamma, direction = current_direction_gamma()
    resistance = hazard_resistance_J_per_m2(
        barrier_J=barrier_J,
        cooperative_hits=float(getattr(engine.f, "m_hits", 1.0)),
        burgers_vector_m=float(engine.b),
        gamma_relative=gamma,
    )
    snapshot = OBSERVER.snapshot or {}
    mat = snapshot.get("mat")
    Eprime = float(getattr(mat, "Eprime", 0.0)) if mat is not None else 0.0
    driving_J_per_m2 = K * K / max(Eprime, 1.0e-300)
    tolerance = OBSERVER.config.relative_energy_tolerance * max(
        abs(driving_J_per_m2), abs(resistance), 1.0
    )
    return {
        "energy_gate_continuum_open": bool(
            driving_J_per_m2 + tolerance >= resistance
        ),
        "event_K_Pa_sqrt_m": K,
        "event_temperature_K": T,
        "event_sigma_tip_Pa": stress,
        "hazard_barrier_J": float(barrier_J),
        "hazard_cooperative_hits": float(getattr(engine.f, "m_hits", 1.0)),
        "hazard_burgers_vector_m": float(engine.b),
        "orientation_gamma_relative": gamma,
        "hazard_resistance_J_per_m2": resistance,
        "continuum_driving_J_per_m2": driving_J_per_m2,
        "continuum_gate_tolerance_J_per_m2": tolerance,
        "direction_audit": direction,
    }


def _damage_for_segment(
    mesh,
    damage: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    kill_r: float,
) -> np.ndarray:
    seg = p1 - p0
    length2 = float(seg @ seg)
    if length2 <= 0.0:
        return np.asarray(damage, dtype=float).copy()
    centroids = mesh.nodes[mesh.elems].mean(axis=1)
    t = np.clip(((centroids - p0[None, :]) @ seg) / length2, 0.0, 1.0)
    projection = p0[None, :] + t[:, None] * seg[None, :]
    distance2 = np.sum((centroids - projection) ** 2, axis=1)
    element_radius = np.sqrt(np.maximum(mesh.area_e, 1.0e-30))
    radius = np.maximum(float(kill_r), 0.7 * element_radius)
    selected = distance2 <= radius * radius
    dnew = np.asarray(damage, dtype=float).copy()
    if np.any(selected):
        dnew[mesh.elems[selected]] = 1.0
    return dnew


def _infer_boundary_opening(boundary, u: np.ndarray) -> tuple[float, float]:
    top = np.asarray(boundary.top_nodes, dtype=int)
    bot = np.asarray(boundary.bot_nodes, dtype=int)
    Uy_top = float(np.mean(u[2 * top + 1])) if top.size else 0.0
    Uy_bot = float(np.mean(u[2 * bot + 1])) if bot.size else 0.0
    return Uy_top, Uy_bot


def _equilibrate_fixed_opening(
    *,
    mesh,
    boundary,
    u_initial: np.ndarray,
    ep_gp: np.ndarray,
    rho_gp: np.ndarray,
    damage: np.ndarray,
    D: np.ndarray,
    mat,
    cohesive_network,
) -> tuple[np.ndarray, np.ndarray, float]:
    if OBSERVER.original_assemble is None:
        raise RuntimeError("energy-gate mechanics observer is not installed")
    assemble = OBSERVER.original_assemble
    u0 = np.asarray(u_initial, dtype=float).copy()
    Uy_top, Uy_bot = _infer_boundary_opening(boundary, u0)
    Kmat, Rint, *_ = assemble(
        mesh,
        u0,
        ep_gp,
        rho_gp,
        damage,
        D,
        mat,
        cohesive_network=cohesive_network,
    )
    u1, _ = solve_dirichlet(Kmat, Rint, u0, boundary, Uy_top, Uy_bot)
    _, _, sigma_gp, *_ = assemble(
        mesh,
        u1,
        ep_gp,
        rho_gp,
        damage,
        D,
        mat,
        cohesive_network=cohesive_network,
    )
    energy = _stored_energy(mesh, u1, ep_gp, sigma_gp, D)
    return u1, np.asarray(sigma_gp, dtype=float), energy


def _probe_K_Pa_sqrt_m(event_K: float) -> float:
    if OBSERVER.latest_probe_K_Pa_sqrt_m is not None and OBSERVER.latest_probe_K_Pa_sqrt_m > 0.0:
        return float(OBSERVER.latest_probe_K_Pa_sqrt_m)
    audit = fixed_deltaK_audit_payload()
    latest = audit.get("latest_incoming_Kmax_Pa_sqrt_m")
    if latest is not None and float(latest) > 0.0:
        return float(latest)
    lo = audit.get("incoming_Kmax_min_Pa_sqrt_m")
    hi = audit.get("incoming_Kmax_max_Pa_sqrt_m")
    if lo is not None and hi is not None and abs(float(hi) - float(lo)) <= 1.0e-12 * max(
        abs(float(hi)), 1.0
    ):
        return max(float(hi), 0.0)
    return max(float(event_K), 0.0)


def energy_gate_event_length(
    *,
    kwargs: dict[str, Any],
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    """Return the largest cumulatively admissible event length."""

    cfg = OBSERVER.config
    proposal = max(float(descriptor["event_advance_m"]), 0.0)
    if proposal <= 0.0:
        raise ValueError("stochastic event proposal must be positive")
    snapshot = OBSERVER.snapshot
    if not isinstance(snapshot, dict) or "mesh" not in snapshot:
        raise RuntimeError("no valid mechanics snapshot is available for energy gating")

    mesh = kwargs["mesh"]
    boundary = kwargs["boundary"]
    damage = np.asarray(kwargs["damage"], dtype=float)
    displacement = np.asarray(kwargs["displacement"], dtype=float)
    if snapshot["mesh"] is not mesh:
        raise RuntimeError("energy-gate mechanics snapshot does not match event mesh")
    if snapshot["damage"].shape != damage.shape or not np.array_equal(
        snapshot["damage"], damage
    ):
        raise RuntimeError("energy-gate mechanics snapshot does not match event damage")

    direction = np.asarray(kwargs["direction"], dtype=float).reshape(2)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-300)
    p0 = np.asarray(kwargs["p0"], dtype=float).reshape(2)
    kill_r = float(kwargs["kill_r"])

    gamma, direction_audit = current_direction_gamma(direction)
    barrier_J = max(float(descriptor.get("hazard_barrier_J", 0.0)), 0.0)
    hits = max(float(descriptor.get("hazard_cooperative_hits", 1.0)), 1.0)
    b = max(abs(float(descriptor.get("hazard_burgers_vector_m", 0.0))), 1.0e-300)
    resistance = hazard_resistance_J_per_m2(
        barrier_J=barrier_J,
        cooperative_hits=hits,
        burgers_vector_m=b,
        gamma_relative=gamma,
    )

    event_K = max(float(descriptor.get("event_K_Pa_sqrt_m", 0.0)), 0.0)
    probe_K = _probe_K_Pa_sqrt_m(event_K)
    energy_scale = (
        (event_K / probe_K) ** 2
        if event_K > 0.0 and probe_K > 0.0
        else 1.0
    )

    ep_gp = np.asarray(snapshot["ep_gp"], dtype=float)
    rho_gp = np.asarray(snapshot["rho_gp"], dtype=float)
    D = np.asarray(snapshot["D"], dtype=float)
    mat = snapshot["mat"]
    cohesive = snapshot.get("cohesive_network")
    u_pre, _, energy_pre_probe = _equilibrate_fixed_opening(
        mesh=mesh,
        boundary=boundary,
        u_initial=displacement,
        ep_gp=ep_gp,
        rho_gp=rho_gp,
        damage=damage,
        D=D,
        mat=mat,
        cohesive_network=cohesive,
    )

    ntrial = max(int(math.ceil(1.0 / cfg.trial_fraction)), 1)
    candidates = [proposal * i / ntrial for i in range(1, ntrial + 1)]
    rows: list[dict[str, Any]] = []
    accepted_length = 0.0
    accepted_u = u_pre
    first_failed = None

    def evaluate(length: float) -> tuple[float, np.ndarray, dict[str, Any]]:
        p1 = p0 + float(length) * direction
        dtrial = _damage_for_segment(mesh, damage, p0, p1, kill_r)
        utrial, _, energy_post_probe = _equilibrate_fixed_opening(
            mesh=mesh,
            boundary=boundary,
            u_initial=u_pre,
            ep_gp=ep_gp,
            rho_gp=rho_gp,
            damage=dtrial,
            D=D,
            mat=mat,
            cohesive_network=cohesive,
        )
        released_probe = max(energy_pre_probe - energy_post_probe, 0.0)
        released = released_probe * energy_scale
        dissipated = resistance * float(length)
        tolerance = max(
            cfg.absolute_energy_tolerance_J_per_m,
            cfg.relative_energy_tolerance
            * max(abs(released), abs(dissipated), 1.0e-300),
        )
        residual = released - dissipated
        row = {
            "trial_length_m": float(length),
            "stored_energy_pre_probe_J_per_m": energy_pre_probe,
            "stored_energy_post_probe_J_per_m": energy_post_probe,
            "elastic_release_probe_J_per_m": released_probe,
            "probe_to_event_energy_scale": energy_scale,
            "elastic_release_event_J_per_m": released,
            "hazard_dissipation_J_per_m": dissipated,
            "energy_residual_J_per_m": residual,
            "energy_tolerance_J_per_m": tolerance,
            "admissible": bool(residual + tolerance >= 0.0),
            "newly_killed_nodes": int(np.count_nonzero((dtrial > damage) & (dtrial > 0.0))),
        }
        return residual + tolerance, utrial, row

    for length in candidates:
        residual_with_tol, utrial, row = evaluate(length)
        rows.append(row)
        if residual_with_tol >= 0.0:
            accepted_length = float(length)
            accepted_u = utrial
        else:
            first_failed = float(length)
            break

    if accepted_length > 0.0 and first_failed is not None:
        lo = accepted_length
        hi = first_failed
        ulo = accepted_u
        for _ in range(cfg.bisection_iterations):
            mid = 0.5 * (lo + hi)
            residual_with_tol, umid, row = evaluate(mid)
            row["bisection"] = True
            rows.append(row)
            if residual_with_tol >= 0.0:
                lo = mid
                ulo = umid
            else:
                hi = mid
        accepted_length = lo
        accepted_u = ulo

    all_proposal = accepted_length >= proposal * (1.0 - 1.0e-12)
    return {
        "energy_gate_model_id": MODEL_ID,
        "stochastic_proposed_event_length_m": proposal,
        "energy_admissible_event_length_m": accepted_length,
        "committed_event_length_m": min(proposal, accepted_length),
        "arrest_reason": (
            "stochastic_proposal_reached"
            if all_proposal
            else (
                "hazard_derived_energy_arrest"
                if accepted_length > 0.0
                else "no_energy_admissible_increment"
            )
        ),
        "hazard_barrier_J": barrier_J,
        "hazard_cooperative_hits": hits,
        "hazard_burgers_vector_m": b,
        "orientation_gamma_relative": gamma,
        "hazard_resistance_J_per_m2": resistance,
        "event_K_Pa_sqrt_m": event_K,
        "probe_K_Pa_sqrt_m": probe_K,
        "probe_to_event_energy_scale": energy_scale,
        "mechanics_serial": int(OBSERVER.mechanics_serial),
        "latest_probe_K_Pa_sqrt_m": OBSERVER.latest_probe_K_Pa_sqrt_m,
        "direction_serial": int(OBSERVER.direction_serial),
        "direction_audit": direction_audit,
        "trial_rows": rows,
        "equilibrated_displacement": accepted_u,
        "athermal_Gc_used": False,
        "independent_toughness_floor_used": False,
        "paris_law_used": False,
    }


def finalize_engine_event(
    descriptor: dict[str, Any],
    moved_m: float,
    gate: dict[str, Any],
) -> None:
    engine = _engine_from_id(int(descriptor.get("energy_gate_engine_id", -1)))
    if engine is None:
        raise RuntimeError("energy-gated event engine is no longer available")
    if not hasattr(engine, "commit_energy_gated_event"):
        raise RuntimeError("event engine lacks commit_energy_gated_event")
    engine.commit_energy_gated_event(
        float(moved_m),
        gate,
        descriptor.get("energy_gate_result_ref"),
    )


class EnergyGatedAvalancheBackend:
    """Apply the hazard-derived energy balance before the existing one-shot commit."""

    name = "sharp_wake"
    diagnostic_name = "hazard_energy_gated_stochastic_event"

    def __init__(self, base_backend, config: HazardEnergyGateConfig):
        self.base_backend = base_backend
        self.cohesive_network = getattr(base_backend, "cohesive_network", None)
        self.config = copy.deepcopy(config).validate()
        self.advance_log: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.base_backend, name)

    def advance(self, **kwargs) -> CrackAdvanceResult:
        if not _avalanche_tip._PENDING_GEOMETRY_EVENTS:
            return CrackAdvanceResult(
                kwargs["mesh"],
                kwargs["boundary"],
                kwargs["damage"],
                kwargs["displacement"],
                0.0,
                False,
                reason="missing_stochastic_event_descriptor",
            )
        descriptor = _avalanche_tip._PENDING_GEOMETRY_EVENTS.popleft()
        gate = energy_gate_event_length(kwargs=kwargs, descriptor=descriptor)
        committed = float(gate["committed_event_length_m"])
        if committed <= 0.0:
            row = {key: value for key, value in gate.items() if key != "equilibrated_displacement"}
            row["inserted"] = False
            self.advance_log.append(row)
            OBSERVER.event_records.append(copy.deepcopy(row))
            return CrackAdvanceResult(
                kwargs["mesh"],
                kwargs["boundary"],
                kwargs["damage"],
                kwargs["displacement"],
                0.0,
                False,
                reason="hazard_energy_gate_no_admissible_increment",
            )

        modified = dict(descriptor)
        modified["event_advance_m"] = committed
        modified["energy_gate_stochastic_proposal_m"] = float(
            descriptor["event_advance_m"]
        )
        modified["energy_gate_committed_m"] = committed
        _avalanche_tip._PENDING_GEOMETRY_EVENTS.appendleft(modified)
        result = self.base_backend.advance(**kwargs)
        if not result.inserted or result.moved <= 0.0:
            return result

        moved = float(result.moved)
        gate["committed_event_length_m"] = moved
        finalize_engine_event(descriptor, moved, gate)
        if getattr(self.base_backend, "advance_log", None):
            base_row = self.base_backend.advance_log[-1]
            for key, value in gate.items():
                if key == "equilibrated_displacement":
                    continue
                base_row[key] = value
        row = {
            key: value
            for key, value in gate.items()
            if key != "equilibrated_displacement"
        }
        row.update(
            {
                "inserted": True,
                "event_advance_m": moved,
                "front_id": int(kwargs.get("front_id", 0)),
            }
        )
        self.advance_log.append(row)
        OBSERVER.event_records.append(copy.deepcopy(row))
        return CrackAdvanceResult(
            mesh=result.mesh,
            boundary=result.boundary,
            damage=result.damage,
            displacement=np.asarray(gate["equilibrated_displacement"], dtype=float),
            moved=moved,
            inserted=True,
            angle_error_deg=float(result.angle_error_deg),
            selected_edge_length=float(result.selected_edge_length),
            reason=result.reason,
            elem_parent_map=result.elem_parent_map,
        )

    def write_diagnostics(self, out_dir: str) -> None:
        try:
            self.base_backend.write_diagnostics(out_dir)
        except Exception:
            pass
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "hazard_energy_gated_events_v10_2_30.json").write_text(
            json.dumps(self.advance_log, indent=2, default=str) + "\n"
        )


def build_energy_gated_avalanche_backend(
    args,
    geom,
    original_builder: Callable,
    default_subsegment_fraction: float = 0.1,
    *,
    original_avalanche_builder: Callable,
):
    global _LAST_BACKEND
    config = config_from_environment(default_subsegment_fraction)
    OBSERVER.config = config
    base = original_avalanche_builder(
        args,
        geom,
        original_builder,
        default_subsegment_fraction=default_subsegment_fraction,
    )
    wrapped = EnergyGatedAvalancheBackend(base, config)
    _LAST_BACKEND = wrapped
    return wrapped


def write_last_energy_gate_diagnostics(out_dir: str | Path) -> None:
    if _LAST_BACKEND is None:
        raise RuntimeError("no v10.2.30 energy-gated backend was constructed")
    _LAST_BACKEND.write_diagnostics(str(out_dir))


def audit_payload() -> dict[str, Any]:
    return {
        "schema": MODEL_ID,
        "config": asdict(OBSERVER.config),
        "mechanics_serial": int(OBSERVER.mechanics_serial),
        "direction_serial": int(OBSERVER.direction_serial),
        "events": list(OBSERVER.event_records),
        "fracture_trigger": "cleavage_first_passage_only",
        "event_reward": "min(stochastic_proposal,hazard_derived_energy_arrest)",
        "hazard_resistance_expression": (
            "gamma_relative*m_hits*DeltaG_cleave_effective/b**2"
        ),
        "athermal_Gc_active": False,
        "generic_FractureResistanceConfig_used": False,
        "independent_toughness_floor_used": False,
        "orientation_rescale_source": "existing_production_gamma_relative",
        "fixed_deltaK_probe_energy_scaling": "(K_event/K_probe)**2",
        "rapid_event_plasticity_frozen": True,
        "external_work_during_trial": 0.0,
    }


__all__ = [
    "EnergyGatedAvalancheBackend",
    "HazardEnergyGateConfig",
    "MODEL_ID",
    "OBSERVER",
    "attach_pending_event_info",
    "audit_payload",
    "build_energy_gated_avalanche_backend",
    "config_from_environment",
    "continuum_gate_diagnostics",
    "hazard_resistance_J_per_m2",
    "register_engine",
    "reset_runtime_state",
    "set_latest_probe_K",
    "wrap_assemble_mechanics",
    "wrap_cleave_direction_competition",
    "wrap_cleavage_branch_candidates",
    "write_last_energy_gate_diagnostics",
]
