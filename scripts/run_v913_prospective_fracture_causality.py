#!/usr/bin/env python3
"""Run prospective v9.13 fracture cases with additive full-state snapshots.

The authoritative v9.13 ``run_autonomous_rcurve`` function is called without
changing its arguments or numerical implementation.  This driver temporarily
wraps the state and event-record classes *inside each worker process* so that
the complete state at every accepted first passage can be written alongside
the normal result.  The wrappers do not alter rates, state updates, loading,
thresholds, or event geometry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


DEFAULT_V913_ROOT = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "Arrhenius_FEM_CZM_MPZ_v9_13_dbtt_temperature_shelf"
)
HISTORICAL_TEMPERATURES_K = (
    700.0,
    800.0,
    900.0,
    950.0,
    1000.0,
    1050.0,
    1100.0,
    1200.0,
    1300.0,
    1400.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v913-root", type=Path, default=DEFAULT_V913_ROOT)
    parser.add_argument("--candidate-ids", nargs="*", default=())
    parser.add_argument(
        "--temperatures-K",
        nargs="+",
        type=float,
        default=(300.0, *HISTORICAL_TEMPERATURES_K),
    )
    parser.add_argument("--target-extension-um", type=float, default=50.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-hazard-increment", type=float, default=0.05)
    parser.add_argument("--translation-action-exponent", type=float, default=0.95)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def load_external(v913_root: Path):
    root = str(v913_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from arrhenius_fracture.emergent_gnd_campaign_v913 import (
        candidate_from_registry_row,
    )
    from arrhenius_fracture.emergent_gnd_contract_v913 import (
        effective_candidate_parameters,
    )
    from arrhenius_fracture.emergent_gnd_rcurve_v913 import RCurveLoadingMap
    import arrhenius_fracture.emergent_gnd_rcurve_v913 as rcurve
    from scripts.run_mpz_v9_13_persistent_top5 import load_physics

    return (
        candidate_from_registry_row,
        effective_candidate_parameters,
        RCurveLoadingMap,
        rcurve,
        load_physics,
    )


def _finite_summary(values: Any, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_sum": float("nan"),
        }
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_sum": float(np.sum(finite)),
    }


def _snapshot(state: Any, event: Any, temperature_K: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    K_applied = float(event.K_MPa_sqrt_m)
    shield = float(state.K_shield_MPa_sqrt_m())
    K_effective = max(K_applied - shield, 0.0)
    radius = float(state.tip_radius_m())
    stress_scale = 1.0e6 / np.sqrt(
        2.0 * np.pi * max(radius, float(state.c.b_m), 1.0e-30)
    )
    sigma_applied = K_applied * stress_scale
    sigma_effective = K_effective * stress_scale
    rates = state.local_rates(K_applied, temperature_K)
    geometry = state.source_geometry()
    rho_back, tau_back, sigma_back = state.backstress_state()
    emission_stress = np.asarray(rates["emission_effective_stress_Pa"], dtype=float)
    emission_barrier = np.asarray(
        state.p.emission.barrier_eV(emission_stress, temperature_K), dtype=float
    )
    peierls_surface = state.p.peierls.surface(state.p.emission)
    taylor_surface = state.p.taylor.surface(state.p.emission)
    tau_effective = np.asarray(rates["tau_eff_Pa"], dtype=float)
    peierls_barrier = np.asarray(
        peierls_surface.barrier_eV(
            state.p.peierls.stress_fraction * tau_effective,
            temperature_K,
        ),
        dtype=float,
    )
    peierls_rate = np.asarray(
        state._signed_rate(
            peierls_surface,
            state.p.peierls.stress_fraction * tau_effective,
            temperature_K,
            state.p.peierls.nu0_s,
        ),
        dtype=float,
    )
    forest = np.asarray(state.forest_density_m2(), dtype=float)
    spacing = 1.0 / (2.0 * np.sqrt(np.maximum(forest, 1.0)))
    taylor_phi = spacing / max(float(state.c.b_m), 1.0e-30)
    if np.isfinite(float(state.c.taylor_phi_max)):
        taylor_phi = np.minimum(taylor_phi, float(state.c.taylor_phi_max))
    taylor_barrier = np.asarray(
        taylor_surface.barrier_eV(
            state.p.taylor.stress_fraction * tau_effective * taylor_phi,
            temperature_K,
        ),
        dtype=float,
    )
    cleavage_barrier = float(
        state.p.cleavage.barrier_eV(sigma_effective, temperature_K)
    )
    cleavage_rate = float(state.cleavage_rate_s(K_applied, temperature_K))
    per_site = np.asarray(rates["emission_rate_per_site_s"], dtype=float)
    multiplicity = float(geometry["multiplicity_per_system"])

    scalar: dict[str, Any] = {
        "event_index": int(event.event_index),
        "temperature_K": float(temperature_K),
        "state_snapshot_phase": (
            "threshold_reached_before_event_record_after_hazard_coupled_translation"
        ),
        "threshold_action": float(event.threshold_action),
        "accumulated_cleavage_action": float(event.threshold_action),
        "applied_displacement_m": float(event.applied_displacement_m),
        "elapsed_time_s": float(event.elapsed_time_s),
        "K_applied_MPa_sqrt_m": K_applied,
        "K_shield_MPa_sqrt_m": shield,
        "K_local_effective_MPa_sqrt_m": K_effective,
        "sigma_applied_Pa": float(sigma_applied),
        "sigma_local_effective_Pa": float(sigma_effective),
        "cleavage_barrier_eV": cleavage_barrier,
        "cleavage_rate_s": cleavage_rate,
        "source_multiplicity_per_system": multiplicity,
        "emission_aggregate_rate_s": float(np.sum(per_site) * multiplicity),
        "peierls_aggregate_rate_s": float(np.sum(np.abs(peierls_rate))),
        "taylor_aggregate_rate_s": float(
            np.sum(np.maximum(np.asarray(rates["taylor_completion_s"], dtype=float), 0.0))
        ),
        "mobile_population_sum_m2": float(np.sum(state.mobile_m2)),
        "mobile_population_unsigned_sum_m2": float(np.sum(np.abs(state.mobile_m2))),
        "retained_population_sum_m2": float(np.sum(state.retained_m2)),
        "retained_population_unsigned_sum_m2": float(np.sum(np.abs(state.retained_m2))),
        "accumulated_slip_sum_m2": float(np.sum(state.accumulated_slip_m2)),
        "backstress_mean_Pa": float(np.mean(sigma_back)),
        "backstress_max_Pa": float(np.max(sigma_back)),
        "tip_radius_m": radius,
        "physical_front_width_m": float(geometry["front_width_m"]),
        "state_path_extension_m": float(state.extension_m),
        "cumulative_projected_extension_m": float(
            event.cumulative_projected_extension_m
        ),
        "cumulative_path_extension_m": float(event.cumulative_path_extension_m),
        "cumulative_source_activations": float(
            np.sum(state.cumulative_source_activations)
        ),
        "cumulative_line_content": float(np.sum(state.cumulative_line_content)),
    }
    scalar.update(_finite_summary(emission_barrier, "emission_barrier_eV"))
    scalar.update(_finite_summary(per_site, "emission_rate_per_site_s"))
    scalar.update(_finite_summary(peierls_barrier, "peierls_barrier_eV"))
    scalar.update(_finite_summary(peierls_rate, "peierls_rate_s"))
    scalar.update(_finite_summary(rates["peierls_velocity_m_s"], "peierls_velocity_m_s"))
    scalar.update(_finite_summary(taylor_barrier, "taylor_barrier_eV"))
    scalar.update(_finite_summary(rates["taylor_completion_s"], "taylor_completion_s"))

    arrays = {
        "mobile_m2": np.asarray(state.mobile_m2, dtype=float).copy(),
        "retained_m2": np.asarray(state.retained_m2, dtype=float).copy(),
        "accumulated_slip_m2": np.asarray(state.accumulated_slip_m2, dtype=float).copy(),
        "rho_back_by_system_m2": np.asarray(rho_back, dtype=float).copy(),
        "tau_back_by_system_Pa": np.asarray(tau_back, dtype=float).copy(),
        "sigma_back_by_system_Pa": np.asarray(sigma_back, dtype=float).copy(),
        "emission_barrier_eV": emission_barrier.copy(),
        "emission_rate_per_site_s": per_site.copy(),
        "peierls_barrier_eV": peierls_barrier.copy(),
        "peierls_rate_s": peierls_rate.copy(),
        "peierls_velocity_m_s": np.asarray(
            rates["peierls_velocity_m_s"], dtype=float
        ).copy(),
        "taylor_barrier_eV": taylor_barrier.copy(),
        "taylor_completion_s": np.asarray(
            rates["taylor_completion_s"], dtype=float
        ).copy(),
        "tau_effective_Pa": tau_effective.copy(),
    }
    return scalar, arrays


def _instrumented_run(
    row: dict[str, str],
    temperature_K: float,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    v913_root = Path(settings["v913_root"])
    (
        candidate_from_registry_row,
        effective_candidate_parameters,
        RCurveLoadingMap,
        rcurve,
        load_physics,
    ) = load_external(v913_root)
    effective_candidate_parameters(row)
    candidate = candidate_from_registry_row(row)
    physics, _ = load_physics(Path(settings["physics_json"]))
    loading_map = RCurveLoadingMap.from_dict(
        json.loads(Path(settings["loading_map"]).read_text())
    )

    original_state = rcurve.EmergentGNDState
    original_event = rcurve.RCurveEvent
    context: dict[str, Any] = {"state": None, "scalars": [], "arrays": []}

    class InstrumentedState(original_state):
        def _make_current(self):
            context["state"] = self

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._make_current()

        def source_geometry(self):
            self._make_current()
            return super().source_geometry()

        def advance_time(self, *args, **kwargs):
            value = super().advance_time(*args, **kwargs)
            self._make_current()
            return value

        def advance_coupled_segment(self, *args, **kwargs):
            value = super().advance_coupled_segment(*args, **kwargs)
            self._make_current()
            return value

        def translate_tip(self, *args, **kwargs):
            value = super().translate_tip(*args, **kwargs)
            self._make_current()
            return value

    def InstrumentedEvent(*args, **kwargs):
        event = original_event(*args, **kwargs)
        state = context["state"]
        if state is None:
            raise RuntimeError("instrumented event has no current state")
        scalar, arrays = _snapshot(state, event, float(temperature_K))
        context["scalars"].append(scalar)
        context["arrays"].append(arrays)
        return event

    rcurve.EmergentGNDState = InstrumentedState
    rcurve.RCurveEvent = InstrumentedEvent
    try:
        result = rcurve.run_autonomous_rcurve(
            candidate,
            physics,
            loading_map,
            float(temperature_K),
            target_projected_extension_m=(
                float(settings["target_extension_um"]) * 1.0e-6
            ),
            max_hazard_increment=float(settings["max_hazard_increment"]),
            translation_mode="hazard_coupled",
            translation_action_exponent=float(
                settings["translation_action_exponent"]
            ),
        )
    finally:
        rcurve.EmergentGNDState = original_state
        rcurve.RCurveEvent = original_event

    if len(context["scalars"]) != len(result.events):
        raise RuntimeError("state snapshot/event count mismatch")
    arrays_out: dict[str, np.ndarray] = {
        "event_index": np.asarray(
            [x["event_index"] for x in context["scalars"]], dtype=np.int64
        )
    }
    if context["arrays"]:
        for key in context["arrays"][0]:
            arrays_out[key] = np.stack([x[key] for x in context["arrays"]])
    payload = result.as_dict()
    payload["full_state_snapshot_count"] = len(context["scalars"])
    payload["full_state_snapshot_schema"] = "v913_prospective_event_state_v1"
    return payload, context["scalars"], arrays_out


def _case_worker(task: tuple[dict[str, str], float, dict[str, Any]]):
    row, temperature_K, settings = task
    started = time.perf_counter()
    payload, scalars, arrays = _instrumented_run(row, temperature_K, settings)
    return row, temperature_K, payload, scalars, arrays, time.perf_counter() - started


def _case_stem(candidate_id: str, temperature_K: float) -> str:
    tag = f"{temperature_K:g}".replace(".", "p")
    return f"{candidate_id}__T{tag}K"


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least one")
    if args.target_extension_um <= 0.0:
        raise ValueError("--target-extension-um must be positive")
    temperatures = sorted(set(float(value) for value in args.temperatures_K))
    if not temperatures or any(not np.isfinite(value) for value in temperatures):
        raise ValueError("temperatures must be finite and nonempty")
    args.out.mkdir(parents=True, exist_ok=True)
    case_root = args.out / "cases"
    state_root = args.out / "event_state_npz"
    case_root.mkdir(exist_ok=True)
    state_root.mkdir(exist_ok=True)

    with args.candidate_registry.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("candidate registry is empty")
    id_field = "candidate_id"
    if "prospective_candidate_id" in rows[0]:
        for row in rows:
            row["candidate_id"] = row["prospective_candidate_id"]
    wanted = set(args.candidate_ids)
    if wanted:
        rows = [row for row in rows if row[id_field] in wanted]
        missing = wanted - {row[id_field] for row in rows}
        if missing:
            raise RuntimeError(f"candidate IDs not found: {sorted(missing)}")

    v913_root = args.v913_root.resolve()
    physics_json = v913_root / "mpz_v9_13_v10222_transfer_common_physics.json"
    loading_map = (
        v913_root
        / "runs/v9_13_long_map_exponential_110um_v2/"
        "v10_2_22_long_rcurve_loading_map_exponential_110um.json"
    )
    policy_json = v913_root / "mpz_v9_12_targeted_local_search_policy.json"
    for required in (physics_json, loading_map, policy_json, args.candidate_registry):
        if not required.is_file():
            raise FileNotFoundError(required)
    settings: dict[str, Any] = {
        "v913_root": str(v913_root),
        "physics_json": str(physics_json),
        "loading_map": str(loading_map),
        "target_extension_um": float(args.target_extension_um),
        "max_hazard_increment": float(args.max_hazard_increment),
        "translation_action_exponent": float(args.translation_action_exponent),
    }
    contract = {
        "schema": "v913_prospective_fracture_causality_run_contract_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_registry": str(args.candidate_registry.resolve()),
        "candidate_registry_sha256": sha256_path(args.candidate_registry),
        "candidate_ids": [row[id_field] for row in rows],
        "temperatures_K": temperatures,
        "historical_temperature_grid_K": list(HISTORICAL_TEMPERATURES_K),
        "qualification_temperature_K": 300.0,
        "settings": settings,
        "physics_json_sha256": sha256_path(physics_json),
        "loading_map_sha256": sha256_path(loading_map),
        "policy_json_sha256": sha256_path(policy_json),
        "v913_git_head": git_head(v913_root),
        "instrumentation_only": True,
        "physics_changed": False,
        "numerical_settings_changed": False,
    }
    contract_hash = stable_hash({k: v for k, v in contract.items() if k != "created_at_utc"})
    contract["run_contract_sha256"] = contract_hash
    contract_path = args.out / "run_contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text())
        if previous.get("run_contract_sha256") != contract_hash:
            raise RuntimeError("existing output has a different run contract")
    else:
        atomic_json(contract_path, contract)

    tasks = []
    payloads: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for row in rows:
        for temperature in temperatures:
            stem = _case_stem(row[id_field], temperature)
            json_path = case_root / f"{stem}.json"
            npz_path = state_root / f"{stem}.npz"
            if args.resume and json_path.exists() and npz_path.exists():
                payload = json.loads(json_path.read_text())
                if payload.get("run_contract_sha256") != contract_hash:
                    raise RuntimeError(f"resume contract mismatch: {json_path}")
                payloads.append(payload)
                state_rows.extend(payload.get("event_state_scalars", []))
            else:
                tasks.append((row, temperature, settings))

    def accept(result) -> None:
        row, temperature, payload, scalars, arrays, wall_s = result
        candidate_id = row[id_field]
        stem = _case_stem(candidate_id, temperature)
        npz_path = state_root / f"{stem}.npz"
        temporary_npz = npz_path.with_suffix(".npz.tmp")
        with temporary_npz.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary_npz.replace(npz_path)
        for scalar in scalars:
            scalar.update(
                {
                    "candidate_id": candidate_id,
                    "state_npz": str(npz_path.resolve()),
                }
            )
        payload.update(
            {
                "candidate_id": candidate_id,
                "run_contract_sha256": contract_hash,
                "case_wall_s": float(wall_s),
                "temperature_grid_role": (
                    "K300_qualification" if temperature == 300.0 else "historical_grid"
                ),
                "event_state_scalars": scalars,
                "event_state_npz": str(npz_path.resolve()),
                "event_state_npz_sha256": sha256_path(npz_path),
            }
        )
        atomic_json(case_root / f"{stem}.json", payload)
        payloads.append(payload)
        state_rows.extend(scalars)
        print(
            "V913_PROSPECTIVE_CASE_COMPLETE "
            f"candidate={candidate_id} T={temperature:g} "
            f"status={payload['status']} events={len(payload.get('events', []))} "
            f"K50={payload.get('K_50um_MPa_sqrt_m')} wall_s={wall_s:.3f}",
            flush=True,
        )

    if tasks and args.jobs == 1:
        for task in tasks:
            accept(_case_worker(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_case_worker, task) for task in tasks]
            for future in as_completed(futures):
                accept(future.result())

    case_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for payload in payloads:
        case_rows.append(
            {
                key: value
                for key, value in payload.items()
                if key not in {"events", "event_state_scalars"}
                and not isinstance(value, (dict, list))
            }
        )
        for event in payload.get("events", []):
            event_rows.append(
                {
                    "candidate_id": payload["candidate_id"],
                    "temperature_K": payload["temperature_K"],
                    "status": payload["status"],
                    **event,
                }
            )
    key = lambda row: (str(row["candidate_id"]), float(row["temperature_K"]))
    atomic_csv(args.out / "prospective_fracture_case_results.csv", sorted(case_rows, key=key))
    atomic_csv(
        args.out / "prospective_fracture_events.csv",
        sorted(event_rows, key=lambda row: (*key(row), int(row["event_index"]))),
    )
    atomic_csv(
        args.out / "prospective_fracture_state_at_first_passage.csv",
        sorted(state_rows, key=lambda row: (*key(row), int(row["event_index"]))),
    )
    expected = len(rows) * len(temperatures)
    complete = len(payloads) == expected and all(
        payload.get("status") == "complete" for payload in payloads
    )
    manifest = {
        "schema": "v913_prospective_fracture_causality_run_manifest_v1",
        "run_contract_sha256": contract_hash,
        "candidate_count": len(rows),
        "temperature_count": len(temperatures),
        "expected_case_count": expected,
        "completed_case_count": len(payloads),
        "complete_grid": complete,
        "full_state_snapshot_count": len(state_rows),
        "instrumentation_only": True,
        "physics_changed": False,
        "numerical_settings_changed": False,
    }
    atomic_json(args.out / "run_manifest.json", manifest)
    print(
        "V913_PROSPECTIVE_FRACTURE_COMPLETE "
        f"complete_grid={str(complete).lower()} cases={len(payloads)}/{expected} "
        f"state_snapshots={len(state_rows)} out={args.out}",
        flush=True,
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
