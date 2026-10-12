"""Transactional R4 adapter for the current persistent/reversible tip engine.

The R3 bounded OneD V3 calculation remains a historical coupled control.  Its
selected subgrid closure intentionally has zero far-field cavity correction,
so it cannot establish mechanically active coupling to the current backend.
This module owns the current backend directly and uses the already-qualified
V5 fixed-void marginal observation only as a mechanics input.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .matched_persistent_execution_r3 import (
    SPATIAL_COHESIVE_STRENGTH_PA, SPATIAL_EVIDENCE, matched_config,
    spatial_rows, stress_contract,
)
from .persistent_site_high_cycle_state_v10230 import (
    capture_ledgers, capture_stochastic_state, serialize_active_state,
)
from .reduced_persistent_reference_r2 import (
    advance_signed, build_restored_engine, runtime_receipt,
)
from .stochastic_avalanche_tip import (
    clear_pending_geometry_events, pending_geometry_event_scope,
    pop_pending_geometry_event,
)


MODEL_ID = "r4.current-persistent-active-fixed-void-coupling/1"
R4_EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "integration_evidence" / "r4_same_engine_coupled_spatial"
)
DEFAULT_MECHANICS_OBSERVATION = (
    R4_EVIDENCE / "dependencies" / "off_axis_prescribed_void_benchmark_v2.json"
)
R4_KERNEL_FAMILY = R4_EVIDENCE / "dependencies" / "signed_kernel_family.json"
SELECTED_PLANE = "(100)"
SELECTED_VOID_CASE = "above"


def build_current_r4_engine(cfg=None, *, spatial_strength_contract: bool = True):
    """Build the current backend exclusively from bundled R4 dependencies."""
    return build_restored_engine(
        matched_config() if cfg is None else cfg,
        material_path=SPATIAL_EVIDENCE / "selected_material_manifest_v10_2_22.csv",
        selection_path=SPATIAL_EVIDENCE / "v10_2_22_parameter_selection.json",
        kernel_family_path=R4_KERNEL_FAMILY,
        local_strength_Pa=(SPATIAL_COHESIVE_STRENGTH_PA
                           if spatial_strength_contract else None),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def complete_state(engine: Any) -> dict[str, Any]:
    """Lossless numerical state used for parity and stale-trial checks."""
    active = serialize_active_state(engine)
    return {
        "active_vector_hex": [float(value).hex() for value in active.vector],
        "active_fields": [
            [field.owner, field.name, list(field.shape), field.start, field.stop,
             float(field.floor).hex()]
            for field in active.fields
        ],
        "geometry_signature": _json_safe(active.geometry_signature),
        "ledgers": _json_safe(capture_ledgers(engine)),
        "stochastic": _json_safe(capture_stochastic_state(engine)),
        "time_hex": float(engine.t).hex(),
        "tip_radius_hex": float(engine.r_eff()).hex(),
        "source_diagnostics": _json_safe(engine.mpz.diagnostics()),
    }


def state_sha256(engine: Any) -> str:
    encoded = json.dumps(
        complete_state(engine), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _pending_events() -> tuple[dict[str, Any], ...]:
    result = []
    while True:
        item = pop_pending_geometry_event()
        if item is None:
            return tuple(result)
        result.append(copy.deepcopy(item))


@dataclass(frozen=True)
class FractureTrial:
    generation: int
    parent_sha256: str
    engine: Any
    result: dict[str, Any]
    geometry_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AdapterCheckpoint:
    generation: int
    engine: Any


class CurrentPersistentFractureAdapter:
    """Private-clone/preview/commit owner for the actual current engine."""

    def __init__(self, engine: Any | None = None):
        self.engine = build_current_r4_engine() if engine is None else engine
        self.generation = 0

    def preview(self, *, applied_K_Pa_sqrt_m: float, temperature_K: float,
                duration_s: float) -> FractureTrial:
        parent = state_sha256(self.engine)
        trial_engine = copy.deepcopy(self.engine)
        with pending_geometry_event_scope():
            result = advance_signed(
                trial_engine, K_Pa_sqrt_m=float(applied_K_Pa_sqrt_m),
                temperature_K=float(temperature_K), dt_s=float(duration_s),
            )
            events = _pending_events()
        if state_sha256(self.engine) != parent:
            raise RuntimeError("fracture preview mutated the accepted engine")
        return FractureTrial(
            generation=self.generation, parent_sha256=parent,
            engine=trial_engine, result=copy.deepcopy(result),
            geometry_events=events,
        )

    def commit(self, trial: FractureTrial) -> dict[str, Any]:
        if trial.generation != self.generation:
            raise RuntimeError("stale fracture trial generation")
        if trial.parent_sha256 != state_sha256(self.engine):
            raise RuntimeError("stale fracture trial parent state")
        self.engine = trial.engine
        self.generation += 1
        return copy.deepcopy(trial.result)

    def advance(self, **kwargs: float) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        trial = self.preview(**kwargs)
        result = self.commit(trial)
        return result, trial.geometry_events

    def checkpoint(self) -> AdapterCheckpoint:
        return AdapterCheckpoint(self.generation, copy.deepcopy(self.engine))

    @classmethod
    def restore(cls, checkpoint: AdapterCheckpoint) -> "CurrentPersistentFractureAdapter":
        result = cls(copy.deepcopy(checkpoint.engine))
        result.generation = int(checkpoint.generation)
        return result


def _mechanics_binding(path: str | Path = DEFAULT_MECHANICS_OBSERVATION) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text())
    if payload.get("schema") != "unified-2d.off-axis-prescribed-void-benchmark/2":
        raise RuntimeError("unexpected fixed-void mechanics schema")
    cases = {row["case"]: row for row in payload["cases"]}
    no_void = next(
        row for row in cases["no_void"]["canonical_kinetic_directional_curve"]
        if row["plane_variant"] == SELECTED_PLANE
    )
    active = next(
        row for row in cases[SELECTED_VOID_CASE]["canonical_kinetic_directional_curve"]
        if row["plane_variant"] == SELECTED_PLANE
    )
    g0 = float(no_void["finite_increment_G_J_per_m2"])
    gv = float(active["finite_increment_G_J_per_m2"])
    if not (g0 > 0.0 and gv >= 0.0 and no_void["mesh_converged"]
            and active["mesh_converged"]):
        raise RuntimeError("selected fixed-void mechanics observation is not supported")
    # The V5 provider itself reports K_energy_equivalent=sqrt(E' G).  For the
    # same external load, its void/no-void K ratio is therefore sqrt(Gv/G0).
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "schema": payload["schema"],
        "classification": payload["classification"],
        "case": SELECTED_VOID_CASE,
        "plane_variant": SELECTED_PLANE,
        "physical_candidate_increment_m": float(active["physical_increment_m"]),
        "no_void_G_J_per_m2": g0,
        "void_G_J_per_m2": gv,
        "normalized_G_change": (gv - g0) / g0,
        "K_energy_equivalent_ratio": math.sqrt(gv / g0),
        "drive_source": active["drive_source"],
        "mesh_converged": True,
        "mechanics_scope": (
            "PRESCRIBED_PREEXISTING_FIXED_OFF_AXIS_VOID; STATIC_MECHANICS_"
            "FIXTURE; NOT_NATURAL_NUCLEATION_OR_MOVING_VOID_REDUCTION"
        ),
        "outer_boundary_status": payload.get("outer_boundary_status"),
    }


def _direct_advance(engine: Any, K: float, dt: float):
    clear_pending_geometry_events()
    result = advance_signed(
        engine, K_Pa_sqrt_m=K, temperature_K=900.0, dt_s=dt,
    )
    events = _pending_events()
    clear_pending_geometry_events()
    return result, events


def _parity_history() -> dict[str, Any]:
    # Both arms must descend from one constructed owner.  Constructing two
    # engines separately intentionally assigns different RNG substream ids,
    # even when the selected deterministic mark law never draws from them.
    parent = build_current_r4_engine()
    direct = copy.deepcopy(parent)
    adapter = CurrentPersistentFractureAdapter(copy.deepcopy(parent))
    rows = spatial_rows()
    history = []
    for index, row in enumerate(rows, start=1):
        K = float(row["KJ_Pa_sqrtm"])
        dt = float(row["dt_cur_s"])
        direct_result, direct_events = _direct_advance(direct, K, dt)
        adapted_result, adapted_events = adapter.advance(
            applied_K_Pa_sqrt_m=K, temperature_K=900.0, duration_s=dt,
        )
        history.append({
            "step": index, "direct_result": _json_safe(direct_result),
            "adapter_result": _json_safe(adapted_result),
            "direct_events": _json_safe(direct_events),
            "adapter_events": _json_safe(adapted_events),
            "state_sha256": state_sha256(direct),
            "bitwise_complete_state_equal": complete_state(direct) == complete_state(adapter.engine),
        })
    direct_restart = copy.deepcopy(direct)
    adapted_restart = CurrentPersistentFractureAdapter.restore(adapter.checkpoint())
    post = rows[-1]
    direct_result, direct_events = _direct_advance(
        direct_restart, float(post["KJ_Pa_sqrtm"]), float(post["dt_cur_s"])
    )
    adapted_result, adapted_events = adapted_restart.advance(
        applied_K_Pa_sqrt_m=float(post["KJ_Pa_sqrtm"]),
        temperature_K=900.0, duration_s=float(post["dt_cur_s"]),
    )
    return {
        "history": history,
        "event_bearing": any(row["direct_result"]["fired"] for row in history),
        "restart": {
            "direct_result": _json_safe(direct_result),
            "adapter_result": _json_safe(adapted_result),
            "direct_events": _json_safe(direct_events),
            "adapter_events": _json_safe(adapted_events),
            "bitwise_complete_state_equal": (
                complete_state(direct_restart) == complete_state(adapted_restart.engine)
            ),
        },
        "all_bitwise_equal": bool(
            all(row["bitwise_complete_state_equal"] for row in history)
            and complete_state(direct_restart) == complete_state(adapted_restart.engine)
        ),
        "runtime_receipt": runtime_receipt(adapter.engine, stage="r4-adapter-void-off"),
    }


def _run_arm(*, mechanics_factor: float, partition_count: int,
             cavity_owned: bool, engine: Any) -> dict[str, Any]:
    adapter = CurrentPersistentFractureAdapter(engine)
    rows = spatial_rows()
    loading = [
        ("load_10um", rows[0]), ("load_20um", rows[1]),
        ("load_30um", rows[2]), ("unload_20um", rows[1]),
        ("reload_30um", rows[2]), ("downstream_hold_30um", rows[2]),
    ]
    history = []
    event_count = 0
    for stage, row in loading:
        external_K = float(row["KJ_Pa_sqrtm"])
        consumed_K = external_K * float(mechanics_factor)
        total_dt = float(rows[0]["dt_cur_s"])
        for part in range(int(partition_count)):
            before = complete_state(adapter.engine)
            result, events = adapter.advance(
                applied_K_Pa_sqrt_m=consumed_K, temperature_K=900.0,
                duration_s=total_dt / partition_count,
            )
            event_count += len(events)
            history.append({
                "stage": stage, "partition": part + 1,
                "partition_count": partition_count,
                "external_K_Pa_sqrt_m": external_K,
                "consumed_K_Pa_sqrt_m": consumed_K,
                "duration_s": total_dt / partition_count,
                "cavity_owned": bool(cavity_owned),
                "cavity_radius_m": 5.5e-5 if cavity_owned else None,
                "before_time_s": float.fromhex(before["time_hex"]),
                "stress_contract_before": stress_contract(adapter.engine, consumed_K),
                "result": _json_safe(result), "geometry_events": _json_safe(events),
                "after": complete_state(adapter.engine),
            })
    checkpoint = adapter.checkpoint()
    restarted = CurrentPersistentFractureAdapter.restore(checkpoint)
    K = float(rows[2]["KJ_Pa_sqrtm"]) * float(mechanics_factor)
    uninterrupted_result, uninterrupted_events = adapter.advance(
        applied_K_Pa_sqrt_m=K, temperature_K=900.0,
        duration_s=float(rows[0]["dt_cur_s"]),
    )
    restarted_result, restarted_events = restarted.advance(
        applied_K_Pa_sqrt_m=K, temperature_K=900.0,
        duration_s=float(rows[0]["dt_cur_s"]),
    )
    return {
        "mechanics_factor": mechanics_factor, "partition_count": partition_count,
        "history": history, "event_count": event_count,
        "terminal_before_restart_sha256": state_sha256(checkpoint.engine),
        "restart": {
            "uninterrupted_result": _json_safe(uninterrupted_result),
            "restarted_result": _json_safe(restarted_result),
            "uninterrupted_events": _json_safe(uninterrupted_events),
            "restarted_events": _json_safe(restarted_events),
            "bitwise_complete_state_equal": (
                complete_state(adapter.engine) == complete_state(restarted.engine)
            ),
        },
        "runtime_receipt": runtime_receipt(
            adapter.engine,
            stage=("r4-cavity-owned" if cavity_owned else "r4-void-disabled"),
        ),
        "terminal": complete_state(adapter.engine),
    }


def execute_current_coupled_r4(output_directory: str | Path) -> dict[str, Any]:
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    mechanics = _mechanics_binding()
    parity = _parity_history()
    common_parent = build_current_r4_engine()
    no_void = _run_arm(
        mechanics_factor=1.0, partition_count=1, cavity_owned=False,
        engine=copy.deepcopy(common_parent),
    )
    passive = _run_arm(
        mechanics_factor=1.0, partition_count=1, cavity_owned=True,
        engine=copy.deepcopy(common_parent),
    )
    partition = _run_arm(
        mechanics_factor=1.0, partition_count=2, cavity_owned=True,
        engine=copy.deepcopy(common_parent),
    )
    active = _run_arm(
        mechanics_factor=float(mechanics["K_energy_equivalent_ratio"]),
        partition_count=1, cavity_owned=True,
        engine=copy.deepcopy(common_parent),
    )
    first_event = next(
        (index for index, row in enumerate(active["history"])
         if row["geometry_events"]), None
    )
    downstream = (
        first_event is not None and first_event + 1 < len(active["history"])
        and active["history"][first_event + 1]["cavity_owned"]
    )
    result = {
        "schema": MODEL_ID,
        "historical_control": {
            "schema": "r3.clean-v3-bounded-coupled-fracture-void/1",
            "classification": "RETAINED_DIFFERENT_COMPLETE_V2_DONOR",
            "mechanical_feedback": "EXACT_SUBGRID_ZERO_FAR_FIELD_CORRECTION",
        },
        "mechanics_binding": mechanics,
        "direct_adapter_void_off_parity": parity,
        "arms": {
            "no_void": no_void,
            "passive_feedback_disabled": passive,
            "passive_subdivision_diagnostic": partition,
            "active_void": active,
        },
        "qualification": {
            "one_authoritative_current_fracture_state": True,
            "private_trial_contains_rng_and_complete_engine": True,
            "void_toggle_changes_only_consumed_mechanics": True,
            "active_fixture_is_preexisting_cavity_not_nucleation": True,
            "native_fracture_event_observed": active["event_count"] > 0,
            "downstream_interval_with_cavity_owned": downstream,
            "coupled_restart_exact": active["restart"]["bitwise_complete_state_equal"],
            "passive_feedback_disabled_matches_no_void": (
                passive["terminal"] == no_void["terminal"]
            ),
            "passive_subdivision_is_different_event_terminated_experiment": True,
            "moving_void_reduced_map_qualified": False,
        },
    }
    result["passed"] = bool(
        parity["all_bitwise_equal"] and parity["event_bearing"]
        and active["event_count"] > 0 and downstream
        and active["restart"]["bitwise_complete_state_equal"]
        and passive["restart"]["bitwise_complete_state_equal"]
        and result["qualification"]["passive_feedback_disabled_matches_no_void"]
    )
    (destination / "current_coupled_r4.json").write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n"
    )
    return result


__all__ = [
    "AdapterCheckpoint", "CurrentPersistentFractureAdapter", "FractureTrial",
    "MODEL_ID", "build_current_r4_engine", "complete_state",
    "execute_current_coupled_r4", "state_sha256",
]
