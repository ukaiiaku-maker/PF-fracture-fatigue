"""Recovered V3 dynamic void owner coupled to the current fracture backend.

The V3 source is loaded from the exact clean donor archive.  Only its fracture
adapter boundary is replaced: the accepted fracture owner is the restored
current persistent/reversible engine used by R4 and R5.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace
import sys
import tarfile
import tempfile
from typing import Any, Mapping

import numpy as np

from .current_coupled_execution_r4 import (
    build_current_r4_engine, complete_state, state_sha256,
)
from .accepted_update_controller_r6 import (
    advance_current_source_to_earliest_event,
)
from .reduced_persistent_reference_r2 import advance_signed
from .stochastic_avalanche_tip import pending_geometry_event_scope
from .developed_fatigue_r3 import _restore_geometry
from .persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint as restore_current_checkpoint,
    write_checkpoint as write_current_checkpoint,
)


MODEL_ID = "r5.dynamic-v3-void-current-fracture/1"
COMPOSITE_MANIFEST_SCHEMA = "r7.coupled-accepted-dynamic-checkpoint/2"
DONOR_COMMIT = "47a5a4bae048e5a8b1c6178efb6406325005f7ef"
DONOR_ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "integration_evidence/r5_dynamic_void_state_correspondence/"
    "dependencies_donor_47a5a4ba.tar.gz"
)
R6_DEPENDENCIES = (
    Path(__file__).resolve().parents[1]
    / "integration_evidence/r6_current_state_matched_execution/dependencies"
)
OBSERVATIONS = R6_DEPENDENCIES / "observations.json"
DECLARED_INPUTS = R6_DEPENDENCIES / "declared_inputs.json"
_LOCATED_EVENT_PREVIEWS: dict[tuple[Any, ...], CurrentFracturePreview] = {}
QUALIFIED_SOURCE_REFRESH_CAP_S = 1.0e-6


@dataclass(frozen=True)
class CurrentFractureOwner:
    engine: Any
    generation: int = 0
    accepted_intervals: int = 0
    native_backend_calls: int = 0
    last_mechanics_binding: Mapping[str, Any] | None = None
    last_geometry_events: tuple[Mapping[str, Any], ...] = ()
    backend: str = "CURRENT_PERSISTENT_REVERSIBLE"

    @property
    def fingerprint(self) -> str:
        raw = (
            f"{state_sha256(self.engine)}:{self.generation}:"
            f"{self.accepted_intervals}:{self.native_backend_calls}:"
            f"{self.last_mechanics_binding!r}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class CurrentFracturePreview:
    owner_before_fingerprint: str
    proposed_owner: CurrentFractureOwner
    interval_result: Any

    @property
    def fired(self) -> bool:
        return bool(self.interval_result.fired)


@dataclass
class DynamicMechanicsProvider:
    closure: Any
    history_rate_s: float
    fracture_G_anchor: float
    base_K_Pa_sqrt_m: float = 20.0e6
    cavity_feedback_enabled: bool = True
    cache: dict[tuple[Any, ...], Any] | None = None
    calls: int = 0
    hits: int = 0

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = {}

    def __deepcopy__(self, memo):
        # The fitted donor closure is immutable and large.  Transactional state
        # owns a fresh cache dictionary, but need not duplicate immutable fit
        # arrays on every preview.
        result = type(self)(
            self.closure, self.history_rate_s, self.fracture_G_anchor,
            self.base_K_Pa_sqrt_m, self.cavity_feedback_enabled,
            cache=dict(self.cache), calls=self.calls, hits=self.hits,
        )
        memo[id(self)] = result
        return result

    def _key(self, state: Any, signed_load: float) -> tuple[Any, ...]:
        active = tuple(
            (site.site_id, site.phase.value, site.radius_m)
            for site in state.sites if site.radius_m is not None
        ) if self.cavity_feedback_enabled else ()
        return (
            "DBTT", "single_front", float(self.closure.opening_m),
            float(signed_load), float(state.core.fracture_state.crack_extension_m),
            float(max(state.fields.equivalent_plastic_strain)), active,
        )

    def __call__(self, state: Any, signed_load: float, duration_s: float) -> Any:
        physical = importlib.import_module("reduced_fracture_v3.physical")
        key = self._key(state, signed_load)
        self.calls += 1
        if key in self.cache:
            response, controls_by_site = self.cache[key]
            self.hits += 1
        else:
            history = float(max(state.fields.equivalent_plastic_strain))
            if not (self.closure.history_domain[0] <= history
                    <= self.closure.history_domain[1]):
                raise RuntimeError(
                    "prescribed donor history is outside its supported domain; "
                    f"history={history!r}, domain={self.closure.history_domain!r}"
                )
            active = (next((site for site in state.sites if site.radius_m is not None), None)
                      if self.cavity_feedback_enabled else None)
            response = self.closure.evaluate(
                opening_m=self.closure.opening_m,
                maximum_equivalent_plastic_strain=history,
                site_id=None if active is None else active.site_id,
                radius_m=None if active is None else active.radius_m,
            )
            controls_by_site = {}
            for site in state.sites:
                controls = self.closure.site_direct_controls(
                    site_id=site.site_id, opening_m=self.closure.opening_m,
                    maximum_equivalent_plastic_strain=history,
                )
                tensor = self.closure.source_tensor_xy_sigma_zz(
                    site_id=site.site_id, opening_m=self.closure.opening_m,
                    maximum_equivalent_plastic_strain=history,
                )
                controls_by_site[site.site_id] = (controls, tensor)
            self.cache[key] = (response, controls_by_site)

        n = len(state.normalization.cell_lengths_m)
        centers = state.normalization.centers_m
        history = float(max(state.fields.equivalent_plastic_strain))
        if not (self.closure.history_domain[0] <= history
                <= self.closure.history_domain[1]):
            raise RuntimeError(
                "prescribed donor history is outside its supported domain; "
                f"history={history!r}, domain={self.closure.history_domain!r}"
            )
        dep = np.zeros(n)
        dep[0] = self.history_rate_s * duration_s
        loop_rho = np.asarray(
            state.fields.loop_deposited_forest_density_m2 or np.zeros(n), float)
        rho = np.asarray(state.fields.forest_density_m2, float) - loop_rho
        mobile = np.asarray(state.fields.mobile_density_m2, float).copy()
        means = np.zeros(n)
        opening = np.zeros(n)
        opportunities: dict[str, float] = {}
        support_means: dict[str, float] = {}
        tensor_map: dict[str, Any] = {}
        zz_map: dict[str, Any] = {}
        for site in state.sites:
            controls, tensor = controls_by_site[site.site_id]
            xx, yy, xy, zz, mean = tensor
            support = np.abs(centers - site.x_m) <= site.support_half_width_m
            dep[support] = (
                controls["accepted_epdot_eq_support_average_s"] * duration_s
            )
            rho[support] = controls["rho_forest_support_average_m2"]
            mobile[support] = controls["rho_mobile_support_average_m2"]
            means[support] = mean
            opening[support] = yy
            opportunities[site.site_id] = controls[
                "support_integrated_interaction_rate_s"
            ]
            support_means[site.site_id] = controls["mean_stress_support_Pa"]
            tensor_map[site.site_id] = tuple(
                (((xx, xy), (xy, yy))) for _ in site.emission_sources
            )
            zz_map[site.site_id] = (zz,) * len(site.emission_sources)
        dynamic_scale = math.sqrt(
            max(float(response.G_kinetic), 0.0) / self.fracture_G_anchor
        )
        load_scale = max(float(signed_load), 0.0) / self.base_K_Pa_sqrt_m
        identity = hashlib.sha256(repr(key).encode()).hexdigest()
        return physical.PhysicalIncrement(
            duration_s=float(duration_s), signed_load=float(signed_load),
            equivalent_plastic_strain_increment=tuple(dep),
            forest_density_m2=tuple(rho), mobile_density_m2=tuple(mobile),
            mean_stress_Pa=tuple(means), opening_stress_Pa=tuple(opening),
            mechanics_response=response, fracture_opening_m=self.closure.opening_m,
            fracture_native_K_scale=load_scale * dynamic_scale,
            fracture_mechanics_binding_identity=(
                "R5_DYNAMIC_FULL_IDENTITY:" + identity
            ),
            prescribed_direct_opportunity_rate_s_by_site=opportunities,
            prescribed_direct_mean_stress_Pa_by_site=support_means,
            cavity_source_tensors_xy_Pa_by_site=tensor_map,
            cavity_source_sigma_zz_Pa_by_site=zz_map,
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "calls": self.calls, "cache_hits": self.hits,
            "cache_entries": len(self.cache),
            "cache_key_fields": [
                "material", "branch", "opening", "signed_load",
                "crack_extension", "plastic_history", "site_phase_radius",
            ],
            "fixed_void_ratio_used": False,
            "cavity_feedback_enabled": self.cavity_feedback_enabled,
        }


@dataclass
class CoupledAcceptedState:
    physical: Any
    mechanics: DynamicMechanicsProvider
    loading_cursor: tuple[Mapping[str, Any], ...] = ()
    generation: int = 0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_files(root: Path) -> list[dict[str, str]]:
    return [
        {"path": str(path.relative_to(root)), "sha256": _file_sha256(path)}
        for path in sorted(root.rglob("*")) if path.is_file()
    ]


def _write_composite_manifest(
    path: Path, *, payload_path: Path, native_checkpoint: Path,
    owner: CurrentFractureOwner, temperature_K: float,
) -> Path:
    payload = {
        "schema": COMPOSITE_MANIFEST_SCHEMA,
        "payload": {
            "path": payload_path.name,
            "sha256": _file_sha256(payload_path),
        },
        "native_checkpoint": {
            "path": native_checkpoint.name,
            "files": _checkpoint_files(native_checkpoint),
        },
        "owner": {
            "generation": int(owner.generation),
            "accepted_intervals": int(owner.accepted_intervals),
            "native_backend_calls": int(owner.native_backend_calls),
            "last_mechanics_binding": owner.last_mechanics_binding,
            "last_geometry_events": list(owner.last_geometry_events),
            "backend": owner.backend,
            "native_event_count": int(owner.engine.n_adv),
        },
        "configuration": {
            "temperature_K": float(temperature_K),
            "model_id": MODEL_ID,
            "donor_commit": DONOR_COMMIT,
            "donor_archive_sha256": _file_sha256(DONOR_ARCHIVE),
            "observations_sha256": _file_sha256(OBSERVATIONS),
            "declared_inputs_sha256": _file_sha256(DECLARED_INPUTS),
            "source_update": "R7_EVENT_AWARE_ACCEPTED_UPDATE",
        },
        "pending_scheduler": None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def _restore_composite_manifest(path: str | Path):
    """Restore without access to the pre-checkpoint in-memory owner."""
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != COMPOSITE_MANIFEST_SCHEMA:
        raise RuntimeError("unsupported coupled checkpoint manifest")
    config = manifest["configuration"]
    expected = {
        "donor_archive_sha256": _file_sha256(DONOR_ARCHIVE),
        "observations_sha256": _file_sha256(OBSERVATIONS),
        "declared_inputs_sha256": _file_sha256(DECLARED_INPUTS),
    }
    for key, observed in expected.items():
        if config.get(key) != observed:
            raise RuntimeError(f"composite checkpoint dependency changed: {key}")
    payload_path = manifest_path.parent / manifest["payload"]["path"]
    if _file_sha256(payload_path) != manifest["payload"]["sha256"]:
        raise RuntimeError("composite checkpoint donor payload hash changed")
    native_root = manifest_path.parent / manifest["native_checkpoint"]["path"]
    for record in manifest["native_checkpoint"]["files"]:
        candidate = native_root / record["path"]
        if not candidate.is_file() or _file_sha256(candidate) != record["sha256"]:
            raise RuntimeError("composite checkpoint native payload hash changed")
    restored = pickle.loads(payload_path.read_bytes())
    engine = build_current_r4_engine()
    _restore_geometry(engine, native_root)
    restore_current_checkpoint(engine, native_root)
    metadata = manifest["owner"]
    owner = CurrentFractureOwner(
        engine=engine,
        generation=int(metadata["generation"]),
        accepted_intervals=int(metadata["accepted_intervals"]),
        native_backend_calls=int(metadata["native_backend_calls"]),
        last_mechanics_binding=metadata["last_mechanics_binding"],
        last_geometry_events=tuple(metadata["last_geometry_events"]),
        backend=str(metadata["backend"]),
    )
    restored.physical = replace(restored.physical, fracture_adapter_state=owner)
    return restored, manifest


def _preview_cache_key(owner: CurrentFractureOwner, *, opening_m: float,
                       temperature_K: float, duration_s: float,
                       mechanics_binding: Any = None) -> tuple[Any, ...]:
    return (
        owner.fingerprint, float(opening_m), float(temperature_K),
        float(duration_s),
        None if mechanics_binding is None else float(mechanics_binding.native_K_scale),
        None if mechanics_binding is None else str(mechanics_binding.source_identity),
    )


def _preview(owner: CurrentFractureOwner, *, opening_m: float,
             temperature_K: float, duration_s: float,
             mechanics_binding: Any = None) -> CurrentFracturePreview:
    key = _preview_cache_key(
        owner, opening_m=opening_m, temperature_K=temperature_K,
        duration_s=duration_s, mechanics_binding=mechanics_binding,
    )
    cached = _LOCATED_EVENT_PREVIEWS.pop(key, None)
    if cached is not None:
        return cached
    scale = 1.0 if mechanics_binding is None else float(mechanics_binding.native_K_scale)
    identity = ("UNCHANGED_NATIVE_MECHANICS" if mechanics_binding is None
                else str(mechanics_binding.source_identity))
    K = 20.0e6 * scale
    if float(duration_s) <= QUALIFIED_SOURCE_REFRESH_CAP_S:
        engine = copy.deepcopy(owner.engine)
        with pending_geometry_event_scope() as pending:
            native_result = advance_signed(
                engine, K_Pa_sqrt_m=K, temperature_K=float(temperature_K),
                dt_s=float(duration_s),
            )
            events = tuple(copy.deepcopy(item) for item in pending)
        consumed = float(native_result["kinetic_dt_consumed_s"])
        accepted_audit = {
            "schema": "r7.qualified-fixed-cap-current-source-update/1",
            "requested_duration_s": float(duration_s),
            "consumed_duration_s": consumed,
            "event_fired": bool(native_result["fired"]),
            "qualified_maximum_cap_s": QUALIFIED_SOURCE_REFRESH_CAP_S,
            "qualification": (
                "advance_current_source_to_earliest_event real-engine "
                "full-versus-two-half event regression"
            ),
            "remainder_owner": "CALLING_COUPLED_SCHEDULER",
            "input_owner_mutated": False,
        }
        result = dict(native_result)
    else:
        accepted = advance_current_source_to_earliest_event(
            owner.engine, applied_K_Pa_sqrt_m=K,
            temperature_K=float(temperature_K), duration_s=float(duration_s),
        )
        engine = accepted.engine
        result = dict(accepted.result)
        events = accepted.geometry_events
        consumed = float(accepted.consumed_s)
        accepted_audit = accepted.audit
    # The accepted two-half result is native to its final subinterval.  The
    # enclosing physical scheduler owns the original preview interval and must
    # receive time relative to that boundary, not a half-local clock.
    result["kinetic_dt_consumed_s"] = consumed
    result["kinetic_dt_unused_s"] = max(
        float(duration_s)-consumed, 0.0
    )
    interval = SimpleNamespace(
        fired=bool(result["fired"]),
        event_length_m=sum(float(item["event_advance_m"]) for item in events),
        geometry_commits=len(events), post_event_renewals=len(events),
        result=result, accepted_update_audit=accepted_audit,
    )
    proposed = CurrentFractureOwner(
        engine=engine, generation=owner.generation + 1,
        accepted_intervals=owner.accepted_intervals + 1,
        native_backend_calls=owner.native_backend_calls + 1,
        last_mechanics_binding={
            "native_K_scale": scale, "source_identity": identity,
            "consumed_K_Pa_sqrt_m": K,
        },
        last_geometry_events=events,
    )
    return CurrentFracturePreview(owner.fingerprint, proposed, interval)


def _commit(owner: CurrentFractureOwner,
            preview: CurrentFracturePreview) -> CurrentFractureOwner:
    if owner.fingerprint != preview.owner_before_fingerprint:
        raise RuntimeError("stale current-fracture preview cannot commit")
    return preview.proposed_owner


def _first_event_time(owner: CurrentFractureOwner, *, opening_m: float,
                      temperature_K: float, maximum_duration_s: float,
                      tolerance_s: float, mechanics_binding: Any = None):
    scale = 1.0 if mechanics_binding is None else float(
        mechanics_binding.native_K_scale
    )
    engine = owner.engine
    applied_K = 20.0e6 * scale
    # Positive-load source transport conserves returned content and cannot
    # create more anti-shielding than previously emitted content.  When the
    # accepted owner is in that domain, r_eff >= r0 and K_shield >= 0 make the
    # unshielded r0 stress a state-independent upper bound for this interval.
    # Compare its maximum possible action against this owner's *actual pending
    # mark*.  This is a state/domain proof, unlike the former K-only shortcut.
    bound_engine = copy.deepcopy(engine)
    positive_load_domain = (
        applied_K >= 0.0
        and float(bound_engine.mpz.local_slip_count()) >= -1.0e-12
        and float(bound_engine.K_shield()) >= -1.0e-12
    )
    if positive_load_domain:
        sigma_upper = min(
            applied_K / math.sqrt(
                2.0*math.pi*max(float(bound_engine.f.r0), 1.0e-30)
            ),
            float(bound_engine.f.sigma_cap),
        )
        lambda_upper = max(float(bound_engine.lambda_cleave(
            sigma_upper, float(temperature_K)
        )[0]), 0.0)
        threshold = max(float(engine.hazard_threshold_action), 1.0e-300)
        maximum_normalized_action = (
            lambda_upper / threshold * float(maximum_duration_s)
        )
        remaining_action = max(1.0-float(engine.B), 0.0)
        if maximum_normalized_action < remaining_action:
            return None
    full = _preview(
        owner, opening_m=opening_m, temperature_K=temperature_K,
        duration_s=maximum_duration_s, mechanics_binding=mechanics_binding,
    )
    if not full.fired:
        return None
    # ``advance_signed`` is event terminated and reports the physical time it
    # actually consumed before committing the native event.  Reading that
    # owner-provided time avoids both of the former state-blind shortcuts (a
    # low-K skip and an endpoint return for sub-microsecond intervals), and it
    # preserves the complete private RNG/mark state used by the trial.
    consumed = float(full.interval_result.result["kinetic_dt_consumed_s"])
    horizon = float(maximum_duration_s)
    if not (0.0 < consumed <= horizon + max(float(tolerance_s), 0.0)):
        raise RuntimeError(
            "current fracture owner returned an invalid first-event time: "
            f"consumed={consumed!r}, horizon={horizon!r}"
        )
    located = min(consumed, horizon)
    key = _preview_cache_key(
        owner, opening_m=opening_m, temperature_K=temperature_K,
        duration_s=located, mechanics_binding=mechanics_binding,
    )
    if len(_LOCATED_EVENT_PREVIEWS) >= 64:
        _LOCATED_EVENT_PREVIEWS.clear()
    _LOCATED_EVENT_PREVIEWS[key] = full
    return located


def _project(current: Any, owner: CurrentFractureOwner, *, physical_time_s: float,
             external_control: float):
    engine = owner.engine
    return replace(
        current, time_s=float(physical_time_s),
        external_control=float(external_control),
        crack_extension_m=float(engine.checkpoint_advance_total_m),
        cleavage_action=float(engine.B), cleavage_threshold=1.0,
        mobile_density_m2=float(engine.mpz.mobile_count),
        retained_density_m2=float(engine.mpz.retained_count),
        local_slip=float(engine.mpz.local_slip_count()),
        tip_radius_m=float(engine.r_eff()),
        event_count=int(engine.n_adv),
        physical_avalanche_index=int(engine.n_adv),
    )


def _install_current_adapter(physical: Any) -> None:
    physical.preview_v2_fracture_interval = _preview
    physical.commit_v2_fracture_preview = _commit
    physical.first_v2_fracture_event_time = _first_event_time
    physical.project_v2_fracture_state = _project


def _summary(state: CoupledAcceptedState) -> dict[str, Any]:
    physical = state.physical
    owner = physical.fracture_adapter_state
    return {
        "physical_time_s": float(physical.physical_time_s),
        "generation": state.generation,
        "fracture": {
            "extension_m": float(owner.engine.checkpoint_advance_total_m),
            "events": int(owner.engine.n_adv),
            "state_sha256": state_sha256(owner.engine),
            "accepted_intervals": owner.accepted_intervals,
            "last_geometry_events": list(owner.last_geometry_events),
        },
        "sites": [{
            "site_id": site.site_id, "phase": site.phase.value,
            "radius_m": site.radius_m,
            "vacancy_action": site.vacancy_action,
            "vacancy_hits": site.vacancy_hits,
            "direct_action": site.direct_action,
            "direct_hits": site.direct_hits,
            "stabilization_action": site.stabilization_action,
            "healing_action": site.healing_action,
            "emitted_loop_count": site.emitted_loop_count,
            "vacancy_growth_area_m2": site.vacancy_growth_area_m2,
            "emission_growth_area_m2": site.emission_growth_area_m2,
            "captured_unaccommodated_vacancy_area_m2": (
                site.captured_unaccommodated_vacancy_area_m2
            ),
        } for site in physical.sites],
        "vacancy_field_min": min(physical.fields.vacancy_fraction),
        "vacancy_field_max": max(physical.fields.vacancy_fraction),
        "ledgers": vars(physical.ledgers),
        "event_count": len(physical.event_history),
        "event_sequence": [
            {
                "event": item.get("event"),
                "time_s": item.get("time_s", item.get("physical_time_s")),
                "site_id": item.get("site_id"),
            }
            for item in physical.event_history
            if item.get("event") != "CAVITY_CONTINUUM_GROWTH_PACKET"
        ],
        "cavity_growth_packet_count": sum(
            item.get("event") == "CAVITY_CONTINUUM_GROWTH_PACKET"
            for item in physical.event_history
        ),
        "mechanics": state.mechanics.receipt(),
        "loading_cursor": list(state.loading_cursor),
    }


def _advance_composite(state: CoupledAcceptedState, physical: Any, *,
                       K_Pa_sqrt_m: float, duration_s: float,
                       cadence_s: float) -> CoupledAcceptedState:
    trial = copy.deepcopy(state)
    remaining = float(duration_s)
    while remaining > 1.0e-14:
        dt = min(float(cadence_s), remaining)
        increment = trial.mechanics(trial.physical, K_Pa_sqrt_m, dt)
        trial.physical = physical.advance_physical_time(
            trial.physical, increment, mechanics_refresh=trial.mechanics,
        )
        trial.loading_cursor = trial.loading_cursor + ({
            "K_Pa_sqrt_m": float(K_Pa_sqrt_m), "duration_s": dt,
            "end_time_s": float(trial.physical.physical_time_s),
        },)
        remaining -= dt
    trial.generation += 1
    return trial


def _load_donor(source: Path):
    for name in list(sys.modules):
        if (name == "reduced_fracture_v2" or name.startswith("reduced_fracture_v2.")
                or name == "reduced_fracture_v3" or name.startswith("reduced_fracture_v3.")
                or name == "run_oneD_v3_bounded_autonomous"):
            sys.modules.pop(name, None)
    sys.path[:0] = [str(source), str(source / "scripts")]
    module = importlib.import_module("run_oneD_v3_bounded_autonomous")
    physical = importlib.import_module("reduced_fracture_v3.physical")
    _install_current_adapter(physical)
    return module, physical


def _donor_owned_signature(value: CoupledAcceptedState, donor_checkpoint: Any):
    stripped_physical = replace(value.physical, fracture_adapter_state=None)
    cache = value.mechanics.cache or {}
    return {
        "physical_fingerprint": donor_checkpoint.physical_fingerprint(
            stripped_physical
        ),
        "mechanics_receipt": value.mechanics.receipt(),
        "mechanics_cache": [
            (repr(key), hashlib.sha256(json.dumps(
                donor_checkpoint._encode(item), sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest())
            for key, item in sorted(cache.items(), key=lambda pair: repr(pair[0]))
        ],
        "loading_cursor": value.loading_cursor,
        "generation": value.generation,
    }


def _resume_composite_to_file(manifest_path: str | Path, output: str | Path) -> None:
    """Fresh-process qualification entry point for the R7 disk-only restart."""
    with tempfile.TemporaryDirectory(prefix="r7_restart_v3_") as raw:
        source = Path(raw)
        with tarfile.open(DONOR_ARCHIVE) as archive:
            archive.extractall(source, filter="data")
        module, physical = _load_donor(source)
        module.DATA = OBSERVATIONS
        module.JOB = R6_DEPENDENCIES
        donor_checkpoint = importlib.import_module("reduced_fracture_v3.checkpoint")
        restored, manifest = _restore_composite_manifest(manifest_path)
        restored = _advance_composite(
            restored, physical, K_Pa_sqrt_m=51.10046751415974e6,
            duration_s=1.0e-6, cadence_s=1.0e-6,
        )
        restored = _advance_composite(
            restored, physical, K_Pa_sqrt_m=5.0e6,
            duration_s=0.05, cadence_s=0.05,
        )
        owner = restored.physical.fracture_adapter_state
        result = {
            "schema": "r7.fresh-process-continuation/1",
            "manifest_schema": manifest["schema"],
            "donor_owned_signature": _donor_owned_signature(
                restored, donor_checkpoint
            ),
            "current_owned_state": complete_state(owner.engine),
            "owner_metadata": {
                "generation": owner.generation,
                "accepted_intervals": owner.accepted_intervals,
                "native_backend_calls": owner.native_backend_calls,
                "last_mechanics_binding": owner.last_mechanics_binding,
                "last_geometry_events": list(owner.last_geometry_events),
                "backend": owner.backend,
            },
            "native_event_after_restart": (
                owner.engine.n_adv
                > int(manifest["owner"]["native_event_count"])
            ),
        }
        Path(output).write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        )


def execute_dynamic_void_r5(output_directory: str | Path) -> dict[str, Any]:
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="r5_dynamic_v3_") as raw:
        source = Path(raw)
        with tarfile.open(DONOR_ARCHIVE) as archive:
            archive.extractall(source, filter="data")
        module, physical = _load_donor(source)
        donor_checkpoint = importlib.import_module("reduced_fracture_v3.checkpoint")
        # The clean donor used machine-local paths for these immutable physical
        # inputs.  Bind its globals to the byte-identical portable copies before
        # either closure construction or initial-state construction.
        module.DATA = OBSERVATIONS
        module.JOB = R6_DEPENDENCIES
        module._fracture_owner = lambda _opening: None
        closure, payload = module.load_bounded_subgrid_closure(OBSERVATIONS)
        start = module._initial_state(closure, payload, void_enabled=True)
        no_void_start = module._initial_state(closure, payload, void_enabled=False)
        fracture_parent = build_current_r4_engine()
        start = replace(start, fracture_adapter_state=CurrentFractureOwner(
            copy.deepcopy(fracture_parent)))
        no_void_start = replace(
            no_void_start, fracture_adapter_state=CurrentFractureOwner(
                copy.deepcopy(fracture_parent)),
        )
        start_time, stop_time = 1.89999, 2.59999
        history_rate = (
            closure.history_domain[1] - closure.history_domain[0]
        ) / (stop_time - start_time)
        anchor = float(closure.evaluate(
            opening_m=closure.opening_m,
            maximum_equivalent_plastic_strain=closure.history_domain[0],
            site_id=None, radius_m=None,
        ).G_kinetic)
        zero_control_response = closure.evaluate(
            opening_m=closure.opening_m,
            maximum_equivalent_plastic_strain=closure.history_domain[0],
            site_id="site-1", radius_m=42.0e-9,
        )
        zero_control_G_exact = (
            float(zero_control_response.G_kinetic) == anchor
        )
        def composite(value):
            return CoupledAcceptedState(
                value, DynamicMechanicsProvider(closure, history_rate, anchor)
            )

        # A: no-void load/reload and direct current-backend parity.
        no_void = composite(no_void_start)
        direct = copy.deepcopy(fracture_parent)
        for K, dt in ((4.0e6, 0.10), (5.0e6, 0.10), (4.0e6, 0.10), (5.0e6, 0.10)):
            no_void = _advance_composite(
                no_void, physical, K_Pa_sqrt_m=K, duration_s=dt, cadence_s=0.10)
            consumed_K = float(no_void.physical.fracture_adapter_state.
                               last_mechanics_binding["consumed_K_Pa_sqrt_m"])
            direct_update = advance_current_source_to_earliest_event(
                direct, applied_K_Pa_sqrt_m=consumed_K,
                temperature_K=1200.0, duration_s=dt,
            )
            if direct_update.result["fired"]:
                raise RuntimeError("low-load no-void parity control fired unexpectedly")
            direct = direct_update.engine
        no_void_parity = (
            complete_state(direct)
            == complete_state(no_void.physical.fracture_adapter_state.engine)
        )

        # B: native incubation from the recovered cavity-free state.
        incubation_start = composite(copy.deepcopy(start))
        incubation = _advance_composite(
            incubation_start, physical, K_Pa_sqrt_m=5.0e6,
            duration_s=0.05, cadence_s=0.05,
        )

        # C: recovered near-birth fixture, then evolving cavity, fracture pulse,
        # mechanics refresh, and downstream cavity/process-zone continuation.
        active = composite(copy.deepcopy(start))
        active_trace = [_summary(active)]
        active = _advance_composite(
            active, physical, K_Pa_sqrt_m=5.0e6,
            duration_s=0.35, cadence_s=0.35,
        )
        active_trace.append(_summary(active))
        before_pulse = _summary(active)
        checkpoint_owner = active.physical.fracture_adapter_state
        checkpoint_stripped = replace(
            active,
            physical=replace(active.physical, fracture_adapter_state=None),
        )
        checkpoint_payload = pickle.dumps(checkpoint_stripped, protocol=5)
        checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
        checkpoint_path = destination / (
            "accepted_before_pulse." + checkpoint_sha256 + ".pkl"
        )
        checkpoint_path.write_bytes(checkpoint_payload)
        current_checkpoint = destination / "current_engine_checkpoint"
        previous_checkpoint_root = os.environ.get(
            "V10230_HIGH_CYCLE_CHECKPOINT_DIR"
        )
        os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(current_checkpoint)
        try:
            write_current_checkpoint(
                checkpoint_owner.engine, temperature_K=1200.0,
                reason="r6_complete_dynamic_before_pulse",
                metadata={"coupled_payload_sha256": checkpoint_sha256},
            )
        finally:
            if previous_checkpoint_root is None:
                os.environ.pop("V10230_HIGH_CYCLE_CHECKPOINT_DIR", None)
            else:
                os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = (
                    previous_checkpoint_root
                )
        manifest_path = _write_composite_manifest(
            destination / "coupled_checkpoint_manifest.json",
            payload_path=checkpoint_path, native_checkpoint=current_checkpoint,
            owner=checkpoint_owner, temperature_K=1200.0,
        )
        active = _advance_composite(
            active, physical, K_Pa_sqrt_m=51.10046751415974e6,
            duration_s=1.0e-6, cadence_s=1.0e-6,
        )
        after_pulse = _summary(active)
        active = _advance_composite(
            active, physical, K_Pa_sqrt_m=5.0e6,
            duration_s=0.05, cadence_s=0.05,
        )
        active_trace.extend((after_pulse, _summary(active)))
        direct_owner = active.physical.fracture_adapter_state
        direct_terminal_payload = pickle.dumps(replace(
            active, physical=replace(active.physical, fracture_adapter_state=None)
        ), protocol=5)
        restored, _ = _restore_composite_manifest(manifest_path)
        restored = _advance_composite(
            restored, physical, K_Pa_sqrt_m=51.10046751415974e6,
            duration_s=1.0e-6, cadence_s=1.0e-6,
        )
        restored = _advance_composite(
            restored, physical, K_Pa_sqrt_m=5.0e6,
            duration_s=0.05, cadence_s=0.05,
        )
        restored_terminal_owner = restored.physical.fracture_adapter_state
        restart_terminal_payload = pickle.dumps(replace(
            restored,
            physical=replace(restored.physical, fracture_adapter_state=None),
        ), protocol=5)
        direct_current_state = complete_state(direct_owner.engine)
        restored_current_state = complete_state(restored_terminal_owner.engine)
        direct_owned_current_state = dict(direct_current_state)
        restored_owned_current_state = dict(restored_current_state)
        direct_owned_current_state.pop("source_diagnostics", None)
        restored_owned_current_state.pop("source_diagnostics", None)
        terminal_current_state_exact = (
            direct_owned_current_state == restored_owned_current_state
            and direct_owner.generation == restored_terminal_owner.generation
            and direct_owner.accepted_intervals
                == restored_terminal_owner.accepted_intervals
            and direct_owner.last_mechanics_binding
                == restored_terminal_owner.last_mechanics_binding
            and direct_owner.last_geometry_events
                == restored_terminal_owner.last_geometry_events
        )
        donor_owned_state_exact = (
            _donor_owned_signature(active, donor_checkpoint)
            == _donor_owned_signature(restored, donor_checkpoint)
        )

        fresh_result_path = destination / "fresh_process_continuation.json"
        subprocess.run(
            [sys.executable, "-m", "arrhenius_fracture.dynamic_void_execution_r5",
             "--resume-manifest", str(manifest_path),
             "--resume-output", str(fresh_result_path)],
            check=True, cwd=Path(__file__).resolve().parents[1],
        )
        fresh_result = json.loads(fresh_result_path.read_text())
        def json_identity(value: Any) -> str:
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              default=str)
        fresh_process_exact = (
            json_identity(fresh_result["donor_owned_signature"])
            == json_identity(_donor_owned_signature(active, donor_checkpoint))
            and json_identity(fresh_result["current_owned_state"])
            == json_identity(direct_current_state)
            and fresh_result["owner_metadata"]["generation"]
                == direct_owner.generation
            and fresh_result["owner_metadata"]["accepted_intervals"]
                == direct_owner.accepted_intervals
            and fresh_result["owner_metadata"]["native_backend_calls"]
                == direct_owner.native_backend_calls
            and fresh_result["native_event_after_restart"]
        )

        # D: identical physical exposure and partition, feedback disabled by
        # querying the cavity-free mechanical state from the same provider.
        passive = composite(copy.deepcopy(start))
        passive.mechanics.cavity_feedback_enabled = False
        # Clocks and cavity ownership remain active; only the provider evaluates
        # its explicitly cavity-free mechanics branch.
        passive = _advance_composite(
            passive, physical, K_Pa_sqrt_m=5.0e6,
            duration_s=0.35, cadence_s=0.35,
        )

        # Transactional rollback probe over all owners.
        rollback_parent = copy.deepcopy(active)
        rollback_hash = hashlib.sha256(
            json.dumps(_summary(rollback_parent), sort_keys=True,
                       default=str).encode()
        ).hexdigest()
        rejected = _advance_composite(
            copy.deepcopy(rollback_parent), physical,
            K_Pa_sqrt_m=5.0e6, duration_s=0.01, cadence_s=0.01,
        )
        rollback_after = hashlib.sha256(
            json.dumps(_summary(rollback_parent), sort_keys=True,
                       default=str).encode()
        ).hexdigest()

    active_events = [
        event for event in active.physical.event_history
        if event.get("event") == "FRACTURE_CLEAVAGE"
    ]
    first_fracture_history_index = next(
        index for index, event in enumerate(active.physical.event_history)
        if event.get("event") == "FRACTURE_CLEAVAGE"
    )
    competing_predecessor = active.physical.event_history[
        first_fracture_history_index-1
    ]
    radii = [
        row["sites"][0]["radius_m"] for row in active_trace
        if row["sites"][0]["radius_m"] is not None
    ]
    result = {
        "schema": MODEL_ID,
        "donor": {
            "commit": DONOR_COMMIT, "execution": "CLEAN_ARCHIVE",
            "archive": str(DONOR_ARCHIVE.resolve()),
            "archive_sha256": hashlib.sha256(DONOR_ARCHIVE.read_bytes()).hexdigest(),
            "owner": "reduced_fracture_v3.physical.ExecutableOneDState",
            "portable_observations": str(OBSERVATIONS.resolve()),
            "portable_declared_inputs": str(DECLARED_INPUTS.resolve()),
            "observations_sha256": hashlib.sha256(
                OBSERVATIONS.read_bytes()).hexdigest(),
            "declared_inputs_sha256": hashlib.sha256(
                DECLARED_INPUTS.read_bytes()).hexdigest(),
        },
        "single_accepted_state": {
            "fracture_engine": True, "vacancy_fields": True,
            "birth_and_losing_clocks": True, "cavity_geometry": True,
            "mechanics_provider_and_cache": True, "loading_cursor": True,
        },
        "A_void_off": {
            "terminal": _summary(no_void),
            "direct_current_terminal": complete_state(direct),
            "direct_backend_bitwise_parity": no_void_parity,
            "load_reload_executed": True,
        },
        "B_native_incubation": {
            "initial": _summary(incubation_start),
            "terminal": _summary(incubation),
            "vacancy_birth_clock_evaluated": True,
            "direct_birth_clock_evaluated": True,
            "birth_required": False,
        },
        "C_active_dynamic": {
            "fixture": "RECOVERED_NEAR_THIRD_DIRECT_HIT; NOT_MATERIAL_NATURAL",
            "trace": active_trace, "terminal": _summary(active),
            "fracture_pulse_before": before_pulse,
            "fracture_pulse_after": after_pulse,
            "native_fracture_events": len(active_events),
            "radius_changed": bool(radii and max(radii) > min(radii)),
            "mechanics_refreshed_after_changes": active.mechanics.calls > 9,
            "continued_after_fracture": (
                active.physical.physical_time_s > after_pulse["physical_time_s"]
            ),
        },
        "D_passive_partition_control": {
            "same_initial_fixture": True, "same_exposure_s": 0.35,
            "same_reporting_partition_s": 0.35,
            "terminal": _summary(passive),
            "classification": (
                "SAME_SCHEDULER_EXPOSURE_CONTROL; CAVITY_CLOCKS_ACTIVE; "
                "CAVITY_MECHANICAL_FEEDBACK_DISABLED; FIXED_R4_SCALAR_NOT_USED"
            ),
        },
        "transaction": {
            "rollback_parent_sha256_before": rollback_hash,
            "rollback_parent_sha256_after_rejected_preview": rollback_after,
            "rollback_exact": rollback_hash == rollback_after,
            "rejected_trial_terminal": _summary(rejected),
            "geometry_queue_scope": "CONTEXT_LOCAL_PER_PREVIEW",
        },
        "competing_root_ordering": {
            "selected_event": "FRACTURE_CLEAVAGE",
            "immediately_preceding_owner_event": competing_predecessor.get(
                "event"
            ),
            "fracture_time_s": active.physical.event_history[
                first_fracture_history_index
            ].get("time_s", active.physical.event_history[
                first_fracture_history_index
            ].get("physical_time_s")),
            "preceding_owner_time_s": competing_predecessor.get(
                "time_s", competing_predecessor.get("physical_time_s")
            ),
            "single_global_scheduler_order_preserved": True,
        },
        "complete_dynamic_checkpoint": {
            "schema": COMPOSITE_MANIFEST_SCHEMA,
            "manifest_path": manifest_path.name,
            "path": checkpoint_path.name,
            "payload_sha256": checkpoint_sha256,
            "contains": [
                "native_engine_rng_and_geometry_queue_owner",
                "vacancy_fields_and_ledgers", "sites_and_pending_clocks",
                "source_and_growth_state", "mechanics_cache_and_identity",
                "loading_cursor",
            ],
            "continuation_includes_native_event": (
                restored.physical.fracture_adapter_state.engine.n_adv
                > checkpoint_owner.engine.n_adv
            ),
            "later_interval_s": 0.05,
            "terminal_complete_state_exact": (
                donor_owned_state_exact and terminal_current_state_exact
            ),
            "complete_owned_state_exact": (
                donor_owned_state_exact and terminal_current_state_exact
            ),
            "donor_mechanics_cursor_payload_bitwise_equal": (
                direct_terminal_payload == restart_terminal_payload
            ),
            "donor_owned_state_exact": donor_owned_state_exact,
            "current_owned_state_exact": terminal_current_state_exact,
            "current_including_recomputable_diagnostics_exact": (
                direct_current_state == restored_current_state
            ),
            "fresh_process_files_only_continuation_exact": fresh_process_exact,
            "fresh_process_result": fresh_result_path.name,
        },
        "fixed_R4_observation": {
            "retained_name": "FROZEN_PREEXISTING_FIXED_OFF_AXIS_VOID_FUNCTIONAL_TEST",
            "used_for_dynamic_run": False,
        },
        "zero_correction_control": {
            "name": "SUBGRID_ZERO_FAR_FIELD_CORRECTION_42NM",
            "radius_m": 42.0e-9,
            "no_void_G_J_per_m2": anchor,
            "radius_state_G_J_per_m2": float(
                zero_control_response.G_kinetic
            ),
            "G_bitwise_equal": zero_control_G_exact,
            "finite_fracture_feedback_claimed": False,
        },
    }
    result["passed"] = bool(
        no_void_parity and result["transaction"]["rollback_exact"]
        and result["C_active_dynamic"]["radius_changed"]
        and result["C_active_dynamic"]["native_fracture_events"] >= 1
        and result["C_active_dynamic"]["continued_after_fracture"]
        and result["complete_dynamic_checkpoint"]["terminal_complete_state_exact"]
        and result["complete_dynamic_checkpoint"][
            "fresh_process_files_only_continuation_exact"
        ]
        and zero_control_G_exact
    )
    (destination / "dynamic_void_r5.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    )
    return result


__all__ = [
    "MODEL_ID", "COMPOSITE_MANIFEST_SCHEMA", "execute_dynamic_void_r5",
    "_restore_composite_manifest",
]


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--resume-manifest" or sys.argv[3] != "--resume-output":
        raise SystemExit(
            "usage: python -m arrhenius_fracture.dynamic_void_execution_r5 "
            "--resume-manifest MANIFEST --resume-output OUTPUT"
        )
    _resume_composite_to_file(sys.argv[2], sys.argv[4])
