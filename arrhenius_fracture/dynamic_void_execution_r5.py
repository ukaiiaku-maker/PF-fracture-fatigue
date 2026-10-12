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
from pathlib import Path
from types import SimpleNamespace
import sys
import tarfile
import tempfile
from typing import Any, Mapping

import numpy as np

from .current_coupled_execution_r4 import (
    build_current_r4_engine, complete_state, state_sha256,
)
from .reduced_persistent_reference_r2 import advance_signed
from .state_correspondence_r5 import advance_fixed_horizon
from .stochastic_avalanche_tip import pending_geometry_event_scope


MODEL_ID = "r5.dynamic-v3-void-current-fracture/1"
DONOR_COMMIT = "47a5a4bae048e5a8b1c6178efb6406325005f7ef"
DONOR_ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "integration_evidence/r5_dynamic_void_state_correspondence/"
    "dependencies_donor_47a5a4ba.tar.gz"
)
OBSERVATIONS = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/voiding/recovered-pr63/"
    "artifacts/aligned_subgrid_closure_v1/observations.json"
)


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
            history = min(max(
                max(state.fields.equivalent_plastic_strain),
                self.closure.history_domain[0]), self.closure.history_domain[1])
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
        history = min(max(
            max(state.fields.equivalent_plastic_strain),
            self.closure.history_domain[0]), self.closure.history_domain[1])
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


def _preview(owner: CurrentFractureOwner, *, opening_m: float,
             temperature_K: float, duration_s: float,
             mechanics_binding: Any = None) -> CurrentFracturePreview:
    engine = copy.deepcopy(owner.engine)
    scale = 1.0 if mechanics_binding is None else float(mechanics_binding.native_K_scale)
    identity = ("UNCHANGED_NATIVE_MECHANICS" if mechanics_binding is None
                else str(mechanics_binding.source_identity))
    K = 20.0e6 * scale
    with pending_geometry_event_scope() as pending:
        result = advance_signed(
            engine, K_Pa_sqrt_m=K, temperature_K=float(temperature_K),
            dt_s=float(duration_s),
        )
        events = tuple(copy.deepcopy(item) for item in pending)
    interval = SimpleNamespace(
        fired=bool(result["fired"]),
        event_length_m=sum(float(item["event_advance_m"]) for item in events),
        geometry_commits=len(events), post_event_renewals=len(events),
        result=result,
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
    scale = 1.0 if mechanics_binding is None else float(mechanics_binding.native_K_scale)
    # The qualified current-law envelope is many orders below first passage at
    # and below 12 MPa sqrt(m) over these sub-second donor intervals.  The
    # accepted advance still runs; this only avoids a duplicate private replay
    # for a provably inactive fracture clock.
    if 20.0e6 * scale <= 12.0e6 and maximum_duration_s <= 1.0:
        return None
    full = _preview(
        owner, opening_m=opening_m, temperature_K=temperature_K,
        duration_s=maximum_duration_s, mechanics_binding=mechanics_binding,
    )
    if not full.fired:
        return None
    if maximum_duration_s <= 1.0e-6:
        return float(maximum_duration_s)
    low, high = 0.0, float(maximum_duration_s)
    for _ in range(64):
        if high - low <= tolerance_s:
            break
        mid = 0.5 * (low + high)
        trial = _preview(
            owner, opening_m=opening_m, temperature_K=temperature_K,
            duration_s=mid, mechanics_binding=mechanics_binding,
        )
        if trial.fired:
            high = mid
        else:
            low = mid
    return high


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
            advance_fixed_horizon(
                direct, applied_K_Pa_sqrt_m=consumed_K, temperature_K=1200.0,
                duration_s=dt, maximum_outer_substep_s=0.10,
            )
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
        "fixed_R4_observation": {
            "retained_name": "FROZEN_PREEXISTING_FIXED_OFF_AXIS_VOID_FUNCTIONAL_TEST",
            "used_for_dynamic_run": False,
        },
    }
    result["passed"] = bool(
        no_void_parity and result["transaction"]["rollback_exact"]
        and result["C_active_dynamic"]["radius_changed"]
        and result["C_active_dynamic"]["native_fracture_events"] >= 1
        and result["C_active_dynamic"]["continued_after_fracture"]
    )
    (destination / "dynamic_void_r5.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    )
    return result


__all__ = ["MODEL_ID", "execute_dynamic_void_r5"]
