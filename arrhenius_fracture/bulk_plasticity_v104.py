"""Production v10.4 bulk-plasticity scope and accepted-load-step adapter.

The crack-tip MPZ is deliberately outside this module.  This adapter owns only
the distributed P0 FEM plastic fields and records their accepted constitutive
provenance in the ordinary topology state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import time
from typing import Any, Mapping

import numpy as np

from .config import SimulationConfig
from .emission_derived_plasticity import bind_dislocation_config_to_material_manifest
from .fem import assemble_mechanics
from .materials import PlasticityModel
from .plasticity import update_plasticity
from .topology_transaction_v11 import equilibrate_fixed_load_with_production_fem
from .unified_fracture_material_v5 import material_manifest, require_bound_identity
from .vacancy_coupling_v1 import equivalent_plastic_strain_rate_s


class PlasticityScope(str, Enum):
    TIP_PROCESS_ONLY = "TIP_PROCESS_ONLY"
    FULL_FIELD_PT_FIXED_RHO = "FULL_FIELD_PT_FIXED_RHO"
    FULL_FIELD_PT_EVOLVING_RHO = "FULL_FIELD_PT_EVOLVING_RHO"
    # Backward-compatible spelling for the completed two-scope integration.
    FULL_FIELD_BULK_PT = "FULL_FIELD_PT_EVOLVING_RHO"
    SOURCES_ONLY_TRANSPORTED = "SOURCES_ONLY_TRANSPORTED"


@dataclass(frozen=True)
class BulkPlasticityManifestV104:
    scope: PlasticityScope
    bulk_kinetics_model: str
    plastic_update_mode: str
    thermo_consistency_mode: str
    bulk_mult_frac: float
    tip_source_rho_per_emit: float
    rho_transport_c: float
    freeze_rho: bool
    exhaustion_enabled: bool
    use_static_recovery: bool
    material_bundle_sha256: str
    plasticity_fingerprint: str
    schema: str = "v10.4.4.bulk-plasticity-manifest/1"

    @property
    def fingerprint(self) -> str:
        payload = {name: (value.value if isinstance(value, Enum) else value)
                   for name, value in self.__dict__.items()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


def production_bulk_configuration(state: Any, scope: PlasticityScope):
    """Map the immutable selected material row to the exact v10.4 settings."""
    scope = PlasticityScope(scope)
    if scope is PlasticityScope.SOURCES_ONLY_TRANSPORTED:
        raise ValueError("SOURCES_ONLY_TRANSPORTED is reserved and disabled")
    identity = dict(require_bound_identity(state))
    from .voiding_production_v5 import _bundle_from_state
    bundle = _bundle_from_state(state)
    selected = material_manifest(bundle)
    cfg = SimulationConfig()
    cfg.material = state.material
    disl = cfg.dislocations
    disl.bulk_kinetics_model = "emission_derived_peierls_taylor_multihit"
    disl.plastic_update_mode = "explicit_rate"
    disl.thermo_consistency_mode = "time_cone"
    disl.bulk_mult_frac = 1.0
    disl.tip_source_rho_per_emit = 0.0
    disl.rho_transport_c = 0.0
    disl.freeze_rho = False
    disl.exhaustion_enabled = False
    # These are selected-row decisions in the v10.4 production manifest, not
    # generic DislocationConfig defaults.
    disl.use_static_recovery = False
    disl.mobile_rho_floor = float(bundle.fracture_row["rho_forest_floor_m2"])
    disl.pt_mobile_density_floor_m2 = disl.mobile_rho_floor
    disl.pt_taylor_corr_rho_c = float(selected.taylor_corr_rho_c_m2)
    disl.pt_taylor_m_scale = float(selected.taylor_corr_scale)
    if scope is PlasticityScope.FULL_FIELD_PT_FIXED_RHO:
        disl.freeze_rho = True
    bind_dislocation_config_to_material_manifest(disl, selected)
    manifest = BulkPlasticityManifestV104(
        scope=scope,
        bulk_kinetics_model=disl.bulk_kinetics_model,
        plastic_update_mode=disl.plastic_update_mode,
        thermo_consistency_mode=disl.thermo_consistency_mode,
        bulk_mult_frac=disl.bulk_mult_frac,
        tip_source_rho_per_emit=disl.tip_source_rho_per_emit,
        rho_transport_c=disl.rho_transport_c,
        freeze_rho=disl.freeze_rho,
        exhaustion_enabled=disl.exhaustion_enabled,
        use_static_recovery=disl.use_static_recovery,
        material_bundle_sha256=identity["material_bundle_sha256"],
        plasticity_fingerprint=identity["plasticity_fingerprint"],
    )
    return cfg, manifest


def _zero_audit(ne: int, rho: np.ndarray) -> dict[str, Any]:
    z = np.zeros(ne)
    return {
        "bulk_pt_active": False, "bulk_kinetics_model": "DISABLED_BY_SCOPE",
        "thermo_mode": "off", "dep_eq_accepted_gp": z,
        "dWp_accepted_gp": z,
        "pt_peierls_rate_gp": z, "pt_taylor_single_hit_rate_gp": z,
        "pt_taylor_completion_rate_gp": z, "pt_series_rate_gp": z,
        "pt_taylor_m_eff_gp": np.ones(ne), "pt_rho_mobile_gp": z,
        "rho_storage_rate_gp_m2_s": z,
        "rho_dynamic_annihilation_rate_gp_m2_s": z,
        "rho_static_recovery_rate_gp_m2_s": z,
        "rho_net_rate_gp_m2_s": z,
        "rho_previous_gp_m2": rho.copy(), "rho_accepted_gp_m2": rho.copy(),
    }


def accepted_bulk_load_step(
    state_n: Any, *, scope: PlasticityScope, load_factor: float,
    duration_s: float, temperature_K: float, maximum_stagger_iterations: int = 40,
    relative_tolerance: float = 1.0e-6, stagger_relaxation: float = 0.25,
    ep_absolute_tolerance: float = 1.0e-12,
    rho_absolute_tolerance_m2: float = 1.0e3,
    equilibrium_provider=None,
):
    """Perform one accepted load increment with fixed-point bulk/mechanics coupling."""
    scope = PlasticityScope(scope)
    if duration_s <= 0.0 or load_factor <= 0.0:
        raise ValueError("load duration and factor must be positive")
    if not 0.0 < stagger_relaxation <= 1.0:
        raise ValueError("stagger_relaxation must satisfy 0 < alpha <= 1")
    wall_start = time.perf_counter()
    equilibrium_wall_s = 0.0
    constitutive_wall_s = 0.0
    observation_assembly_wall_s = 0.0
    cfg, manifest = production_bulk_configuration(state_n, scope)
    equilibrate = (equilibrate_fixed_load_with_production_fem
                   if equilibrium_provider is None else equilibrium_provider)
    tip_before = state_n.tip_process_state
    stage_start = time.perf_counter()
    trial = equilibrate(replace(
        state_n, displacement=np.asarray(state_n.displacement)*float(load_factor),
    ))
    equilibrium_wall_s += time.perf_counter()-stage_start
    ne = trial.mesh.ne
    if scope is PlasticityScope.TIP_PROCESS_ONLY:
        audit = _zero_audit(ne, np.asarray(state_n.rho_gp))
        iterations, residual = 1, 0.0
    else:
        base_ep = np.asarray(state_n.ep_gp).copy()
        base_rho = np.asarray(state_n.rho_gp).copy()
        iter_ep = base_ep.copy()
        iter_rho = base_rho.copy()
        model = PlasticityModel(cfg.plasticity_barrier, state_n.material)
        for iterations in range(1, int(maximum_stagger_iterations)+1):
            stage_start = time.perf_counter()
            _, _, sigma, *_ = assemble_mechanics(
                trial.mesh, trial.displacement, trial.ep_gp, trial.rho_gp,
                trial.damage, trial.elasticity_D, trial.material,
                cohesive_network=trial.cohesive_network,
            )
            observation_assembly_wall_s += time.perf_counter()-stage_start
            stage_start = time.perf_counter()
            ep, rho, _, audit = update_plasticity(
                base_ep.copy(), base_rho.copy(), sigma, trial.material,
                float(temperature_K), float(duration_s), model, cfg.dislocations,
                return_info=True,
            )
            constitutive_wall_s += time.perf_counter()-stage_start
            ep_delta = ep-iter_ep
            rho_delta = rho-iter_rho
            ep_scale = max(float(np.max(np.abs(ep))),
                           float(np.max(np.abs(iter_ep))), 1.0)
            rho_scale = max(float(np.max(np.abs(rho))),
                            float(np.max(np.abs(iter_rho))), 1.0)
            ep_residual = float(np.max(np.abs(ep_delta))) / max(
                ep_absolute_tolerance+relative_tolerance*ep_scale,
                np.finfo(float).tiny,
            )
            rho_residual = float(np.max(np.abs(rho_delta))) / max(
                rho_absolute_tolerance_m2+relative_tolerance*rho_scale,
                np.finfo(float).tiny,
            )
            residual = max(ep_residual, rho_residual)
            if residual <= 1.0:
                stage_start = time.perf_counter()
                trial = equilibrate(
                    replace(trial, ep_gp=ep, rho_gp=rho)
                )
                equilibrium_wall_s += time.perf_counter()-stage_start
                break
            iter_ep = iter_ep+float(stagger_relaxation)*ep_delta
            iter_rho = iter_rho+float(stagger_relaxation)*rho_delta
            stage_start = time.perf_counter()
            trial = equilibrate(
                replace(trial, ep_gp=iter_ep, rho_gp=iter_rho)
            )
            equilibrium_wall_s += time.perf_counter()-stage_start
        else:
            raise RuntimeError("bulk/mechanics stagger failed to converge")
    stage_start = time.perf_counter()
    _, _, sigma_start, *_ = assemble_mechanics(
        state_n.mesh, state_n.displacement, state_n.ep_gp, state_n.rho_gp,
        state_n.damage, state_n.elasticity_D, state_n.material,
        cohesive_network=state_n.cohesive_network,
    )
    _, _, sigma_end, *_ = assemble_mechanics(
        trial.mesh, trial.displacement, trial.ep_gp, trial.rho_gp,
        trial.damage, trial.elasticity_D, trial.material,
        cohesive_network=trial.cohesive_network,
    )
    observation_assembly_wall_s += time.perf_counter()-stage_start
    delta_ep = np.asarray(trial.ep_gp)-np.asarray(state_n.ep_gp)
    endpoint_work_gp = np.sum(0.5*(sigma_start+sigma_end)*delta_ep, axis=0)
    work_scale = float(np.sum(np.abs(endpoint_work_gp)*np.asarray(trial.mesh.area_e)))
    work_total = float(np.sum(endpoint_work_gp*np.asarray(trial.mesh.area_e)))
    negative_tolerance = max(1.0e-14*max(work_scale, 1.0),
                             64.0*np.finfo(float).eps*max(work_scale, 1.0))
    if work_total < -negative_tolerance:
        raise RuntimeError("accepted endpoint-path plastic work is materially negative")
    work_total = max(work_total, 0.0)
    epdot = equivalent_plastic_strain_rate_s(state_n.ep_gp, trial.ep_gp, duration_s)
    owned = {
        "schema": "v10.4.4.accepted-bulk-plasticity-state/1",
        "scope": scope.value,
        "manifest": {**manifest.__dict__, "scope": manifest.scope.value},
        "manifest_fingerprint": manifest.fingerprint,
        "dot_ep_gp_s": epdot,
        "stagger_iterations": iterations,
        "stagger_relative_residual": residual,
        "stagger_converged": True,
        "stagger_relaxation": float(stagger_relaxation),
        "stagger_rtol": float(relative_tolerance),
        "stagger_ep_atol": float(ep_absolute_tolerance),
        "stagger_rho_atol_m2": float(rho_absolute_tolerance_m2),
        "accepted_endpoint_path_plastic_work_gp_J_m3": endpoint_work_gp,
        "accepted_endpoint_path_plastic_work_J_per_m": work_total,
        "accepted_work_source": "equilibrated_endpoint_trapezoid_sigma_colon_delta_ep",
        "equilibrium_provider": ("NATIVE_DEFAULT" if equilibrium_provider is None
                                 else getattr(equilibrium_provider,"provider_identity",
                                              type(equilibrium_provider).__name__)),
        "constitutive": audit,
    }
    junction = dict(trial.junction_process_state)
    junction["accepted_bulk_plasticity_v104"] = owned
    accepted = replace(trial, junction_process_state=junction)
    if accepted.tip_process_state != tip_before:
        raise RuntimeError("bulk load step mutated the legacy tip-process owner")
    returned_audit = dict(owned)
    returned_audit["wall_profile_s"] = {
        "equilibrium": equilibrium_wall_s,
        "constitutive_evolution": constitutive_wall_s,
        "constitutive_observation_assembly": observation_assembly_wall_s,
        "accepted_bulk_load_step_total": time.perf_counter()-wall_start,
    }
    return accepted, returned_audit


def vacancy_source_diagnostics(previous: Any, accepted: Any, duration_s: float) -> Mapping[str, Any]:
    """Resolve baseline and rho enhancement from accepted bulk fields only."""
    transport = previous.vacancy_transport_state_v2
    parent = np.asarray(transport.parent_mechanics_element_ids, int)
    epdot_all = equivalent_plastic_strain_rate_s(previous.ep_gp, accepted.ep_gp, duration_s)
    epdot = epdot_all[parent]
    rho = np.asarray(accepted.rho_gp)[parent]
    c = np.asarray(transport.vacancy_fraction_gp)
    row = transport.parameter_row
    k = float(row.plastic_vacancy_coefficient_per_strain)
    baseline = k*epdot*(1.0-c)
    enhancement = baseline*float(row.dislocation_production_boost)*np.sqrt(
        rho/float(row.rho_reference_m2))
    mobile_enhancement = enhancement*float(row.interaction_mobile_vacancy_fraction)
    direct_equivalent = enhancement*float(row.interaction_direct_cavitation_fraction)
    benign_equivalent = enhancement*float(row.interaction_benign_fraction)
    area = np.asarray(transport.transport_mesh.area_e)
    centers = np.asarray(accepted.mesh.nodes)[np.asarray(accepted.mesh.elems)].mean(axis=1)
    tip = np.asarray(accepted.crack_network.branches[0].tip)
    from .voiding_production_v5 import _bundle_from_state
    from .unified_fracture_material_v5 import front_config
    near_parent = np.linalg.norm(centers-tip, axis=1) <= front_config(_bundle_from_state(accepted)).L_pz
    near = near_parent[parent]
    total = baseline+mobile_enhancement
    return {
        "baseline_rate_gp_s": baseline,
        "dislocation_density_enhancement_rate_gp_s": enhancement,
        "mobile_vacancy_interaction_rate_gp_s": mobile_enhancement,
        "direct_cavitation_interaction_equivalent_rate_gp_s": direct_equivalent,
        "benign_interaction_equivalent_rate_gp_s": benign_equivalent,
        "total_rate_gp_s": total,
        "integrated_source_fraction_area_per_s": float(np.sum(area*total)),
        "near_tip_source_fraction_area_per_s": float(np.sum(area[near]*total[near])),
        "far_field_source_fraction_area_per_s": float(np.sum(area[~near]*total[~near])),
        "source": "accepted_bulk_ep_gp_rho_gp_only",
    }


__all__ = ["PlasticityScope", "BulkPlasticityManifestV104",
           "production_bulk_configuration", "accepted_bulk_load_step",
           "vacancy_source_diagnostics"]
