#!/usr/bin/env python3
"""Generate the fixed aligned DBTT 18-state mechanics oracle."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cavity_free_boundary_verification import verify_cavity_free_boundary
from arrhenius_fracture.finalization_v3_schema import canonical_hash
from arrhenius_fracture.hybrid_directional_drive_v5 import _tip_rows_by_owner, _topology_request
from arrhenius_fracture.live_topology_kernel_v11 import evaluate_exact_topology
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, require_equilibrium_observables,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle, require_bound_identity
from arrhenius_fracture.voiding_production_v5 import (
    build_production_void_state, cavity_source_resolution_metrics,
    deterministic_trajectory, downstream_front_transaction,
    equilibrate_fixed_load_with_production_fem, fresh_sharp_front_engine,
    ligament_transaction, observables, refine_downstream_source,
    sharp_front_load_provider,
)
from arrhenius_fracture.voiding_v5 import VoidPhase


SCHEMA = "unified-2d.one-void-aligned-oracle/1"
GEOMETRIES = {
    "LOW": 52.5e-6,
    "CENTRAL": 55.0e-6,
    # The frozen finite defect-inventory ledger caps this bundle at
    # 56.41895835477563 um.  HIGH is an interior oracle coordinate, not a
    # request to modify or bypass that physical inventory limit.
    "HIGH": 56.0e-6,
}
OPENINGS_M = (0.8e-6, 1.0e-6)
CRACK_PATH_M = (
    (0.0, 8.572527594031472e-20),
    (0.0005725993004046688, 1.5602085896855586e-20),
)
REGIMES = ("PRECONNECTION", "CONNECTED_CAVITY", "DOWNSTREAM_CHILD")
IMPLEMENTATION_FILES = (
    "arrhenius_fracture/voiding_production_v5.py",
    "arrhenius_fracture/cavity_free_boundary_verification.py",
    "arrhenius_fracture/cavity_source_recovery_v3.py",
    "arrhenius_fracture/hybrid_directional_drive_v5.py",
    "arrhenius_fracture/topology_transaction_v11.py",
    "arrhenius_fracture/unified_fracture_material_v5.py",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("oracle payload contains a nonfinite float")
    return value


def _implementation_identity():
    digest = hashlib.sha256()
    files = {}
    for relative in IMPLEMENTATION_FILES:
        payload = (ROOT / relative).read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        files[relative] = checksum
        digest.update(relative.encode()); digest.update(b"\0"); digest.update(payload)
    return {
        "repository": "ukaiiaku-maker/PF-fracture-fatigue",
        "branch": subprocess.check_output(("git", "branch", "--show-current"), cwd=ROOT, text=True).strip(),
        "git_head": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "frozen_source_sha256": digest.hexdigest(),
        "files": files,
    }


def _set_opening(state, opening_m):
    displacement = np.asarray(state.displacement).copy()
    displacement[2*np.asarray(state.boundary.top_nodes, dtype=int)+1] = opening_m/2
    displacement[2*np.asarray(state.boundary.bot_nodes, dtype=int)+1] = -opening_m/2
    return equilibrate_fixed_load_with_production_fem(replace(state, displacement=displacement))


def _topology_fingerprint(state):
    return hashlib.sha256(pickle.dumps((
        state.crack_network, state.v12_support_state,
        tuple((c.cavity_id, c.center_m, c.radius_m, c.phase.value,
               c.connection_entry_m, c.connection_exit_m)
              for c in (() if state.void_state is None else state.void_state.cavities)),
    ), protocol=5)).hexdigest()


def _active_path(state, regime):
    root = state.crack_network.branches[0]
    if regime != "DOWNSTREAM_CHILD":
        return tuple(root.path)
    child = state.crack_network.branch(state.crack_network.active_tip_ids[0])
    return tuple(root.path) + tuple(child.path)


def _drive(state, regime):
    if regime == "CONNECTED_CAVITY":
        return {
            "G_local_J_per_m2": None, "G_marginal_J_per_m2": None,
            "G_kinetic_used_J_per_m2": None, "drive_source": "CAVITY_SOURCE_TENSOR_NO_SHARP_TIP_G",
            "K_energy_equivalent_Pa_sqrt_m": None,
            "status": "NOT_APPLICABLE_NO_ACTIVE_SHARP_TIP",
        }
    branch_id = state.crack_network.active_tip_ids[0]
    rows = sharp_front_load_provider(state, branch_id=branch_id)["directional"]
    if len(rows) != 1:
        raise RuntimeError("oracle requires exactly one aligned directional drive")
    row = rows[0]
    return {key: _jsonable(row.get(key)) for key in (
        "G_local_J_per_m2", "G_marginal_J_per_m2", "G_kinetic_used_J_per_m2",
        "drive_source", "K_energy_equivalent_Pa_sqrt_m", "directional_drive_status",
    )} | {"status": "MEASURED_HYBRID_DIRECTIONAL_PROVIDER"}


def _companion(bundle, state, regime, opening_m):
    path = _active_path(state, regime)
    companion, _ = build_production_void_state(
        bundle=bundle, enabled=False, crack_path_m=path,
        boundary_segments=64, radial_layers=24, source_conforming_geometry=True,
    )
    companion = _set_opening(companion, opening_m)
    branch_id = companion.crack_network.active_tip_ids[0]
    engine = fresh_sharp_front_engine(companion.material, bundle)
    request = _topology_request(
        companion, {branch_id: companion.competition.candidates},
        contour_radius_m=max(float(engine.f.L_pz), 1.0e-6),
        provider_contract_contour_radius_m=max(float(engine.f.L_pz), 1.0e-6),
    )
    provider = evaluate_exact_topology(request)
    rows = _tip_rows_by_owner(companion, provider)[branch_id]["directional"]
    if len(rows) != 1:
        raise RuntimeError("matched no-void companion has nonunique aligned G0")
    row = rows[0]
    G0 = float(row["J_provider_contract_positive_J_per_m2"])
    if not math.isfinite(G0) or G0 < 0.0:
        raise RuntimeError("matched no-void provider did not produce finite nonnegative G0")
    return companion, {
        "accepted_state_fingerprint": complete_accepted_state_fingerprint(companion),
        "topology_fingerprint": _topology_fingerprint(companion),
        "G0_J_per_m2": G0,
        "K0_energy_equivalent_Pa_sqrt_m": float(row["K_provider_contract_Pa_sqrt_m"]),
        "G0_source": "J_PROVIDER_CONTRACT_POSITIVE_J",
        "observables": _mechanical_observables(companion),
    }


def _mechanical_observables(state):
    eq = require_equilibrium_observables(state)
    return {
        "top_reaction_N_per_m": float(eq.top_normal_resultant_N_per_m),
        "bottom_reaction_N_per_m": float(eq.bottom_normal_resultant_N_per_m),
        "reaction_balance_relative": float(eq.top_bottom_balance_residual),
        "mean_reaction_magnitude_N_per_m": float(eq.mean_reaction_magnitude_N_per_m),
        "compliance_m2_per_N": float(eq.compliance_m2_per_N),
        "recoverable_energy_J_per_m": float(eq.recoverable_elastic_energy_J_per_m),
        "total_stored_internal_energy_J_per_m": float(eq.total_stored_internal_energy_J_per_m),
        "plastic_eigenstrain_half_work_J_per_m": float(eq.plastic_eigenstrain_half_work_J_per_m),
        "plastic_internal_dissipation_J_per_m": eq.plastic_internal_dissipation_J_per_m,
        "topology_event_dissipation_J_per_m": eq.topology_event_dissipation_J_per_m,
        "external_reaction_work_J_per_m": float(eq.external_reaction_work_J_per_m),
        "energy_identity_reference_J_per_m": float(eq.energy_identity_reference_J_per_m),
        "energy_identity_residual": float(eq.energy_identity_residual),
        "free_dof_residual_relative": float(eq.free_dof_residual_relative),
    }


def _source_metrics(state, regime):
    cavity = state.void_state.cavities[0]
    position = cavity.connection_exit_m
    if position is None:
        position = (cavity.center_m[0] + cavity.radius_m, cavity.center_m[1])
    evaluate_piecewise = cavity.connection_exit_m is not None
    metrics = cavity_source_resolution_metrics(
        state,
        source_position_m=position,
        final_traction_verification=evaluate_piecewise,
    )
    weak = verify_cavity_free_boundary(state)
    if not weak.passed:
        raise RuntimeError(f"{regime} weak cavity certificate failed: {weak.predicates}")
    weak_full = _jsonable(asdict(weak))
    weak_certificate = {
        key: weak_full[key] for key in (
            "schema", "accepted_state_fingerprint", "geometry_fingerprint",
            "assembled_boundary_residual_relative", "normalized_virtual_work",
            "total_boundary_resultant_relative", "total_boundary_moment_relative",
            "global_top_bottom_reaction_balance", "free_dof_residual_relative",
            "predicates", "passed",
        )
    }
    weak_certificate.update({
        "boundary_edge_count": len(weak.cavity_boundary_edge_ids),
        "boundary_node_count": len(weak.boundary_node_ids),
        "full_certificate_sha256": canonical_hash(weak_full),
    })
    probe_ids = tuple(map(int, metrics["probe_element_ids"]))
    return {
        "position_m": list(map(float, position)),
        "tensor_global_xy_Pa": metrics["source_tensor_global_xy_Pa"],
        "tensor_local_nt_Pa": metrics["source_tensor_local_nt_Pa"],
        "sigma_nn_Pa": float(metrics["sigma_nn_Pa"]),
        "sigma_tt_Pa": float(metrics["sigma_tt_Pa"]),
        "sigma_nt_Pa": float(metrics["sigma_nt_Pa"]),
        "sigma_zz_Pa": float(metrics["recovery_record"]["plane_strain_sigma_zz_Pa"]),
        "mean_stress_Pa": float(metrics["recovery_record"]["mean_stress_Pa"]),
        "mean_stress_status": "DIAGNOSTIC_ONLY_NOT_CONSUMED_BY_DOWNSTREAM_SOURCE_LAW",
        "weak_cavity_boundary_certificate": weak_certificate,
        "raw_CST_traction_diagnostic": metrics["RAW_CST_BOUNDARY_TRACTION_DIAGNOSTIC"],
        "piecewise_traction_diagnostic": (
            metrics["PIECEWISE_EQUILIBRATED_TRACTION_DIAGNOSTIC"]
            if evaluate_piecewise else {
                "classification": "NON_AUTHORITATIVE_DIAGNOSTIC",
                "status": "NOT_AVAILABLE_BEFORE_CAVITY_CONNECTION",
            }
        ),
        "resolution": {key: _jsonable(metrics[key]) for key in (
            "eta_n_max", "eta_n_median", "eta_t_max", "eta_t_median",
            "local_aspect_ratio_max", "local_aspect_ratio_median", "minimum_quality",
            "local_minimum_quality", "constrained_fit_residual", "patch_rank", "patch_condition",
            "boundary_node_id", "recovery_operator",
        )} | {
            "probe_element_count": len(probe_ids),
            "probe_element_ids_sha256": canonical_hash(probe_ids),
        },
    }


def _geometry(state, regime, l_star):
    cavity = state.void_state.cavities[0]
    if regime == "DOWNSTREAM_CHILD":
        tip = np.asarray(state.crack_network.branch(state.crack_network.active_tip_ids[0]).tip)
        ligament = float(tip[0] - (cavity.center_m[0] + cavity.radius_m))
    else:
        root_tip = np.asarray(state.crack_network.branches[0].tip)
        tip = root_tip
        ligament = float(cavity.center_m[0] - cavity.radius_m - tip[0])
    return {
        "x_tip_m": float(tip[0]), "x_void_m": float(cavity.center_m[0]),
        "R_void_m": float(cavity.radius_m), "ligament_m": ligament,
        "L_over_R_void": ligament / float(cavity.radius_m),
        "R_void_over_l_star": float(cavity.radius_m) / l_star,
        "l_star_m": l_star,
    }


def _row(*, implementation, bundle, level, load_index, opening_m, regime,
         state, companion, ligament_result, l_star):
    obs = observables(state, "oracle:" + regime)
    mechanical = _mechanical_observables(state)
    geometry = _geometry(state, regime, l_star)
    source = _source_metrics(state, regime)
    drive = _drive(state, regime)
    ledgers = dict(state.void_state.length_ledgers)
    required_ledgers = (
        "ordinary_crack_fractured_length_m", "traversed_void_free_span_m",
        "active_front_coordinate_advance_m",
    )
    if not all(name in ledgers for name in required_ledgers):
        raise RuntimeError("accepted oracle state lacks separate length ledgers")
    return {
        "oracle_row_id": f"{regime}:{level}:LOAD_{load_index}",
        "oracle_schema": SCHEMA,
        "implementation_identity": implementation,
        "accepted_state_fingerprint": complete_accepted_state_fingerprint(state),
        "material_bundle_fingerprint": dict(require_bound_identity(state)),
        "topology_regime": regime,
        "physical_void_phase": state.void_state.cavities[0].phase.value,
        "topology_fingerprint": _topology_fingerprint(state),
        "geometry_level": level,
        "geometry": geometry,
        "load_index": load_index,
        "applied_opening_m": float(opening_m),
        "applied_mean_reaction_N_per_m": mechanical["mean_reaction_magnitude_N_per_m"],
        "matched_no_void": companion,
        "G_local_J_per_m2": drive.get("G_local_J_per_m2"),
        "G_marginal_J_per_m2": drive.get("G_marginal_J_per_m2"),
        "G_kinetic_used_J_per_m2": drive.get("G_kinetic_used_J_per_m2"),
        "drive_source": drive["drive_source"],
        "drive_status": drive,
        "K_energy_equivalent_Pa_sqrt_m": drive.get("K_energy_equivalent_Pa_sqrt_m"),
        "cavity_source": source,
        "mechanics": mechanical,
        "ligament_energy_release_J_per_m": float(ligament_result.energy_release_J_per_m),
        "ligament_hazard_dissipation_J_per_m": float(ligament_result.hazard_dissipation_J_per_m),
        "length_ledgers": ledgers,
        "fractured_material_ledger_m": float(ledgers["ordinary_crack_fractured_length_m"]),
        "free_span_ledger_m": float(ledgers["traversed_void_free_span_m"]),
        "front_coordinate_ledger_m": float(ledgers["active_front_coordinate_advance_m"]),
        "mesh": {
            "node_count": int(len(state.mesh.nodes)), "element_count": int(len(state.mesh.elems)),
            "minimum_quality": float(obs["mesh_minimum_quality"]),
            "maximum_aspect_ratio": float(obs["mesh_maximum_aspect_ratio"]),
        },
    }


def _validate(rows):
    expected = {
        (regime, level, index)
        for regime in REGIMES for level in GEOMETRIES for index in (1, 2)
    }
    observed = {(row["topology_regime"], row["geometry_level"], row["load_index"]) for row in rows}
    if len(rows) != 18 or observed != expected:
        raise RuntimeError("oracle registry is incomplete or duplicated")
    for row in rows:
        required_common = (
            "implementation_identity", "accepted_state_fingerprint",
            "material_bundle_fingerprint", "topology_regime", "topology_fingerprint",
            "geometry", "applied_opening_m", "matched_no_void", "drive_source",
            "cavity_source", "mechanics", "ligament_energy_release_J_per_m",
            "fractured_material_ledger_m", "free_span_ledger_m",
            "front_coordinate_ledger_m", "mesh",
        )
        missing = [name for name in required_common if name not in row or row[name] is None]
        if missing:
            raise RuntimeError(f"{row.get('oracle_row_id')} incomplete fields: {missing}")
        encoded = json.dumps(_jsonable(row), sort_keys=True, allow_nan=False)
        if not encoded:
            raise RuntimeError("oracle row did not serialize")
        mechanics = row["mechanics"]
        if mechanics["compliance_m2_per_N"] <= 0.0:
            raise RuntimeError("oracle row has nonphysical compliance")
        if mechanics["top_reaction_N_per_m"] == 0.0 or mechanics["bottom_reaction_N_per_m"] == 0.0:
            raise RuntimeError("oracle row contains a zero/default reaction")
        companion = row["matched_no_void"]
        if companion["G0_J_per_m2"] is None or companion["observables"]["compliance_m2_per_N"] <= 0.0:
            raise RuntimeError("oracle row lacks a measured no-void G0/compliance companion")
        source = row["cavity_source"]
        tensor = np.asarray(source["tensor_global_xy_Pa"], dtype=float)
        if tensor.shape != (2, 2) or not np.isfinite(tensor).all():
            raise RuntimeError("oracle row lacks a complete finite in-plane source tensor")
        if source.get("raw_CST_traction_diagnostic") is None or source.get("piecewise_traction_diagnostic") is None:
            raise RuntimeError("oracle row lacks required nonauthoritative traction diagnostics")
        if not row["cavity_source"]["weak_cavity_boundary_certificate"]["passed"]:
            raise RuntimeError("oracle row lacks a passing weak certificate")
        if row["topology_regime"] == "CONNECTED_CAVITY":
            if any(row[name] is not None for name in (
                "G_local_J_per_m2", "G_marginal_J_per_m2",
                "G_kinetic_used_J_per_m2", "K_energy_equivalent_Pa_sqrt_m",
            )):
                raise RuntimeError("connected cavity must not expose a sharp-tip G")
        elif row["G_local_J_per_m2"] is None or row["G_kinetic_used_J_per_m2"] is None:
            raise RuntimeError("sharp-front oracle row lacks measured local/kinetic G")


def _scaling(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["topology_regime"], row["geometry_level"]), []).append(row)
    checks = []
    for (regime, level), pair in sorted(groups.items()):
        low, high = sorted(pair, key=lambda row: row["applied_opening_m"])
        K_low = float(low["matched_no_void"]["K0_energy_equivalent_Pa_sqrt_m"])
        K_high = float(high["matched_no_void"]["K0_energy_equivalent_Pa_sqrt_m"])
        amplitude = K_high / K_low
        opening_amplitude = high["applied_opening_m"] / low["applied_opening_m"]
        K_opening_linearity_error = abs(K_high-opening_amplitude*K_low)/max(abs(K_high), 1e-300)
        sigma_low = np.asarray(low["cavity_source"]["tensor_global_xy_Pa"])
        sigma_high = np.asarray(high["cavity_source"]["tensor_global_xy_Pa"])
        stress_error = float(np.linalg.norm(sigma_high-amplitude*sigma_low)
                             / max(np.linalg.norm(sigma_high), 1e-300))
        delta_energy_low = (
            low["mechanics"]["recoverable_energy_J_per_m"]
            - low["matched_no_void"]["observables"]["recoverable_energy_J_per_m"]
        )
        delta_energy_high = (
            high["mechanics"]["recoverable_energy_J_per_m"]
            - high["matched_no_void"]["observables"]["recoverable_energy_J_per_m"]
        )
        energy_error = abs(
            delta_energy_high-amplitude**2*delta_energy_low
        )/max(abs(delta_energy_high), 1e-300)
        delta_compliance_low = (
            low["mechanics"]["compliance_m2_per_N"]
            - low["matched_no_void"]["observables"]["compliance_m2_per_N"]
        )
        delta_compliance_high = (
            high["mechanics"]["compliance_m2_per_N"]
            - high["matched_no_void"]["observables"]["compliance_m2_per_N"]
        )
        compliance_error = abs(delta_compliance_high-delta_compliance_low)/max(
            abs(delta_compliance_high), 1e-300)
        delta_total_energy_low = (
            low["mechanics"]["total_stored_internal_energy_J_per_m"]
            - low["matched_no_void"]["observables"]["total_stored_internal_energy_J_per_m"]
        )
        delta_total_energy_high = (
            high["mechanics"]["total_stored_internal_energy_J_per_m"]
            - high["matched_no_void"]["observables"]["total_stored_internal_energy_J_per_m"]
        )
        total_energy_error = abs(
            delta_total_energy_high-amplitude**2*delta_total_energy_low
        )/max(abs(delta_total_energy_high), 1e-300)
        g_error = None
        if low["G_kinetic_used_J_per_m2"] is not None and high["G_kinetic_used_J_per_m2"] is not None:
            g_error = abs(high["G_kinetic_used_J_per_m2"]-amplitude**2*low["G_kinetic_used_J_per_m2"])/max(
                abs(high["G_kinetic_used_J_per_m2"]), 1e-300)
        passed = (
            stress_error <= 0.02
            and K_opening_linearity_error <= 0.02
            and energy_error <= 0.02
            and total_energy_error <= 0.02
            and compliance_error <= 0.02
        )
        if g_error is not None:
            passed &= g_error <= 0.05
        checks.append({
            "topology_regime": regime, "geometry_level": level,
            "applied_load_coordinate": "MATCHED_NO_VOID_K0_ENERGY_EQUIVALENT",
            "K0_low_Pa_sqrt_m": K_low, "K0_high_Pa_sqrt_m": K_high,
            "K0_amplitude_ratio": amplitude,
            "K0_opening_linearity_relative_error": K_opening_linearity_error,
            "source_stress_linear_relative_error": stress_error,
            "void_induced_recoverable_energy_increment_quadratic_relative_error": energy_error,
            "void_induced_total_energy_increment_quadratic_relative_error": total_energy_error,
            "void_induced_compliance_increment_load_invariance_relative_error": compliance_error,
            "G_quadratic_relative_error": g_error, "passed": bool(passed),
        })
    passed = all(row["passed"] for row in checks)
    return {
        "passed": passed,
        "map_load_coordinate_policy": "ANALYTICALLY_FACTORED" if passed else "RETAIN_APPLIED_OPENING_COORDINATE",
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "analysis_outputs/unified_one_void_18_state_oracle")
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/unified-one-void-oracle-cache"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    implementation = _implementation_identity()
    progress_path = args.out / "oracle_progress.json"
    reusable_rows = {}
    if progress_path.exists():
        prior = json.loads(progress_path.read_text())
        for row in prior.get("rows", ()):
            if row.get("implementation_identity", {}).get("frozen_source_sha256") == implementation["frozen_source_sha256"]:
                reusable_rows[row["oracle_row_id"]] = row
    bundle = material_bundle("DBTT")
    l_star = float(fresh_sharp_front_engine(
        build_production_void_state(bundle=bundle, enabled=False)[0].material, bundle,
    ).f.L_pz)
    rows = []
    for level, radius in GEOMETRIES.items():
        for load_index, opening in enumerate(OPENINGS_M, start=1):
            cache = args.cache_dir / f"{implementation['frozen_source_sha256']}-{level}-{load_index}.pickle"
            if cache.exists():
                print(f"[oracle] {level} load {load_index}: restoring temporary solve cache", flush=True)
                cached = pickle.loads(cache.read_bytes())
                pre, connected, child, ligament = cached["states"]
            else:
                print(f"[oracle] {level} load {load_index}: building preconnection", flush=True)
                pre, _ = deterministic_trajectory(
                    bundle=bundle, stop_before_ligament=True, crack_path_m=CRACK_PATH_M,
                    boundary_segments=128, radial_layers=48, shape_regular_local_level="B",
                    resolved_radius_m=radius,
                )
                pre = _set_opening(pre, opening)
                connected, ligament = ligament_transaction(pre)
                if not ligament.accepted:
                    raise RuntimeError(f"{level} load {load_index} ligament event rejected")
                print(f"[oracle] {level} load {load_index}: qualifying connected source", flush=True)
                connected, audit = refine_downstream_source(
                    connected, max_refinement_levels=1,
                    refinement_region="complete_cavity_ring", quality_improvement="constrained_v1",
                )
                if audit["status"] != "SOURCE_TENSOR_QUALIFIED":
                    raise RuntimeError(f"{level} load {load_index} source unqualified: {audit}")
                child, result, _, _ = downstream_front_transaction(connected)
                if result is None or not result.accepted:
                    raise RuntimeError(f"{level} load {load_index} downstream child rejected")
                cache.write_bytes(pickle.dumps({
                    "implementation_sha256": implementation["frozen_source_sha256"],
                    "states": (pre, connected, child, ligament),
                }, protocol=5))
            states = {
                "PRECONNECTION": pre,
                "CONNECTED_CAVITY": connected,
                "DOWNSTREAM_CHILD": child,
            }
            for regime, state in states.items():
                row_id = f"{regime}:{level}:LOAD_{load_index}"
                reusable = reusable_rows.get(row_id)
                if (
                    reusable is not None
                    and reusable.get("accepted_state_fingerprint")
                    == complete_accepted_state_fingerprint(state)
                    and reusable.get("topology_fingerprint") == _topology_fingerprint(state)
                ):
                    print(f"[oracle] {row_id}: reusing measured mechanics; refreshing diagnostics", flush=True)
                    rows.append({**reusable, "cavity_source": _source_metrics(state, regime)})
                else:
                    companion_state, companion = _companion(bundle, state, regime, opening)
                    rows.append(_row(
                        implementation=implementation, bundle=bundle, level=level,
                        load_index=load_index, opening_m=opening, regime=regime,
                        state=state, companion=companion, ligament_result=ligament, l_star=l_star,
                    ))
            progress = {"schema": SCHEMA, "complete": False, "rows": _jsonable(rows)}
            progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True)+"\n")
            print(f"[oracle] accepted rows: {len(rows)}/18", flush=True)
    _validate(rows)
    scaling = _scaling(rows)
    payload = {
        "schema": SCHEMA,
        "implementation_identity": implementation,
        "material_family": "DBTT",
        "geometry_registry": {key: {"R_void_m": value} for key, value in GEOMETRIES.items()},
        "load_registry": list(OPENINGS_M),
        "load_scaling": scaling,
        "oracle_completeness": "18/18",
        "rows": _jsonable(rows),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n"
    oracle = args.out / "oracle_18_state.json"
    oracle.write_text(encoded)
    manifest = {
        "schema": SCHEMA+".manifest",
        "oracle_path": oracle.name,
        "oracle_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "row_count": 18,
        "row_ids": [row["oracle_row_id"] for row in rows],
        "implementation_identity": implementation,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    columns = (
        "oracle_row_id", "topology_regime", "geometry_level", "load_index",
        "applied_opening_m", "G_local_J_per_m2", "G_marginal_J_per_m2",
        "G_kinetic_used_J_per_m2", "K_energy_equivalent_Pa_sqrt_m",
    )
    with (args.out / "oracle_18_state.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows({key: row.get(key) for key in columns} for row in rows)
    print("ORACLE_COMPLETENESS = 18/18")
    print("LOAD_SCALING = " + ("PASS" if scaling["passed"] else "FAIL_LOAD_RETAINED_AS_MAP_COORDINATE"))


if __name__ == "__main__":
    main()
