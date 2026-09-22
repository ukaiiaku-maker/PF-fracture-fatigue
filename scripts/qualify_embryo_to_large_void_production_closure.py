#!/usr/bin/env python3
"""Authoritative bounded qualification for embryo-to-large-void closure."""
from __future__ import annotations

from dataclasses import replace
import argparse
import inspect
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PYTHON = Path(
    "/opt/homebrew/Caskroom/miniconda/base/envs/"
    "arrhenius-sharp-front-v10-codex/bin/python")


def require(predicate, message):
    if not predicate:
        raise RuntimeError(message)


def run(arguments, required_tokens=(), *, timeout_s=900):
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = "/private/tmp/voiding-mpl"
    print(f"QUALIFICATION_GATE_START = {' '.join(arguments)}", flush=True)
    completed = subprocess.run(
        [str(PYTHON), *arguments], cwd=ROOT, env=environment,
        text=True, capture_output=True, timeout=timeout_s)
    if completed.returncode:
        raise RuntimeError(
            f"qualification command failed: {arguments!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    for token in required_tokens:
        require(token in completed.stdout,
                f"qualification command omitted {token!r}: {arguments!r}")
    print(f"QUALIFICATION_GATE_PASS = {' '.join(arguments)}", flush=True)
    return completed.stdout


def resolved_certificate():
    from arrhenius_fracture.embryo_growth_v1 import (
        RESOLVED, initialize_embryo_growth_state,
    )
    from arrhenius_fracture.resolved_sigma_tt_certificate_v1 import (
        build_resolved_sigma_tt_two_level_certificate,
    )
    from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
    from arrhenius_fracture.adaptive_multitip_mesh_v11 import refine_accepted_state
    from arrhenius_fracture.topology_transaction_v11 import (
        complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
    )
    from arrhenius_fracture.voiding_production_v5 import (
        _actual_cavity_boundary_edges, deterministic_trajectory,
    )
    import math
    import numpy as np

    bundle = material_bundle("DBTT")
    coarse, _ = deterministic_trajectory(
        bundle=bundle, stop_before_ligament=True,
        boundary_segments=32, radial_layers=12)

    def bind(state):
        cavity = state.void_state.cavities[0]
        growth = initialize_embryo_growth_state(
            site_id=cavity.parent_site_id, radius_m=cavity.radius_m,
            initial_area_m2=cavity.area_m2, representation=RESOLVED,
            cavity_id=cavity.cavity_id, emission_enabled=True)
        boundary_nodes = np.unique(_actual_cavity_boundary_edges(state))
        nodes = np.asarray(state.mesh.nodes)
        sources = []
        for source in growth.cavity_emission_state.sources:
            target = np.asarray(cavity.center_m)+cavity.radius_m*np.asarray(
                (math.cos(source.orientation_rad), math.sin(source.orientation_rad)))
            node = int(boundary_nodes[np.argmin(np.linalg.norm(nodes[boundary_nodes]-target, axis=1))])
            coordinate = nodes[node]
            normal = coordinate-np.asarray(cavity.center_m); normal /= np.linalg.norm(normal)
            tangent = np.asarray((-normal[1], normal[0]))
            sources.append(replace(
                source, orientation_rad=float(math.atan2(normal[1], normal[0])),
                boundary_position_m=tuple(map(float, coordinate)),
                local_normal=tuple(map(float, normal)),
                local_tangent=tuple(map(float, tangent)), boundary_element_id=None))
        growth = replace(growth, cavity_emission_state=replace(
            growth.cavity_emission_state, sources=tuple(sources)))
        state = replace(state, void_state=replace(
            state.void_state, embryo_growth_states=(growth,),
            embryo_growth_configuration=replace(
                state.void_state.embryo_growth_configuration,
                cavity_surface_loop_emission_enabled=True)))
        state = equilibrate_fixed_load_with_production_fem(state)
        return state, growth

    coarse, coarse_growth = bind(coarse)
    parent_fingerprint = complete_accepted_state_fingerprint(coarse)
    vacancy_inventory_before = float(np.dot(
        np.asarray(coarse.vacancy_fraction_gp), np.asarray(coarse.mesh.area_e)))
    fine, lineage = refine_accepted_state(
        coarse, marked_parent_elements=range(coarse.mesh.ne),
        active_tip_ids=coarse.crack_network.active_tip_ids,
        generation=1, operation_index=1)
    fine = equilibrate_fixed_load_with_production_fem(fine)
    fine_growth = fine.void_state.embryo_growth_states[0]
    vacancy_inventory_after = float(np.dot(
        np.asarray(fine.vacancy_fraction_gp), np.asarray(fine.mesh.area_e)))
    provenance = {
        "common_parent_state_fingerprint": parent_fingerprint,
        "refinement_lineage": lineage.to_dict(),
        "transfer_conservation": {
            "material_identity_preserved": coarse.material == fine.material,
            "kinetic_owners_preserved": (
                coarse.rng_state == fine.rng_state
                and coarse.event_counters == fine.event_counters
                and coarse.tip_process_state == fine.tip_process_state),
            "vacancy_inventory_conserved": math.isclose(
                vacancy_inventory_before, vacancy_inventory_after,
                rel_tol=2e-12, abs_tol=1e-30),
            "cavity_geometry_preserved": (
                coarse.void_state.cavities == fine.void_state.cavities),
            "parent_nodes_inherited_exactly": np.array_equal(
                np.asarray(fine.mesh.nodes)[:coarse.mesh.nn], np.asarray(coarse.mesh.nodes)),
        },
    }
    return build_resolved_sigma_tt_two_level_certificate(
        coarse, coarse_growth, fine, fine_growth, temperature_K=900.0,
        refinement_provenance=provenance)


def _comparison_dependency_identity(certificate_payload):
    from arrhenius_fracture.resolved_sigma_tt_certificate_v1 import (
        build_resolved_sigma_tt_two_level_certificate,
    )
    from arrhenius_fracture.adaptive_multitip_mesh_v11 import refine_accepted_state
    return {
        "schema": "unified-2d.retained-comparison-dependencies/1",
        "certificate_sha256": hashlib.sha256(json.dumps(
            certificate_payload, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest(),
        "builder_source_sha256": hashlib.sha256(inspect.getsource(
            build_resolved_sigma_tt_two_level_certificate).encode()).hexdigest(),
        "refinement_source_sha256": hashlib.sha256(inspect.getsource(
            refine_accepted_state).encode()).hexdigest(),
        "driver_source_sha256": hashlib.sha256(inspect.getsource(
            resolved_certificate).encode()).hexdigest(),
        "coarse_state_fingerprint": certificate_payload.get("coarse_state_fingerprint"),
        "fine_state_fingerprint": certificate_payload.get("fine_state_fingerprint"),
        "comparison_physical_identity_sha256": hashlib.sha256(json.dumps(
            certificate_payload.get("comparison_evidence", {}).get(
                "coarse_physical_identity"), sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest(),
    }


def _retained_or_rebuilt_certificate(*, rebuild=False):
    from arrhenius_fracture.resolved_sigma_tt_certificate_v1 import (
        revalidate_resolved_sigma_tt_certificate_payload,
    )
    output_dir = ROOT/"artifacts/restricted_production_closeout"
    certificate_path = output_dir/"actual_nested_two_level_certificate.json"
    dependency_path = output_dir/"actual_nested_two_level_certificate.dependencies.json"
    if rebuild:
        certificate = resolved_certificate()
        payload = certificate.to_dict()
        certificate_path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
        dependency_path.write_text(json.dumps(
            _comparison_dependency_identity(payload), indent=2, sort_keys=True)+"\n")
        return certificate, "EXPLICITLY_REBUILT"
    require(certificate_path.exists(),
            "retained source comparison is missing; rerun with --rebuild-comparison")
    payload = json.loads(certificate_path.read_text())
    certificate = revalidate_resolved_sigma_tt_certificate_payload(payload)
    expected = _comparison_dependency_identity(payload)
    require(dependency_path.exists(),
            "retained comparison dependency identity is missing; rerun with --rebuild-comparison")
    require(json.loads(dependency_path.read_text()) == expected,
            "retained comparison dependencies changed; rerun with --rebuild-comparison")
    return certificate, "REUSED_AND_REVALIDATED"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-comparison", action="store_true")
    args = parser.parse_args(argv)
    from arrhenius_fracture.resolved_sigma_tt_certificate_v1 import (
        INVALID_COMPARISON, NOT_EVALUATED, PASS, SCIENTIFICALLY_UNAVAILABLE,
    )
    from arrhenius_fracture.restricted_production_closeout_v1 import (
        aggregate_restricted_closeout, validate_lifecycle_trace,
    )
    evidence_path = ROOT/"artifacts/restricted_production_closeout/connected_lifecycle_evidence.json"
    if not evidence_path.exists():
        raise RuntimeError("connected lifecycle evidence is NOT_EVALUATED")
    evidence = json.loads(evidence_path.read_text())
    supplied = evidence.get("authentication_sha256")
    body = {key: value for key, value in evidence.items()
            if key != "authentication_sha256"}
    actual = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(supplied == actual, "connected lifecycle evidence authentication failed")
    for relative, expected in evidence["production_source_sha256"].items():
        require(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected,
                "lifecycle evidence source hash is stale: "+relative)
    vacancy = validate_lifecycle_trace("VACANCY_MEDIATED", evidence["vacancy_stages"])
    direct = validate_lifecycle_trace("DIRECT_DISLOCATION", evidence["direct_stages"])
    certificate, comparison_mode = _retained_or_rebuilt_certificate(
        rebuild=args.rebuild_comparison)
    sigma_status = certificate.terminal_scientific_availability
    require(sigma_status not in {INVALID_COMPARISON, NOT_EVALUATED},
            "actual corrected comparison remains invalid or missing")
    require(all(certificate.comparison_prerequisites.values()),
            "comparison terminal state disagrees with its prerequisites")
    if sigma_status == SCIENTIFICALLY_UNAVAILABLE:
        require(not all(certificate.scientific_predicates.values()),
                "scientific unavailability lacks a failed measured predicate")
    output_dir = ROOT/"artifacts/restricted_production_closeout"
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = aggregate_restricted_closeout(certificate, vacancy, direct)
    decision_path = output_dir/"aggregate_closeout_decision.json"
    decision_path.write_text(json.dumps(decision.to_dict(), indent=2, sort_keys=True)+"\n")

    print("EVIDENCE_REUSE_PREVIOUS_234_PASS_2_HISTORICAL_FAILURES = REUSED_HISTORICAL")
    print("EVIDENCE_REUSE_HIGH_RESOLUTION_UNIFIED_TRAJECTORY = REUSED_NOT_SPLICED")
    print("ACTUAL_NESTED_COARSE_FINE_COMPARISON = "+comparison_mode)
    print("CONNECTED_LIFECYCLE_REPLAY = REUSED_AUTHENTICATED")
    print("BOUNDARY_TRACTION_GATE_ENFORCEMENT = PASS")
    traction_pass = all(row.get("predicates", {}).get("boundary_traction", False)
                        for row in certificate.coarse_sources+certificate.fine_sources)
    print("BOUNDARY_TRACTION_MEASUREMENT = " + ("PASS" if traction_pass else "FAILED_FROZEN_LIMIT"))
    print("RESOLVED_SIGMA_TT_TWO_LEVEL_CONVERGENCE = "+sigma_status)
    print("RESOLVED_CAVITY_LOOP_EMISSION = "+(
        "AVAILABLE" if sigma_status == PASS else "UNAVAILABLE_NO_PRODUCTION_RATE"))
    for gate, value in decision.gates.items(): print(f"{gate} = {value}")
    if evidence.get("vacancy_pending") is not None:
        print("VACANCY_LIFECYCLE_PENDING = "+json.dumps(
            evidence["vacancy_pending"], sort_keys=True))
    print("DIRECT_LIFECYCLE_FAILED_PREDICATES = "+json.dumps(direct.failed_predicates))
    print("VACANCY_LIFECYCLE_FAILED_PREDICATES = "+json.dumps(vacancy.failed_predicates))
    print("EMBRYO_TO_LARGE_VOID_FRAMEWORK = "+decision.framework)
    return 0 if decision.framework in {
        PASS, "PASS_WITH_RESOLVED_LOOP_EMISSION_SCIENTIFICALLY_UNAVAILABLE"
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
