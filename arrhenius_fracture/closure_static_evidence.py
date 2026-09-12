"""Source-resolved static branch of the closure ontology (not lifecycle states).

The registry and recovery operators are frozen before the new execution. The
resolution screen is a numerical hypothesis based on preserved traction_a,
not a scientific acceptance tolerance and NOT permission to fire an event.
"""
from __future__ import annotations

import hashlib
import math
import platform
import sys

import numpy as np
import scipy

from .explicit_cavity_v5 import cavity_edge_traction_geometry
from .finalization_v3_schema import canonical_hash, SCIENTIFIC_ACCEPTANCE_TOLERANCES as LIMITS

SCHEMA = "v12.voiding-v5-closure-static/2"
COUPLED = ((32, 12), (64, 24), (128, 48), (256, 96), (512, 192))
CROSSED = ((128, 48), (256, 48), (512, 48), (256, 96), (256, 192), (512, 192))
CRACKED = ((128, 48), (256, 96), (512, 192))
ARC_FRACTIONS = (0., .125, .25, .375, .5)
CANDIDATE_NORMALS = ((1., 0.), (0., 1.), (2**-.5, 2**-.5), (2**-.5, -2**-.5))
RESOLUTION_SCREEN = {"eta_n_max": .03, "eta_t_max": .025,
                     "local_aspect_ratio_max": 7., "minimum_quality": .05}
SOURCE_GROUPS = {
    "mesh": ("nodes", "elements", "cavity_edges", "exterior_edges", "top_nodes", "bottom_nodes"),
    "support": ("support_mask",),
    "assembled_system": ("K_data", "K_indices", "K_indptr", "K_shape", "prescribed_dofs", "elasticity_D"),
    "solution": ("displacement", "stress", "assembled_residual"),
}


def configuration(cracked, segments, layers):
    # Keys are actual solve_crack_void_case arguments; the runner passes all.
    return dict(cavity_center_m=[7e-4, 0.], cavity_radius_m=5e-5,
                specimen_width_m=1e-3, specimen_height_m=1e-3, far_h_m=5e-5,
                material_E_Pa=210e9, material_nu=.3, opening_m=4e-7,
                residual_stiffness_kappa=1e-6 if cracked else 0.,
                boundary_segments=segments, radial_layers=layers, tip_layer=3,
                crack_enabled=cracked, cavity_enabled=True, ligament_ratio=None,
                crack_orientation_deg=0., crack_root_m=None, crack_tip_m=None,
                crack_path_m=[[0., 0.], [5e-4, 0.]] if cracked else None,
                geometry_mode="V3_FIXED_LABORATORY_GEOMETRY", capture_source=True)


def base_id(cracked, segments, layers):
    return f"{'cracked' if cracked else 'cavity'}:{segments}:{layers}"


TRACTION_REGISTRY = {
    base_id(cracked, segments, layers): configuration(cracked, segments, layers)
    for cracked, pairs in ((False, tuple(dict.fromkeys(COUPLED + CROSSED))), (True, CRACKED))
    for segments, layers in pairs
}


def array_hash(value):
    array = np.ascontiguousarray(value)
    # Fixed little-endian representation across workers; shape and dtype bound.
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    header = canonical_hash({"dtype": array.dtype.str, "shape": list(array.shape)})
    return hashlib.sha256(header.encode() + array.tobytes()).hexdigest()


def source_fingerprints(source):
    if set(source) != {key for group in SOURCE_GROUPS.values() for key in group}:
        raise ValueError("source array inventory differs from frozen contract")
    return {name: canonical_hash({key: array_hash(source[key]) for key in keys})
            for name, keys in SOURCE_GROUPS.items()}


def validate_solver_capture(raw, cfg):
    """Reassemble with captured u and frozen inputs; hashes alone are not proof."""
    from dataclasses import replace
    from .config import ElasticProperties
    from .mesh import rebuild_tri_mesh
    from .fem import assemble_mechanics, plane_strain_D, stress_state
    mesh = rebuild_tri_mesh(raw["nodes"], raw["elements"])
    mesh = replace(mesh, element_damage_gp=raw["support_mask"].astype(float))
    material = ElasticProperties(E=cfg["material_E_Pa"], nu=cfg["material_nu"])
    D = plane_strain_D(material)
    ep, rho, damage = np.zeros((3, mesh.ne)), np.zeros(mesh.ne), np.zeros(mesh.nn)
    K, residual, *_ = assemble_mechanics(mesh, raw["displacement"], ep, rho, damage, D,
                                       material, kappa=cfg["residual_stiffness_kappa"])
    K = K.tocsr(); K.sum_duplicates(); K.sort_indices()
    stress, *_ = stress_state(mesh, raw["displacement"], ep, damage, D, material,
                             kappa=cfg["residual_stiffness_kappa"])
    for name, calculated in (("K_data", K.data), ("K_indices", K.indices), ("K_indptr", K.indptr),
                             ("K_shape", K.shape), ("assembled_residual", residual),
                             ("stress", stress), ("elasticity_D", D)):
        if not np.array_equal(raw[name], calculated):
            raise ValueError("source reassembly mismatch: "+name)
    if not np.isfinite(raw["displacement"]).all():
        raise ValueError("nonfinite solution")
    top, bottom = raw["top_nodes"], raw["bottom_nodes"]
    if (not np.all(raw["displacement"][2*top+1] == cfg["opening_m"]/2) or
        not np.all(raw["displacement"][2*bottom+1] == -cfg["opening_m"]/2)):
        raise ValueError("captured solution violates prescribed opening")


def environment_identity():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": sys.platform,
            "machine": platform.machine(), "solver": "scipy.sparse.linalg.spsolve",
            "assembly": "production_CST_plane_strain_P0_V12_support"}


def _tensor_record(tensor, normal):
    normal = np.asarray(normal); tangent = np.array([-normal[1], normal[0]])
    return {"tensor_Pa": tensor.tolist(), "sigma_nn_Pa": float(normal @ tensor @ normal),
            "sigma_tt_Pa": float(tangent @ tensor @ tangent),
            "sigma_nt_Pa": float(normal @ tensor @ tangent),
            "principal_stresses_Pa": np.linalg.eigvalsh(tensor).tolist(),
            "candidate_planes": [{"normal": list(n),
                "normal_opening_Pa": float(np.asarray(n) @ tensor @ n),
                "shear_Pa": float(np.array([-n[1], n[0]]) @ tensor @ n)} for n in CANDIDATE_NORMALS]}


def fixed_arc_probes(edges, center, radius):
    """Periodic linear interpolation of adjacent-CST tensors at edge mid-angles.

    Physical coordinates and directions, not variable-length mesh indices,
    control comparisons. No traction projection or stress-dependent selection.
    """
    angles = np.asarray([math.atan2(*(np.mean(e["edge_endpoints_m"], axis=0) - center)[::-1]) % (2*math.pi)
                         for e in edges])
    order = np.argsort(angles, kind="stable"); angles = angles[order]
    rows = []
    for fraction in ARC_FRACTIONS:
        angle = 2*math.pi*fraction
        upper = int(np.searchsorted(angles, angle, side="right"))
        lo, hi = (upper-1) % len(edges), upper % len(edges)
        left = angles[lo] - (2*math.pi if upper == 0 else 0.)
        right = angles[hi] + (2*math.pi if upper == len(edges) else 0.)
        w = float((angle-left)/(right-left))
        a, b = edges[int(order[lo])], edges[int(order[hi])]
        tensor = (1-w)*np.asarray(a["adjacent_element_stress_tensor_Pa"]) + w*np.asarray(b["adjacent_element_stress_tensor_Pa"])
        normal = np.array([math.cos(angle), math.sin(angle)])
        rows.append({"arc_fraction": fraction, "physical_position_m": (center+radius*normal).tolist(),
                     "operator": "periodic_linear_edge_mid_angle_adjacent_CST_v1",
                     "element_ids": [a["adjacent_element_id"], b["adjacent_element_id"]],
                     "edge_node_ids": [a["edge_node_ids"], b["edge_node_ids"]],
                     "weights": [1-w, w], **_tensor_record(tensor, normal)})
    return rows


def fixed_tip_probe(source, cfg):
    if not cfg["crack_enabled"]:
        return None
    path = np.asarray(cfg["crack_path_m"])
    tangent = path[-1]-path[-2]; tangent /= np.linalg.norm(tangent)
    # A fixed physical point, 25 um ahead, avoids sampling the singular tip.
    point = path[-1] + 2.5e-5*tangent
    tri = source["nodes"][source["elements"]]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v, w, p = b-a, c-a, point-a
    det = v[:, 0]*w[:, 1]-v[:, 1]*w[:, 0]
    x = (p[:, 0]*w[:, 1]-p[:, 1]*w[:, 0])/det
    y = (v[:, 0]*p[:, 1]-v[:, 1]*p[:, 0])/det
    chosen = np.flatnonzero((x >= -1e-12) & (y >= -1e-12) & (x+y <= 1+1e-12) & ~source["support_mask"])
    if not len(chosen):
        raise ValueError("fixed physical crack-tip probe is outside live solid")
    weights = np.abs(det[chosen]); weights /= weights.sum()
    s = source["stress"][:, chosen] @ weights
    tensor = np.array([[s[0], s[2]], [s[2], s[1]]])
    return {"operator": "area_weighted_containing_live_CST_at_fixed_25um_ahead_v1",
            "physical_position_m": point.tolist(), "element_ids": chosen.tolist(),
            "weights": weights.tolist(), **_tensor_record(tensor, [-tangent[1], tangent[0]])}


def recompute_measurements(source, cfg):
    """Only raw mesh, solution and assembly arrays feed these measurements."""
    nodes, elements, sigma = source["nodes"], source["elements"], source["stress"]
    center, radius = np.asarray(cfg["cavity_center_m"]), cfg["cavity_radius_m"]
    owners = {}
    for ei, elem in enumerate(elements):
        for a, b in zip(elem, np.roll(elem, -1)):
            owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(ei)
    tri = nodes[elements]
    sides = np.linalg.norm(tri[:, [1, 2, 0]]-tri[:, [0, 1, 2]], axis=2)
    cross = (tri[:, 1, 0]-tri[:, 0, 0])*(tri[:, 2, 1]-tri[:, 0, 1])-(tri[:, 1, 1]-tri[:, 0, 1])*(tri[:, 2, 0]-tri[:, 0, 0])
    quality = 2*np.sqrt(3)*np.abs(cross)/np.sum(sides*sides, axis=1)
    edges, normal_spacing, aspects = [], [], []
    for a, b in source["cavity_edges"]:
        adjacent = owners.get(tuple(sorted((int(a), int(b)))), [])
        if len(adjacent) != 1 or source["support_mask"][adjacent[0]]:
            raise ValueError("cavity edge must have exactly one live solid owner")
        ei = adjacent[0]; s = sigma[:, ei]
        tensor = np.array([[s[0], s[2]], [s[2], s[1]]])
        geometry = cavity_edge_traction_geometry(nodes[a], nodes[b], center, tensor)
        third = next(v for v in elements[ei] if v not in (a, b))
        spacing = float((nodes[third]-(nodes[a]+nodes[b])/2) @ geometry["cavity_outward_into_solid_normal"])
        if spacing <= 0: raise ValueError("nonpositive live-solid normal spacing")
        normal_spacing.append(spacing/radius)
        aspects.append(float(sides[ei].max()/sides[ei].min()))
        edges.append({"edge_node_ids": [int(a), int(b)], "adjacent_element_id": ei,
                      "adjacent_solid_element_count": len(adjacent),
                      "edge_endpoints_m": nodes[[a, b]].tolist(),
                      "edge_length_m": geometry["edge_length_m"],
                      "adjacent_element_stress_tensor_Pa": tensor.tolist(),
                      "traction_Pa": geometry["traction"].tolist(),
                      "first_layer_normal_spacing_m": spacing})
    if not edges: raise ValueError("static traction capture has no cavity edges")
    lengths = np.asarray([e["edge_length_m"] for e in edges])
    top = float(np.sum(source["assembled_residual"][2*source["top_nodes"]+1]))
    bottom = float(np.sum(source["assembled_residual"][2*source["bottom_nodes"]+1]))
    remote = abs(top)/float(np.ptp(nodes[:, 0]))
    t2 = sum(float(np.dot(e["traction_Pa"], e["traction_Pa"]))*e["edge_length_m"] for e in edges)
    boundary_nodes = np.unique(source["cavity_edges"])
    dofs = np.c_[2*boundary_nodes, 2*boundary_nodes+1].ravel()
    energy = float(.5*source["displacement"] @ source["assembled_residual"])
    result = {"edge_owner_valid": True, "minimum_quality": float(quality.min()),
              "eta_n_max": float(max(normal_spacing)), "eta_n_median": float(np.median(normal_spacing)),
              "eta_t_max": float(lengths.max()/radius), "eta_t_median": float(np.median(lengths)/radius),
              "local_aspect_ratio_max": max(aspects), "local_aspect_ratio_median": float(np.median(aspects)),
              "normalized_traction": math.sqrt(t2)/(remote*math.sqrt(float(lengths.sum()))),
              "assembled_cavity_node_residual_relative": float(np.linalg.norm(source["assembled_residual"][dofs]))/abs(top),
              "free_residual_relative": float(np.linalg.norm(source["assembled_residual"][~source["prescribed_dofs"]]))/abs(top),
              "reaction_top_N_per_m": top, "reaction_bottom_N_per_m": bottom,
              "stored_energy_J_per_m": energy, "compliance_m2_per_N": cfg["opening_m"]/abs(top),
              "reaction_balance_relative": abs(top+bottom)/abs(top),
              "energy_reaction_identity_relative": abs(energy-.5*cfg["opening_m"]*top)/abs(energy),
              "fixed_arc_probes": fixed_arc_probes(edges, center, radius),
              "fixed_tip_probe": fixed_tip_probe(source, cfg), "edge_records": edges}
    return result


def resolution_screen(measurements):
    return (all(measurements[k] <= RESOLUTION_SCREEN[k] for k in
                ("eta_n_max", "eta_t_max", "local_aspect_ratio_max")) and
            measurements["minimum_quality"] >= RESOLUTION_SCREEN["minimum_quality"])


def derived_rows(bases):
    rows = []
    def add(name, ids, inputs, result):
        rows.append({"predicate_id": name, "source_row_ids": ids,
                     "predicate_inputs": inputs, "predicate_result": bool(result)})
    for key, base in bases.items():
        m = base["measurements"]
        for name, metric, limit, minimum in (
            ("mesh_quality", "minimum_quality", LIMITS["mesh_minimum_quality"], True),
            ("assembled_cavity_residual", "assembled_cavity_node_residual_relative", LIMITS["free_residual_relative"], False),
            ("normalized_traction", "normalized_traction", LIMITS["cavity_traction_normalized"], False)):
            add(key+"/"+name, [key], {"value": m[metric], "frozen_limit": limit},
                m[metric] >= limit if minimum else m[metric] <= limit)
        add(key+"/edge_owner", [key], {"edge_count": len(m["edge_records"]), "valid": m["edge_owner_valid"]}, m["edge_owner_valid"])
        add(key+"/resolution_screen", [key], {k: m[k] for k in RESOLUTION_SCREEN}, resolution_screen(m))
    for name, cracked, pairs, decreasing in (
        ("coupled_convergence", False, COUPLED, True),
        ("angular_refinement", False, ((128,48),(256,48),(512,48)), False),
        ("radial_refinement", False, ((256,48),(256,96),(256,192)), True),
        ("cracked_refinement", True, CRACKED, True)):
        ids = [base_id(cracked, *pair) for pair in pairs]
        values = [bases[key]["measurements"]["normalized_traction"] for key in ids]
        orders = [math.log(a/b, 2) for a, b in zip(values, values[1:])]
        add(name, ids, {"values": values, "observed_orders": orders,
                      "expected_trend": "decreasing" if decreasing else "increasing_error"},
            all(a > b if decreasing else a < b for a, b in zip(values, values[1:])))
    ids = [base_id(False, *pair) for pair in CROSSED]
    add("crossed_resolution_screen_agrees_with_traction", ids, {"screen_limits": RESOLUTION_SCREEN},
        all(resolution_screen(bases[key]["measurements"]) ==
            (bases[key]["measurements"]["normalized_traction"] <= LIMITS["cavity_traction_normalized"]) for key in ids))
    # A static solve is not a production candidate transfer execution.
    add("production_resolution_qualified", list(bases),
        {"actual_production_transfer_executions": 0, "classification": "NOT_RUN_REQUIRES_PRODUCTION_TRANSFER"}, False)
    for cracked, pairs in ((False, COUPLED), (True, CRACKED)):
        fine = base_id(cracked, *pairs[-1]); ref = bases[fine]["measurements"]
        for pair in pairs[:-1]:
            key = base_id(cracked, *pair); m = bases[key]["measurements"]
            # Relative full tensor error uses the nonzero reference field norm;
            # traction components are separately normalized by that same norm.
            errors = []
            for probe, reference in zip(m["fixed_arc_probes"], ref["fixed_arc_probes"]):
                norm = max(np.linalg.norm(reference["tensor_Pa"]), 1e-300)
                errors.append({"arc_fraction": probe["arc_fraction"],
                    "tensor_relative": float(np.linalg.norm(np.asarray(probe["tensor_Pa"])-reference["tensor_Pa"])/norm),
                    "tt_relative": abs(probe["sigma_tt_Pa"]-reference["sigma_tt_Pa"])/norm,
                    "nn_normalized": abs(probe["sigma_nn_Pa"])/norm,
                    "nt_normalized": abs(probe["sigma_nt_Pa"])/norm})
            add(key+"/fixed_arc_tensor", [key, fine], {"errors": errors, "frozen_limit": LIMITS["tensor_probe_relative"]},
                all(max(e[k] for k in ("tensor_relative", "tt_relative", "nn_normalized", "nt_normalized")) <= LIMITS["tensor_probe_relative"] for e in errors))
            if cracked:
                error = float(np.linalg.norm(np.asarray(m["fixed_tip_probe"]["tensor_Pa"])-ref["fixed_tip_probe"]["tensor_Pa"])/np.linalg.norm(ref["fixed_tip_probe"]["tensor_Pa"]))
                add(key+"/fixed_tip_tensor", [key, fine], {"error": error, "frozen_limit": LIMITS["tensor_probe_relative"]}, error <= LIMITS["tensor_probe_relative"])
    return rows


def validate_static_evidence(payload, sources, *, executed_code_sha):
    if payload.get("schema") != SCHEMA or payload.get("executed_code_sha") != executed_code_sha:
        raise ValueError("static schema or implementation identity mismatch")
    if len(executed_code_sha) != 40 or any(c not in "0123456789abcdef" for c in executed_code_sha):
        raise ValueError("invalid exact implementation SHA")
    rows = payload["base_rows"]
    if len(rows) != len(TRACTION_REGISTRY) or {r["case_id"] for r in rows} != set(TRACTION_REGISTRY):
        raise ValueError("unique physical base registry mismatch")
    if set(sources) != set(TRACTION_REGISTRY):
        raise ValueError("raw source registry mismatch")
    bases = {}
    for row in rows:
        key = row["case_id"]; cfg = TRACTION_REGISTRY[key]; raw = sources[key]
        if row["input_configuration"] != cfg or row["input_hash"] != canonical_hash(cfg):
            raise ValueError("input configuration mismatch: "+key)
        if row["execution_id"] != key+":"+executed_code_sha or row["executed_code_sha"] != executed_code_sha:
            raise ValueError("base execution identity mismatch")
        if row["source_fingerprints"] != source_fingerprints(raw):
            raise ValueError("raw solver source fingerprint mismatch: "+key)
        validate_solver_capture(raw, cfg)
        geometry = {k: row["source_fingerprints"][k] for k in ("mesh", "support")}
        if row["realized_geometry_fingerprint"] != canonical_hash(geometry):
            raise ValueError("realized geometry fingerprint mismatch")
        if row["measurement_source"] != "sources/"+key.replace(":", "_")+".npz":
            raise ValueError("measurement source must resolve to its physical solve")
        if not row.get("environment_identity") or row["operation_trace"] != ["build_mesh", "realize_V12_support", "assemble", "solve", "recover"]:
            raise ValueError("missing static execution provenance")
        measured = recompute_measurements(raw, cfg)
        if canonical_hash(row["measurements"]) != canonical_hash(measured):
            raise ValueError("raw observable recomputation mismatch: "+key)
        bases[key] = row
    if payload["derived_rows"] != derived_rows(bases):
        raise ValueError("registered predicate recomputation or multiplicity mismatch")
    return {"valid": True, "unique_physical_solve_count": len(rows),
            "derived_predicate_count": len(payload["derived_rows"])}
