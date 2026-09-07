"""Frozen unique complete-static matrix and directly recomputed predicates."""
import math
from dataclasses import asdict, is_dataclass
import numpy as np

from .closure_static_evidence import (
    configuration, recompute_measurements, fixed_tip_probe, source_fingerprints,
    validate_solver_capture,
)
from .finalization_v3_schema import canonical_hash, SCIENTIFIC_ACCEPTANCE_TOLERANCES as LIMITS

SCHEMA = "v12.voiding-v5-closure-mechanics/1"
MESHES = ((128, 48), (256, 96))
REGISTRY = {}
GROUPS = {}


def canonical_data(value):
    if is_dataclass(value): return canonical_data(asdict(value))
    if isinstance(value, np.ndarray): return canonical_data(value.tolist())
    if isinstance(value, np.generic): return canonical_data(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "-infinity" if value < 0 else "NaN"
    if isinstance(value, dict): return {str(k): canonical_data(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [canonical_data(v) for v in value]
    return value


def register(label, cfg):
    # The physical configuration, not a family label, controls uniqueness.
    key = canonical_hash(cfg)
    REGISTRY.setdefault(key, cfg)
    GROUPS[label] = key


for n, layers in MESHES:
    mesh_id = f"{n}:{layers}"
    for radius in (4e-5, 5e-5, 6e-5):
        for ligament_ratio in (1., 2., 3.):
            cfg = dict(configuration(True, n, layers), cavity_radius_m=radius,
                       cavity_center_m=[round(5e-4+radius*(1+ligament_ratio), 15), 0.])
            label = f"matrix:{radius}:{ligament_ratio}:{mesh_id}"
            register(label, cfg)
            register(label+":matched_crack_only", dict(cfg, cavity_enabled=False))
    cfg = configuration(True, n, layers)
    register("centered:"+mesh_id, cfg)
    for offset in (-4e-5, 4e-5):
        register(f"offset:{offset}:{mesh_id}", dict(cfg, cavity_center_m=[7e-4, offset]))
    for distance in (1.2e-3, 1.5e-3):
        far = dict(cfg, specimen_width_m=3e-3, specimen_height_m=2e-3,
                   cavity_center_m=[distance, 0.])
        register(f"far:{distance}:{mesh_id}", far)
        register(f"far:{distance}:{mesh_id}:matched_crack_only", dict(far, cavity_enabled=False))
    for epsilon in (2.5e-6, 1.25e-6):
        for sign in (-1, 1):
            register(f"crack_perturb:{epsilon}:{sign}:{mesh_id}",
                     dict(cfg, crack_path_m=[[0., 0.], [5e-4+sign*epsilon, 0.]]))
            register(f"radius_perturb:{epsilon}:{sign}:{mesh_id}",
                     dict(cfg, cavity_radius_m=5e-5+sign*epsilon))


def measurements(raw, cfg):
    from .mesh import rebuild_tri_mesh
    from .crack_network_v11 import CrackNetworkState
    from .mechanically_separating_sharp_wake_v12 import independent_intact_path_certificate
    from .voiding_production_v5 import _convex_polygon_interior_overlaps_triangle
    tri = raw["nodes"][raw["elements"]]
    side = np.linalg.norm(tri[:, [1,2,0]]-tri[:, [0,1,2]], axis=2)
    v, w = tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]
    area = np.abs(v[:,0]*w[:,1]-v[:,1]*w[:,0])/2
    quality = 4*np.sqrt(3)*area/np.sum(side**2, axis=1)
    top = float(np.sum(raw["assembled_residual"][2*raw["top_nodes"]+1]))
    bottom = float(np.sum(raw["assembled_residual"][2*raw["bottom_nodes"]+1]))
    energy = float(.5*raw["displacement"] @ raw["assembled_residual"])
    result = {"reaction": top, "compliance": cfg["opening_m"]/abs(top), "energy": energy,
              "mesh_quality": float(quality.min()),
              "reaction_balance": abs(top+bottom)/abs(top),
              "energy_identity": abs(energy-.5*cfg["opening_m"]*top)/abs(energy),
              "free_residual_relative": float(np.linalg.norm(raw["assembled_residual"][~raw["prescribed_dofs"]]))/abs(top),
              "cavity_enabled": cfg["cavity_enabled"], "cavity_fields": None}
    selected = np.flatnonzero(raw["support_mask"])
    network = CrackNetworkState.one_tip(tuple(map(tuple, cfg["crack_path_m"])))
    result["independent_intact_path_certificate"] = independent_intact_path_certificate(
        rebuild_tri_mesh(raw["nodes"], raw["elements"]), network, selected)
    result["fixed_crack_vertex_error_m"] = max(float(np.linalg.norm(raw["nodes"]-point, axis=1).min())
        for point in cfg["crack_path_m"])
    # A probe rejection is a measured failure, not permission to move the probe.
    try:
        result["fixed_tip_probe"] = fixed_tip_probe(raw, cfg)
    except ValueError as error:
        result["fixed_tip_probe"] = {"failure": str(error)}
    if cfg["cavity_enabled"]:
        try:
            result["cavity_fields"] = recompute_measurements(raw, cfg)
        except ValueError as error:
            result["cavity_fields"] = {"failure": str(error)}
        nodes = raw["nodes"][np.unique(raw["cavity_edges"])]
        theta = np.arctan2(nodes[:,1]-cfg["cavity_center_m"][1], nodes[:,0]-cfg["cavity_center_m"][0])
        nodes = nodes[np.argsort(theta)]
        result["support_cavity_polygon_overlap_element_ids"] = [int(e) for e in selected
            if _convex_polygon_interior_overlaps_triangle(nodes, tri[e])]
        polygon_area = abs(.5*np.sum(nodes[:,0]*np.roll(nodes[:,1],-1)-nodes[:,1]*np.roll(nodes[:,0],-1)))
        perimeter = float(np.linalg.norm(nodes-np.roll(nodes,-1,axis=0), axis=1).sum())
        result.update(cavity_area_relative=abs(polygon_area-math.pi*cfg["cavity_radius_m"]**2)/(math.pi*cfg["cavity_radius_m"]**2),
                      cavity_perimeter_relative=abs(perimeter-2*math.pi*cfg["cavity_radius_m"])/(2*math.pi*cfg["cavity_radius_m"]))
    return canonical_data(result)


def predicates(bases):
    rows = []
    def add(name, sources, inputs, passed):
        rows.append({"predicate_id": name, "source_row_ids": sources,
                     "predicate_inputs": inputs, "predicate_result": bool(passed)})
    def get(label):
        return bases[GROUPS[label]]["measurements"]
    for key, row in bases.items():
        m = row["measurements"]
        if "failure" in m:
            add(key+"/static_solve", [key], m, False); continue
        gates = {"free_residual_relative": LIMITS["free_residual_relative"],
                 "reaction_balance": LIMITS["reaction_balance_relative"],
                 "energy_identity": LIMITS["energy_reaction_identity_relative"]}
        if m["cavity_enabled"]:
            gates.update(cavity_area_relative=LIMITS["cavity_area_relative"], cavity_perimeter_relative=LIMITS["cavity_perimeter_relative"])
        for field, limit in gates.items():
            add(key+"/"+field, [key], {"value": m[field], "limit": limit}, m[field] <= limit)
        add(key+"/quality", [key], {"value": m["mesh_quality"], "limit": LIMITS["mesh_minimum_quality"]}, m["mesh_quality"] >= LIMITS["mesh_minimum_quality"])
        certificate = m["independent_intact_path_certificate"]
        add(key+"/no_intact_cross_crack_path", [key], certificate,
            not certificate["intact_cross_graph_path_exists"]
            and not certificate["insufficient_seed_segment_ids"])
        add(key+"/fixed_crack_vertices", [key], {"error_m": m["fixed_crack_vertex_error_m"],
            "limit": LIMITS["requested_realized_geometry_abs_m"]},
            m["fixed_crack_vertex_error_m"] <= LIMITS["requested_realized_geometry_abs_m"])
        if m["cavity_enabled"]:
            fields = m["cavity_fields"]
            add(key+"/traction", [key], {"value": fields.get("normalized_traction"), "limit": LIMITS["cavity_traction_normalized"]},
                "normalized_traction" in fields and fields["normalized_traction"] <= LIMITS["cavity_traction_normalized"])
            add(key+"/unique_live_edge_owner", [key], {"valid": fields.get("edge_owner_valid", False)}, fields.get("edge_owner_valid", False))
            overlap = m["support_cavity_polygon_overlap_element_ids"]
            add(key+"/no_support_cavity_interior_overlap", [key], {"overlap_element_ids": overlap}, not overlap)
    derivatives = {}
    for n, layers in MESHES:
        mesh_id = f"{n}:{layers}"; base = get("centered:"+mesh_id)
        for distance in (1.2e-3, 1.5e-3):
            label = f"far:{distance}:{mesh_id}"; m, control = get(label), get(label+":matched_crack_only")
            errors = {k: abs(m[k]-control[k])/abs(control[k]) for k in ("reaction", "compliance", "energy")}
            add(label+"/global_response", [GROUPS[label], GROUPS[label+":matched_crack_only"]],
                {"errors": errors, "limit": LIMITS["far_void_relative"]}, max(errors.values()) <= LIMITS["far_void_relative"])
        positive, negative = get(f"offset:{4e-5}:{mesh_id}"), get(f"offset:{-4e-5}:{mesh_id}")
        errors = {k: abs(positive[k]-negative[k])/max(abs(positive[k]),abs(negative[k])) for k in ("reaction", "compliance")}
        add("fixed_crack_mirror_offsets:"+mesh_id,
            [GROUPS[f"offset:{offset}:{mesh_id}"] for offset in (-4e-5,4e-5)],
            {"errors": errors, "limit": LIMITS["offset_reaction_relative"]}, max(errors.values()) <= LIMITS["offset_reaction_relative"])
        for family in ("crack_perturb", "radius_perturb"):
            for epsilon in (2.5e-6, 1.25e-6):
                labels = [f"{family}:{epsilon}:{sign}:{mesh_id}" for sign in (-1,1)]
                low, high = map(get, labels)
                dU = (high["energy"]-low["energy"])/(2*epsilon)
                dC = (high["compliance"]-low["compliance"])/(2*epsilon)
                energy_derivative = -dU
                compliance_derivative = .5*(4e-7)**2*dC/base["compliance"]**2
                relative = abs(energy_derivative-compliance_derivative)/max(abs(energy_derivative),abs(compliance_derivative),1e-300)
                name = f"{family}:{epsilon}:{mesh_id}"
                derivatives[name] = (energy_derivative, [GROUPS[label] for label in labels])
                add(name+"/energy_compliance", derivatives[name][1]+[GROUPS["centered:"+mesh_id]],
                    {"minus_dU_dp": energy_derivative, "compliance_derivative": compliance_derivative,
                     "parameter": "crack_length_m" if family == "crack_perturb" else "cavity_radius_m",
                     "relative_error": relative, "limit": LIMITS["derivative_energy_compliance_relative"]}, relative <= LIMITS["derivative_energy_compliance_relative"])
            large, small = (derivatives[f"{family}:{eps}:{mesh_id}"] for eps in (2.5e-6,1.25e-6))
            error = abs(large[0]-small[0])/max(abs(small[0]),1e-300)
            add(f"{family}:{mesh_id}/perturbation_convergence", large[1]+small[1],
                {"relative_error": error, "limit": LIMITS["derivative_perturbation_relative"]}, error <= LIMITS["derivative_perturbation_relative"])
    # Same physical input across both mesh levels; no variable-length edge
    # array comparison or geometry relocation is used as a field comparator.
    coarse_suffix, fine_suffix = (f"{n}:{r}" for n,r in MESHES)
    for label, coarse_key in GROUPS.items():
        if not label.endswith(coarse_suffix): continue
        peer_label = label[:-len(coarse_suffix)]+fine_suffix
        if peer_label not in GROUPS: continue
        fine_key = GROUPS[peer_label]; coarse, fine = bases[coarse_key]["measurements"], bases[fine_key]["measurements"]
        errors = {field: abs(coarse[field]-fine[field])/max(abs(fine[field]),1e-300)
                  for field in ("reaction", "energy", "compliance")}
        add(label+"/mesh_global", [coarse_key,fine_key], {"errors": errors,
            "limit": LIMITS["static_mesh_reaction_relative"]}, max(errors.values()) <= LIMITS["static_mesh_reaction_relative"])
        if "tensor_Pa" in coarse["fixed_tip_probe"] and "tensor_Pa" in fine["fixed_tip_probe"]:
            a,b = np.asarray(coarse["fixed_tip_probe"]["tensor_Pa"]), np.asarray(fine["fixed_tip_probe"]["tensor_Pa"])
            error = float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
            add(label+"/mesh_tip_tensor", [coarse_key,fine_key], {"relative_error": error,
                "limit": LIMITS["tensor_probe_relative"]}, error <= LIMITS["tensor_probe_relative"])
        else:
            add(label+"/mesh_tip_tensor", [coarse_key,fine_key], {"failure": "FIXED_PROBE_UNAVAILABLE"}, False)
        if coarse["cavity_fields"] and "fixed_arc_probes" in coarse["cavity_fields"] and "fixed_arc_probes" in fine["cavity_fields"]:
            errors = [float(np.linalg.norm(np.asarray(a["tensor_Pa"])-b["tensor_Pa"])/max(np.linalg.norm(b["tensor_Pa"]),1e-300))
                for a,b in zip(coarse["cavity_fields"]["fixed_arc_probes"],fine["cavity_fields"]["fixed_arc_probes"])]
            add(label+"/mesh_cavity_tensor", [coarse_key,fine_key], {"relative_errors": errors,
                "limit": LIMITS["tensor_probe_relative"]}, max(errors) <= LIMITS["tensor_probe_relative"])
    return rows


def validate(payload, sources, *, executed_code_sha):
    if payload["schema"] != SCHEMA or payload["executed_code_sha"] != executed_code_sha:
        raise ValueError("mechanics schema/source identity")
    rows = payload["base_rows"]
    if len(rows) != len(REGISTRY) or {r["case_id"] for r in rows} != set(REGISTRY):
        raise ValueError("unique mechanics registry mismatch")
    bases = {}
    for row in rows:
        key = row["case_id"]; cfg = REGISTRY[key]
        if row["input_configuration"] != cfg or row["input_hash"] != key:
            raise ValueError("mechanics input identity")
        if row["executed_code_sha"] != executed_code_sha or row["execution_id"] != key+":"+executed_code_sha:
            raise ValueError("mechanics execution identity")
        raw = sources[key]
        if source_fingerprints(raw) != row["source_fingerprints"]:
            raise ValueError("mechanics raw source binding")
        validate_solver_capture(raw, cfg)
        if canonical_hash(measurements(raw, cfg)) != canonical_hash(row["measurements"]):
            raise ValueError("mechanics measurement recomputation")
        bases[key] = row
    if payload["derived_rows"] != predicates(bases):
        raise ValueError("mechanics predicate recomputation")
    return {"valid": True, "unique_physical_solves": len(rows), "derived_predicates": len(payload["derived_rows"])}
