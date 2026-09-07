#!/usr/bin/env python3
"""Source-resolved V3-closure cavity-traction convergence qualification."""
from __future__ import annotations
import hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from arrhenius_fracture.crack_void_mechanics_v5 import solve_crack_void_case
from arrhenius_fracture.finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES, canonical_hash

COUPLED=((32,12),(64,24),(128,48),(256,96),(512,192))
CROSSED=((128,48),(256,48),(512,48),(256,96),(256,192),(512,192))
CRACKED=((128,48),(256,96),(512,192))
PATH=((0.,0.),(5e-4,0.))

def clean(v):
    if isinstance(v,float) and not math.isfinite(v): return None
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    return v

def main(argv=None):
    out=Path((argv or sys.argv[1:])[0]); out.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip()
    cases=[]
    for family,pairs,cracked in (("cavity_only_coupled",COUPLED,False),("cavity_only_crossed",CROSSED,False),("crack_void_coupled",CRACKED,True)):
        for segments,layers in pairs:
            cfg={"family":family,"boundary_segments":segments,"radial_layers":layers,
                 "specimen_width_m":1e-3,"specimen_height_m":1e-3,"cavity_center_m":[7e-4,0.],
                 "cavity_radius_m":5e-5,"opening_m":4e-7,"material_E_Pa":210e9,"material_nu":.3,
                 "crack_path_m":list(PATH) if cracked else None,"residual_stiffness_kappa":1e-6 if cracked else 0.}
            kwargs={"boundary_segments":segments,"radial_layers":layers,"crack_enabled":cracked}
            if cracked: kwargs.update(crack_path_m=PATH,geometry_mode="V3_FIXED_LABORATORY_GEOMETRY")
            result=solve_crack_void_case(**kwargs); obs=result["observables"]; edge=obs["cavity_edge_traction_records"]
            realized={"crack_path_m":[obs["crack_root_m"],obs["crack_tip_m"]] if cracked else None,
              "cavity_center_m":obs["cavity_center_m"],"cavity_radius_m":5e-5,
              "polygon_intersection_m":obs["realized_polygon_intersection_m"],
              "ligament_m":obs["ligament_length_m"],"ligament_over_R":obs["ligament_over_radius"],
              "R_over_h_max":obs["R_over_h_cavity_max"],"ligament_over_h_max":obs["ligament_over_h_max"]}
            cases.append({"case_id":f"{family}:{segments}:{layers}","execution_id":f"traction:{family}:{segments}:{layers}:{head[:12]}",
              "executed_code_sha":head,"input_configuration":cfg,"input_hash":canonical_hash(cfg),
              "actual_realized_geometry":realized,"actual_geometry_fingerprint":canonical_hash(realized),
              "mesh":{"nodes":obs["mesh_nodes"],"elements":obs["mesh_elements"],"minimum_quality":obs["mesh_minimum_quality"],
                      "maximum_aspect_ratio":obs["mesh_maximum_aspect_ratio"],
                      "first_layer_normal_spacing_m":[x["first_layer_normal_spacing_m"] for x in edge]},
              "measurements":{k:obs[k] for k in ("assembled_cavity_node_residual_relative","cavity_traction_l2_dimensional_Pa_sqrt_m",
                "cavity_traction_normal_l2_dimensional_Pa_sqrt_m","cavity_traction_tangential_l2_dimensional_Pa_sqrt_m",
                "nominal_remote_stress_Pa","traction_diagnostic_cavity_perimeter_m","cavity_traction_l2_normalized",
                "cavity_traction_normal_l2_normalized","cavity_traction_tangential_l2_normalized","cavity_traction_resultant_normalized",
                "cavity_traction_moment_normalized","reaction_top_N_per_m","reaction_bottom_N_per_m","compliance_m2_per_N",
                "stored_energy_J_per_m","cavity_area_m2","cavity_perimeter_m","hoop_stress_concentration","crack_tip_sigma_yy_Pa")},
              "cavity_surface_tensor_probe_Pa":[x["adjacent_element_stress_tensor_Pa"] for x in edge],
              "edge_owner_certificate":all(x["adjacent_solid_element_count"]==1 for x in edge),
              "edge_records":edge,"support_audit":result["support_audit"],
              "traction_gate":obs["cavity_traction_l2_normalized"]<=SCIENTIFIC_ACCEPTANCE_TOLERANCES["cavity_traction_normalized"],
              "mesh_quality_gate":obs["mesh_minimum_quality"]>=SCIENTIFIC_ACCEPTANCE_TOLERANCES["mesh_minimum_quality"]})
    coupled=[r for r in cases if r["case_id"].startswith("cavity_only_coupled")]
    orders=[math.log(a["measurements"]["cavity_traction_l2_normalized"]/b["measurements"]["cavity_traction_l2_normalized"],2) for a,b in zip(coupled,coupled[1:])]
    manifest={"schema":"v12.voiding-v5-closure-traction/1","implementation_sha":head,"case_count":len(cases),
      "coupled_convergence_orders":orders,"fine_static_traction_recovery_qualified":all(
        next(r for r in cases if r["case_id"]==f"{family}:512:192")[gate]
        for family in ("cavity_only_coupled","crack_void_coupled") for gate in ("traction_gate","mesh_quality_gate")),
      "production_resolution_cavity_tensor_qualified":False,
      "production_classification":"FAIL_CLOSED_PENDING_RESOLUTION_TRANSFER"}
    payload=clean({"manifest":manifest,"rows":cases}); path=out/"traction_convergence.json"
    path.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
    (out/"sha256_manifest.json").write_text(json.dumps({path.name:hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+"\n")
    print(json.dumps(manifest,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
