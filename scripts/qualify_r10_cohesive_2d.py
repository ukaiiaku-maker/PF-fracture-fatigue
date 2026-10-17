#!/usr/bin/env python3
"""Small production-assembly qualification for one exact cohesive material."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammainc
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive import CohesiveElement, CohesiveNetwork
from arrhenius_fracture.cohesive_arrhenius import COHESIVE_KINETIC_CZM
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.fem import assemble_mechanics, plane_strain_D
from arrhenius_fracture.mesh import rebuild_tri_mesh
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding


def specimen(thermal):
    L = 1.0e-6; H = 2.0e-6
    nodes = np.array([[0,0],[L,0],[0,H],[L,H],
                      [0,0],[L,0],[0,-H],[L,-H]], float)
    elems = np.array([[0,1,2],[1,3,2],[4,6,5],[5,6,7]], int)
    mesh = rebuild_tri_mesh(nodes, elems)
    elem = CohesiveElement((0,1),(4,5),np.array([0.,1.]),np.array([1.,0.]),
                           L, thermal)
    return mesh, CohesiveNetwork([elem], contact_penalty_Pa_per_m=1e23), elem


def equilibrium(mesh, network, imposed_jump, u0=None):
    mat = ElasticProperties(); D = plane_strain_D(mat)
    u = np.zeros(mesh.ndof) if u0 is None else np.asarray(u0, float).copy()
    fixed = np.r_[2*np.arange(mesh.nn), [2*2+1,2*3+1,2*6+1,2*7+1]]
    fixed = np.unique(fixed)
    values = np.zeros_like(fixed, dtype=float)
    prescribed = {2*2+1: imposed_jump/2, 2*3+1: imposed_jump/2,
                  2*6+1: -imposed_jump/2, 2*7+1: -imposed_jump/2}
    for i,dof in enumerate(fixed): values[i] = prescribed.get(int(dof), 0.)
    u[fixed] = values
    free = np.setdiff1d(np.arange(mesh.ndof), fixed)
    converged = False
    for iteration in range(80):
        K,R,*_ = assemble_mechanics(mesh,u,np.zeros((3,mesh.ne)),np.zeros(mesh.ne),
                                    np.zeros(mesh.nn),D,mat,cohesive_network=network)
        norm = float(np.linalg.norm(R[free]))
        if norm <= max(1e-8, 1e-10*max(np.linalg.norm(R[fixed]),1.0)):
            converged = True; break
        du = spsolve(K[free][:,free], -R[free])
        # Damping keeps the local solve on the connected equilibrium branch.
        scale = min(1.0, 0.25*network.elements[0].thermal.delta0_m/
                    max(float(np.max(np.abs(du))), 1e-30))
        u[free] += scale*du; u[fixed] = values
    if not converged: raise RuntimeError("cohesive qualification Newton solve failed")
    K,R,*_ = assemble_mechanics(mesh,u,np.zeros((3,mesh.ne)),np.zeros(mesh.ne),
                                np.zeros(mesh.nn),D,mat,cohesive_network=network)
    diag = dict(network.elements[0].metadata)
    diag.update({"imposed_jump_m": imposed_jump, "newton_iterations": iteration+1,
                 "free_residual_N_per_m": float(np.linalg.norm(R[free])),
                 "top_reaction_N_per_m": float(R[2*2+1]+R[2*3+1])})
    return u, diag


def atomic_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n"); tmp.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--candidate-id",required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--temperature-K",type=float,default=300.); a=p.parse_args(argv)
    a.out.mkdir(parents=True,exist_ok=True)
    binding=load_cohesive_binding(a.registry,a.candidate_id,execution_identity=COHESIVE_KINETIC_CZM)
    barrier=binding.manifest.cleavage; thermal=barrier.thermal
    mesh,network,elem=specimen(thermal); elem.temperature_K=a.temperature_K
    rows=[]; u=None
    # Load and unload below the peak before accepting rupture.
    for factor in (0.,0.15,0.35,0.60,0.35,0.60):
        u,d=equilibrium(mesh,network,factor*thermal.delta0_m,u); rows.append({"stage":"INTACT",**d})
    traction=max(float(elem.metadata["normal_traction_Pa"]),0.)
    raw=float(barrier.rate(traction,a.temperature_K))
    inherited=float(gammainc(binding.reduced_config.cleavage_hits,
                              raw*binding.reduced_config.cleavage_tau_s)
                    /binding.reduced_config.cleavage_tau_s)
    threshold=0.5; dwell=threshold/max(inherited,1e-300)
    elem.accept_rupture(1)
    u,d=equilibrium(mesh,network,0.60*thermal.delta0_m,u)
    rows.append({"stage":"ACCEPTED_KINETIC_EVENT_REEQUILIBRATED",**d})
    u,d=equilibrium(mesh,network,-0.05*thermal.delta0_m,u)
    rows.append({"stage":"BROKEN_CONTACT",**d})
    checkpoint={"schema":"r10.cohesive-kinetic-czm-checkpoint/1",
        "candidate_id":a.candidate_id,"family":binding.provenance["family"],
        "execution_identity":COHESIVE_KINETIC_CZM,"rupture_state":elem.rupture_state,
        "accepted_event_index":elem.accepted_event_index,"u":u.tolist(),
        "temperature_K":a.temperature_K,"hazard_threshold_action":threshold,
        "hazard_action_accepted":threshold,"pending_action":0.0}
    atomic_json(a.out/"restart.json",checkpoint)
    mesh2,network2,elem2=specimen(thermal); elem2.temperature_K=a.temperature_K
    elem2.rupture_state=checkpoint["rupture_state"]; elem2.accepted_event_index=checkpoint["accepted_event_index"]
    u2,d2=equilibrium(mesh2,network2,-0.05*thermal.delta0_m,np.asarray(checkpoint["u"]))
    restart_error=float(np.max(np.abs(u2-u)))
    mat=ElasticProperties();D=plane_strain_D(mat)
    _,_,sigma_gp,seq_gp,s1_gp,psi_gp=assemble_mechanics(
        mesh2,u2,np.zeros((3,mesh2.ne)),np.zeros(mesh2.ne),np.zeros(mesh2.nn),D,mat,
        cohesive_network=network2)
    np.savez_compressed(a.out/"spatial_fields.npz",nodes_m=mesh2.nodes,elems=mesh2.elems,
        displacement_m=u2,bulk_sigma_gp_Pa=sigma_gp,bulk_von_mises_gp_Pa=seq_gp,
        bulk_sigma1_gp_Pa=s1_gp,bulk_elastic_energy_gp_J_m3=psi_gp,
        cohesive_normal_jump_m=np.array([d2["normal_jump_m"]]),
        cohesive_normal_traction_Pa=np.array([d2["normal_traction_Pa"]]),
        cohesive_broken=np.array([elem2.broken],dtype=bool))
    summary={"schema":"r10.cohesive-kinetic-czm-production-qualification/1",
        "status":"QUALIFIED" if restart_error<=1e-18 and elem2.broken and d2["contact_active"] else "FAILED",
        "candidate_id":a.candidate_id,"family":binding.provenance["family"],
        "execution_identity":COHESIVE_KINETIC_CZM,
        "same_complete_material_sha256":binding.provenance["complete_row_sha256"],
        "barrier_only_comparison":False,"actual_nonlinear_interface_assembly":True,
        "scope":"SMALL_PRODUCTION_ASSEMBLY_QUALIFICATION_NOT_SCIENTIFIC_TRANSFER",
        "retained_dislocation_state_exercised":False,
        "accepted_event_owned_rupture":True,"deterministic_failure_rule":False,
        "cohesive_surface_energy_charged_once":True,"contact_does_not_heal":True,
        "raw_elementary_rate_s-1":raw,"inherited_cooperative_rate_s-1":inherited,
        "dwell_to_accepted_event_s":dwell,"restart_displacement_linf_m":restart_error,
        "history":rows,"restart_contact":d2,"binding":dict(binding.provenance)}
    atomic_json(a.out/"qualification.json",summary)
    with (a.out/"traction_separation_history.csv").open("w",newline="") as stream:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2,sort_keys=True,default=str)); return 0 if summary["status"]=="QUALIFIED" else 2

if __name__=="__main__": raise SystemExit(main())
