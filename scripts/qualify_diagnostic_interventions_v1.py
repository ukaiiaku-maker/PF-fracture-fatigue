#!/usr/bin/env python3
"""Qualify orthogonal R06 interventions on a populated spatial checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType,SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.diagnostic_interventions_v1 import (
    apply_history_reset_reequilibrated,install_diagnostic_intervention,
)
from arrhenius_fracture.fem import assemble_mechanics,plane_strain_D,solve_dirichlet
from arrhenius_fracture.mesh import BoundaryData,rebuild_tri_mesh
from arrhenius_fracture.persistent_site_source_v10221 import _source_backstress
from arrhenius_fracture.signed_burgers_shared_v1025 import _active_K,_sync_active,_sync_wake
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
)
from arrhenius_fracture.unified_mpz import UnifiedMPZState
from arrhenius_fracture.tip_bulk_population_transfer_v1 import (
    conservative_tip_bulk_transfer,
)


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _generation(root):
    manifest=json.loads((root/"run_state_checkpoint.json").read_text())
    directory=root/"run_state_generations"/manifest["generation"]
    for name,expected in manifest["files"].items():
        if _sha(directory/name)!=expected: raise RuntimeError(f"checkpoint hash mismatch: {name}")
    return manifest,directory


def _state(directory,kernel,run_args,selection):
    kinetic=json.loads((directory/"kinetic.json").read_text())
    with np.load(directory/"state.npz",allow_pickle=False) as archive:
        vector=np.asarray(archive["kinetic_active_vector"],float)
    state=SimpleNamespace()
    for field in kinetic["fields"]:
        if field["owner"]!="mpz": continue
        value=vector[field["start"]:field["stop"]].reshape(field["shape"]).copy()
        setattr(state,field["name"],value)
    state.n_systems,state.n_bins=state.mobile_positive.shape
    state.wake_n_bins=state.wake_mobile_positive.shape[1]
    state.dx=float(kernel.active_x_m[1]-kernel.active_x_m[0])
    state.x=np.asarray(kernel.active_x_m,float).copy()
    source_wake=np.asarray(kernel.wake_x_m,float)
    state.wake_x=(source_wake.copy() if source_wake.size==state.wake_n_bins
                  else (np.arange(state.wake_n_bins)+.5)*state.dx)
    state.wake_dx=(float(state.wake_x[1]-state.wake_x[0]) if state.wake_n_bins>1 else state.dx)
    row=selection["exact_registry_row"]
    state.cfg=SimpleNamespace(
        mobile_shield_fraction=float(run_args["mpz_mobile_shield_fraction"]),
        blunting_length_m=float(run_args["mpz_blunting_length_m"]),
        taylor_stress_fraction=float(row["taylor_stress_fraction"]),
        forest_density_floor_m2=float(row["rho_forest_floor_m2"]),
        wake_shielding=True,
    )
    state.manifest=SimpleNamespace(c_blunt=float(row["c_blunt"]))
    mat=ElasticProperties()
    state._campaign_G_Pa=mat.G; state._campaign_b=mat.b
    state._campaign_backstress_scale=1.0
    state._persistent_r0_m=float(run_args["r_pz"]); state._persistent_b=mat.b
    state._signed_kernel=kernel
    state.local_slip_count=MethodType(UnifiedMPZState.local_slip_count,state)
    state.blunted_radius=MethodType(UnifiedMPZState.blunted_radius,state)
    for name,value in kinetic["ledgers"].items():
        if name.startswith("mpz."): setattr(state,name[4:],float(value))
    _sync_active(state); _sync_wake(state)
    return state,kinetic,vector


def _mechanics_reequilibrator(directory,outer):
    archive=np.load(directory/"state.npz",allow_pickle=False)
    mesh=rebuild_tri_mesh(archive["mesh_nodes"],archive["mesh_elems"])
    u=np.asarray(archive["displacement"],float).copy()
    ep=np.asarray(archive["ep_gp"],float); rho=np.asarray(archive["rho_gp"],float)
    damage=np.asarray(archive["damage"],float)
    y=mesh.nodes[:,1]; tol=max(np.ptp(mesh.nodes[:,0]),np.ptp(y))*1e-12
    top=np.flatnonzero(abs(y-y.max())<=tol); bot=np.flatnonzero(abs(y-y.min())<=tol)
    boundary=BoundaryData(top,bot,int(bot[np.argmin(mesh.nodes[bot,0])]),
                          int(bot[np.argmax(mesh.nodes[bot,0])]),np.asarray([],int))
    opening=float(outer["history"]["hist"]["Uapp"][-1]); mat=ElasticProperties(); D=plane_strain_D(mat)
    def solve():
        nonlocal u
        K,R,*_=assemble_mechanics(mesh,u,ep,rho,damage,D,mat)
        old=u.copy(); u,reaction=solve_dirichlet(K,R,u,boundary,opening/2,-opening/2)
        _K2,R2,*_=assemble_mechanics(mesh,u,ep,rho,damage,D,mat)
        fixed=np.zeros(mesh.ndof,bool); fixed[2*top+1]=True; fixed[2*bot+1]=True
        fixed[2*boundary.left_bot]=True; fixed[2*boundary.right_bot]=True
        residual=float(np.linalg.norm(R2[~fixed],ord=np.inf))
        return {"converged":bool(np.isfinite(residual) and residual<1e-3),
                "residual_norm_N_per_m":residual,
                "displacement_correction_norm_m":float(np.linalg.norm(u-old)),
                "reaction_N_per_m":float(reaction),"native_mechanics":True}
    return solve


def qualify(checkpoint,kernel_path,out):
    manifest,directory=_generation(checkpoint)
    run_root=checkpoint.parent
    run_args=json.loads((run_root/"run_args.json").read_text())
    selection=json.loads((run_root/"v10_2_22_parameter_selection.json").read_text())
    outer=json.loads((directory/"outer.json").read_text())
    kernel=ActiveOnlySigned2DShieldingKernelFamily.from_json(kernel_path)
    kernel.resolve(r_eff_over_r0=1.0,opening_strength_fraction=0.0,crack_extension_m=0.0)
    base,kinetic,vector=_state(directory,kernel,run_args,selection)
    kernel.resolve(
        r_eff_over_r0=base.blunted_radius(base._persistent_r0_m,base._persistent_b)
                         /base._persistent_r0_m,
        opening_strength_fraction=0.0,crack_extension_m=0.0,
    )
    baseline={
        "active_shielding_Pa_sqrt_m":_active_K(base),
        "tip_radius_m":base.blunted_radius(base._persistent_r0_m,base._persistent_b),
        "source_backstress_Pa":_source_backstress(base)[2].tolist(),
        "mobile_count":float(np.sum(base.mobile)),"retained_count":float(np.sum(base.retained)),
        "active_vector_nonzero_count":int(np.count_nonzero(vector)),
    }
    shield=copy.deepcopy(base); shield_receipt=install_diagnostic_intervention(shield,"retained_shielding_off")
    source=copy.deepcopy(base); source_receipt=install_diagnostic_intervention(source,"source_backstress_off")
    fixed=copy.deepcopy(base); fixed_receipt=install_diagnostic_intervention(fixed,"blunting_fixed")
    fixed.accumulated_slip_positive*=2.0; _sync_active(fixed)
    transport=copy.deepcopy(base); transport_receipt=install_diagnostic_intervention(transport,"transport_retention_off")
    reset=copy.deepcopy(base)
    reset_receipt=apply_history_reset_reequilibrated(
        reset,reequilibrate=_mechanics_reequilibrator(directory,outer),
        accumulated_work={"W_emit_J_per_m":float(outer["history"]["hist"]["W_emit"][-1])},
    )
    with np.load(directory/"state.npz",allow_pickle=False) as archive:
        transfer_mesh=rebuild_tri_mesh(archive["mesh_nodes"],archive["mesh_elems"])
    tip_transfer=np.asarray(base.retained_positive,float).ravel()
    requested=.1*float(np.sum(tip_transfer))
    _tip_after,_bulk_after,transfer_receipt=conservative_tip_bulk_transfer(
        tip_transfer,np.zeros(transfer_mesh.ne),transfer_mesh.area_e,
        direction="tip_to_bulk",requested_line_content=requested,
        tip_support_weights=np.ones_like(tip_transfer),
        bulk_support_weights=np.ones(transfer_mesh.ne),
    )
    ledger=kinetic["ledgers"]
    checks={
        "checkpoint_is_populated":baseline["active_vector_nonzero_count"]>0 and baseline["mobile_count"]+baseline["retained_count"]>0,
        "retained_shielding_removed":_active_K(shield)==0.0 and baseline["active_shielding_Pa_sqrt_m"]!=0.0,
        "shielding_population_preserved":shield_receipt["population_state_unchanged"],
        "source_backstress_removed":bool(np.all(_source_backstress(source)[2]==0.0)),
        "source_population_preserved":source_receipt["population_state_unchanged"],
        "fixed_blunting_holds_developed_radius":fixed.blunted_radius(fixed._persistent_r0_m,fixed._persistent_b)==fixed_receipt["fixed_blunting_radius_m"],
        "transport_retention_terms_were_active":float(ledger["mpz.gross_trapping_total"])>0.0 and float(ledger["mpz.gross_release_total"])>0.0 and float(ledger["mpz.mobile_distance_population_total_m"])>0.0,
        "transport_install_preserves_population":transport_receipt["population_state_unchanged"],
        "history_reset_removed_populated_state":reset_receipt["removed_population_total"]>0.0,
        "history_reset_recorded_work":reset_receipt["counterfactual_removed_work"]["W_emit_J_per_m"]>0.0,
        "history_reset_not_physical_recovery":reset_receipt["physical_loss_ledgers_unchanged"],
        "history_reset_native_reequilibrated":reset_receipt["native_reequilibration"]["converged"],
        "no_tip_bulk_transfer_added":all(r["tip_bulk_population_transfer_added"] is False for r in (shield_receipt,source_receipt,fixed_receipt,transport_receipt,reset_receipt)),
        "parameterized_transfer_is_conservative":abs(transfer_receipt["conservation_residual_line_content"])<1e-12,
        "parameterized_transfer_has_no_default_coefficient":transfer_receipt["physical_transfer_coefficient"] is None,
    }
    report={
        "schema":"r06-native-diagnostic-intervention-qualification/1",
        "status":"PASS" if all(checks.values()) else "FAIL",
        "source_commit":subprocess.check_output(("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip(),
        "production_execution_authorized":False,
        "checkpoint":{"path":str(checkpoint.resolve()),"generation":manifest["generation"],
                      "manifest_sha256":_sha(checkpoint/"run_state_checkpoint.json")},
        "kernel":{"path":str(kernel_path.resolve()),"sha256":_sha(kernel_path)},
        "baseline":baseline,"checks":checks,
        "receipts":{"retained_shielding_off":shield_receipt,
                    "source_backstress_off":source_receipt,"blunting_fixed":fixed_receipt,
                    "transport_retention_off":transport_receipt,
                    "history_reset_reequilibrated":reset_receipt},
        "parameterized_transfer":transfer_receipt,
        "transfer_status":"GENERIC_CONSERVATIVE_OPERATOR_QUALIFIED_PHYSICAL_COEFFICIENT_UNAVAILABLE",
        "scope":"populated accepted original spatial PF state; orthogonal diagnostic consumers; no production calibration or tip/bulk transfer",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"qualification_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--kernel-family",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv); r=qualify(a.checkpoint.resolve(),a.kernel_family.resolve(),a.out.resolve())
    print(json.dumps(r,indent=2,sort_keys=True)); return 0 if r["status"]=="PASS" else 1


if __name__=="__main__": raise SystemExit(main())
