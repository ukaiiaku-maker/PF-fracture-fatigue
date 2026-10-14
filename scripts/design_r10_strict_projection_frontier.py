#!/usr/bin/env python3
"""Project retained P25/P40 targets into class-parent strict EXP-floor rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    ExpFloorBounds, RenewalControls, exact_power_law_target_barrier,
    exact_target_instantaneous_rate, project_exp_floor,
)
from arrhenius_fracture.retained_material_r10 import (
    STRICT_FAMILY, load_parent_binding, load_strict_original_binding,
)


CLASSES={
    "Peak":"v913_paper_peak01_0242980_persistent_sites",
    "DBTT":"v913_paper_dbtt01_0202500_persistent_sites",
    "WeakT":"v913_paper_weakT01_0129902_persistent_sites",
}
TARGETS={
    "P25":{"M":2.5,"Klo":4.,"Khi":30.,"nK":19,"Kref":8.685353406927238,
           "gref":3.5753577430049013e-9},
    "P40":{"M":4.,"Klo":5.,"Khi":20.,"nK":17,"Kref":8.685353406927238,
           "gref":3.5753577430049013e-9},
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def at(surface,T):
    d=T-surface.Tref_K
    return surface.G00_eV+surface.gT_eV_per_K*d, surface.sigc0_Pa+surface.sT_Pa_per_K*d


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--original-registry",type=Path,required=True)
    p.add_argument("--parent-registry",type=Path,required=True);p.add_argument("--out-root",type=Path,required=True)
    a=p.parse_args(argv);a.out_root.mkdir(parents=True,exist_ok=True); rows=[]; diagnostics=[]
    with a.original_registry.open(newline="") as stream:
        original_rows={r["option_key"]:r for r in csv.DictReader(stream)}
    retained={name:load_parent_binding(a.parent_registry,f"{name}_TRANSFER_V1_RANK1" if name=="P25" else "P40_TRANSFER_CALIBRATED_GEN2") for name in TARGETS}
    for class_name,option in CLASSES.items():
        binding=load_strict_original_binding(a.original_registry,option); template=binding.manifest.cleavage
        for target_name,spec in TARGETS.items():
            donor=retained[target_name].manifest.cleavage
            controls=RenewalControls(hits=3.,tau_s=1e-6,event_length_m=5e-6,
                frequency_Hz=1000.,temperature_K=300.,radius_m=1e-6,n_phase=2048,stress_cap_Pa=30e9)
            K=np.geomspace(spec["Klo"],spec["Khi"],spec["nK"]); M=spec["M"]
            target_growth=spec["gref"]*(K/spec["Kref"])**M
            sigma=np.geomspace(.1*spec["Klo"]*1e6/np.sqrt(2*np.pi*1e-6),
                               spec["Khi"]*1e6/np.sqrt(2*np.pi*1e-6),129)
            rate=exact_target_instantaneous_rate(sigma,M=M,R=.1,
                K_ref_MPa_sqrt_m=spec["Kref"],g_ref_m_per_cycle=spec["gref"],controls=controls)
            barrier=exact_power_law_target_barrier(sigma,M=M,R=.1,
                K_ref_MPa_sqrt_m=spec["Kref"],g_ref_m_per_cycle=spec["gref"],
                attempt_frequency_s=template.attempt_frequency_s,controls=controls)
            d=template.Tref_K-300.
            start=[donor.G00_eV+template.gT_eV_per_K*d,
                   (donor.sigc0_Pa+template.sT_Pa_per_K*d)/1e9,
                   donor.alpha,donor.exponent,donor.floor_fraction]
            bounds=ExpFloorBounds(
                G00_eV=(max(.3,template.gT_eV_per_K*d+.25),max(6.,template.G00_eV*1.2)),
                sigc_GPa=(.2,40.),alpha=(.001,20.),exponent=(.25,12.),floor_fraction=(.001,.95))
            fit=project_exp_floor(sigma_grid_Pa=sigma,target_barrier_eV=barrier,
                target_rate_s=rate,K_grid_MPa_sqrt_m=K,target_growth=target_growth,
                target_slope=np.full_like(K,M),R=.1,template=template,controls=controls,
                bounds=bounds,weights={"barrier":.1,"derivative":.3,"rate":1.,
                    "growth":5.,"slope":5.,"parameter":.001},start=start)
            vector=np.asarray(fit["vector"],float); G300=vector[0]+template.gT_eV_per_K*(300-template.Tref_K)
            S300=vector[1]*1e9+template.sT_Pa_per_K*(300-template.Tref_K)
            G1200_parent,S1200_parent=at(template,1200.)
            gT=(G1200_parent-G300)/900.;sT=(S1200_parent-S300)/900.
            G00=G300-gT*(300-template.Tref_K);sigc0=(S300-sT*(300-template.Tref_K))/1e9
            cid=f"R10_{class_name.upper()}_{target_name}_THERMAL_ANCHOR_V1"
            complete=dict(original_rows[option]);complete.update({
                "candidate_id":cid,"cleave_G00_eV":G00,"cleave_gT_eV_per_K":gT,
                "cleave_sigc0_GPa":sigc0,"cleave_sT_GPa_per_K":sT/1e9,
                "cleave_exp_a":vector[2],"cleave_exp_n":vector[3],
                "cleave_floor_frac":vector[4],"physics__cleavage_hits":3.,
                "physics__cleavage_correlation_time_s":1e-6,
            })
            row_hash=canonical_hash(complete)
            lower=bounds.lower();upper=bounds.upper();margin=np.minimum((vector-lower)/(upper-lower),(upper-vector)/(upper-lower))
            changed={k:{"parent":original_rows[option].get(k),"offspring":complete[k]} for k in (
                "cleave_G00_eV","cleave_gT_eV_per_K","cleave_sigc0_GPa","cleave_sT_GPa_per_K",
                "cleave_exp_a","cleave_exp_n","cleave_floor_frac")}
            record={"candidate_id":cid,"parent_option_key":option,
                "parent_candidate_id":binding.optimized_row["candidate_id"],
                "parent_complete_row_sha256":binding.provenance["complete_row_sha256"],
                "family":STRICT_FAMILY,"target":target_name,"cleavage_hits":3.,"cleavage_tau_s":1e-6,
                "complete_row_sha256":row_hash,"changed_fields_json":json.dumps(changed,sort_keys=True),
                "complete_parameter_vector":json.dumps(complete,sort_keys=True),
                "projection_log_growth_RMS":fit["log_growth_RMS"],"projection_slope_RMS":fit["slope_RMS"],
                "projection_local_slope_min":float(np.min(fit["slope"])),
                "projection_local_slope_max":float(np.max(fit["slope"])),
                "projection_growth_width_decades":float(np.log10(fit["growth"][-1]/fit["growth"][0])),
                "minimum_relative_bound_margin":float(np.min(margin)),
                "interior":bool(np.min(margin)>1e-5),"thermal_anchor":"parent opening G0 and sigc at 1200 K"}
            rows.append(record);diagnostics.append({k:v for k,v in record.items() if k not in ("complete_parameter_vector","changed_fields_json")})
    fieldnames=list(rows[0]);out=a.out_root/"strict_projection_frontier.csv"
    with out.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
    decision={"schema":"r10.strict-projection-frontier/1","family":STRICT_FAMILY,
        "method":"existing cooperative inverse plus bounded project_exp_floor; no production-law correction",
        "protected_anchor":"each original class parent opening G0 and sigc at 1200 K",
        "improvement_objectives":TARGETS,"rows":diagnostics,
        "selection":"NO_AUTOMATIC_PROMOTION_WHEN_BOUND_ACTIVE; evolve representatives to measure state/fracture consequence",
        "source_hashes":{"original_registry":hashlib.sha256(a.original_registry.read_bytes()).hexdigest(),
                         "parent_registry":hashlib.sha256(a.parent_registry.read_bytes()).hexdigest()}}
    (a.out_root/"strict_projection_frontier_decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")
    print(json.dumps(decision,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
