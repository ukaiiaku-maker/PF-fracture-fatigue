#!/usr/bin/env python3
"""Fail-closed verifier for the permanent v9.14 candidate atlas."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARAMS=["cleave_G00_eV","cleave_sigc0_GPa","cleave_exp_a","cleave_exp_n","cleave_floor_frac","emit_G00_eV","emit_sigc0_GPa","emit_exp_a","emit_exp_n","emit_floor_frac","peierls_H0_eV","peierls_activation_entropy_kB","peierls_exp_a","peierls_exp_n","taylor_H0_eV","taylor_activation_entropy_kB","taylor_exp_a","taylor_exp_n","rho_source0_m2","taylor_corr_rho_c_m2","taylor_corr_scale","c_blunt"]

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--atlas",type=Path,required=True);ap.add_argument("--prior-root",type=Path,required=True);a=ap.parse_args();d=a.atlas
    expected={"master_candidate_registry.parquet":1572864,"fatigue_analytical_points.parquet":25165824,"fatigue_candidate_properties.parquet":1572864}
    required=[*expected,"fatigue_window_catalog.parquet","candidate_property_observations.parquet","n64_candidate_queue.csv","n64_round01_candidate_ids.csv","n64_round01_cases.csv","n64_candidate_results.parquet","n128_candidate_results.parquet","parameter_space_clusters.parquet","property_space_clusters.parquet","candidate_atlas_coverage.csv","candidate_atlas_coverage_report.md","candidate_atlas_schema.json","candidate_generation_provenance.json","joint_candidate_property_atlas_schema.json","candidate_atlas_finalization.json","m_vs_C.png","m_vs_window_width.png","m_vs_three_hit_amplification.png"]
    require(all((d/x).is_file() and (d/x).stat().st_size>0 for x in required),"required atlas outputs are missing or empty")
    for name,n in expected.items():require(pq.ParquetFile(d/name).metadata.num_rows==n,f"wrong row count: {name}")
    require(pq.ParquetFile(d/"fatigue_window_catalog.parquet").metadata.num_rows>0,"no analytical windows")
    identity=pq.read_table(d/"master_candidate_registry.parquet",columns=["candidate_id","candidate_sha256","domain_label",*PARAMS]).to_pandas()
    require(identity.candidate_id.is_unique,"candidate IDs are not unique");require(identity.candidate_sha256.is_unique,"candidate hashes are not unique")
    require(identity.candidate_id.iloc[0]=="V914_ATLAS_000000001" and identity.candidate_id.iloc[-1]=="V914_ATLAS_001572864","stable ID bounds wrong")
    specs=[("historical_domain",0,524288),("cleavage_extension",1048576,131072)]
    for name,start,n in specs:
        old=pq.read_table(a.prior_root/name/"large_search_candidates.parquet",columns=PARAMS).slice(0,n).to_pandas().to_numpy()
        new=identity.iloc[start:start+n][PARAMS].to_numpy()
        require(np.array_equal(old,new,equal_nan=True),f"Sobol prefix mismatch: {name}")
    queue=pd.read_csv(d/"n64_candidate_queue.csv")
    require(len(queue)==250 and queue.candidate_id.is_unique,"n64 queue must contain 250 unique candidates")
    require(set(queue.candidate_id)<=set(identity.candidate_id),"queue contains candidates outside registry")
    require(set(queue.n64_launch_status)=={"QUEUED_NOT_LAUNCHED"},"analytical finalizer must not mark queue launched")
    cases=pd.read_csv(d/"n64_round01_cases.csv");require(len(cases)==75 and cases.candidate_id.nunique()==25,"round 1 must be 25 candidates x 3 loads")
    require(set(cases.DeltaK_MPa_sqrt_m)=={10.8,16.2,21.6},"round 1 descriptor loads changed")
    roles=queue.atlas_sampling_role.value_counts().to_dict();require(roles=={"realistic_paris_exploitation":100,"distinct_low_m_basin":50,"property_space_coverage":50,"analytical_uncertainty":25,"maximin_global_exploration":25},f"queue role mismatch: {roles}")
    require(pq.ParquetFile(d/"parameter_space_clusters.parquet").metadata.num_rows==1572864,"parameter clustering incomplete")
    require(pq.ParquetFile(d/"property_space_clusters.parquet").metadata.num_rows==1572864,"property clustering incomplete")
    print(json.dumps({"status":"PASS","atlas_candidates":1572864,"analytical_points":25165824,"windows":pq.ParquetFile(d/"fatigue_window_catalog.parquet").metadata.num_rows,"queue_roles":roles,"sobol_prefixes_exact":True},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
