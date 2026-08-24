#!/usr/bin/env python3
"""Build the immutable, extensible v9.14 multiphysics candidate atlas."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.audit_v914_historical_analytical_model import COORDINATES
from scripts.search_v914_large_analytical_paris import BOUNDS,transform,cleavage_extension_bounds,evaluate

U=[f"u__{x}" for x in COORDINATES]
RESERVED_FRACTURE=["fracture_Kinit_MPa_sqrt_m","fracture_KJ_MPa_sqrt_m","fracture_class_probability_peakT","fracture_class_probability_DBTT","fracture_class_probability_weakT","fracture_class_probability_ceramic"]
RESERVED_RCURVE=["Rcurve_K_25um","Rcurve_K_50um","Rcurve_K_100um","Rcurve_K_250um","Rcurve_K_500um","Rcurve_K_1000um","Rcurve_initial_dK_da","Rcurve_max_dK_da","Rcurve_total_rise_100um","Rcurve_total_rise_1mm","Rcurve_plateau_K","Rcurve_curvature","Rcurve_characteristic_extension_um"]
RESERVED_MECHANISM=["mechanism_plastic_activity","mechanism_source_activity","mechanism_mobile_content","mechanism_retained_content","mechanism_shielding","mechanism_blunting","mechanism_backstress","mechanism_fracture_hazard"]

def sha(row,domain,index):
    payload=domain+"|"+str(index)+"|"+"|".join(float(row[x]).hex() for x in COORDINATES)
    return hashlib.sha256(payload.encode()).hexdigest()

def windows(loads,y,feasible,cfg):
    found=[]; wid=0
    for i in range(len(loads)):
        if not feasible[i]:continue
        for j in range(i+int(cfg["minimum_Paris_points"]),len(loads)+1):
            if not np.all(feasible[i:j]):break
            span=loads[j-1]/loads[i]-1
            if span<float(cfg["minimum_relative_DeltaK_span"]):continue
            x=np.log10(loads[i:j]);z=y[i:j];m,b=np.polyfit(x,z,1);fit=b+m*x;ss=np.sum((z-z.mean())**2);r2=1-np.sum((z-fit)**2)/ss if ss>0 else 0
            if r2<float(cfg["minimum_fit_R2"]):continue
            wid+=1;found.append({"window_id":wid,"DeltaK_min":loads[i],"DeltaK_max":loads[j-1],"number_points":j-i,"m":m,"C":10**b,"log10_C":b,"R2":r2,"residual_rmse_log10":np.sqrt(np.mean((z-fit)**2)),"DeltaK_span_fraction":span,"da_dN_span_decades":z.max()-z.min(),"predicted_rate_min":z.min(),"predicted_rate_max":z.max(),"sustained_probability":np.mean(feasible[i:j])})
    # Keep non-dominated windows: no wider window with equal-or-better residual.
    return [w for w in found if not any(v["DeltaK_min"]<=w["DeltaK_min"] and v["DeltaK_max"]>=w["DeltaK_max"] and v["residual_rmse_log10"]<=w["residual_rmse_log10"] and v is not w for v in found)]

def write_batch(writer,path,frame):
    table=pa.Table.from_pandas(frame,preserve_index=False)
    if writer is None:writer=pq.ParquetWriter(path,table.schema,compression="zstd")
    writer.write_table(table);return writer

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--target-config",type=Path,required=True);ap.add_argument("--physics",type=Path,required=True);ap.add_argument("--calibration-points",type=Path,required=True);ap.add_argument("--cleavage-shell",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--historical-power",type=int,default=20);ap.add_argument("--extension-power",type=int,default=19);ap.add_argument("--chunk-size",type=int,default=32768);ap.add_argument("--creation-commit",required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.target_config.read_text());phys=json.loads(a.physics.read_text())["common_physics"];cal=pd.read_csv(a.calibration_points);use=cal.developed_point_usable.astype(bool)&(pd.to_numeric(cal.developed_da_dN_m_per_cycle,errors="coerce")>0);delta=np.log10(cal.loc[use,"developed_da_dN_m_per_cycle"])-np.log10(cal.loc[use,"analytical_Gamma_3hit_s"].clip(lower=1e-300));offset=float(delta.mean());unc=float(delta.std(ddof=1));loads=np.geomspace(8.,32.,16)
    specs=[("HISTORICAL_DOMAIN",2**a.historical_power,1914,BOUNDS,0),("CLEAVAGE_EXTENSION",2**a.extension_power,2914,cleavage_extension_bounds(a.cleavage_shell),2**a.historical_power)]
    master_w=points_w=window_w=props_w=None;csv_path=a.out/"master_candidate_registry.csv";csv_stream=csv_path.open("w",newline="");csv_writer=None
    counts={}; legacy_checks={}
    for domain,count,seed,bounds,offset_id in specs:
        sobol=qmc.Sobol(d=22,scramble=True,seed=seed).random_base2(int(math.log2(count)));counts[domain]=count
        for start in range(0,count,a.chunk_size):
            end=min(start+a.chunk_size,count);frame=transform(sobol[start:end],bounds);indices=np.arange(start,end,dtype=np.int64);frame.insert(0,"candidate_id",[f"V914_ATLAS_{offset_id+i+1:09d}" for i in indices]);frame.insert(1,"domain_label",domain);frame.insert(2,"sobol_sequence_index",indices);frame.insert(3,"generation_seed",seed);frame.insert(4,"parent_domain_version","v914_historical_22D_v1" if domain=="HISTORICAL_DOMAIN" else "v914_valid_cleavage_shell_v1");frame.insert(5,"creation_commit",a.creation_commit);frame.insert(6,"candidate_sha256",[sha(r,domain,int(i)) for (_,r),i in zip(frame.iterrows(),indices)]);frame.insert(7,"legacy_candidate_id",[f"v914_large_analytical_{i:07d}" if domain=="HISTORICAL_DOMAIN" and i<524288 else (f"v914_cleavage_extension_analytical_{i:07d}" if domain=="CLEAVAGE_EXTENSION" and i<131072 else "") for i in indices])
            master_w=write_batch(master_w,a.out/"master_candidate_registry.parquet",frame)
            if csv_writer is None:csv_writer=csv.DictWriter(csv_stream,fieldnames=list(frame));csv_writer.writeheader()
            csv_writer.writerows(frame.to_dict("records"))
            arr=evaluate(frame,loads,phys,cfg,offset,unc);n=len(frame);flat={"candidate_id":np.repeat(frame.candidate_id.to_numpy(),len(loads)),"DeltaK_MPa_sqrt_m":np.tile(loads,n),"Kmax_MPa_sqrt_m":np.tile(loads/.9,n),"Kmin_MPa_sqrt_m":np.tile(loads/.9*.1,n),"Gstar_eV":arr["Gstar"].ravel(),"Gamma_raw_s":arr["raw"].ravel(),"Gamma_raw_tau":arr["raw_tau"].ravel(),"Gamma_prod_s":arr["rate3"].ravel(),"three_hit_amplification":arr["three_hit_amplification"].ravel(),"local_raw_slope":arr["mraw"].ravel(),"local_production_slope":arr["mlocal"].ravel(),"predicted_log10_da_dN":arr["predicted"].ravel(),"feasible_regime":arr["feasible"].ravel(),"cleavage_emission_log10_ratio":arr["competition"].ravel(),"plastic_activity_proxy":arr["plastic"].ravel(),"transport_barrier_proxy_eV":arr["transport"].ravel()};points_w=write_batch(points_w,a.out/"fatigue_analytical_points.parquet",pd.DataFrame(flat))
            wr=[];pr=[]
            for i,cid in enumerate(frame.candidate_id):
                ws=windows(loads,arr["predicted"][i],arr["feasible"][i],cfg)
                for w in ws:wr.append({"candidate_id":cid,**w})
                if ws:
                    broad=max(ws,key=lambda w:(w["DeltaK_span_fraction"],w["number_points"],-w["residual_rmse_log10"]));ms=[w["m"] for w in ws];Cs=[w["C"] for w in ws]
                    finite=arr["predicted"][i][arr["feasible"][i]]
                    rec={"candidate_id":cid,"minimum_local_m":float(np.min(arr["mlocal"][i])),"median_developed_m":float(np.median(ms)),"best_broad_window_m":broad["m"],"best_window_C":broad["C"],"maximum_valid_window_width":max(w["DeltaK_span_fraction"] for w in ws),"number_valid_windows":len(ws),"minimum_predicted_developed_log10_da_dN":float(finite.min()),"maximum_predicted_developed_log10_da_dN":float(finite.max())}
                else:rec={"candidate_id":cid,"minimum_local_m":float(np.min(arr["mlocal"][i])),"median_developed_m":np.nan,"best_broad_window_m":np.nan,"best_window_C":np.nan,"maximum_valid_window_width":0.,"number_valid_windows":0,"minimum_predicted_developed_log10_da_dN":np.nan,"maximum_predicted_developed_log10_da_dN":np.nan}
                feasible=float(np.mean(arr["feasible"][i]));rec.update({"predicted_censor_inert_probability":1-feasible,"sustained_growth_probability":feasible,"burst_arrest_probability_proxy":float(np.mean((arr["competition"][i]<-14))),"instability_probability_proxy":float(np.mean(arr["predicted"][i]>cfg["target_log10_da_dN_max_m_per_cycle"]+2*unc)),"analytical_initiation_sensitivity":float(arr["mlocal"][i,0]),"raw_Arrhenius_stress_sensitivity_median":float(np.median(arr["mraw"][i])),"three_hit_amplification_median":float(np.median(arr["three_hit_amplification"][i])),"cleavage_emission_competition_median":float(np.median(arr["competition"][i]))});rec.update({x:np.nan for x in RESERVED_FRACTURE+RESERVED_RCURVE+RESERVED_MECHANISM});pr.append(rec)
            if wr:window_w=write_batch(window_w,a.out/"fatigue_window_catalog.parquet",pd.DataFrame(wr))
            props_w=write_batch(props_w,a.out/"fatigue_candidate_properties.parquet",pd.DataFrame(pr));print(f"ATLAS {domain} {end}/{count}",flush=True)
        legacy_checks[domain]={"preserved_prefix_count":524288 if domain=="HISTORICAL_DOMAIN" else 131072,"sobol_seed":seed}
    for w in (master_w,points_w,window_w,props_w):
        if w:w.close()
    csv_stream.close()
    schema={"schema":"v914_joint_candidate_property_atlas_schema_v1","candidate_identity":{"stable_id":"candidate_id","digest":"candidate_sha256","parameter_coordinates":list(COORDINATES)},"groups":{"fatigue_analytical":["minimum_local_m","median_developed_m","best_broad_window_m","best_window_C","maximum_valid_window_width"],"fatigue_n64":["measured_m64","measured_C64","measured_window_span64"],"fatigue_n128":["measured_m128","measured_C128","measured_window_span128"],"fracture":RESERVED_FRACTURE,"R_curve":RESERVED_RCURVE,"mechanism":RESERVED_MECHANISM},"observation_long_schema":["candidate_id","physics_model","fidelity","condition","property_name","value","uncertainty","status","provenance"]};(a.out/"joint_candidate_property_atlas_schema.json").write_text(json.dumps(schema,indent=2)+"\n");(a.out/"candidate_atlas_schema.json").write_text(json.dumps(schema,indent=2)+"\n")
    provenance={"schema":"v914_candidate_atlas_generation_v1","creation_commit":a.creation_commit,"counts":counts,"sobol_prefix_preservation":legacy_checks,"target_config":str(a.target_config.resolve()),"physics":str(a.physics.resolve()),"analytical_model_only":True,"production_physics_modified":False};(a.out/"candidate_generation_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n");print(json.dumps(provenance,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
