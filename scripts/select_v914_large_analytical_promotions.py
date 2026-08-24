#!/usr/bin/env python3
"""Cluster a large analytical population and select a diverse v7 batch."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import sys
import numpy as np,pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from scripts.audit_v914_historical_analytical_model import COORDINATES

U=[f"u__{x}" for x in COORDINATES]

def greedy(pool: pd.DataFrame, count: int, anchors: np.ndarray, quality_weight=.2):
    chosen=[]; available=pool.copy(); A=np.asarray(anchors,float).reshape((-1,len(U)))
    for _ in range(min(count,len(available))):
        X=available[U].to_numpy(float); dist=np.min(cdist(X,A),axis=1) if len(A) else np.ones(len(X))
        q=available.quality01.to_numpy(float); score=(1-quality_weight)*dist+quality_weight*(1-q)
        j=int(np.argmax(score)); chosen.append(available.index[j]); A=np.vstack([A,X[j]]) if len(A) else X[j:j+1]; available=available.drop(available.index[j])
    return chosen

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--population",type=Path,required=True); ap.add_argument("--extension-population",type=Path); ap.add_argument("--tested-registry",type=Path,required=True); ap.add_argument("--target-config",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--batch-size",type=int,default=40); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(a.population)
    if a.extension_population: d=pd.concat([d,pd.read_parquet(a.extension_population)],ignore_index=True,sort=False)
    cfg=json.loads(a.target_config.read_text()); valid=d[d.valid_analytical_Paris_window & d.population_role.isin(["HISTORICAL_DOMAIN","CLEAVAGE_EXTENSION"])].copy()
    valid["target_m_distance"]=np.maximum(cfg["target_m_min"]-valid.m,0)+np.maximum(valid.m-cfg["target_m_max"],0)
    preferred_min=cfg.get("preferred_m_min",cfg["target_m_min"]); preferred_max=cfg.get("preferred_m_max",cfg["target_m_max"])
    valid["preferred_m_distance"]=np.maximum(preferred_min-valid.m,0)+np.maximum(valid.m-preferred_max,0)
    valid["quality_score"]=valid.target_m_distance+valid.preferred_m_distance+valid.shape_score+.25*valid.level_score-.25*valid.DeltaK_fractional_span-.1*valid.predicted_sustained_growth_probability
    valid["quality01"]=valid.quality_score.rank(pct=True)
    top=valid.nsmallest(min(5000,len(valid)),"quality_score").copy(); centroids,labels=kmeans2(top[U].to_numpy(float),30,minit="points",seed=1914,iter=50); top["cluster_id"]=labels
    reps=top.sort_values("quality_score").groupby("cluster_id",as_index=False).head(1); reps.to_csv(a.out/"analytical_candidate_clusters.csv",index=False); reps.to_csv(a.out/"large_search_clusters.csv",index=False)
    tested=pd.read_csv(a.tested_registry); tested=tested[[c for c in U if c in tested]].dropna(); anchors=tested.to_numpy(float) if len(tested) else np.empty((0,len(U)))
    exploit_pool=top[top.m.between(cfg["target_m_min"],cfg["target_m_max"])].nsmallest(2500,"quality_score")
    preferred_pool=exploit_pool[exploit_pool.m.between(preferred_min,preferred_max)]
    exploit_idx=greedy(preferred_pool,14,anchors,quality_weight=.7)
    preferred=top.loc[exploit_idx]; preferred_anchors=np.vstack([anchors,preferred[U].to_numpy(float)]) if len(anchors) else preferred[U].to_numpy(float)
    exploit_idx += greedy(exploit_pool.drop(exploit_idx),20-len(exploit_idx),preferred_anchors,quality_weight=.65)
    exploit=top.loc[exploit_idx].copy(); exploit["acquisition_role"]="best_predicted_realistic_m"
    A=np.vstack([anchors,exploit[U].to_numpy(float)]) if len(anchors) else exploit[U].to_numpy(float)
    remaining=top.drop(exploit.index); remaining=remaining[remaining.m.between(cfg["target_m_min"],cfg["target_m_max"])].copy(); ci=greedy(remaining,8,A,quality_weight=.35); clusters=top.loc[ci].copy(); clusters["acquisition_role"]="distinct_low_m_cluster"
    A=np.vstack([A,clusters[U].to_numpy(float)]); remain=top.drop(exploit.index.union(clusters.index)).copy(); remain["distance_from_tested"]=np.min(cdist(remain[U],anchors),axis=1) if len(anchors) else 1.
    uncertain=remain.nlargest(min(1000,len(remain)),"distance_from_tested"); ui=greedy(uncertain,6,A,quality_weight=.25); uncertainty=top.loc[ui].copy(); uncertainty["acquisition_role"]="high_epistemic_uncertainty"
    A=np.vstack([A,uncertainty[U].to_numpy(float)]); remain=valid.drop(exploit.index.union(clusters.index).union(uncertainty.index)); ei=greedy(remain.sample(min(15000,len(remain)),random_state=1914),6,A,quality_weight=.05); explore=valid.loc[ei].copy(); explore["acquisition_role"]="global_maximin_exploration"
    batch=pd.concat([exploit,clusters,uncertainty,explore]).head(a.batch_size).copy(); batch["promotion_order"]=range(1,len(batch)+1)
    # Fresh n64 descriptor loads lie inside each predicted window and are retained explicitly.
    for j,f in enumerate((0.,.5,1.)):
        batch[f"descriptor_DeltaK_{j+1}_MPa_sqrt_m"]=batch.DeltaK_min*(batch.DeltaK_max/batch.DeltaK_min)**f
    batch["Tref_K"]=300.; batch["cleave_gT_eV_per_K"]=0.; batch["cleave_sT_GPa_per_K"]=0.; batch["emit_gT_eV_per_K"]=0.; batch["emit_sT_GPa_per_K"]=0.; batch["peierls_nu0_s"]=1e12; batch["taylor_nu0_s"]=1e11
    batch["source_historical_sobol_index"]=""; batch["paris_design_role"]="LARGE_ANALYTICAL_PROMOTION_BATCH_01"
    batch.to_csv(a.out/"v7_promotion_batch_01.csv",index=False)
    cases=[]
    for _,row in batch.iterrows():
        for j in range(1,4): cases.append({"candidate_id":row.candidate_id,"DeltaK_MPa_sqrt_m":row[f"descriptor_DeltaK_{j}_MPa_sqrt_m"],"R":.1,"frequency_Hz":1000.,"temperature_K":300.,"seed":1720,"analytical_m":row.m,"analytical_window_min":row.DeltaK_min,"analytical_window_max":row.DeltaK_max,"acquisition_role":row.acquisition_role,"population_role":row.population_role})
    pd.DataFrame(cases).to_csv(a.out/"v7_promotion_batch_01_cases.csv",index=False)
    hist=pd.DataFrame([{"iteration":0,"analytical_population":int((d.population_role=="sobol_historical_domain").sum()),"v7_promoted":0,"best_v7_m":math.nan,"status":"analytical_population_ranked"},{"iteration":1,"analytical_population":int((d.population_role=="sobol_historical_domain").sum()),"v7_promoted":len(batch),"best_v7_m":math.nan,"status":"promotion_batch_selected_not_launched"}]); hist.to_csv(a.out/"active_learning_history.csv",index=False); hist[["iteration","best_v7_m","status"]].to_csv(a.out/"best_m_vs_iteration.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,4)); ax.hist(valid.m.clip(-10,80),bins=80); ax.axvspan(cfg["target_m_min"],cfg["target_m_max"],alpha=.15,color="green"); ax.set(xlabel="predicted Paris m",ylabel="candidates"); fig.tight_layout(); fig.savefig(a.out/"predicted_m_distribution.png",dpi=180); plt.close(fig)
    for y,name,label in (("DeltaK_fractional_span","predicted_m_vs_window_span.png","fractional DeltaK span"),("predicted_log10_da_dN_median","predicted_m_vs_rate_magnitude.png","median log10 da/dN")):
        fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(valid.m,valid[y],s=3,alpha=.25); ax.set(xlabel="predicted Paris m",ylabel=label); fig.tight_layout(); fig.savefig(a.out/name,dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(top[U[0]],top[U[1]],c=top.cluster_id,s=7,cmap="tab20"); ax.scatter(batch[U[0]],batch[U[1]],facecolors="none",edgecolors="black",s=40); ax.set(xlabel=U[0],ylabel=U[1]); fig.tight_layout(); fig.savefig(a.out/"top_candidate_clusters.png",dpi=180); plt.close(fig)
    summary={"schema":"v914_large_analytical_promotion_v1","valid_candidates":len(valid),"batch_size":len(batch),"roles":batch.acquisition_role.value_counts().to_dict(),"high_dimensional_learned_residual_primary":False,"selection":"quality plus normalized-22D diversity"}; (a.out/"promotion_batch_01_audit.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
