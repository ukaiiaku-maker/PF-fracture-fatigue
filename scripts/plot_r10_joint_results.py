#!/usr/bin/env python3
"""Plot the decision-controlling R10 fatigue and fracture bridge data."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    out=a.root/"figures";out.mkdir(parents=True,exist_ok=True)
    fatigue=pd.read_csv(a.root/"current_bridge_case_table.csv")
    keep=fatigue[fatigue.rate_20_60_m_per_cycle.notna() & fatigue.candidate_id.isin([
        "P40_TRANSFER_CALIBRATED_GEN2","R10_DBTT_P40_THERMAL_ANCHOR_V1"])]
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    for cid,g in keep.groupby("candidate_id"):
        label="retained P40" if cid.startswith("P40_") else "DBTT-parent P40 projection"
        ax.loglog(g.DeltaK_MPa_sqrt_m,g.rate_20_60_m_per_cycle,"o-",label=label)
    ax.set_xlabel(r"$\Delta K$ (MPa $\sqrt{m}$)");ax.set_ylabel(r"developed $da/dN$ (m cycle$^{-1}$)")
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"current_developed_fatigue.png",dpi=180);plt.close(fig)
    fracture=pd.read_csv(a.root/"fracture/current_fracture_case_table.csv")
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    for cid,g in fracture.groupby("candidate_id"):
        label="original DBTT" if cid.startswith("v913_") else "DBTT-parent P40 projection"
        y=g.K_init_MPa_sqrt_m.fillna(g.K_init_lower_bound_MPa_sqrt_m)
        ax.plot(g.temperature_K,y,"o-",label=label)
        cens=g.K_init_MPa_sqrt_m.isna();ax.scatter(g.loc[cens,"temperature_K"],y[cens],marker="^",s=55)
    ax.set_xlabel("temperature (K)");ax.set_ylabel(r"initiation $K$ (MPa $\sqrt{m}$)")
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"current_dbtt_fracture_bridge.png",dpi=180);plt.close(fig)
if __name__=="__main__":main()
