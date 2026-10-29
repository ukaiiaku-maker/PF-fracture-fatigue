#!/usr/bin/env python3
"""Report the fair independently refitted bulk/no-bulk 1-D comparison."""
from pathlib import Path
import csv,json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"literature_data/fatigue_recovered_review/model_conditional/emission_owned_blunting/resolved_increment_1d"
NO=BASE/"no_bulk_matched_refit_20261028"; BULK=BASE/"signed_bulk_joint_refit_20261028"
def read(path): return list(csv.DictReader(open(path,newline="",encoding="utf-8")))
def main():
    nf=json.loads((NO/"fit.json").read_text());bf=json.loads((BULK/"fit.json").read_text())
    n=read(NO/"predictions.csv");b=read(BULK/"predictions_signed_bulk.csv")
    fig,ax=plt.subplots(figsize=(9,6.2));colors={.1:"#1f77b4",.3:"#ff7f0e",.5:"#2ca02c",.7:"#d62728"}
    for R,c in colors.items():
        nn=[x for x in n if float(x["R"])==R];bb=[x for x in b if float(x["R"])==R]
        x=np.asarray([float(z["DeltaK_MPa_sqrt_m"]) for z in nn]);o=np.argsort(x)
        y=np.asarray([float(z["observed_dadn_m_per_cycle"]) for z in nn])
        pn=np.asarray([float(z["predicted_dadn_m_per_cycle"]) for z in nn]);pb=np.asarray([float(z["predicted_dadn_m_per_cycle"]) for z in bb])
        ax.plot(x[o],y[o],c=c,lw=2,label=f"R={R:.1f} experiment")
        ax.scatter(x[::max(len(x)//18,1)],pn[::max(len(x)//18,1)],marker="s",facecolors="none",edgecolors=c,s=24)
        ax.scatter(x[::max(len(x)//18,1)],pb[::max(len(x)//18,1)],marker="o",c=c,s=12)
    ax.set(xscale="log",yscale="log",xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",ylabel=r"$da/dN$ (m/cycle)",title="Fair independently refitted 1-D controls\nfilled circles: signed bulk; open squares: no bulk")
    ax.legend(frameon=False,ncol=2,fontsize=8);fig.tight_layout();fig.savefig(BASE/"matched_bulk_no_bulk_comparison.png",dpi=220);plt.close(fig)
    rmse_no=nf["metrics"]["overall_rmse_log10_rate"];rmse_bulk=bf["signed_bulk_metrics"]["overall_rmse_log10_rate"]
    delta=rmse_bulk-rmse_no
    decision=("signed bulk gives a modest aggregate improvement under matched "
              "independent optimization" if delta < 0 else
              "signed bulk does not improve aggregate fit under matched independent optimization")
    result={"schema":"qste340tm-matched-1d-control-comparison/1",
        "no_bulk_rmse_log10":rmse_no,"signed_bulk_rmse_log10":rmse_bulk,
        "signed_bulk_minus_no_bulk_rmse_log10":delta,
        "relative_RMSE_reduction_fraction":-delta/rmse_no,
        "no_bulk_block_metrics":nf["metrics"]["blocks"],
        "signed_bulk_block_metrics":bf["signed_bulk_metrics"]["blocks"],
        "decision":decision,"comparison_contract":nf["comparison_contract"],
        "no_bulk_fit":str(NO/"fit.json"),"signed_bulk_fit":str(BULK/"fit.json")}
    (BASE/"MATCHED_CONTROL_COMPARISON.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
