"""Representative Appendix A2 displacement-control protocol check."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(path):
    with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))


def make_a2_check(data_dir:Path,run_dir:Path,scenario:str):
    source=_rows(data_dir/"data/A2_time_history.csv")
    curves=defaultdict(list)
    for row in source:curves[row["curve_id"]].append(row)
    fig,axes=plt.subplots(2,1,figsize=(7,7))
    for ax,cid in zip(axes,("A2_nominal_K","A2_plotted_Kdot")):
        obs=sorted(curves[cid],key=lambda r:float(r["time_s"]))
        ax.scatter([float(r["time_s"]) for r in obs],
                   [float(r["value"]) for r in obs],s=5,color="black",
                   alpha=.55,label="Ast A2 digitized")
        for level,color in (("reduced","#1f77b4"),("spatial","#d95f02")):
            file=run_dir/f"confirmation_{scenario.lower()}/{level}/Tp21C_rate1p6/ast_observer.csv"
            if not file.exists():continue
            history=_rows(file)
            t=np.array([float(r["time_s"]) for r in history])
            K=np.array([float(r["K_nominal_MPa_sqrt_m"]) for r in history])
            unique,idx=np.unique(t,return_index=True)
            K=K[idx]
            if len(unique)<2:continue
            if cid=="A2_nominal_K":
                ax.plot(unique,K,color=color,lw=1,label="0-D; 21°C assumption" if level=="reduced" else "1-D; 21°C assumption")
            else:
                bins=np.arange(0,min(max(unique),5)+.1,.1)
                if len(bins)>2:
                    sampled=np.interp(bins,unique,K)
                    rate=np.gradient(sampled,bins)
                    ax.plot(bins,rate,color=color,lw=1,
                            label="0-D 0.1 s derivative" if level=="reduced" else "1-D 0.1 s derivative")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Nominal initial-notch K (MPa√m)" if cid=="A2_nominal_K"
                      else "Nominal K rate (MPa√m/s)")
        ax.text(.03,.95,"A2 nominal K" if cid=="A2_nominal_K" else "A2 plotted rate",
                transform=ax.transAxes,va="top",fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    output=run_dir/f"figures_{scenario.lower()}";output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/"A2_protocol_check.png",dpi=180)
    fig.savefig(output/"A2_protocol_check.svg")
    plt.close(fig)
    return output
