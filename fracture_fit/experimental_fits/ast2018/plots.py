"""Publication-oriented Ast data, prediction and residual figures."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from .data import AstData


def _csv(path):
    if not path.exists():return []
    with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))


def _save(fig,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"),dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _summary_index(run_dir:Path,scenario:str):
    file=run_dir/f"confirmation_{scenario.lower()}/summary.json"
    rows=json.loads(file.read_text())
    return {(r["level"],r["condition_id"]):r for r in rows}


def _history(run_dir,scenario,level,condition_id):
    return _csv(run_dir/f"confirmation_{scenario.lower()}/{level}/{condition_id}/ast_observer.csv")


def _curve_j(figdir,data,run_dir,scenario):
    curves=defaultdict(list)
    for row in data.J_curves:curves[row["curve_id"]].append(row)
    fig,axes=plt.subplots(4,2,figsize=(11,13))
    for ax,(curve_id,rows) in zip(axes.flat,sorted(curves.items())):
        rows=sorted(rows,key=lambda r:float(r["J_N_per_m"]))
        cond=rows[0]["condition_id"]
        ax.scatter([float(r["crack_extension_nm"])*1e-3 for r in rows],
                   [float(r["J_N_per_m"]) for r in rows],s=5,color="black",
                   alpha=.45,label="Ast digitized")
        for level,color in (("reduced","#1f77b4"),("spatial","#d95f02")):
            history=_history(run_dir,scenario,level,cond)
            points=[r for r in history if r["event"]=="True"]
            if points:
                ax.plot([float(r["da_m"])*1e6 for r in points],
                        [float(r["J_continuous_N_per_m"]) for r in points],
                        lw=1.2,color=color,label="0-D" if level=="reduced" else "1-D")
        ax.text(.03,.95,curve_id,transform=ax.transAxes,va="top",fontsize=8)
        ax.set_xlabel("Crack extension Δa (µm)")
        ax.set_ylabel("Continuous J (N/m)")
        ax.set_xlim(-.1,max(float(r["crack_extension_nm"])*1e-3 for r in rows)+.15)
        ax.legend(fontsize=7,loc="best")
    _save(fig,figdir/"J_resistance_curves")


def _onsets(figdir,data,run_dir,scenario,summary):
    byrate=defaultdict(list)
    for row in data.toughness_scenario(scenario):
        if row["observable"]=="Kq_Jcontinuous_onset":
            byrate[float(row["initial_elastic_Kdot_MPa_sqrt_m_per_s"])].append(row)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(rate,rows) in zip(axes.flat,sorted(byrate.items())):
        rows=sorted(rows,key=lambda r:float(r["temperature_C"]))
        x=np.array([float(r["temperature_C"]) for r in rows])
        ax.scatter(x,[float(r["value"]) for r in rows],marker="o",s=30,color="black",label="Ast onset")
        for level,color,marker in (("reduced","#1f77b4","s"),("spatial","#d95f02","^")):
            xx=[];yy=[]
            for r in rows:
                pred=summary.get((level,r["condition_id"]),{}).get("onset_K_Jcontinuous_MPa_sqrt_m")
                if pred is not None:xx.append(float(r["temperature_C"]));yy.append(pred)
            ax.scatter(xx,yy,marker=marker,s=25,color=color,label="0-D" if level=="reduced" else "1-D")
        ax.set_title(f"initial K̇ = {rate:g} MPa√m/s",loc="left",fontsize=8)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Onset Kq,J at Δa=0.2 µm (MPa√m)")
        ax.legend(fontsize=7)
    _save(fig,figdir/"onset_by_rate")


def _cleavage(figdir,data,summary):
    byrate=defaultdict(list)
    for row in data.toughness:
        if row["observable"] in ("Kq_LEFM","Kq_Jbasic_cleavage"):
            byrate[float(row["initial_elastic_Kdot_MPa_sqrt_m_per_s"])].append(row)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(rate,rows) in zip(axes.flat,sorted(byrate.items())):
        for kind,marker in (("Kq_LEFM","o"),("Kq_Jbasic_cleavage","D")):
            selected=[r for r in rows if r["observable"]==kind]
            ax.scatter([float(r["temperature_C"]) for r in selected],
                       [float(r["value"]) for r in selected],s=30,marker=marker,
                       label="Ast LEFM" if kind=="Kq_LEFM" else "Ast basic J")
        for level,color in (("reduced","#1f77b4"),("spatial","#d95f02")):
            xx=[];yy=[]
            for r in rows:
                pred=summary.get((level,r["condition_id"]),{}).get(
                    "LEFM_Kq_MPa_sqrt_m" if r["observable"]=="Kq_LEFM" else
                    "basic_J_Kq_a0_MPa_sqrt_m")
                if pred is not None:xx.append(float(r["temperature_C"]));yy.append(pred)
            ax.scatter(xx,yy,s=22,marker="x",color=color,
                       label="0-D ligament endpoint" if level=="reduced" else "1-D ligament endpoint")
        ax.set_title(f"initial K̇ = {rate:g} MPa√m/s",loc="left",fontsize=8)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Final cleavage Kq (MPa√m)")
        ax.legend(fontsize=6)
    _save(fig,figdir/"cleavage_validation_only")


def _trace_shift(rows,history,figure):
    if figure!="2a":return 0.0
    x=np.array([float(r["displacement_um"])*1e-6 for r in rows])
    y=np.array([float(r["K_MPa_sqrt_m"]) for r in rows])
    xm=np.array([float(r["u_m"]) for r in history])
    ym=np.array([float(r["K_nominal_MPa_sqrt_m"]) for r in history])
    def cost(shift):
        return np.mean((np.interp(x+shift,xm,ym)-y)**2)+.0001*(shift/.3e-6)**2
    return float(minimize_scalar(cost,bounds=(-.3e-6,.3e-6),method="bounded",
                                 options={"xatol":1e-10}).x)


def _nominal_traces(figdir,data,run_dir,scenario):
    grouped=defaultdict(list)
    for row in data.K_displacement:
        if row["fit_role"]=="loading_history_candidate":grouped[row["curve_id"]].append(row)
    curves=sorted(grouped.items())
    fig,axes=plt.subplots(6,3,figsize=(13,18))
    for ax,(cid,rows) in zip(axes.flat,curves):
        rows=sorted(rows,key=lambda r:float(r["displacement_um"]))
        cond=rows[0]["condition_id"]
        ax.scatter([float(r["displacement_um"]) for r in rows],
                   [float(r["K_MPa_sqrt_m"]) for r in rows],s=4,color="black",
                   alpha=.55,label="Ast loading")
        for level,color in (("reduced","#1f77b4"),("spatial","#d95f02")):
            h=_history(run_dir,scenario,level,cond)
            if h:
                shift=_trace_shift(rows,h,rows[0]["figure"])
                ax.plot([float(r["u_m"])*1e6-shift*1e6 for r in h],
                        [float(r["K_nominal_MPa_sqrt_m"]) for r in h],
                        color=color,lw=1,label="0-D" if level=="reduced" else "1-D")
        ax.text(.03,.95,cid,transform=ax.transAxes,va="top",fontsize=7)
        ax.set_xlabel("Indenter displacement (µm; Fig. 2a relative)")
        ax.set_ylabel("Nominal initial-notch K (MPa√m)")
        ax.legend(fontsize=6)
    _save(fig,figdir/"nominal_K_loading_traces")


def _stiffness(figdir,data,run_dir,scenario):
    grouped=defaultdict(list)
    for row in data.stiffness:grouped[row["curve_id"]].append(row)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,(cid,rows) in zip(axes,sorted(grouped.items())):
        cond=rows[0]["condition_id"]
        ax.scatter([float(r["displacement_nm"])*1e-3 for r in rows],
                   [float(r["relative_stiffness"]) for r in rows],
                   s=5,color="black",alpha=.5,label="Ast")
        for level,color in (("reduced","#1f77b4"),("spatial","#d95f02")):
            h=_history(run_dir,scenario,level,cond)
            if h:
                ax.plot([float(r["u_m"])*1e6 for r in h],
                        [float(r["relative_stiffness"]) for r in h],
                        color=color,lw=1,label="0-D" if level=="reduced" else "1-D")
        ax.text(.03,.95,cid,transform=ax.transAxes,va="top",fontsize=8)
        ax.set_xlabel("Indenter displacement (µm)")
        ax.set_ylabel("Relative stiffness")
        ax.legend(fontsize=7)
    _save(fig,figdir/"relative_stiffness")


def _residuals(figdir,summary):
    families=sorted({b["family"] for row in summary.values() for b in row.get("residual_blocks",[])})
    conditions=sorted({cid for level,cid in summary})
    fig,axes=plt.subplots(1,2,figsize=(13,8))
    for ax,level in zip(axes,("reduced","spatial")):
        mat=np.full((len(conditions),len(families)),np.nan)
        for i,cid in enumerate(conditions):
            for block in summary.get((level,cid),{}).get("residual_blocks",[]):
                mat[i,families.index(block["family"])]=block["cost"]
        image=ax.imshow(np.log10(np.maximum(mat,1e-6)),aspect="auto",cmap="viridis")
        ax.set_xticks(range(len(families)),families,rotation=45,ha="right",fontsize=8)
        ax.set_yticks(range(len(conditions)),conditions,fontsize=6)
        ax.set_xlabel("Observable family")
        ax.set_ylabel("Temperature/rate condition")
        ax.text(.03,.03,level,transform=ax.transAxes,color="white",fontsize=9)
        fig.colorbar(image,ax=ax,label="log10 block working cost")
    _save(fig,figdir/"residual_blocks")


def _unchanged_transfer(figdir,data,run_dir,scenario):
    """Keep the unchanged-vector priority confirmation visually distinct."""
    root=run_dir/f"confirmation_{scenario.lower()}_priority11"
    summaries=json.loads((root/"summary.json").read_text())
    index={(r["level"],r["condition_id"]):r for r in summaries}
    available={r["condition_id"] for r in summaries}
    onsets=[r for r in data.toughness_scenario(scenario)
            if r["observable"]=="Kq_Jcontinuous_onset" and
            r["condition_id"] in available]
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot([0,30],[0,30],color="gray",lw=.7,ls="--",label="equal")
    for level,color,marker in (("reduced","#1f77b4","o"),
                               ("spatial","#d95f02","^")):
        x=[];y=[]
        for row in onsets:
            prediction=index.get((level,row["condition_id"]),{}).get(
                "onset_K_Jcontinuous_MPa_sqrt_m")
            if prediction is not None:
                x.append(float(row["value"]));y.append(prediction)
        ax.scatter(x,y,color=color,marker=marker,s=28,
                   label=f"unchanged material {level}")
    ax.set_xlabel("Ast continuous-J onset K (MPa√m)")
    ax.set_ylabel("Predicted continuous-J onset K (MPa√m)")
    ax.legend(fontsize=8)
    _save(fig,figdir/"unchanged_transfer_onset")
    grouped=defaultdict(list)
    for row in data.J_curves:grouped[row["curve_id"]].append(row)
    fig,axes=plt.subplots(4,2,figsize=(11,13))
    for ax,(curve_id,rows) in zip(axes.flat,sorted(grouped.items())):
        cond=rows[0]["condition_id"]
        ax.scatter([float(r["crack_extension_nm"])*1e-3 for r in rows],
                   [float(r["J_N_per_m"]) for r in rows],s=5,color="black",
                   alpha=.45,label="Ast digitized")
        for level,color in (("reduced","#1f77b4"),("spatial","#d95f02")):
            history=_csv(root/level/cond/"ast_observer.csv")
            events=[r for r in history if r["event"]=="True"]
            if events:
                ax.plot([float(r["da_m"])*1e6 for r in events],
                        [float(r["J_continuous_N_per_m"]) for r in events],
                        color=color,lw=1.1,label=f"unchanged {level}")
        ax.text(.03,.95,curve_id,transform=ax.transAxes,va="top",fontsize=8)
        ax.set_xlabel("Crack extension Δa (µm)")
        ax.set_ylabel("Continuous J (N/m)")
        ax.legend(fontsize=7)
    _save(fig,figdir/"unchanged_transfer_J_curves")
    state_cases=("Tp21C_rate0p2","Tp100C_rate1p6","Tp500C_rate1p6")
    metrics=(("cumulative_activations","Cumulative source activations"),
             ("backstress_pre_Pa","Mean backstress (GPa)"),
             ("source_multiplicity_pre","Source multiplicity / system"))
    fig,axes=plt.subplots(3,3,figsize=(12,9))
    for row_index,condition_id in enumerate(state_cases):
        for level,color in (("reduced","#1f77b4"),("spatial","#d95f02")):
            path=root/level/condition_id/"events.json"
            events=[r for r in json.loads(path.read_text())
                    if not r.get("terminal_ligament_failure")]
            extension=[float(r["cumulative_extension_m"])*1e6 for r in events]
            for col_index,(key,ylabel) in enumerate(metrics):
                values=[float(r.get(key,0))*(1e-9 if key=="backstress_pre_Pa" else 1)
                        for r in events]
                ax=axes[row_index,col_index]
                ax.plot(extension,values,color=color,lw=1,
                        label="0-D" if level=="reduced" else "1-D")
                ax.set_xlabel("Crack extension Δa (µm)")
                ax.set_ylabel(ylabel)
                ax.text(.03,.95,condition_id,transform=ax.transAxes,va="top",fontsize=7)
                if row_index==0:ax.legend(fontsize=7)
    _save(fig,figdir/"unchanged_transfer_native_state")


def make_all(data_dir:Path,run_dir:Path,scenario:str):
    data=AstData.load(data_dir)
    summary=_summary_index(run_dir,scenario)
    figdir=run_dir/f"figures_{scenario.lower()}"
    _onsets(figdir,data,run_dir,scenario,summary)
    _cleavage(figdir,data,summary)
    _curve_j(figdir,data,run_dir,scenario)
    _nominal_traces(figdir,data,run_dir,scenario)
    _stiffness(figdir,data,run_dir,scenario)
    _residuals(figdir,summary)
    _unchanged_transfer(figdir,data,run_dir,scenario)
    return figdir
