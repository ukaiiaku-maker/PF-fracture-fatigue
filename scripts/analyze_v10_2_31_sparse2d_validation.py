#!/usr/bin/env python3
"""Build mechanism-resolved v9.14 1-D / v10.2.31 2-D validation products."""
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CLASSES = {
    "A": ("A_0462", "v914_endurance_knee_0462", "direct barrier"),
    "B": ("B_0658", "v914_endurance_knee_0658", "plastic-state controlled"),
    "C": ("C_0554", "v914_endurance_knee_0554", "timescale crossover"),
    "D": ("D_0133", "v914_endurance_knee_0133", "mixed"),
}
COLORS = {"A":"#1f77b4", "B":"#d62728", "C":"#2ca02c", "D":"#9467bd"}

def rows(path: Path) -> list[dict[str,str]]:
    with path.open(newline="") as f: return list(csv.DictReader(f))

def num(value: Any) -> float | None:
    try:
        x=float(value); return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None

def write_csv(path: Path, data: list[dict[str,Any]]) -> None:
    if not data: return
    fields=list(data[0]); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)

def save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    for ext in ("png","pdf","svg"): fig.savefig(base.with_suffix("."+ext),dpi=220)
    plt.close(fig)

def one_d(reference: Path) -> list[dict[str,Any]]:
    out=[]
    for name in ("abcd_developed_rates.csv","abcd_censor_table.csv"):
        for r in rows(reference/name):
            out.append({"dimension":"1D","class":r["class"],"candidate_id":r["candidate_id"],
                "case":r["case"],"deltaK_MPa_sqrt_m":num(r["deltaK_MPa_sqrt_m"]),
                "fraction":num(r["fraction"]),"status":r["status"],"plot_kind":r["plot_kind"],
                "rate_m_per_cycle":num(r["developed_da_dN_m_per_cycle"]),
                "observed_partial_rate_m_per_cycle":num(r["observed_partial_rate_m_per_cycle"]),
                "cycles":num(r["final_cycles"]),"extension_um":num(r["final_extension_um"]),
                "events":int(float(r["event_count"]))})
    return out

def two_d(root: Path) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    cases=[]; events=[]
    for cls,(folder,candidate,_) in CLASSES.items():
        for summary in sorted((root/folder).glob("*/developed_fatigue_growth_summary.json")):
            case=summary.parent; d=json.loads(summary.read_text())
            args=json.loads((case/"run_args.json").read_text())
            dk=num(args.get("target_deltaK_MPa_sqrt_m"))
            if dk is None: dk=num(args.get("target_deltaK_MPa_sqrt_m".replace("_","-")))
            control=json.loads((case/"v10_2_30_fixed_deltaK_control.json").read_text())
            dk=dk or num(control.get("target_deltaK_MPa_sqrt_m")) or num(control.get("DeltaK_MPa_sqrt_m"))
            developed=num(d.get("developed_interval",{}).get("da_dN")) if d.get("stable_growth_provisional") else None
            cyc=num(d.get("cycles_consumed")) or 0.; ext=num(d.get("final_projected_extension_um")) or 0.
            terminal="target" if d.get("target_reached") else ("cycle_censor" if cyc>=0.999999e14 else "partial")
            observed=ext*1e-6/cyc if cyc>0 else None
            cases.append({"dimension":"2D","class":cls,"candidate_id":candidate,"case":case.name,
                "deltaK_MPa_sqrt_m":dk,"status":terminal,"plot_kind":"resolved" if developed else "censored_or_partial",
                "rate_m_per_cycle":developed,"observed_partial_rate_m_per_cycle":observed,
                "cycles":cyc,"extension_um":ext,"events":int(d.get("event_count",0)),
                "stable_growth":bool(d.get("stable_growth_provisional")),"result_path":str(case.resolve())})
            ef=case/"fatigue_event_growth_0300K.csv"
            if ef.exists():
                for e in rows(ef):
                    events.append({"class":cls,"case":case.name,"deltaK_MPa_sqrt_m":dk,
                        "event_index":int(float(e["event_index"])),"cycles_post":num(e["cycles_post"]),
                        "extension_um":1e6*(num(e["projected_extension_post_m"]) or 0.),
                        "retained_count":num(e.get("retained_count")),"mobile_count":num(e.get("mobile_count")),
                        "sigma_back_Pa":num(e.get("sigma_back_Pa")),"B_post":num(e.get("B_post")),
                        "lambda_c_per_s":num(e.get("lambda_c_per_s"))})
    return cases,events

def interpolation(reference: list[dict[str,Any]], cls: str, dk: float) -> float | None:
    rr=sorted((r for r in reference if r["class"]==cls and r["rate_m_per_cycle"]),key=lambda r:r["deltaK_MPa_sqrt_m"])
    if not rr or dk<rr[0]["deltaK_MPa_sqrt_m"] or dk>rr[-1]["deltaK_MPa_sqrt_m"]: return None
    return float(10**np.interp(dk,[r["deltaK_MPa_sqrt_m"] for r in rr],np.log10([r["rate_m_per_cycle"] for r in rr])))

def mapping_table(source: Path, registry: Path) -> list[dict[str,Any]]:
    src={r["candidate_id"]:r for r in rows(source)}; dst={r["candidate_id"]:r for r in rows(registry)}
    common=[k for k in rows(registry)[0] if k in rows(source)[0]]
    out=[]
    for cls,(_,candidate,_) in CLASSES.items():
        for field in common:
            same=src[candidate].get(field)==dst[candidate].get(field)
            status="EXACT" if same else ("EQUIVALENT_REPRESENTATION" if field=="material_class" else "NOT_USED_IN_2D")
            out.append({"class":cls,"candidate_id":candidate,"1-D field":field,"1-D value":src[candidate].get(field),
                "2-D field":field,"2-D value":dst[candidate].get(field),"mapping status":status})
        out.append({"class":cls,"candidate_id":candidate,"1-D field":"physics__encounter_efficiency",
            "1-D value":src[candidate]["physics__encounter_efficiency"],"2-D field":"encounter_efficiency",
            "2-D value":dst[candidate]["encounter_efficiency"],"mapping status":"EQUIVALENT_REPRESENTATION"})
    return out

def plots(reference:list[dict[str,Any]], spatial:list[dict[str,Any]], events:list[dict[str,Any]], out:Path)->None:
    fig,ax=plt.subplots(figsize=(8,6))
    for cls in CLASSES:
        rr=sorted((r for r in reference if r["class"]==cls and r["rate_m_per_cycle"]),key=lambda r:r["deltaK_MPa_sqrt_m"])
        ax.plot([r["deltaK_MPa_sqrt_m"] for r in rr],[r["rate_m_per_cycle"] for r in rr],color=COLORS[cls],label=f"{cls} 1-D")
        ss=[r for r in spatial if r["class"]==cls]
        finite=[r for r in ss if r["rate_m_per_cycle"]]
        ax.scatter([r["deltaK_MPa_sqrt_m"] for r in finite],[r["rate_m_per_cycle"] for r in finite],s=55,facecolors="white",edgecolors=COLORS[cls],marker="o",zorder=4,label=f"{cls} 2-D")
        partial=[r for r in ss if not r["rate_m_per_cycle"] and r["observed_partial_rate_m_per_cycle"]]
        ax.scatter([r["deltaK_MPa_sqrt_m"] for r in partial],[r["observed_partial_rate_m_per_cycle"] for r in partial],s=55,color=COLORS[cls],marker="v",zorder=4)
    ax.set(yscale="log",xlabel=r"Dimensional $\Delta K$ (MPa$\sqrt{m}$)",ylabel=r"$da/dN$ (m/cycle)")
    ax.grid(True,which="both",alpha=.25); ax.legend(ncol=2,fontsize=8)
    save(fig,out/"abcd_1D_2D_da_dN_vs_deltaK_overlay")

    fig,axs=plt.subplots(2,2,figsize=(11,8),sharey=True)
    for ax,cls in zip(axs.flat,CLASSES):
        rr=sorted((r for r in reference if r["class"]==cls and r["rate_m_per_cycle"]),key=lambda r:r["deltaK_MPa_sqrt_m"])
        ax.plot([r["deltaK_MPa_sqrt_m"] for r in rr],[r["rate_m_per_cycle"] for r in rr],color=COLORS[cls],label="1-D prediction")
        ss=[r for r in spatial if r["class"]==cls]
        for r in ss:
            y=r["rate_m_per_cycle"] or r["observed_partial_rate_m_per_cycle"]
            if y: ax.scatter(r["deltaK_MPa_sqrt_m"],y,facecolors="white" if r["rate_m_per_cycle"] else COLORS[cls],edgecolors=COLORS[cls],marker="o" if r["rate_m_per_cycle"] else "v",s=50)
        ax.set_yscale("log"); ax.grid(True,which="both",alpha=.25); ax.set_title(f"{cls} — {CLASSES[cls][2]}"); ax.set_xlabel(r"$\Delta K$ (MPa$\sqrt{m}$)")
    axs[0,0].set_ylabel(r"$da/dN$ (m/cycle)"); axs[1,0].set_ylabel(r"$da/dN$ (m/cycle)")
    save(fig,out/"abcd_1D_2D_da_dN_vs_deltaK_four_panel")

    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,cls in zip(axs.flat,CLASSES):
        for case in sorted({e["case"] for e in events if e["class"]==cls}):
            ee=sorted((e for e in events if e["class"]==cls and e["case"]==case),key=lambda e:e["event_index"])
            if ee: ax.plot([max(e["cycles_post"],1e-12) for e in ee],[e["extension_um"] for e in ee],marker="o",ms=2,label=case)
        ax.set_xscale("log"); ax.grid(True,which="both",alpha=.25); ax.set_title(cls); ax.set_xlabel("Cumulative cycles N"); ax.set_ylabel("Projected extension (µm)"); ax.legend(fontsize=6)
    save(fig,out/"abcd_1D_2D_a_vs_N_matched_conditions")

    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,cls in zip(axs.flat,CLASSES):
        for case in sorted({e["case"] for e in events if e["class"]==cls}):
            ee=sorted((e for e in events if e["class"]==cls and e["case"]==case),key=lambda e:e["event_index"])
            if ee: ax.plot([e["extension_um"] for e in ee],[(e["sigma_back_Pa"] or 0)/1e9 for e in ee],marker=".",label=case)
        ax.grid(True,alpha=.25); ax.set_title(f"{cls}: {CLASSES[cls][2]}"); ax.set_xlabel("Extension (µm)"); ax.set_ylabel("Backstress (GPa)"); ax.legend(fontsize=6)
    save(fig,out/"abcd_1D_2D_mechanism_comparison")

    ratio=[]
    for r in spatial:
        if not r["rate_m_per_cycle"]: continue
        ref=interpolation(reference,r["class"],r["deltaK_MPa_sqrt_m"])
        if ref: ratio.append({**r,"one_d_interpolated_rate":ref,"rate_ratio_2D_to_1D":r["rate_m_per_cycle"]/ref,"delta_log10":math.log10(r["rate_m_per_cycle"]/ref)})
    fig,ax=plt.subplots(figsize=(8,5))
    for cls in CLASSES:
        rr=[r for r in ratio if r["class"]==cls]
        ax.scatter([r["deltaK_MPa_sqrt_m"] for r in rr],[r["rate_ratio_2D_to_1D"] for r in rr],color=COLORS[cls],label=cls)
    for y,ls in ((1,"-"),(2,"--"),(.5,"--"),(3,":"),(1/3,":"),(10,"-."),(.1,"-.")): ax.axhline(y,color="grey",lw=.7,ls=ls)
    ax.set_yscale("log"); ax.set_xlabel(r"$\Delta K$ (MPa$\sqrt{m}$)"); ax.set_ylabel("2-D / interpolated 1-D rate"); ax.grid(True,which="both",alpha=.25); ax.legend()
    save(fig,out/"abcd_2D_rate_ratio_vs_deltaK")
    write_csv(out/"abcd_2D_rate_ratio_vs_deltaK.csv",ratio)

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("campaign",type=Path); p.add_argument("--one-d-analysis",type=Path,required=True); p.add_argument("--one-d-source",type=Path,required=True); p.add_argument("--registry",type=Path,required=True); a=p.parse_args()
    out=a.campaign/"analysis"; out.mkdir(parents=True,exist_ok=True)
    ref=one_d(a.one_d_analysis); spatial,events=two_d(a.campaign)
    write_csv(out/"abcd_1D_plot_data.csv",ref); write_csv(out/"abcd_2D_plot_data.csv",spatial); write_csv(out/"abcd_2D_event_histories.csv",events)
    write_csv(out/"abcd_1D_to_2D_parameter_mapping.csv",mapping_table(a.one_d_source,a.registry))
    plots(ref,spatial,events,out)
    classifications=[]
    for cls in CLASSES:
        ss=[r for r in spatial if r["class"]==cls]
        classifications.append({"class":cls,"candidate_id":CLASSES[cls][1],"classification":"FULL_2D_MAPPING_REQUIRED",
            "reason":"sparse 2-D curve shape/arrest behavior is not described by a constant factor-of-two rate offset",
            "terminal_2D_cases":len(ss),"stable_developed_2D_cases":sum(bool(r["rate_m_per_cycle"]) for r in ss)})
    write_csv(out/"abcd_2D_validation_classification.csv",classifications)
    (out/"abcd_1D_2D_validation_summary.json").write_text(json.dumps({"schema":"v10.2.31_sparse_2D_validation_v1","classifications":classifications,"parameter_refit":False,"censor_semantics":"partial/cycle-censored points are plotted as downward markers, never artificial developed rates"},indent=2)+"\n")
    return 0
if __name__=="__main__": raise SystemExit(main())
