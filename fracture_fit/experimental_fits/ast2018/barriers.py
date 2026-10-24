"""Executed barrier, derivative and elementary/effective-rate outputs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammainc

from .data import AstData
from .native import historical_start,load_native
from .objective import material_from_normalized,physically_admissible
from .surfaces import surface_diagnostics


EVENT_STRESSES={
    "cleavage":("cleavage_effective_stress_Pa",),
    "emission":("emission_effective_stress_min_Pa","emission_effective_stress_max_Pa"),
    "Peierls":("peierls_effective_stress_min_Pa","peierls_effective_stress_max_Pa"),
    "Taylor":("taylor_effective_stress_min_Pa","taylor_effective_stress_max_Pa"),
}
EVENT_RATES={
    "cleavage":"cleavage_effective_rate_s",
    "emission":"emission_effective_activations_s",
    "Peierls":"peierls_effective_encounter_s",
    "Taylor":"taylor_effective_release_s",
}


def _visited_events(run_dir:Path,scenario:str):
    root=run_dir/f"confirmation_{scenario.lower()}"
    data=AstData.load(run_dir/"input/Ast2018_joint_fit_data")
    rows=[]
    for path in sorted((root/"spatial").glob("*/events.json")):
        condition_id=path.parent.name
        T=float(data.condition(condition_id)["temperature_K"])
        for event in json.loads(path.read_text()):
            if event.get("terminal_ligament_failure"):
                continue
            if not all(key in event for keys in EVENT_STRESSES.values() for key in keys):
                raise RuntimeError(f"missing state-dependent event stresses: {path}")
            rows.append((condition_id,T,event))
    if not rows:
        raise RuntimeError(f"no confirmed spatial events for barrier audit: {root}")
    return rows


def _stress_domains(events):
    domains={}
    for mechanism,keys in EVENT_STRESSES.items():
        values=[float(event[key]) for _,_,event in events for key in keys]
        if not values or not all(math.isfinite(value) and value>=0 for value in values):
            raise RuntimeError(f"invalid visited {mechanism} stress domain")
        domains[mechanism]=(min(values),max(values))
    return domains


def make_barrier_outputs(source:Path,run_dir:Path,scenario:str,vector_path:Path):
    mods,receipt=load_native(source)
    base,physics,_=historical_start(source,mods[2],mods[3].CommonPhysics)
    x=json.loads(vector_path.read_text())["x"]
    candidate=material_from_normalized(base,x)
    surfaces={"cleavage":candidate.cleavage,"emission":candidate.emission,
              "Peierls":candidate.peierls.surface(candidate.emission),
              "Taylor":candidate.taylor.surface(candidate.emission)}
    attempts={"cleavage":physics.cleavage_nu0_s,"emission":physics.emission_nu0_s,
              "Peierls":candidate.peierls.nu0_s,"Taylor":candidate.taylor.nu0_s}
    events=_visited_events(run_dir,scenario)
    domains=_stress_domains(events)
    stress_grids={name:np.linspace(lo,hi if hi>lo else lo+1.0,81)
                  for name,(lo,hi) in domains.items()}
    T_min=min(T for _,T,_ in events);T_max=max(T for _,T,_ in events)
    T_grid=np.linspace(T_min,T_max,25) if T_max>T_min else np.array([T_min])
    output=run_dir/f"barriers_{scenario.lower()}"
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,surface in surfaces.items():
        for T in T_grid:
            for stress in stress_grids[name]:
                values=surface_diagnostics(surface,float(stress),float(T),attempts[name])
                values["mechanism"]=name
                if name=="cleavage":
                    exposure=min(values["elementary_rate_s"]*physics.cleavage_correlation_time_s,1e12)
                    values["effective_cleavage_rate_s"]=(
                        values["elementary_rate_s"] if physics.cleavage_hits<=1+1e-12
                        else float(gammainc(physics.cleavage_hits,exposure)/
                                   physics.cleavage_correlation_time_s))
                else:values["effective_cleavage_rate_s"]=""
                rows.append(values)
    with (output/"barrier_surfaces.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    event_rows=[]
    for condition_id,T,event in events:
        for name,surface in surfaces.items():
            key=EVENT_STRESSES[name][-1]
            stress=float(event[key])
            row=surface_diagnostics(surface,stress,T,attempts[name])
            row.update({"scenario":scenario,"condition_id":condition_id,
                        "event_index":event["index"],"mechanism":name,
                        "effective_rate_kind":EVENT_RATES[name],
                        "effective_rate_s":event.get(EVENT_RATES[name]),
                        "state_stress_min_Pa":event[EVENT_STRESSES[name][0]],
                        "state_stress_max_Pa":event[key],
                        "source_multiplicity_pre":event.get("source_multiplicity_pre"),
                        "rho_mobile_sum_m2":event.get("rho_mobile_sum_m2"),
                        "rho_retained_sum_m2":event.get("rho_retained_sum_m2")})
            event_rows.append(row)
    with (output/"barrier_visited_event_rates.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(event_rows[0]))
        writer.writeheader();writer.writerows(event_rows)
    profile_path=run_dir/f"identifiability_{scenario.lower()}/profiles.json"
    analytical=json.loads((run_dir/f"best_{scenario.lower()}.json").read_text())
    cost_limit=float(analytical["cost"])+max(1.0,.1*abs(float(analytical["cost"])))
    profile_rows=json.loads(profile_path.read_text())
    vectors=[list(x)]
    for profile in profile_rows:
        trial=profile["optimized_x"]
        if (float(profile["cost"])<=cost_limit and trial not in vectors and
            physically_admissible(material_from_normalized(base,trial))):
            vectors.append(trial)
    candidates=[material_from_normalized(base,trial) for trial in vectors]
    envelope=[]
    for name in surfaces:
        for T in T_grid:
            for stress in stress_grids[name]:
                diagnostics=[]
                for trial in candidates:
                    mechanism={"cleavage":trial.cleavage,"emission":trial.emission,
                               "Peierls":trial.peierls.surface(trial.emission),
                               "Taylor":trial.taylor.surface(trial.emission)}[name]
                    diagnostics.append(surface_diagnostics(mechanism,float(stress),float(T),
                                                          attempts[name]))
                row={"mechanism":name,"temperature_K":float(T),
                     "stress_Pa":float(stress),"n_profile_vectors":len(vectors),
                     "working_cost_limit":cost_limit}
                for field in ("G_eV","H_eV","S_eV_per_K","V_m3","elementary_rate_s"):
                    values=[float(d[field]) for d in diagnostics if np.isfinite(d[field])]
                    row[f"{field}_min"]=min(values) if values else ""
                    row[f"{field}_max"]=max(values) if values else ""
                envelope.append(row)
    with (output/"barrier_sensitivity_envelope.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(envelope[0]))
        writer.writeheader();writer.writerows(envelope)
    receipt.update({"visited_event_stress_domains_Pa":domains,
                    "stress_coordinate":"native mechanism-specific nonnegative barrier argument; Peierls and Taylor include local resolved stress and Taylor forest-spacing factor",
                    "temperature_min_K":T_min,"temperature_max_K":T_max,
                    "n_spatial_events":len(events),
                    "profile_sensitivity_envelope_vectors":vectors,
                    "profile_sensitivity_working_cost_limit":cost_limit,
                    "profile_sensitivity_interpretation":"conditional working-cost envelope, not a confidence interval",
                    "barrier_parameter_keys":{
                        "cleavage":"candidate.cleavage",
                        "emission":"candidate.emission",
                        "Peierls":"candidate.peierls.surface(candidate.emission)",
                        "Taylor":"candidate.taylor.surface(candidate.emission)"},
                    "effective_rate_available":"state-dependent native rates sampled at confirmed spatial events; Peierls/Taylor weighted rates are undefined when their carrier density is zero"})
    (output/"barrier_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    colors=["#1f77b4","#d95f02","#2ca02c","#9467bd"]
    plot_T=np.linspace(T_min,T_max,4)
    fig,axes=plt.subplots(4,4,figsize=(15,12))
    metrics=[("G_eV","G* (eV)"),("H_eV","H* (eV)"),
             ("S_over_kB","S*/kB"),("V_m3","V* (m³)")]
    for col,(name,surface) in enumerate(surfaces.items()):
        for j,T in enumerate(plot_T):
            stress_grid=stress_grids[name]
            d=[surface_diagnostics(surface,float(stress),T,attempts[name])
               for stress in stress_grid]
            for row,(key,ylabel) in enumerate(metrics):
                axes[row,col].plot(stress_grid*1e-9,[v[key] for v in d],
                                   color=colors[j],lw=1,label=f"{T:.0f} K")
        for row,(key,ylabel) in enumerate(metrics):
            ax=axes[row,col]
            ax.set_xlabel("Effective barrier stress (GPa)")
            ax.set_ylabel(ylabel)
            ax.text(.03,.95,name,transform=ax.transAxes,va="top",fontsize=8)
            if row==0:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/"barrier_derivatives.png",dpi=180)
    fig.savefig(output/"barrier_derivatives.svg");plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(name,surface) in zip(axes.flat,surfaces.items()):
        stress_grid=stress_grids[name]
        for j,T in enumerate(plot_T):
            d=[surface_diagnostics(surface,float(stress),T,attempts[name])
               for stress in stress_grid]
            ax.plot(stress_grid*1e-9,[max(v["elementary_rate_s"],1e-300) for v in d],
                    color=colors[j],lw=1,label=f"elementary {T:.0f} K")
            if name=="cleavage":
                ax.plot(stress_grid*1e-9,[max(v["elementary_rate_s"] if physics.cleavage_hits<=1+1e-12
                    else float(gammainc(physics.cleavage_hits,
                        min(v["elementary_rate_s"]*physics.cleavage_correlation_time_s,1e12)) /
                        physics.cleavage_correlation_time_s),1e-300) for v in d],
                    color=colors[j],lw=1,ls="--",label=f"multi-hit {T:.0f} K")
        ax.set_yscale("log")
        ax.set_xlabel("Effective barrier stress (GPa)")
        ax.set_ylabel("Rate (s⁻¹)")
        ax.text(.03,.95,name,transform=ax.transAxes,va="top",fontsize=8)
        ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(output/"barrier_rates.png",dpi=180)
    fig.savefig(output/"barrier_rates.svg");plt.close(fig)
    return output
