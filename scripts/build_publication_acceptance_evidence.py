#!/usr/bin/env python3
"""Build claim-specific evidence and corrected publication figures.

This is a read-only analysis of retained accepted runs.  It never advances a
solver state or changes a native checkpoint.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "review_packet" / "publication_acceptance_0c72ca2b"
FINAL = ROOT / "integration_evidence" / "final_fracture_campaign"
REVIEW = ROOT / "review_packet" / "final_fracture_campaign_a5a5883"


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _attestation(case: Path) -> dict:
    return read_json(case / "shared_runtime_attestation.post.json")


def build_method_map() -> dict:
    families = [
        ("baseline_and_reused_anchor", ROOT / "integration_evidence/current_science_pilots_6abbf10/pf/DBTT_1200K/accommodation_enabled"),
        ("focused_seeds", FINAL / "focused_seed_dbtt_1200K_31415_accommodation_enabled"),
        ("rate", FINAL / "sensitivity_rate_dbtt_1200K_100x_accommodation_enabled"),
        ("orientation", FINAL / "sensitivity_orientation_dbtt_1200K_45deg_accommodation_enabled"),
        ("bulk_uniform", FINAL / "sensitivity_source_dbtt_1200K_bulk_uniform_run2"),
        ("tip_plus_bulk", FINAL / "sensitivity_source_dbtt_1200K_tip_plus_bulk"),
        ("developed_growth", ROOT / "integration_evidence/current_science_pilots_6abbf10/pf/DBTT_1200K/developed_1000um_current_v2_run3"),
    ]
    rows = []
    for name, case in families:
        att = _attestation(case)
        lifecycle = next(x for x in att["observed_bindings"] if x["role"] == "source_lifecycle")
        rows.append({
            "family": name,
            "case": str(case.relative_to(ROOT)),
            "producer_commit": att["source"]["commit"],
            "execution_verified": att["execution_verified"],
            "source_location": att["options"]["source_location"],
            "source_lifecycle": att["options"]["source_lifecycle"],
            "plasticity": att["options"]["plasticity"],
            "integration": att["options"]["integration"],
            "lifecycle_module": lifecycle["module"],
            "lifecycle_attribute": lifecycle["attribute"],
            "native_argv": att["native_argv"],
        })
    cavity_request = read_json(FINAL / "cavity_interaction_ramp_0p2s_run1/shared_execution_request.json")
    cavity_lifecycle = next(x for x in cavity_request["plan"]["bindings"] if x["role"] == "source_lifecycle")
    rows.append({
        "family": "cavity_current_source",
        "case": "integration_evidence/final_fracture_campaign/cavity_interaction_ramp_0p2s_run1",
        "producer_commit": read_json(FINAL / "cavity_interaction_ramp_0p1s/shared_runtime_attestation.post.json")["source"]["commit"],
        "execution_verified": "accepted checkpoint restored after audit-export failure",
        "source_location": cavity_request["options"]["source_location"],
        "source_lifecycle": cavity_request["options"]["source_lifecycle"],
        "plasticity": cavity_request["options"]["plasticity"],
        "integration": cavity_request["options"]["integration"],
        "lifecycle_module": cavity_lifecycle["module"],
        "lifecycle_attribute": cavity_lifecycle["attribute"],
        "native_argv": cavity_request["native_args"],
    })
    return {
        "schema": "fracture-method-to-result-map/2",
        "disposition": "CALCULATIONS_CORRECT_PROSE_STALE_FOR_PERSISTENT_FAMILIES",
        "families": rows,
        "persistent_source": {
            "model_id": "v10.2.21_persistent_site_backstress_blunting",
            "source_model": "persistent_areal_sites_backstress_limited_no_inventory",
            "multiplicity": "M_s = rho_site0 * c_arc * r_tip * w_eff",
            "mean_activation_root": "Delta N_s = M_s lambda_e(max(sigma_drive,s-k_back sqrt(rho_s+a_s Delta N_s),0),T) Delta t",
            "root_condition": "implicit backward-Euler complementarity; bounded by the density at which resolved drive is blocked",
            "inventory_consumed": False,
            "advance_refresh": False,
            "discrete_emission_event": False,
            "line_content": "Delta L_s = a_s Delta N_s; signed by resolved shear and deposited into the source-zone bins",
            "state_update_order": "mean persistent emission, exact mobile/retained exchange, physical mobile advection/escape; adaptive outer accepted-state transaction",
            "signed_state": "positive and negative mobile, retained, accumulated-slip, wake-mobile, wake-retained, and wake-slip arrays",
            "recovery": "selected monotonic parameter rows set explicit mobile and retained temporal recovery to zero; transport escape and moving-frame wake transfer remain active",
            "shielding": "2-D unit signed perturbation kernels act on signed retained content; mobile shielding fraction and wake-shielding selection are explicit runtime inputs",
            "source_files": [
                "arrhenius_fracture/persistent_site_source_v10221.py",
                "arrhenius_fracture/signed_burgers_shared_v1025.py",
                "arrhenius_fracture/current_source_runtime_bindings.py",
            ],
        },
        "opening_process": {
            "sampled_process": "cleavage/opening first-passage threshold Xi=-ln(R)",
            "survival": "exp[-integral sum_j lambda_j dt] over the discrete candidate channels",
            "mean_field_not_extra_hazard": "persistent source population evolution changes conditional rates but is not counted as an additional sampled event channel",
            "partial_progress": "d_xi=L_c dB is an executed hazard-linear moving-coordinate closure, not a probability expectation or resolved crack surface",
            "commit": "geometry changes only at B=1 by the sampled threshold-correlated event length",
        },
        "bulk_and_cavity": {
            "bulk_uniform": "bulk full-field source/evolution replaces the tip process contribution because --no-tip-plasticity is active",
            "tip_plus_bulk": "tip persistent state and full-field bulk update are composed; their effects are not additive scalar strengths",
            "cavity": "separate current-source topology owner with explicit global event scheduling and positive conservative vacancy transport; it is not the persistent reusable source closure",
        },
        "impact": {
            "rerun_required": False,
            "reason": "runtime receipts and installed lifecycle bindings consistently select the intended persistent implementation for the paper PF families; correction is equation/prose/figure provenance only",
        },
    }


PF_FEM_CASES = {
    ("tip_only", "PF"): FINAL / "pf_peak1000_theta0_seed8666_tip_only_matched",
    ("tip_only", "FEM_CZM"): FINAL / "fem_czm_peak1000_theta0_seed8666_tip_only_matched_run5",
    ("full_field", "PF"): FINAL / "pf_peak1000_theta0_seed8666_full_field_exact_matched",
    ("full_field", "FEM_CZM"): FINAL / "fem_czm_peak1000_theta0_seed8666_full_field_exact_matched",
}


def _pre_event(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if float(r["crack_extension_m"]) <= 0.0]


def _interp(rows: list[dict[str, str]], key: str, opening_um: float) -> float:
    x = np.array([float(r["Uapp_m"])*1e6 for r in rows])
    y = np.array([float(r[key]) for r in rows])
    if opening_um < x.min()-1e-12 or opening_um > x.max()+1e-12:
        return math.nan
    return float(np.interp(opening_um, x, y))


def build_pf_fem_common_load() -> tuple[dict, list[dict]]:
    histories = {key: read_csv(path / "steps_1000K.csv") for key, path in PF_FEM_CASES.items()}
    openings = [0.2, 2.0, 5.0, 10.0, 12.0]
    rows = []
    for scope in ("tip_only", "full_field"):
        pf = _pre_event(histories[(scope, "PF")]); fem = _pre_event(histories[(scope, "FEM_CZM")])
        for u in openings:
            values = {}
            for backend, data in (("PF", pf), ("FEM_CZM", fem)):
                values[backend] = {
                    "reaction_N_per_m": _interp(data, "Ftop_N", u),
                    "KJ_MPa_sqrt_m": _interp(data, "KJ_Pa_sqrtm", u)*1e-6,
                    "sigma_tip_GPa": _interp(data, "sigma_tip_Pa", u)*1e-9,
                    "source_backstress_GPa": _interp(data, "sigma_back_Pa", u)*1e-9,
                    "mobile_plus_retained": _interp(data, "N_em", u),
                }
            rows.append({
                "scope": scope, "total_opening_um": u,
                **{f"PF_{k}": v for k,v in values["PF"].items()},
                **{f"FEM_CZM_{k}": v for k,v in values["FEM_CZM"].items()},
                "reaction_ratio_PF_over_FEM": values["PF"]["reaction_N_per_m"] / values["FEM_CZM"]["reaction_N_per_m"],
                "KJ_ratio_PF_over_FEM": values["PF"]["KJ_MPa_sqrt_m"] / values["FEM_CZM"]["KJ_MPa_sqrt_m"],
            })
    elastic = read_json(REVIEW / "PF_FEM_COMMON_SPECIMEN.json")
    cohesive = {}
    for scope in ("tip_only", "full_field"):
        base = PF_FEM_CASES[(scope, "FEM_CZM")]
        elements = read_csv(base / "czm_1000K/cohesive_elements.csv")
        events = read_json(base / "czm_1000K/czm_advance_log.json")
        cohesive[scope] = {
            "cohesive_subsegments": len(elements),
            "fully_separated_subsegments": sum(float(r["damage"]) == 1.0 for r in elements),
            "physical_event_count": len({int(x["physical_event_index"]) for x in events}),
            "cohesive_length_um": sum(float(r["length_m"]) for r in elements)*1e6,
            "state_fields": list(elements[0]),
            "interpretation": "native accepted cohesive topology state; each exported segment is fully separated (damage=1), and numerical subsegments are grouped by physical_event_index",
        }
    low = [r for r in rows if r["scope"] == "tip_only" and r["total_opening_um"] <= 12]
    return ({
        "schema": "production-pf-fem-common-load/1",
        "same_input_openings_um": openings,
        "table": "PF_FEM_COMMON_LOAD.csv",
        "common_mesh_elastic_control": elastic,
        "native_cohesive_state": cohesive,
        "causal_disposition": {
            "finding": "production K(U) and reaction(U) differ before fracture and before substantial state evolution; the tip-only difference is already present at 0.2 um",
            "cause_supported": "different production discretizations and native J supports/operators, not the bulk law alone",
            "cause_excluded": "loading convention, specimen dimensions, plane-strain setting, and base constitutive tensor are excluded by the identical-mesh control",
            "scope_limit": "the retained production histories do not by themselves provide a two-level mesh sequence for each backend; therefore no production-mesh convergence rate is claimed",
            "tip_only_reaction_ratio_range": [min(r["reaction_ratio_PF_over_FEM"] for r in low), max(r["reaction_ratio_PF_over_FEM"] for r in low)],
            "tip_only_K_ratio_range": [min(r["KJ_ratio_PF_over_FEM"] for r in low), max(r["KJ_ratio_PF_over_FEM"] for r in low)],
        },
        "event_K_is_not_mechanical_parity": True,
    }, rows)


def plot_pf_fem(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    colors = {"PF": "#2166ac", "FEM_CZM": "#b2182b"}
    for col, scope in enumerate(("tip_only", "full_field")):
        data = [r for r in rows if r["scope"] == scope]
        for backend in ("PF", "FEM_CZM"):
            label = "sharp-wake PF" if backend == "PF" else "FEM/CZM"
            axes[0,col].plot([r["total_opening_um"] for r in data], [r[f"{backend}_reaction_N_per_m"]/1e3 for r in data], "o-", ms=3, lw=1.1, color=colors[backend], label=label)
            axes[1,col].plot([r["total_opening_um"] for r in data], [r[f"{backend}_KJ_MPa_sqrt_m"] for r in data], "o-", ms=3, lw=1.1, color=colors[backend])
        axes[0,col].text(.04,.92, "tip-only" if scope == "tip_only" else "full field", transform=axes[0,col].transAxes, va="top")
        axes[1,col].set_xlabel("Total opening (µm)")
    axes[0,0].set_ylabel("Reaction (kN m$^{-1}$)")
    axes[1,0].set_ylabel(r"Native $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
    axes[0,0].legend(frameon=False, fontsize=8)
    for tag, ax in zip("abcd", axes.flat):
        ax.text(.02,.98, tag, transform=ax.transAxes, va="top", fontweight="bold")
        ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout(); fig.savefig(path, dpi=350); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def plot_baseline(path: Path) -> None:
    rows = read_csv(REVIEW / "PF_BASELINE_96_DATA.csv")
    families = ["DBTT", "Peak", "Weak-T", "Ceramic-like"]
    fig, axes = plt.subplots(2,2,figsize=(7.2,5.6), sharey=True)
    for ax, family, tag in zip(axes.flat, families, "abcd"):
        for arm, color, marker in (("accommodation_enabled","#b2182b","o"),("opening_only","#2166ac","s")):
            data=sorted((r for r in rows if r["material_row"]==family and r["arm"]==arm), key=lambda r:float(r["temperature_K"]))
            ax.plot([float(r["temperature_K"]) for r in data],[float(r["KJ_MPa_sqrt_m"]) for r in data],marker+"-",ms=3,lw=1.0,color=color,label=arm.replace("_"," "))
        ax.text(.04,.92,f"{tag}  {family}",transform=ax.transAxes,va="top",fontweight="bold")
        ax.set_xlabel("Temperature (K)"); ax.tick_params(direction="in",top=True,right=True)
    axes[0,0].set_ylabel(r"First-event $K_J$ (MPa$\sqrt{\mathrm{m}}$)"); axes[1,0].set_ylabel(r"First-event $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
    axes[0,0].legend(frameon=False,fontsize=7)
    fig.tight_layout(); fig.savefig(path,dpi=350); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def plot_focused(path: Path) -> dict:
    rows=read_csv(REVIEW/"PF_FOCUSED_SEEDS_PAIRED.csv")
    groups=[("DBTT",1200.0,"DBTT, 1200 K"),("Peak",900.0,"Peak, 900 K")]
    fig,ax=plt.subplots(figsize=(6.6,3.8))
    stats={}
    for i,(family,temp,label) in enumerate(groups):
        data=[r for r in rows if r["material_row"]==family and float(r["temperature_K"])==temp]
        y=np.array([float(r["paired_gap_MPa_sqrt_m"]) for r in data]); x=i+np.linspace(-.10,.10,len(y))
        ax.scatter(x,y,s=22,color=("#2166ac" if i==0 else "#b2182b"),zorder=3)
        mean=float(y.mean()); sem=float(y.std(ddof=1)/math.sqrt(len(y))); t=2.364624251
        ax.errorbar(i,mean,yerr=t*sem,fmt="D",mfc="white",mec="black",color="black",capsize=4,lw=1.2,zorder=4)
        stats[label]={"n":len(y),"mean":mean,"sample_sd":float(y.std(ddof=1)),"ci95":[mean-t*sem,mean+t*sem]}
    ax.set_xticks(range(2),[g[2] for g in groups]); ax.set_xlabel("Paired condition")
    ax.set_ylabel(r"Enabled minus opening-only $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
    ax.tick_params(direction="in",top=True,right=True); fig.tight_layout(); fig.savefig(path,dpi=350); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)
    return stats


def plot_developed(path: Path) -> None:
    rows=read_csv(REVIEW/"DEVELOPED_DRIVING_HISTORY.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7)); colors={"cold_300K":"#2166ac","hot_1200K":"#b2182b"}
    for condition,color in colors.items():
        data=[r for r in rows if r["condition"]==condition and r["observation_stage"]!="committed"]
        t=np.array([float(r["model_time_s"]) for r in data]); opening=np.array([float(r["opening_um"]) for r in data]); reaction=np.array([float(r["reaction_N"]) for r in data])/1e3; ext=np.array([float(r["stage_extension_um"]) for r in data])
        axes[0,0].plot(t,opening,lw=.9,color=color,label=condition.replace("_"," "))
        axes[0,1].plot(t,reaction,lw=.8,color=color)
        axes[1,0].plot(t,ext,lw=.9,color=color)
        mask=t>=np.quantile(t,.90)
        axes[1,1].plot(t[mask],ext[mask],lw=.9,color=color)
    axes[0,0].set_ylabel("Total opening (µm)"); axes[0,1].set_ylabel("Reaction (kN m$^{-1}$)")
    axes[1,0].set_ylabel("Projected extension (µm)"); axes[1,1].set_ylabel("Projected extension (µm)")
    for ax in axes.flat: ax.set_xlabel("Accepted model time (s)"); ax.tick_params(direction="in",top=True,right=True)
    axes[0,0].legend(frameon=False,fontsize=7)
    for tag,ax in zip("abcd",axes.flat): ax.text(.02,.98,tag,transform=ax.transAxes,va="top",fontweight="bold")
    axes[1,1].text(.98,.04,"final 10% of stored time",transform=axes[1,1].transAxes,ha="right",va="bottom",fontsize=7)
    fig.tight_layout(); fig.savefig(path,dpi=350); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def plot_native_pf_stages(path: Path) -> list[dict]:
    base=ROOT.parent/"fracture_only_arrhenius_paper/analysis/c17_pf_led_completion_v1/runs/propagation_50um/propagation__dbtt__T1200/field_snapshot_data_1200K"
    stages=[("pre-event",120),("first committed advance",127),("late propagation",185)]
    arrays=[]; side=[]
    for label,step in stages:
        source=next(base.glob(f"*step_{step:06d}.npz")); z=np.load(source); arrays.append((label,step,source,z))
    stress_max=np.percentile(np.concatenate([z["sigma1_Pa"] for _,_,_,z in arrays]),99.5)/1e9
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.7),sharex=True,sharey=True)
    for col,(label,step,source,z) in enumerate(arrays):
        nodes=z["nodes_m"]*1e6; elems=z["elems"]
        s=axes[0,col].tripcolor(nodes[:,0],nodes[:,1],elems,z["sigma1_Pa"]/1e9,shading="flat",cmap="magma",vmin=0,vmax=stress_max)
        d_elem=np.mean(z["damage"][elems],axis=1)
        d=axes[1,col].tripcolor(nodes[:,0],nodes[:,1],elems,d_elem,shading="flat",cmap="gray_r",vmin=0,vmax=1)
        for row,ax in enumerate(axes[:,col]):
            ax.set_xlim(450,650); ax.set_ylim(-100,100); ax.set_aspect("equal"); ax.tick_params(direction="in",top=True,right=True,labelsize=7)
            ax.text(.03,.96,f"{'abc'[col] if row==0 else 'def'[col]}  {label}",transform=ax.transAxes,va="top",color="white" if row==0 else "black",fontsize=7,fontweight="bold")
        axes[1,col].set_xlabel("x (µm)")
        side.append({"stage":label,"step":step,"source":str(source),"KJ_MPa_sqrt_m":float(z["KJ_Pa_sqrt_m"])*1e-6,"tip_x_um":float(z["a_tip_m"])*1e6,"stress_max_GPa":float(np.max(z["sigma1_Pa"]))*1e-9,"rho_min_m2":float(np.min(z["rho_m2"])),"rho_max_m2":float(np.max(z["rho_m2"]))})
    axes[0,0].set_ylabel("y (µm)"); axes[1,0].set_ylabel("y (µm)")
    fig.subplots_adjust(left=.08,right=.89,bottom=.10,top=.98,wspace=.08,hspace=.10)
    c1=fig.add_axes([.91,.57,.018,.32]); fig.colorbar(s,cax=c1).set_label("Maximum principal stress (GPa)",fontsize=7)
    c2=fig.add_axes([.91,.15,.018,.32]); fig.colorbar(d,cax=c2).set_label("Native damage",fontsize=7)
    fig.savefig(path,dpi=350); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)
    return side


def _state_sigma1(state):
    from arrhenius_fracture.fem import stress_state
    return stress_state(state.mesh,state.displacement,state.ep_gp,state.damage,state.elasticity_D,state.material)[2]


def plot_cavity(path: Path) -> list[dict]:
    from arrhenius_fracture.shared_v5_cavity_adapter import _restore_pair
    start,_=_restore_pair(FINAL/"cavity_interaction_ramp_0p1s/checkpoint/accepted_v5_pair.json")
    end,_=_restore_pair(FINAL/"cavity_interaction_ramp_0p2s_run1/checkpoint/accepted_v5_pair.json")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.5)); side=[]
    all_states=[start["enabled"],end["enabled"],start["disabled"],end["disabled"]]
    vmax=np.percentile(np.concatenate([_state_sigma1(s) for s in all_states]),99)
    for ax,state,label,stage,tag in zip(axes.flat,all_states,["cavity enabled","cavity enabled","cavity suppressed","cavity suppressed"],["start","end","start","end"],"abcd"):
        sigma=_state_sigma1(state)/1e9; tri=ax.tripcolor(state.mesh.nodes[:,0]*1e6,state.mesh.nodes[:,1]*1e6,state.mesh.elems,sigma,shading="flat",cmap="magma",vmin=0,vmax=vmax/1e9)
        for branch in state.crack_network.branches:
            p=np.asarray(branch.path)*1e6; ax.plot(p[:,0],p[:,1],color="cyan",lw=1.2)
        cavity=None
        if state.void_state is not None and state.void_state.cavities:
            cavity=state.void_state.cavities[0]; circle=plt.Circle(np.asarray(cavity.center_m)*1e6,cavity.radius_m*1e6,fill=False,color="white",lw=1.0); ax.add_patch(circle)
        tip=np.asarray(state.crack_network.branches[0].tip)*1e6
        ax.set_xlim(tip[0]-120,tip[0]+180); ax.set_ylim(-100,100); ax.set_aspect("equal")
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.tick_params(direction="in",top=True,right=True)
        ax.text(.02,.98,f"{tag}  {label}, {stage}",transform=ax.transAxes,va="top",color="white",fontsize=8,fontweight="bold")
        if cavity is not None:
            # An equal-scale strip makes the 42 nm resolved cavity visible while
            # retaining the full 7.4 µm solid ligament to the accepted front.
            inset=ax.inset_axes([.32,.08,.62,.17])
            inset.set_facecolor("#202020")
            inset.plot([tip[0],cavity.center_m[0]*1e6],[tip[1],cavity.center_m[1]*1e6],color="cyan",lw=1.0)
            inset.add_patch(plt.Circle(np.asarray(cavity.center_m)*1e6,cavity.radius_m*1e6,fill=False,color="white",lw=1.2))
            inset.set_xlim(tip[0]-.2,cavity.center_m[0]*1e6+.2); inset.set_ylim(-.25,.25)
            inset.set_aspect("equal"); inset.tick_params(direction="in",colors="white",labelsize=5,length=2)
            for spine in inset.spines.values(): spine.set_color("white")
            inset.text(.02,.88,"actual-scale ligament",transform=inset.transAxes,color="white",fontsize=5,va="top")
        side.append({"arm":label,"stage":stage,"tip_x_um":float(tip[0]),"tip_y_um":float(tip[1]),"cavity_center_x_um":None if cavity is None else cavity.center_m[0]*1e6,"cavity_center_y_um":None if cavity is None else cavity.center_m[1]*1e6,"cavity_radius_nm":None if cavity is None else cavity.radius_m*1e9,"sigma1_min_GPa":float(np.min(sigma)),"sigma1_max_GPa":float(np.max(sigma)),"nodes":state.mesh.nn,"elements":state.mesh.ne})
    fig.subplots_adjust(left=.09,right=.84,bottom=.10,top=.98,wspace=.30,hspace=.28)
    cax=fig.add_axes([.87,.20,.025,.60]); cb=fig.colorbar(tri,cax=cax); cb.set_label("Maximum principal stress (GPa)")
    fig.savefig(path,dpi=350); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)
    return side


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True)
    method=build_method_map(); write_json(OUT/"METHOD_TO_RESULT_MAP.json",method)
    lines=["# Method-to-result closure","",f"**Disposition:** `{method['disposition']}`","","The retained paper PF families execute the persistent reusable source law. The finite depleted inventory and advance-refresh prose was stale. The cavity calculation intentionally uses the separately declared current-source topology owner.","","| Family | Source support | Lifecycle | Plasticity | Producer |","|---|---|---|---|---|"]
    for r in method["families"]: lines.append(f"| {r['family']} | {r['source_location']} | {r['source_lifecycle']} (`{r['lifecycle_attribute']}`) | {r['plasticity']} | `{r['producer_commit'][:12]}` |")
    lines += ["","No numerical trajectory is rerun: the correction is text/equation/provenance-only. Persistent sites are not consumed or renewed by advance; their state-dependent multiplicity and the implicit backstress complementarity determine the mean emitted population."]
    (OUT/"METHOD_TO_RESULT_MAP.md").write_text("\n".join(lines)+"\n")
    report,rows=build_pf_fem_common_load(); write_json(OUT/"PF_FEM_PRODUCTION_COMMON_LOAD.json",report); write_csv(OUT/"PF_FEM_COMMON_LOAD.csv",rows); plot_pf_fem(rows,OUT/"Figure_8_PF_FEM_production.png")
    plot_baseline(OUT/"Figure_2_PF_baseline_96.png")
    stats=plot_focused(OUT/"Figure_3_PF_focused_gaps.png"); write_json(OUT/"FIGURE_3_ESTIMATOR_CHECK.json",{"schema":"paired-gap-figure-check/1","estimator":"paired enabled-minus-opening-only gap; diamond is mean and bar is two-sided Student-t 95% CI","groups":stats})
    plot_developed(OUT/"Figure_5_developed_physical_history.png")
    native=plot_native_pf_stages(OUT/"Figure_7_native_fields.png"); write_csv(OUT/"PF_NATIVE_FIELD_STAGE_SIDECAR.csv",native)
    cavity=plot_cavity(OUT/"Figure_9_current_cavity_fields.png"); write_csv(OUT/"CAVITY_CURRENT_FIELD_SIDECAR.csv",cavity)
    print(json.dumps({"output":str(OUT),"method_disposition":method["disposition"],"pf_fem_rows":len(rows),"cavity_panels":len(cavity)},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
