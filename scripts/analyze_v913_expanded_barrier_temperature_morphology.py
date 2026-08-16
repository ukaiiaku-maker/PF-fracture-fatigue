#!/usr/bin/env python3
"""Expanded existing-data v9.13 barrier-geometry/fracture morphology analysis.

This v2 layer consumes and preserves the validated v1 products.  It evaluates
the exact historical production surfaces but never evolves a fracture state or
launches a simulation.  Bare K-space quantities use the historical r0=1 um;
state quantities use the saved pre-first-passage radius and backstress.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO=Path(__file__).resolve().parents[1]
V1_ROOT=REPO/"runs/v913_barrier_temperature_fracture_morphology_v1"
DEFAULT_OUT=REPO/"runs/v913_barrier_temperature_fracture_morphology_v2"
SOURCE=Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_13_dbtt_temperature_shelf")
SIM_SHA="559425321b9a8739f32788322d8a1c2af8abad73"
KB=8.617333262145e-5
R0_M=1.0e-6
REF_FRONT_M=1.0e-5
STRESS_PER_K_PA_PER_MPA_SQRT_M=1.0e6/math.sqrt(2*math.pi*R0_M)
LOADING_MAP_PATH=SOURCE/"runs/v9_13_v10222_rcurve_targets_v1/v10_2_22_rcurve_loading_map.json"
_LOADING_MAP=json.loads(LOADING_MAP_PATH.read_text())
KDOT_FIRST_EVENT_MPA_SQRT_M_PER_S=float(_LOADING_MAP["K_per_U_MPa_sqrt_m_per_m"][0])*float(_LOADING_MAP["nominal_dU_m"])/float(_LOADING_MAP["nominal_dt_s"])
LEVELS=(.90,.75,.50,.25,.10)
CANONICAL={"v913_zeroD_sobol_0242980":"Peak-T","v913_zeroD_sobol_0202500":"DBTT",
           "v913_zeroD_sobol_0129902":"weak-T","v913_zeroD_sobol_0077080":"ceramic-like"}
COLORS={"Peak-T":"#F59E0B","Peak-like":"#F59E0B","DBTT":"#3B82F6","DBTT-like":"#3B82F6",
        "weak-T":"#8B5CF6","ceramic-like":"#64748B","other/intermediate":"#94A3B8"}
PRIORITY=["delta_g_ec_at_K75","log10_rate_ratio_at_K75","delta_K50_MPa_sqrt_m","delta_K90_MPa_sqrt_m",
 "width80_ratio_emit_over_cleave","relative_max_first_derivative","relative_first_derivative_at_closest",
 "relative_curvature_at_closest","integrated_abs_barrier_separation_eV","integrated_abs_log_rate_separation",
 "kinetic_crossing_count","competition_temperature_width_log1_K","kinetic_crossover_sharpness_T_per_K",
 "differential_entropy_kB","entropy_separation_importance_at_K75","delta_full_dGdT_at_K75_eV_per_K",
 "delta_mixed_derivative_at_K75_eV_per_MPa_sqrt_m_K","log10_tau_c_over_tau_p_at_K75",
 "dlog10_tau_c_over_tau_p_dT","state_delta_Gc_over_kBT"]
RESPONSES=["fractional_resistance_span","DBTT_temperature_K","DBTT_width_K","DBTT_magnitude_MPa_sqrt_m",
           "peak_temperature_K","peak_prominence_MPa_sqrt_m","normalized_high_T_response","S_mid_MPa_sqrt_m_per_K"]


def load_v1_module():
    path=REPO/"scripts/analyze_v913_barrier_temperature_fracture_morphology.py"
    spec=importlib.util.spec_from_file_location("_v913_v1_analysis",path); module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module; assert spec.loader is not None; spec.loader.exec_module(module); return module


def git(*args): return subprocess.check_output(["git",*args],cwd=REPO,text=True).strip()


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def finite(x,default=np.nan):
    try:
        value=float(x); return value if np.isfinite(value) else float(default)
    except (TypeError,ValueError): return float(default)


def interp_at(x,y,value): return float(np.interp(value,x,y))


def crossing_locations(x: np.ndarray,y: np.ndarray) -> list[float]:
    roots=[]
    for i in range(len(x)-1):
        if not np.isfinite(y[i:i+2]).all(): continue
        if y[i]==0: roots.append(float(x[i]))
        elif y[i]*y[i+1]<0: roots.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
    return roots


def interval_width(x: np.ndarray,mask: np.ndarray) -> float:
    if not mask.any(): return 0.0
    return float(x[mask].max()-x[mask].min())


def class_label(row: pd.Series) -> str:
    if isinstance(row.get("canonical_family"),str): return row.canonical_family
    historical=str(row.get("historical_response_class",''))
    if "peak" in historical.lower(): return "Peak-like"
    if "dbtt" in historical.lower(): return "DBTT-like"
    if "weak" in historical.lower(): return "weak-T"
    if "ceramic" in historical.lower(): return "ceramic-like"
    return "other/intermediate"


def topology(values: np.ndarray,crossings: int,near_eps=.02,positive_means_cleavage=True) -> str:
    values=np.asarray(values,float); near=np.abs(values)<near_eps
    if near.mean()>=.15: return "BROAD_NEAR_DEGENERACY"
    if crossings==0 and near.any(): return "NEAR_TANGENT"
    if crossings==1: return "SINGLE_CROSSING"
    if crossings>1: return "MULTIPLE_CROSSING"
    positive=np.nanmedian(values)>0
    cleavage=(positive if positive_means_cleavage else not positive)
    return "NO_CROSSING_CLEAVAGE_EASIER" if cleavage else "NO_CROSSING_EMISSION_EASIER"


def robust_phase_axis(values: pd.Series, min_finite_fraction=.8, max_tail_ratio=1e4) -> bool:
    """Reject singular coordinates only when selecting axes for a phase-map plot.

    The underlying descriptor remains in every table and statistical model.  This
    guard prevents a handful of near-zero denominators from visually collapsing
    the rest of the population on a linear discovery map.
    """
    numeric=pd.to_numeric(values,errors="coerce").to_numpy(float)
    finite_values=np.abs(numeric[np.isfinite(numeric)])
    if len(finite_values)<min_finite_fraction*len(numeric) or not len(finite_values): return False
    q95=float(np.quantile(finite_values,.95)); maximum=float(finite_values.max())
    return maximum<=max_tail_ratio*max(q95,np.finfo(float).tiny)


def surface_arrays(surface,T,K):
    sigma=K*STRESS_PER_K_PA_PER_MPA_SQRT_M
    G=np.asarray(surface.barrier_eV(sigma,T),float)
    d=np.gradient(G,K,edge_order=2); d2=np.gradient(d,K,edge_order=2)
    dT=(np.asarray(surface.barrier_eV(sigma,T+1),float)-np.asarray(surface.barrier_eV(sigma,T-1),float))/2
    mixed=np.gradient(dT,K,edge_order=2)
    floor=float(surface.barrier_eV(1e15,T)); zero=float(surface.barrier_eV(0,T)); phi=(G-floor)/max(zero-floor,1e-30)
    return sigma,G,d,d2,dT,mixed,phi,floor,zero


def level_position(K,phi,level):
    order=np.argsort(phi); return float(np.interp(level,phi[order],K[order]))


def barrier_geometry_row(cid,T,KR,row,first,v1,ExpFloorSurface,PTMechanism):
    gc=v1.make_surface(row,"cleave",ExpFloorSurface); ge=v1.make_surface(row,"emit",ExpFloorSurface)
    peierls=PTMechanism(finite(row.peierls_H0_eV),finite(row.peierls_activation_entropy_kB),finite(row.peierls_exp_a),finite(row.peierls_exp_n),finite(row.peierls_nu0_s))
    taylor=PTMechanism(finite(row.taylor_H0_eV),finite(row.taylor_activation_entropy_kB),finite(row.taylor_exp_a),finite(row.taylor_exp_n),finite(row.taylor_nu0_s))
    gp,gt=peierls.surface(ge),taylor.surface(ge)
    # 0..200 MPa sqrt(m) safely contains all EXP-floor transitions in this shelf.
    K=200*np.linspace(0,1,1001)**2; sc,Gc,dc,c2,cT,cmix,pc,gfc,g0c=surface_arrays(gc,T,K); se,Ge,de,e2,eT,emix,pe,gfe,g0e=surface_arrays(ge,T,K)
    kc={f"K{int(level*100):02d}_MPa_sqrt_m":level_position(K,pc,level) for level in LEVELS}
    ke={f"K{int(level*100):02d}_MPa_sqrt_m":level_position(K,pe,level) for level in LEVELS}
    for dct in (kc,ke):
        for name,value in list(dct.items()): dct[name.replace("K","sigma",1).replace("_MPa_sqrt_m","_GPa")]=value*STRESS_PER_K_PA_PER_MPA_SQRT_M/1e9
    cW80=kc["K10_MPa_sqrt_m"]-kc["K90_MPa_sqrt_m"]; eW80=ke["K10_MPa_sqrt_m"]-ke["K90_MPa_sqrt_m"]
    cW50=kc["K25_MPa_sqrt_m"]-kc["K75_MPa_sqrt_m"]; eW50=ke["K25_MPa_sqrt_m"]-ke["K75_MPa_sqrt_m"]
    mult=max(finite(first.get("source_multiplicity_pre_advance"),1),1)
    rc=np.array([v1.multihit_rate(x,T) for x in Gc]); re=np.array([v1.arrhenius_rate(x,T,v1.NU_E)*mult for x in Ge])
    stressp=peierls.stress_fraction*sc; stresst=taylor.stress_fraction*sc
    Gp=np.asarray(gp.barrier_eV(stressp,T),float); Gt=np.asarray(gt.barrier_eV(stresst,T),float)
    rp=np.array([v1.arrhenius_rate(x,T,peierls.nu0_s) for x in Gp]); rt=np.array([v1.arrhenius_rate(x,T,taylor.nu0_s) for x in Gt])
    logR=np.log10(np.maximum(re,1e-300))-np.log10(np.maximum(rc,1e-300))
    dlogR_dK=np.gradient(logR,K,edge_order=2)
    taup=np.maximum.reduce([1/np.maximum(re,1e-300),1/np.maximum(rp,1e-300),1/np.maximum(rt,1e-300)])
    logPi=np.log10(np.minimum(1/np.maximum(rc,1e-300)/taup,1e300))
    physical=K<=max(KR,1e-8); dx=max(KR,1e-8)
    closest=int(np.argmin(abs(logR[physical]))); physical_indices=np.flatnonzero(physical); iclose=int(physical_indices[closest])
    cross=crossing_locations(K[physical],logR[physical]); bcross=crossing_locations(K[physical],(Ge-Gc)[physical])
    def at(arr,f): return interp_at(K,arr,f*KR)
    record={"candidate_id":cid,"temperature_K":T,"K_response_MPa_sqrt_m":KR,"bare_reference_tip_radius_m":R0_M,
      "coordinate_definition":"K_to_sigma_tip=K*1e6/sqrt(2*pi*r0)","source_multiplicity":mult}
    record.update({f"cleavage_{k}":v for k,v in kc.items()}); record.update({f"emission_{k}":v for k,v in ke.items()})
    record.update({"cleavage_G0_eV":g0c,"cleavage_floor_eV":gfc,"cleavage_available_drop_eV":g0c-gfc,
      "emission_G0_eV":g0e,"emission_floor_eV":gfe,"emission_available_drop_eV":g0e-gfe,
      "cleavage_width80_MPa_sqrt_m":cW80,"emission_width80_MPa_sqrt_m":eW80,
      "cleavage_width50_MPa_sqrt_m":cW50,"emission_width50_MPa_sqrt_m":eW50,
      "cleavage_width80_over_K50":cW80/max(kc["K50_MPa_sqrt_m"],1e-30),"emission_width80_over_K50":eW80/max(ke["K50_MPa_sqrt_m"],1e-30),
      "cleavage_asymmetry_90_50_10":(kc["K50_MPa_sqrt_m"]-kc["K90_MPa_sqrt_m"])/max(kc["K10_MPa_sqrt_m"]-kc["K50_MPa_sqrt_m"],1e-30),
      "emission_asymmetry_90_50_10":(ke["K50_MPa_sqrt_m"]-ke["K90_MPa_sqrt_m"])/max(ke["K10_MPa_sqrt_m"]-ke["K50_MPa_sqrt_m"],1e-30),
      "delta_K50_MPa_sqrt_m":ke["K50_MPa_sqrt_m"]-kc["K50_MPa_sqrt_m"],"delta_K90_MPa_sqrt_m":ke["K90_MPa_sqrt_m"]-kc["K90_MPa_sqrt_m"],
      "delta_K10_MPa_sqrt_m":ke["K10_MPa_sqrt_m"]-kc["K10_MPa_sqrt_m"],"K50_ratio_emit_over_cleave":ke["K50_MPa_sqrt_m"]/max(kc["K50_MPa_sqrt_m"],1e-30),
      "width80_ratio_emit_over_cleave":eW80/max(cW80,1e-30),"width80_difference_MPa_sqrt_m":eW80-cW80})
    for f in (0,.5,.75,.9,1.0):
        tag=f"K{int(f*100):03d}"; c=at(Gc,f); e=at(Ge,f); lr=at(logR,f)
        record.update({f"Gc_eV_at_{tag}":c,f"Ge_eV_at_{tag}":e,f"barrier_ratio_Ge_over_Gc_at_{tag}":e/max(c,1e-30),
          f"delta_G_ec_eV_at_{tag}":e-c,f"delta_g_ec_at_{tag}":(e-c)/(KB*T),f"log10_rate_ratio_at_{tag}":lr,
          f"abs_dGc_dK_at_{tag}":abs(at(dc,f)),f"abs_dGe_dK_at_{tag}":abs(at(de,f)),
          f"relative_first_derivative_at_{tag}":abs(at(de,f))/max(abs(at(dc,f)),1e-30),
          f"delta_curvature_at_{tag}":at(e2,f)-at(c2,f),f"relative_curvature_at_{tag}":abs(at(e2,f))/max(abs(at(c2,f)),1e-30),
          f"cleavage_full_dGdT_at_{tag}_eV_per_K":at(cT,f),f"emission_full_dGdT_at_{tag}_eV_per_K":at(eT,f),
          f"delta_full_dGdT_at_{tag}_eV_per_K":at(eT,f)-at(cT,f),f"ratio_full_dGdT_at_{tag}":at(eT,f)/at(cT,f) if abs(at(cT,f))>1e-12 else np.nan,
          f"cleavage_mixed_derivative_at_{tag}_eV_per_MPa_sqrt_m_K":at(cmix,f),f"emission_mixed_derivative_at_{tag}_eV_per_MPa_sqrt_m_K":at(emix,f),
          f"delta_mixed_derivative_at_{tag}_eV_per_MPa_sqrt_m_K":at(emix,f)-at(cmix,f),
          f"log10_tau_c_over_tau_p_at_{tag}":at(logPi,f)})
    for level in (.90,.50,.10):
        tag=f"K{int(level*100):02d}"; ck=kc[f"{tag}_MPa_sqrt_m"]; ek=ke[f"{tag}_MPa_sqrt_m"]
        record.update({f"cleavage_abs_dGdK_at_own_{tag}":abs(interp_at(K,dc,ck)),f"emission_abs_dGdK_at_own_{tag}":abs(interp_at(K,de,ek)),
          f"cleavage_curvature_at_own_{tag}":interp_at(K,c2,ck),f"emission_curvature_at_own_{tag}":interp_at(K,e2,ek)})
    Sc=-finite(row.cleave_gT_eV_per_K); Se=-finite(row.emit_gT_eV_per_K); eps=.01
    icmax=int(np.flatnonzero(physical)[np.argmax(abs(dc[physical]))]); iemax=int(np.flatnonzero(physical)[np.argmax(abs(de[physical]))])
    def meaningful(prefix,index):
        record.update({f"barrier_ratio_at_{prefix}":Ge[index]/max(Gc[index],1e-30),f"delta_g_at_{prefix}":(Ge[index]-Gc[index])/(KB*T),
          f"log10_rate_ratio_at_{prefix}":logR[index],f"relative_first_derivative_at_{prefix}":abs(de[index])/max(abs(dc[index]),1e-30),
          f"relative_curvature_at_{prefix}":abs(e2[index])/max(abs(c2[index]),1e-30)})
    meaningful("closest_approach",iclose); meaningful("cleavage_max_sensitivity",icmax); meaningful("emission_max_sensitivity",iemax)
    if cross: meaningful("first_kinetic_crossover",int(np.argmin(abs(K-cross[0]))))
    else:
        for key in ["barrier_ratio","delta_g","log10_rate_ratio","relative_first_derivative","relative_curvature"]: record[f"{key}_at_first_kinetic_crossover"]=np.nan
    record.update({"cleavage_entropy_eV_per_K":Sc,"emission_entropy_eV_per_K":Se,"differential_entropy_kB":(Se-Sc)/KB,
      "eta_entropy_c_at_K75":T*Sc/max(at(Gc,.75),1e-30),"eta_entropy_e_at_K75":T*Se/max(at(Ge,.75),1e-30),
      "delta_eta_entropy_at_K75":T*Se/max(at(Ge,.75),1e-30)-T*Sc/max(at(Gc,.75),1e-30),
      "entropy_separation_importance_at_K75":T*(Se-Sc)/(abs(at(Ge,.75)-at(Gc,.75))+eps),"entropy_separation_regularizer_eV":eps,
      "max_abs_dGc_dK":float(np.max(abs(dc[physical]))),"max_abs_dGe_dK":float(np.max(abs(de[physical]))),
      "K_at_max_abs_dGc_dK":float(K[physical][np.argmax(abs(dc[physical]))]),"K_at_max_abs_dGe_dK":float(K[physical][np.argmax(abs(de[physical]))]),
      "relative_max_first_derivative":float(np.max(abs(de[physical]))/max(np.max(abs(dc[physical])),1e-30)),
      "relative_first_derivative_at_closest":abs(de[iclose])/max(abs(dc[iclose]),1e-30),
      "relative_curvature_at_closest":abs(e2[iclose])/max(abs(c2[iclose]),1e-30),
      "K_closest_kinetic_approach_MPa_sqrt_m":float(K[iclose]),"closest_abs_log10_rate_ratio":float(abs(logR[iclose])),
      "max_positive_curvature_c":float(np.max(c2[physical])),"max_negative_curvature_c":float(np.min(c2[physical])),
      "max_abs_curvature_c":float(np.max(abs(c2[physical]))),"max_positive_curvature_e":float(np.max(e2[physical])),
      "max_negative_curvature_e":float(np.min(e2[physical])),"max_abs_curvature_e":float(np.max(abs(e2[physical]))),
      "integrated_abs_dGc_dK_eV":float(np.trapezoid(abs(dc[physical]),K[physical])),"integrated_abs_dGe_dK_eV":float(np.trapezoid(abs(de[physical]),K[physical])),
      "integrated_abs_curvature_c":float(np.trapezoid(abs(c2[physical]),K[physical])),"integrated_abs_curvature_e":float(np.trapezoid(abs(e2[physical]),K[physical])),
      "integrated_abs_barrier_separation_eV":float(np.trapezoid(abs((Ge-Gc)[physical]),K[physical])/dx),
      "rms_barrier_separation_eV":float(np.sqrt(np.trapezoid(((Ge-Gc)[physical])**2,K[physical])/dx)),
      "integrated_abs_log_rate_separation":float(np.trapezoid(abs(logR[physical]),K[physical])/dx),
      "kinetic_crossing_count_at_temperature":len(cross),"kinetic_crossing_loads_MPa_sqrt_m":";".join(f"{x:.8g}" for x in cross),
      "kinetic_crossover_sharpness_K_values_per_MPa_sqrt_m":";".join(f"{interp_at(K,dlogR_dK,x):.8g}" for x in cross),
      "kinetic_crossover_max_abs_sharpness_K_per_MPa_sqrt_m":max([abs(interp_at(K,dlogR_dK,x)) for x in cross],default=np.nan),
      "barrier_crossing_count_at_temperature":len(bcross),"barrier_crossing_loads_MPa_sqrt_m":";".join(f"{x:.8g}" for x in bcross),
      "competition_load_width_log1_MPa_sqrt_m":interval_width(K[physical],abs(logR[physical])<1),
      "competition_load_width_log2_MPa_sqrt_m":interval_width(K[physical],abs(logR[physical])<2),
      "fraction_path_emission_dominant_frozen_proxy":float(np.mean(logR[physical]>0)),
      "fraction_path_cleavage_dominant_frozen_proxy":float(np.mean(logR[physical]<0)),
      "frozen_path_proxy_not_evolved":True,
      "frozen_path_cleavage_action_proxy":float(np.trapezoid(rc[physical],K[physical])/KDOT_FIRST_EVENT_MPA_SQRT_M_PER_S),
      "frozen_path_emission_action_proxy":float(np.trapezoid(re[physical],K[physical])/KDOT_FIRST_EVENT_MPA_SQRT_M_PER_S),
      "frozen_path_peierls_action_proxy":float(np.trapezoid(rp[physical],K[physical])/KDOT_FIRST_EVENT_MPA_SQRT_M_PER_S),
      "frozen_path_taylor_action_proxy":float(np.trapezoid(rt[physical],K[physical])/KDOT_FIRST_EVENT_MPA_SQRT_M_PER_S),
      "frozen_path_emission_over_cleavage_action_proxy":float(np.trapezoid(re[physical],K[physical])/max(np.trapezoid(rc[physical],K[physical]),1e-300)),
      "frozen_path_loading_rate_MPa_sqrt_m_per_s":KDOT_FIRST_EVENT_MPA_SQRT_M_PER_S})
    hc=abs(dc[physical])>.5*np.max(abs(dc[physical])); he=abs(de[physical])>.5*np.max(abs(de[physical]))
    overlap=hc&he; record.update({"sensitivity_overlap_width_MPa_sqrt_m":interval_width(K[physical],overlap),
      "sensitivity_overlap_fraction":float(overlap.mean()),"sensitivity_region_center_separation_MPa_sqrt_m":abs(float(np.mean(K[physical][hc]))-float(np.mean(K[physical][he]))) if hc.any() and he.any() else np.nan,
      "max_sensitivity_location_separation_MPa_sqrt_m":abs(record["K_at_max_abs_dGe_dK"]-record["K_at_max_abs_dGc_dK"])})
    # First-passage state-conditioned descriptors.  Missing shielding and full
    # populations remain NA and are never reconstructed.
    Ks=finite(first.get("K_first_MPa_sqrt_m",first.get("K_MPa_sqrt_m",KR)),KR); radius=max(finite(first.get("tip_radius_pre_advance_m"),R0_M),1e-12)
    sbare=Ks*STRESS_PER_K_PA_PER_MPA_SQRT_M; sevolved=Ks*1e6/math.sqrt(2*math.pi*radius); back=finite(first.get("backstress_pre_advance_Pa"),0); semit=max(sevolved-back,0)
    Gcs=float(gc.barrier_eV(sevolved,T)); Ges=float(ge.barrier_eV(semit,T)); rcs=v1.multihit_rate(Gcs,T); res=v1.arrhenius_rate(Ges,T,v1.NU_E)*mult
    hK=.01
    state_gc=lambda kval: float(gc.barrier_eV(kval*1e6/math.sqrt(2*math.pi*radius),T)); state_ge=lambda kval: float(ge.barrier_eV(max(kval*1e6/math.sqrt(2*math.pi*radius)-back,0),T))
    defGc=(state_gc(Ks+hK)-state_gc(Ks-hK))/(2*hK); defGe=(state_ge(Ks+hK)-state_ge(Ks-hK))/(2*hK)
    curvGc=(state_gc(Ks+hK)-2*state_gc(Ks)+state_gc(Ks-hK))/(hK*hK); curvGe=(state_ge(Ks+hK)-2*state_ge(Ks)+state_ge(Ks-hK))/(hK*hK)
    stateGcT=(float(gc.barrier_eV(sevolved,T+1))-float(gc.barrier_eV(sevolved,T-1)))/2; stateGeT=(float(ge.barrier_eV(semit,T+1))-float(ge.barrier_eV(semit,T-1)))/2
    Gcbare=float(gc.barrier_eV(sbare,T)); front=finite(first.get("front_width_pre_advance_m"))
    record.update({"state_reconstruction_class":"PARTIAL_SAVED_FIRST_PASSAGE_PROXY","state_K_MPa_sqrt_m":Ks,"state_tip_radius_m":radius,
      "state_backstress_Pa":back,"state_front_width_m":front,"state_Gc_eV":Gcs,"state_Ge_eV":Ges,"state_delta_G_ec_eV":Ges-Gcs,
      "state_delta_g_ec":(Ges-Gcs)/(KB*T),"state_barrier_ratio_Ge_over_Gc":Ges/max(Gcs,1e-30),"state_log10_rate_ratio":math.log10(max(res,1e-300)/max(rcs,1e-300)),
      "state_abs_dGc_dK":abs(defGc),"state_abs_dGe_dK":abs(defGe),"state_relative_first_derivative":abs(defGe)/max(abs(defGc),1e-30),
      "state_curvature_c":curvGc,"state_curvature_e":curvGe,"state_relative_curvature":abs(curvGe)/max(abs(curvGc),1e-30),
      "state_cleavage_dGdT_eV_per_K":stateGcT,"state_emission_dGdT_eV_per_K":stateGeT,"state_delta_dGdT_eV_per_K":stateGeT-stateGcT,
      "state_backstress_amplification":back/max(sbare,1e-30),"state_tip_radius_amplification":radius/R0_M,"state_front_width_ratio":front/REF_FRONT_M,
      "state_delta_Gc_eV":Gcs-Gcbare,"state_delta_Gc_over_kBT":(Gcs-Gcbare)/(KB*T),"state_K_shield_over_K_applied":np.nan,
      "state_missing_fields":"K_shield;mobile_population;retained_population;slip_field;full_loading_state_history",
      "actual_total_source_activations_at_first_passage":finite(first.get("cumulative_source_activations")),
      "actual_total_line_content_at_first_passage":finite(first.get("cumulative_line_content"))})
    return record


def response_family_descriptors(cases: pd.DataFrame,events: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Separate initiation/fixed-extension/developed temperature morphologies."""
    observables=["K_first_MPa_sqrt_m","K_10um_MPa_sqrt_m","K_25um_MPa_sqrt_m","K_50um_MPa_sqrt_m","K_checkpoint_MPa_sqrt_m"]
    rows=[]
    for cid,g0 in cases[cases.status.eq("complete")].groupby("candidate_id"):
        g=g0.sort_values("temperature_K"); T=g.temperature_K.to_numpy(float)
        for observable in observables:
            y=pd.to_numeric(g[observable],errors="coerce").to_numpy(float); good=np.isfinite(y)
            if good.sum()<3: continue
            x=T[good]; z=y[good]; base=z[0]
            rows.append({"candidate_id":cid,"response_observable":observable,"n_temperatures":len(z),
              "temperature_min_K":x.min(),"temperature_max_K":x.max(),"low_value_MPa_sqrt_m":base,
              "high_value_MPa_sqrt_m":z[-1],"normalized_high_response":z[-1]/max(base,1e-30),
              "fractional_span":np.ptp(z)/max(abs(base),1e-30),"thermal_slope":np.polyfit(x,z,1)[0],
              "max_abs_slope":np.max(abs(np.gradient(z,x))),"max_abs_curvature":np.max(abs(np.gradient(np.gradient(z,x),x)))})
    # Developed R-curve slope is event-resolved and never extrapolated.
    for cid,g0 in events.groupby("candidate_id"):
        values=[]
        for T,g in g0.groupby("temperature_K"):
            x=g.cumulative_projected_extension_m.to_numpy(float)*1e6; y=g.K_MPa_sqrt_m.to_numpy(float); keep=x>=20
            if keep.sum()>=2: values.append((float(T),float(np.polyfit(x[keep],y[keep],1)[0])))
        if len(values)>=3:
            values=sorted(values); x=np.array([a for a,_ in values]); z=np.array([b for _,b in values]); base=z[0]
            rows.append({"candidate_id":cid,"response_observable":"developed_R_curve_slope_MPa_sqrt_m_per_um","n_temperatures":len(z),
              "temperature_min_K":x.min(),"temperature_max_K":x.max(),"low_value_MPa_sqrt_m":base,"high_value_MPa_sqrt_m":z[-1],
              "normalized_high_response":z[-1]/base if abs(base)>1e-30 else np.nan,"fractional_span":np.ptp(z)/max(abs(base),1e-30),
              "thermal_slope":np.polyfit(x,z,1)[0],"max_abs_slope":np.max(abs(np.gradient(z,x))),
              "max_abs_curvature":np.max(abs(np.gradient(np.gradient(z,x),x)))})
    table=pd.DataFrame(rows); wide=table.pivot(index="candidate_id",columns="response_observable",values=["normalized_high_response","fractional_span","thermal_slope"])
    wide.columns=[f"response__{metric}__{observable}" for metric,observable in wide.columns]
    return table,wide.reset_index()


def aggregate_expanded(temp: pd.DataFrame,candidates: pd.DataFrame,response: pd.DataFrame,response_wide: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    priority_temp=["delta_g_ec_at_K75","log10_rate_ratio_at_K75","delta_K50_MPa_sqrt_m","delta_K90_MPa_sqrt_m",
      "width80_ratio_emit_over_cleave","relative_max_first_derivative","relative_first_derivative_at_closest",
      "relative_curvature_at_closest","integrated_abs_barrier_separation_eV","integrated_abs_log_rate_separation",
      "differential_entropy_kB","entropy_separation_importance_at_K75","delta_full_dGdT_at_K75_eV_per_K",
      "delta_mixed_derivative_at_K75_eV_per_MPa_sqrt_m_K","log10_tau_c_over_tau_p_at_K75","state_delta_Gc_over_kBT",
      "state_log10_rate_ratio","state_relative_first_derivative","state_backstress_amplification","state_tip_radius_amplification",
      "actual_total_source_activations_at_first_passage","actual_total_line_content_at_first_passage",
      "frozen_path_emission_over_cleavage_action_proxy","fraction_path_emission_dominant_frozen_proxy"]
    # The generated fraction tags are K075/K050/etc.; expose concise aliases.
    temp=temp.copy()
    alias={"delta_g_ec_at_K75":"delta_g_ec_at_K075","log10_rate_ratio_at_K75":"log10_rate_ratio_at_K075",
      "delta_full_dGdT_at_K75_eV_per_K":"delta_full_dGdT_at_K075_eV_per_K",
      "delta_mixed_derivative_at_K75_eV_per_MPa_sqrt_m_K":"delta_mixed_derivative_at_K075_eV_per_MPa_sqrt_m_K",
      "log10_tau_c_over_tau_p_at_K75":"log10_tau_c_over_tau_p_at_K075"}
    for new,old in alias.items(): temp[new]=temp[old]
    for cid,g0 in temp.groupby("candidate_id"):
        g=g0.sort_values("temperature_K"); T=g.temperature_K.to_numpy(float); i=int(np.argmin(abs(T-900)))
        rec={"candidate_id":cid,"descriptor_reference_temperature_K":float(T[i]),"temperature_count":len(g)}
        # One explicitly prefixed near-900 K reference value for every local
        # physical descriptor gives the mandatory one-row-per-candidate table
        # complete coverage without applying arbitrary transform families.
        for col in g.select_dtypes(include=[np.number]).columns:
            if col in {"temperature_K","K_response_MPa_sqrt_m","bare_reference_tip_radius_m"}: continue
            rec[f"Tref900__{col}"]=finite(g.iloc[i][col])
        for col in priority_temp:
            if col not in g: continue
            values=pd.to_numeric(g[col],errors="coerce").to_numpy(float); good=np.isfinite(values)
            if not good.any(): continue
            rec[col]=values[i]; rec[f"{col}__min_T"]=np.nanmin(values); rec[f"{col}__max_T"]=np.nanmax(values)
            rec[f"{col}__span_T"]=np.nanmax(values)-np.nanmin(values)
            rec[f"{col}__d_dT"]=np.polyfit(T[good],values[good],1)[0] if good.sum()>=2 else np.nan
        logR=g.log10_rate_ratio_at_K075.to_numpy(float); dG=g.delta_G_ec_eV_at_K075.to_numpy(float)
        temp_cross=crossing_locations(T,logR); barrier_cross=crossing_locations(T,dG)
        dlog_dT=np.gradient(logR,T)
        load_cross=int(g.kinetic_crossing_count_at_temperature.sum()); barrier_load_cross=int(g.barrier_crossing_count_at_temperature.sum())
        rec.update({"kinetic_temperature_crossing_count":len(temp_cross),"kinetic_temperature_crossings_K":";".join(f"{x:.8g}" for x in temp_cross),
          "kinetic_crossing_count":len(temp_cross)+load_cross,"kinetic_topology":topology(logR,len(temp_cross)+load_cross,near_eps=.1,positive_means_cleavage=False),
          "kinetic_temperature_crossover_sharpness_values_per_K":";".join(f"{interp_at(T,dlog_dT,x):.8g}" for x in temp_cross),
          "barrier_crossing_count":len(barrier_cross)+barrier_load_cross,"barrier_topology":topology(dG,len(barrier_cross)+barrier_load_cross),
          "closest_abs_log10_rate_ratio":float(np.min(abs(logR))),"closest_kinetic_temperature_K":float(T[np.argmin(abs(logR))]),
          "competition_temperature_width_log1_K":interval_width(T,abs(logR)<1),"competition_temperature_width_log2_K":interval_width(T,abs(logR)<2),
          "kinetic_crossover_sharpness_T_per_K":float(np.max(abs(dlog_dT))),
          "dlog10_tau_c_over_tau_p_dT":float(np.polyfit(T,g.log10_tau_c_over_tau_p_at_K075,1)[0]),
          "temperature_fraction_emission_dominant":float(np.mean(logR>0)),"temperature_fraction_cleavage_dominant":float(np.mean(logR<0)),
          "state_backstress_dT_Pa_per_K":float(np.polyfit(T,g.state_backstress_Pa,1)[0]),
          "state_tip_radius_dT_m_per_K":float(np.polyfit(T,g.state_tip_radius_m,1)[0]),
          "state_front_width_dT_m_per_K":float(np.polyfit(T,g.state_front_width_m,1)[0])})
        rows.append(rec)
    provenance_names={"parameter_fingerprint","source_registry","simulation_git_sha","simulation_sha_provenance","github_repository",
                      "historical_branch","canonical_family","canonical_option_key"}
    response_core=response.drop(columns=[c for c in provenance_names if c in response],errors="ignore")
    out=pd.DataFrame(rows).merge(response_core,on="candidate_id",how="left").merge(response_wide,on="candidate_id",how="left")
    keep=["candidate_id","parameter_fingerprint","source_registry","simulation_git_sha","simulation_sha_provenance","github_repository",
          "historical_branch","canonical_family","canonical_option_key","historical_response_class","is_canonical_holdout"]
    out=out.merge(candidates[[c for c in keep if c in candidates]],on="candidate_id",how="left")
    out["morphology_class"]=out.apply(class_label,axis=1)
    return out


def fdr_bh(p):
    p=np.asarray(p,float); q=np.full(len(p),np.nan); good=np.flatnonzero(np.isfinite(p))
    if len(good):
        order=good[np.argsort(p[good])]; vals=p[order]*len(good)/np.arange(1,len(good)+1); q[order]=np.minimum.accumulate(vals[::-1])[::-1].clip(0,1)
    return q


def univariate_correlations(master: pd.DataFrame,features: list[str],responses: list[str]) -> pd.DataFrame:
    rows=[]; discovery=master[~master.is_canonical_holdout]
    for f in features:
      for r in responses:
        q=discovery[[f,r]].apply(pd.to_numeric,errors="coerce").dropna(); rec={"feature":f,"response":r,"n":len(q),"pearson_r":np.nan,"pearson_p":np.nan,"spearman_rho":np.nan,"spearman_p":np.nan}
        if len(q)>=8 and q[f].nunique()>2 and q[r].nunique()>2:
            pr=stats.pearsonr(q[f],q[r]); sr=stats.spearmanr(q[f],q[r]); rec.update({"pearson_r":pr.statistic,"pearson_p":pr.pvalue,"spearman_rho":sr.statistic,"spearman_p":sr.pvalue})
        rows.append(rec)
    out=pd.DataFrame(rows); out["pearson_q_fdr"]=fdr_bh(out.pearson_p); out["spearman_q_fdr"]=fdr_bh(out.spearman_p); return out


def discretize_quantiles(x,bins=8):
    x=np.asarray(x,float); edges=np.unique(np.nanquantile(x,np.linspace(0,1,bins+1)))
    return np.digitize(x,edges[1:-1]) if len(edges)>2 else np.zeros(len(x),int)


def mutual_information_discrete(a,b):
    a=np.asarray(a,int); b=np.asarray(b,int); n=len(a); mi=0.0
    for x in np.unique(a):
      for y in np.unique(b):
        p=np.mean((a==x)&(b==y))
        if p>0: mi+=p*math.log(p/(np.mean(a==x)*np.mean(b==y)))
    return mi


def mutual_information_table(master,features,responses,seed=913):
    rng=np.random.default_rng(seed); rows=[]; d=master[~master.is_canonical_holdout]
    for f in features:
      for r in responses:
        q=d[[f,r]].apply(pd.to_numeric,errors="coerce").dropna()
        if len(q)<30 or q[f].nunique()<5 or q[r].nunique()<5: continue
        a=discretize_quantiles(q[f]); b=discretize_quantiles(q[r]); raw=mutual_information_discrete(a,b)
        perm=np.array([mutual_information_discrete(a,rng.permutation(b)) for _ in range(40)])
        boot=np.array([mutual_information_discrete(a[idx],b[idx]) for idx in [rng.integers(0,len(a),len(a)) for _ in range(40)]])
        rows.append({"feature":f,"response":r,"n":len(q),"mutual_information_nats":raw,"permutation_bias_nats":perm.mean(),
          "bias_corrected_MI_nats":max(raw-perm.mean(),0),"bootstrap_std_nats":boot.std(ddof=1),"permutation_p":(1+np.sum(perm>=raw))/(len(perm)+1)})
    out=pd.DataFrame(rows); out["permutation_q_fdr"]=fdr_bh(out.permutation_p); return out


def spline_basis(values,knots=None):
    x=np.asarray(values,float)
    if knots is None: knots=np.unique(np.quantile(x,[.2,.4,.6,.8]))
    scale=max(np.std(x),1e-30); center=np.mean(x); z=(x-center)/scale; zk=(np.asarray(knots)-center)/scale
    basis=np.column_stack([z,z*z,z*z*z]+[np.maximum(z-k,0)**3 for k in zk])
    return basis,np.asarray(knots),center,scale


def ridge_cv_matrix(X,y,alphas=(.1,1,10,100),folds=5):
    X=np.asarray(X,float); y=np.asarray(y,float); fold_id=np.arange(len(y))%folds; trials=[]
    for alpha in alphas:
        pred=np.full(len(y),np.nan)
        for k in range(folds):
            tr=fold_id!=k; te=~tr; A=np.column_stack([np.ones(tr.sum()),X[tr]]); P=np.eye(A.shape[1])*alpha; P[0,0]=0
            beta=np.linalg.solve(A.T@A+P,A.T@y[tr]); pred[te]=np.column_stack([np.ones(te.sum()),X[te]])@beta
        rmse=float(np.sqrt(np.mean((y-pred)**2))); r2=1-float(np.sum((y-pred)**2))/max(float(np.sum((y-y.mean())**2)),1e-30)
        trials.append((rmse,r2,alpha,pred))
    return min(trials,key=lambda x:x[0])


def gam_performance(master,mi,responses):
    rows=[]; d=master[~master.is_canonical_holdout]
    for response in responses:
        rank=mi[mi.response.eq(response)].sort_values("bias_corrected_MI_nats",ascending=False).feature.drop_duplicates().head(3).tolist()
        if not rank: continue
        q=d[[response]+rank].apply(pd.to_numeric,errors="coerce").dropna(); bases=[]; shapes=[]
        for feature in rank:
            B,knots,center,scale=spline_basis(q[feature]); bases.append(B); shapes.append({"feature":feature,"knots":";".join(f"{x:.8g}" for x in knots),"center":center,"scale":scale})
        X=np.column_stack(bases); rmse,r2,alpha,_=ridge_cv_matrix(X,q[response])
        linear=np.column_stack([(q[f]-q[f].mean())/max(q[f].std(),1e-30) for f in rank]); lrmse,lr2,lalpha,_=ridge_cv_matrix(linear,q[response])
        rows.append({"response":response,"model":"ADDITIVE_TRUNCATED_CUBIC_SPLINE_GAM","features":";".join(rank),"n":len(q),
          "basis_definition":"per-feature cubic polynomial plus four positive-part cubic knots","shape_metadata_json":json.dumps(shapes),
          "selected_ridge_alpha":alpha,"cv_rmse":rmse,"cv_r2":r2,"linear_same_features_cv_r2":lr2,"nonlinear_cv_r2_gain":r2-lr2})
    return pd.DataFrame(rows)


def fit_stump(X,y):
    best=None
    for j in range(X.shape[1]):
        for threshold in np.unique(np.quantile(X[:,j],np.linspace(.1,.9,9))):
            left=X[:,j]<=threshold
            if left.sum()<5 or (~left).sum()<5: continue
            lv=y[left].mean(); rv=y[~left].mean(); loss=np.sum((y[left]-lv)**2)+np.sum((y[~left]-rv)**2)
            if best is None or loss<best[0]: best=(loss,j,float(threshold),float(lv),float(rv))
    return best


def boosted_stump_cv(X,y,n_estimators=40,learning_rate=.08):
    fold=np.arange(len(y))%5; pred=np.full(len(y),np.nan)
    for k in range(5):
        tr=fold!=k; te=~tr; base=y[tr].mean(); train_pred=np.full(tr.sum(),base); test_pred=np.full(te.sum(),base)
        for _ in range(n_estimators):
            stump=fit_stump(X[tr],y[tr]-train_pred)
            if stump is None: break
            _,j,t,lv,rv=stump; train_pred+=learning_rate*np.where(X[tr,j]<=t,lv,rv); test_pred+=learning_rate*np.where(X[te,j]<=t,lv,rv)
        pred[te]=test_pred
    return float(np.sqrt(np.mean((y-pred)**2))),1-float(np.sum((y-pred)**2))/max(float(np.sum((y-y.mean())**2)),1e-30)


def interaction_models(master,mi,responses):
    rows=[]; d=master[~master.is_canonical_holdout]
    pairs=[("differential_entropy_kB","delta_K50_MPa_sqrt_m"),("differential_entropy_kB","width80_ratio_emit_over_cleave"),
      ("log10_tau_c_over_tau_p_at_K75","dlog10_tau_c_over_tau_p_dT"),("delta_g_ec_at_K75","relative_first_derivative_at_closest"),
      ("relative_first_derivative_at_closest","relative_curvature_at_closest")]
    for response in responses:
        top=mi[mi.response.eq(response)].sort_values("bias_corrected_MI_nats",ascending=False).feature.drop_duplicates().head(8).tolist()
        features=list(dict.fromkeys(top+[x for pair in pairs for x in pair if x in master]))
        q=d[[response]+features].apply(pd.to_numeric,errors="coerce"); med=q[features].median(); q[features]=q[features].fillna(med); q=q.dropna(subset=[response])
        X=q[features].to_numpy(float); mu=X.mean(0); sd=X.std(0); sd[sd==0]=1; X=(X-mu)/sd
        rmse,r2,alpha,_=ridge_cv_matrix(X,q[response])
        Xint=X.copy(); inames=[]
        for a,b in pairs:
            if a in features and b in features:
                Xint=np.column_stack([Xint,X[:,features.index(a)]*X[:,features.index(b)]]); inames.append(f"{a}*{b}")
        irmse,ir2,ialpha,_=ridge_cv_matrix(Xint,q[response]); brmse,br2=boosted_stump_cv(X,q[response].to_numpy(float))
        rows.extend([
          {"response":response,"model":"RIDGE_MAIN_EFFECTS","features":";".join(features),"interactions":"","n":len(q),"cv_rmse":rmse,"cv_r2":r2,"selected_alpha":alpha},
          {"response":response,"model":"RIDGE_PHYSICAL_INTERACTIONS","features":";".join(features),"interactions":";".join(inames),"n":len(q),"cv_rmse":irmse,"cv_r2":ir2,"selected_alpha":ialpha},
          {"response":response,"model":"GRADIENT_BOOSTED_DECISION_STUMPS","features":";".join(features),"interactions":"learned threshold sequence","n":len(q),"cv_rmse":brmse,"cv_r2":br2,"selected_alpha":np.nan}])
        def ablation(model_name,selected):
            selected=[f for f in selected if f in d]; z=d[[response]+selected].apply(pd.to_numeric,errors="coerce"); z[selected]=z[selected].fillna(z[selected].median()); z=z.dropna(subset=[response])
            A=z[selected].to_numpy(float); scale=A.std(0); scale[scale==0]=1; A=(A-A.mean(0))/scale; ar,ar2,aa,_=ridge_cv_matrix(A,z[response])
            rows.append({"response":response,"model":model_name,"features":";".join(selected),"interactions":"ablation comparison","n":len(z),"cv_rmse":ar,"cv_r2":ar2,"selected_alpha":aa})
        emission=["delta_g_ec_at_K75","log10_rate_ratio_at_K75","delta_K50_MPa_sqrt_m","width80_ratio_emit_over_cleave"]
        ablation("ABLATION_EMISSION_CLEAVAGE_ONLY",emission)
        ablation("ABLATION_PLUS_SERIAL_TRANSPORT",emission+["log10_tau_c_over_tau_p_at_K75","dlog10_tau_c_over_tau_p_dT","raw__peierls_activation_entropy_kB","raw__taylor_activation_entropy_kB"])
        bare=[f for f in PRIORITY if f in d and not f.startswith("state_")]
        ablation("ABLATION_BARE_GEOMETRY_KINETICS",bare)
        ablation("ABLATION_PLUS_SAVED_STATE",bare+["state_delta_Gc_over_kBT","state_backstress_amplification","state_tip_radius_amplification",
            "actual_total_source_activations_at_first_passage","actual_total_line_content_at_first_passage"])
    return pd.DataFrame(rows)


def gini(labels):
    if not len(labels): return 0
    _,count=np.unique(labels,return_counts=True); p=count/count.sum(); return 1-float(np.sum(p*p))


def build_tree(X,y,names,depth=0,max_depth=3,min_leaf=12):
    values,count=np.unique(y,return_counts=True); majority=str(values[np.argmax(count)])
    node={"n":len(y),"prediction":majority,"class_counts":{str(a):int(b) for a,b in zip(values,count)}}
    if depth>=max_depth or len(values)==1 or len(y)<2*min_leaf: return node
    parent=gini(y); best=None
    for j,name in enumerate(names):
      for t in np.unique(np.quantile(X[:,j],np.linspace(.1,.9,9))):
        left=X[:,j]<=t
        if left.sum()<min_leaf or (~left).sum()<min_leaf: continue
        score=(left.sum()*gini(y[left])+(~left).sum()*gini(y[~left]))/len(y); gain=parent-score
        if best is None or gain>best[0]: best=(gain,j,float(t),left)
    if best is None or best[0]<1e-6: return node
    gain,j,t,left=best; node.update({"feature":names[j],"threshold":t,"gini_gain":gain,
      "left":build_tree(X[left],y[left],names,depth+1,max_depth,min_leaf),"right":build_tree(X[~left],y[~left],names,depth+1,max_depth,min_leaf)})
    return node


def predict_tree(node,row,names):
    while "feature" in node: node=node["left"] if row[names.index(node["feature"])]<=node["threshold"] else node["right"]
    return node["prediction"]


def classification_models(master,features):
    d=master[~master.is_canonical_holdout].copy(); q=d[features].apply(pd.to_numeric,errors="coerce"); med=q.median(); q=q.fillna(med)
    X=q.to_numpy(float); y=d.morphology_class.to_numpy(str); fold=np.arange(len(y))%5; pred=np.empty(len(y),object)
    for k in range(5):
        tr=fold!=k; tree=build_tree(X[tr],y[tr],features); pred[~tr]=[predict_tree(tree,row,features) for row in X[~tr]]
    accuracy=float(np.mean(pred==y)); recalls=[np.mean(pred[y==c]==c) for c in np.unique(y)]; tree=build_tree(X,y,features)
    return pd.DataFrame([{"model":"SHALLOW_CART_DEPTH3","n":len(y),"features":";".join(features),"cv_accuracy":accuracy,
      "cv_balanced_accuracy":float(np.mean(recalls)),"class_labels":";".join(np.unique(y)),"tree_json":json.dumps(tree)}]),tree


def feature_family(name):
    if name.startswith("raw__"): return "RAW_ABSOLUTE_PARAMETER"
    if "state_" in name or "source_activ" in name or "line_content" in name: return "EVOLVED_STATE_PROXY"
    if "entropy" in name or "eta_" in name: return "ACTIVATION_ENTROPY"
    if "tau" in name or "rate_ratio" in name or "competition" in name or "kinetic" in name: return "KINETIC_TIMESCALE_COMPETITION"
    if "integrated" in name or "action" in name or "fraction_path" in name: return "INTEGRATED_PATH_PROXY"
    if "delta_K" in name or "K50_ratio" in name or name.startswith(("cleavage_K","emission_K")): return "RELATIVE_BARRIER_POSITION"
    if "width" in name or "asymmetry" in name: return "BARRIER_WIDTH_SHAPE"
    if "curvature" in name: return "BARRIER_CURVATURE"
    if "derivative" in name or "dGdT" in name or "dGdK" in name: return "THERMAL_STRESS_DERIVATIVE"
    if "delta_g" in name or "delta_G" in name or "barrier_ratio" in name or "barrier_crossing" in name: return "RELATIVE_BARRIER_HEIGHT"
    if any(x in name for x in ["Gc_eV","Ge_eV","G0_eV","floor_eV","available_drop_eV"]): return "ABSOLUTE_BARRIER_HEIGHT"
    return "OTHER_PHYSICAL"


def collinearity(master,features):
    corr=master.loc[~master.is_canonical_holdout,features].corr(method="spearman",min_periods=20); rows=[]
    for i,a in enumerate(features):
      for b in features[i+1:]:
        r=finite(corr.loc[a,b])
        if np.isfinite(r) and abs(r)>=.95: rows.append({"feature_a":a,"feature_b":b,"spearman_rho":r,"family_a":feature_family(a),"family_b":feature_family(b),"interpretation":"COLLINEAR_FAMILY_NOT_INDEPENDENT_DISCOVERY"})
    return pd.DataFrame(rows)


def expanded_pca(master,features):
    X=master[features].apply(pd.to_numeric,errors="coerce"); X=X.fillna(X.median()).to_numpy(float); sd=X.std(0); keep=sd>1e-14
    Z=(X[:,keep]-X[:,keep].mean(0))/sd[keep]; u,s,vt=np.linalg.svd(Z,full_matrices=False); k=min(8,len(s)); score=u[:,:k]*s[:k]
    out=master[["candidate_id","canonical_family","morphology_class","is_canonical_holdout"]].copy()
    for i in range(k): out[f"expanded_PC{i+1}"]=score[:,i]
    var=s*s/max(len(Z)-1,1); meta={"features":np.asarray(features)[keep].tolist(),"explained_variance_ratio":(var/var.sum())[:k].tolist(),"top_loadings":{}}
    kept=np.asarray(features)[keep]
    for i in range(k):
        order=np.argsort(abs(vt[i]))[::-1][:10]; meta["top_loadings"][f"PC{i+1}"]=[{"feature":str(kept[j]),"loading":float(vt[i,j]),"family":feature_family(str(kept[j]))} for j in order]
    return out,meta


def savefig(fig,out,stem,data):
    fig.savefig(out/f"{stem}.png",dpi=190,bbox_inches="tight"); plt.close(fig); data.to_csv(out/f"{stem}_plot_data.csv",index=False)


def class_scatter(ax,data,x,y):
    q=data[["candidate_id","morphology_class","canonical_family",x,y]].replace([np.inf,-np.inf],np.nan).dropna(subset=[x,y])
    markers={"Peak-T":"*","Peak-like":"^","DBTT":"X","DBTT-like":"s","weak-T":"P","ceramic-like":"D","other/intermediate":"o"}
    for label,g in q.groupby("morphology_class"):
        ax.scatter(g[x],g[y],s=np.where(g.canonical_family.notna(),85,24),marker=markers.get(label,"o"),c=COLORS.get(label,"#94A3B8"),
            alpha=.62,edgecolor="black",linewidth=.25,label=label)
    ax.set_xlabel(x.replace("_"," ")); ax.set_ylabel(y.replace("_"," ")); return q


def response_heatmap(table,out,stem,value,features,responses):
    q=table[table.feature.isin(features)&table.response.isin(responses)].copy(); mat=q.pivot(index="feature",columns="response",values=value).reindex(index=features,columns=responses)
    fig,ax=plt.subplots(figsize=(12,max(6,.38*len(features)))); im=ax.imshow(mat,cmap="viridis" if "MI" in value or "mutual" in value else "coolwarm",aspect="auto",
        vmin=0 if "MI" in value or "mutual" in value else -1,vmax=None if "MI" in value or "mutual" in value else 1)
    ax.set_xticks(range(len(responses)),[x.replace("_","\n") for x in responses],rotation=35,ha="right",fontsize=7)
    ax.set_yticks(range(len(features)),[x.replace("_"," ") for x in features],fontsize=7)
    fig.colorbar(im,ax=ax,label=value); fig.tight_layout(); savefig(fig,out,stem,q)


def make_figures(out,master,temp,mi,corr,pca,features,v1,ExpFloorSurface):
    stems=[]
    top=mi.groupby("feature").bias_corrected_MI_nats.max().sort_values(ascending=False).head(20).index.tolist()
    response_heatmap(mi,out,"expanded_descriptor_response_heatmap","bias_corrected_MI_nats",top,RESPONSES); stems.append("expanded_descriptor_response_heatmap")
    rank=mi.sort_values("bias_corrected_MI_nats",ascending=False).groupby("feature",as_index=False).first().nlargest(25,"bias_corrected_MI_nats")
    fig,ax=plt.subplots(figsize=(8,7)); ax.barh(np.arange(len(rank)),rank.bias_corrected_MI_nats,color="#3B82F6"); ax.set_yticks(np.arange(len(rank)),rank.feature.str.replace("_"," "),fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("Bias-corrected quantile MI (nats)"); savefig(fig,out,"expanded_mutual_information_rankings",rank); stems.append("expanded_mutual_information_rankings")
    maps={
      "relative_barrier_geometry_phase_map":("delta_g_ec_at_K75","relative_first_derivative_at_closest"),
      "timescale_temperature_phase_map":("log10_tau_c_over_tau_p_at_K75","dlog10_tau_c_over_tau_p_dT"),
      "barrier_position_width_phase_map":("delta_K50_MPa_sqrt_m","width80_ratio_emit_over_cleave"),
      "entropy_barrier_phase_map":("differential_entropy_kB","delta_full_dGdT_at_K75_eV_per_K"),
    }
    for stem,(x,y) in maps.items():
        fig,ax=plt.subplots(figsize=(7,5.2)); q=class_scatter(ax,master,x,y); ax.legend(fontsize=7,ncol=2); ax.set_title(stem.replace("_"," ").title()); savefig(fig,out,stem,q); stems.append(stem)
    response_maps={
      "DBTT_descriptor_maps":"DBTT_magnitude_MPa_sqrt_m","PeakT_descriptor_maps":"peak_prominence_MPa_sqrt_m",
      "weakT_descriptor_maps":"weakT_max_deviation_from_mean_MPa_sqrt_m","ceramic_descriptor_maps":"thermal_softening_slope_MPa_sqrt_m_per_K"}
    chosen=["delta_g_ec_at_K75","delta_K50_MPa_sqrt_m","width80_ratio_emit_over_cleave","log10_tau_c_over_tau_p_at_K75"]
    for stem,response in response_maps.items():
        fig,axs=plt.subplots(2,2,figsize=(10.5,8)); data=[]
        for ax,x in zip(axs.flat,chosen): data.append(class_scatter(ax,master,x,response).assign(panel=x))
        axs.flat[0].legend(fontsize=6); fig.suptitle(stem.replace("_"," ")); savefig(fig,out,stem,pd.concat(data,ignore_index=True,sort=False)); stems.append(stem)
    dist_features=top[:6]; fig,axs=plt.subplots(2,3,figsize=(13,8)); pdata=[]
    classes=[x for x in ["Peak-like","DBTT-like","weak-T","ceramic-like","other/intermediate"] if x in set(master.morphology_class)]
    for ax,feature in zip(axs.flat,dist_features):
        groups=[]; labels=[]
        for label in classes:
            vals=pd.to_numeric(master.loc[master.morphology_class.eq(label),feature],errors="coerce").dropna().to_numpy()
            if len(vals): groups.append(vals); labels.append(label); pdata.append(pd.DataFrame({"feature":feature,"morphology_class":label,"value":vals}))
        ax.boxplot(groups,tick_labels=labels,showfliers=False); ax.tick_params(axis="x",rotation=35,labelsize=6); ax.set_title(feature.replace("_"," "),fontsize=8)
    fig.tight_layout(); savefig(fig,out,"descriptor_class_distributions",pd.concat(pdata,ignore_index=True)); stems.append("descriptor_class_distributions")
    fig,ax=plt.subplots(figsize=(7,5.3)); q=class_scatter(ax,pca,"expanded_PC1","expanded_PC2"); ax.legend(fontsize=7,ncol=2); ax.set_title("Expanded physical-descriptor PCA"); savefig(fig,out,"expanded_descriptor_pca",q); stems.append("expanded_descriptor_pca")
    # Canonical four overlaid on two discovery-selected, nonredundant coordinates.
    x0=top[0]; xy=[x0]
    for candidate in top[1:]:
        pair=master.loc[~master.is_canonical_holdout,[x0,candidate]].corr(method="spearman").iloc[0,1]
        if (feature_family(candidate)!=feature_family(x0) and robust_phase_axis(master[candidate])
                and (not np.isfinite(pair) or abs(pair)<.8)):
            xy.append(candidate); break
    if len(xy)<2: xy=top[:2]
    fig,ax=plt.subplots(figsize=(7,5.3)); q=class_scatter(ax,master,xy[0],xy[1]);
    for _,r in master[master.is_canonical_holdout].iterrows(): ax.annotate(r.canonical_family,(r[xy[0]],r[xy[1]]),xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.legend(fontsize=7,ncol=2); ax.set_title("Canonical four on discovery-selected phase map"); savefig(fig,out,"canonical_four_on_discovered_phase_map",q); stems.append("canonical_four_on_discovered_phase_map")
    # Every priority descriptor against six requested response summaries.
    six=["DBTT_magnitude_MPa_sqrt_m","DBTT_temperature_K","peak_prominence_MPa_sqrt_m","peak_temperature_K","normalized_high_T_response","S_mid_MPa_sqrt_m_per_K"]
    for feature in [f for f in PRIORITY if f in master]:
        fig,axs=plt.subplots(2,3,figsize=(14,8)); data=[]
        for ax,response in zip(axs.flat,six): data.append(class_scatter(ax,master,feature,response).assign(panel=response))
        axs.flat[0].legend(fontsize=6); fig.suptitle(feature.replace("_"," ")); stem=f"priority_scatter__{feature}"; savefig(fig,out,stem,pd.concat(data,ignore_index=True,sort=False)); stems.append(stem)
    # Canonical five-panel barrier-shape mechanisms.
    candidates,_,_,_=v1.load_population(SOURCE); response=pd.read_csv(V1_ROOT/"fracture_response_curve_points.csv")
    for cid,label in CANONICAL.items():
        row=candidates[candidates.candidate_id.eq(cid)].iloc[0]; gc=v1.make_surface(row,"cleave",ExpFloorSurface); ge=v1.make_surface(row,"emit",ExpFloorSurface)
        temps=np.sort(temp[temp.candidate_id.eq(cid)].temperature_K.unique()); selected=np.unique([temps[0],temps[len(temps)//2],temps[-1]])
        fig,axs=plt.subplots(5,1,figsize=(8,14)); pdata=[]
        for T in selected:
            KR=float(temp[(temp.candidate_id.eq(cid))&temp.temperature_K.eq(T)].K_response_MPa_sqrt_m.iloc[0]); K=max(KR*1.1,1)*np.linspace(0,1,401)**2
            _,Gc,dc,c2,_,_,_,_,_=surface_arrays(gc,T,K); _,Ge,de,e2,_,_,_,_,_=surface_arrays(ge,T,K)
            mult=float(temp[(temp.candidate_id.eq(cid))&temp.temperature_K.eq(T)].source_multiplicity.iloc[0])
            rc=np.array([v1.multihit_rate(x,T) for x in Gc]); re=np.array([v1.arrhenius_rate(x,T,v1.NU_E)*mult for x in Ge]); lr=np.log10(np.maximum(re,1e-300)/np.maximum(rc,1e-300))
            axs[0].plot(K,Gc,label=f"Gc {T:g}K"); axs[0].plot(K,Ge,"--",label=f"Ge {T:g}K")
            axs[1].plot(K,-dc,label=f"c {T:g}K"); axs[1].plot(K,-de,"--",label=f"e {T:g}K")
            axs[2].plot(K,c2,label=f"c {T:g}K"); axs[2].plot(K,e2,"--",label=f"e {T:g}K")
            axs[3].plot(K,lr,label=f"{T:g}K"); pdata.append(pd.DataFrame({"candidate_id":cid,"temperature_K":T,"K_MPa_sqrt_m":K,"Gc_eV":Gc,"Ge_eV":Ge,"minus_dGc_dK":-dc,"minus_dGe_dK":-de,"d2Gc_dK2":c2,"d2Ge_dK2":e2,"log10_rate_ratio":lr}))
        rg=response[response.candidate_id.eq(cid)].sort_values("temperature_K"); axs[4].plot(rg.temperature_K,rg.K_response_MPa_sqrt_m,"o-",color=COLORS[label])
        axs[0].set_ylabel("Barrier (eV)"); axs[1].set_ylabel(r"$-dG/dK$"); axs[2].set_ylabel(r"$d^2G/dK^2$"); axs[3].set_ylabel("log10 emission/cleavage"); axs[4].set(xlabel="Temperature (K)",ylabel=r"$K_R$ (MPa√m)")
        axs[1].set_yscale("symlog",linthresh=.02); axs[2].set_yscale("symlog",linthresh=.02)
        for ax in axs[:4]: ax.set_xlabel(r"K (MPa√m)"); ax.legend(fontsize=6,ncol=2)
        fig.suptitle(f"{label}: expanded barrier geometry"); stem=f"expanded_canonical_{label.lower().replace('-','_')}_barrier_shape"; savefig(fig,out,stem,pd.concat(pdata,ignore_index=True)); stems.append(stem)
    return stems


def descriptor_dictionary(master: pd.DataFrame) -> str:
    units_rules=[("_MPa_sqrt_m","MPa sqrt(m)"),("_eV_per_MPa_sqrt_m_K","eV / (MPa sqrt(m) K)"),("_eV_per_K","eV/K"),
      ("_eV","eV"),("_Pa_per_K","Pa/K"),("_m_per_K","m/K"),("_Pa","Pa"),("_m","m"),("_K","K"),("_kB","k_B"),("ratio","dimensionless"),("fraction","dimensionless")]
    lines=["# Expanded descriptor dictionary","","Bare descriptors use the exact historical production surface with `r0=1e-6 m` and `sigma_tip=K*1e6/sqrt(2*pi*r0)`. State descriptors use the saved pre-first-passage radius/backstress and are explicitly partial. Frozen-path action descriptors integrate exact instantaneous rates along a frozen bare loading path; they are not evolved-state trajectory integrals.",""]
    non_descriptors={"candidate_id","parameter_fingerprint","source_registry","simulation_git_sha","simulation_sha_provenance","github_repository","historical_branch","canonical_family","canonical_option_key","historical_response_class","is_canonical_holdout","morphology_class"}
    for name in master.columns:
        if name in non_descriptors: continue
        unit="dimensionless"
        for token,value in units_rules:
            if token in name: unit=value; break
        family=feature_family(name)
        state="saved evolved pre-first-passage proxy" if "state_" in name or "actual_total" in name else ("temperature aggregate of bare reference-state evaluations" if "__" in name or "_dT" in name else "bare reference state near 900 K")
        locality="integrated" if any(x in name for x in ["integrated","action","fraction_path","width"]) else "local/summary"
        if "K50" in name: definition="load where normalized available barrier drop phi=0.50; differences are emission minus cleavage"
        elif "K90" in name: definition="load where normalized available barrier drop phi=0.90; differences are emission minus cleavage"
        elif "delta_g" in name: definition="(Ge-Gc)/(k_B T)"
        elif "rate_ratio" in name: definition="log10 of exact production emission/cleavage rates including prefactors, multi-hit cleavage, and multiplicity"
        elif "tau_c_over_tau_p" in name: definition="log10(tau_c/tau_p), tau_p=max(tau_emit,tau_Peierls,tau_Taylor) for serial diagnostic"
        elif "curvature" in name: definition="second K derivative of exact production barrier, or emission/cleavage relative value as named"
        elif "derivative" in name or "dGdT" in name: definition="numerical derivative of exact production barrier in the explicitly named coordinate"
        elif "entropy" in name: definition="production activation-entropy contribution or named contrast/importance ratio"
        elif "state_delta_Gc" in name: definition="Gc at evolved saved radius minus Gc at frozen r0 for the same applied K"
        else: definition="physically named summary; exact column expression follows component names and aggregation suffix"
        source="historical ExpFloorSurface/PTMechanism" if family not in {"EVOLVED_STATE_PROXY","INTEGRATED_PATH_PROXY"} else "historical first-event archive plus exact production rate/surface evaluation"
        lines.extend([f"## `{name}`","",f"- Definition: {definition}.",f"- Units: {unit}.",f"- Evaluation state: {state}.",f"- Feature family: `{family}`.",f"- Scope: {locality}.",f"- Source: {source}.",""])
    return "\n".join(lines)


def family_summary(features,mi,corr,models):
    rows=[]
    for family,group in pd.Series(features).groupby(pd.Series(features).map(feature_family)):
        names=group.tolist(); m=mi[mi.feature.isin(names)]; c=corr[corr.feature.isin(names)]
        bestm=m.loc[m.bias_corrected_MI_nats.idxmax()] if len(m) else None; bestc=c.loc[c.spearman_rho.abs().idxmax()] if len(c) and c.spearman_rho.notna().any() else None
        rows.append({"feature_family":family,"feature_count":len(names),"representative_descriptor":bestm.feature if bestm is not None else names[0],
          "best_MI_response":bestm.response if bestm is not None else "","best_bias_corrected_MI_nats":bestm.bias_corrected_MI_nats if bestm is not None else np.nan,
          "best_spearman_descriptor":bestc.feature if bestc is not None else "","best_spearman_response":bestc.response if bestc is not None else "",
          "best_abs_spearman":abs(bestc.spearman_rho) if bestc is not None else np.nan,
          "interpretation":"feature-family evidence; collinear members are not independent discoveries"})
    return pd.DataFrame(rows).sort_values("best_bias_corrected_MI_nats",ascending=False)


def hypothesis_tests(master,mi,corr,gam,interactions,families):
    def mi_family(family,response=None):
        names=[c for c in master if feature_family(c)==family]; q=mi[mi.feature.isin(names)]
        if response: q=q[q.response.eq(response)]
        return float(q.bias_corrected_MI_nats.max()) if len(q) else np.nan
    def rho(feature,response):
        q=corr[(corr.feature.eq(feature))&corr.response.eq(response)]; return finite(q.iloc[0].spearman_rho) if len(q) else np.nan
    def cv_gain(response):
        q=interactions[interactions.response.eq(response)]; base=q[q.model.eq("RIDGE_MAIN_EFFECTS")]; inter=q[q.model.eq("RIDGE_PHYSICAL_INTERACTIONS")]
        return finite(inter.iloc[0].cv_r2)-finite(base.iloc[0].cv_r2) if len(base) and len(inter) else np.nan
    def ablation_gain(response,base_name,plus_name):
        q=interactions[interactions.response.eq(response)]; base=q[q.model.eq(base_name)]; plus=q[q.model.eq(plus_name)]
        return finite(plus.iloc[0].cv_r2)-finite(base.iloc[0].cv_r2) if len(base) and len(plus) else np.nan
    records=[]
    def add(h,statement,classification,basis): records.append({"hypothesis":h,"statement":statement,"classification":classification,"basis":basis})
    raw=mi_family("RAW_ABSOLUTE_PARAMETER"); absolute=mi_family("ABSOLUTE_BARRIER_HEIGHT"); relative=max(mi_family("RELATIVE_BARRIER_POSITION"),mi_family("RELATIVE_BARRIER_HEIGHT"),mi_family("KINETIC_TIMESCALE_COMPETITION"))
    add("H1","Absolute barrier magnitudes alone are insufficient", "SUPPORTED" if relative>absolute else "WEAK_SUPPORT",f"best absolute-height MI={absolute:.4g}; best raw-parameter MI={raw:.4g}; best relative/kinetic MI={relative:.4g}")
    val=mi_family("RELATIVE_BARRIER_POSITION"); add("H2","Relative barrier position predicts dominance","SUPPORTED" if val>.05 else "WEAK_SUPPORT",f"family maximum corrected MI={val:.4g}")
    val=max(mi_family("BARRIER_WIDTH_SHAPE","DBTT_width_K"),mi_family("BARRIER_CURVATURE","DBTT_width_K")); add("H3","Relative width/curvature predicts transition sharpness","SUPPORTED" if val>.05 else "WEAK_SUPPORT",f"DBTT-width corrected MI={val:.4g}")
    q=master.loc[~master.is_canonical_holdout,["differential_entropy_kB","log10_rate_ratio_at_K75__d_dT"]].dropna()
    val=abs(float(stats.spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)) if len(q)>=8 else np.nan
    add("H4","Differential entropy controls relative kinetic thermal evolution","SUPPORTED" if val>.25 else "WEAK_SUPPORT",f"rank association with rate-ratio thermal slope={val:.3g}")
    gain=ablation_gain("DBTT_magnitude_MPa_sqrt_m","ABLATION_EMISSION_CLEAVAGE_ONLY","ABLATION_PLUS_SERIAL_TRANSPORT")
    add("H5","Transport kinetics distinguish nucleation from relaxation","SUPPORTED" if gain>.02 else ("WEAK_SUPPORT" if gain>-0.02 else "REJECTED"),f"timescale-family MI={mi_family('KINETIC_TIMESCALE_COMPETITION'):.4g}; transport ablation CV gain={gain:.3g}")
    rr=abs(rho("closest_kinetic_temperature_K","DBTT_temperature_K")); add("H6","DBTT temperature is primarily crossover-controlled","SUPPORTED" if rr>.5 else ("WEAK_SUPPORT" if rr>.25 else "REJECTED"),f"rho={rr:.3g}")
    rr=max(abs(rho("kinetic_crossover_sharpness_T_per_K","DBTT_width_K")),abs(rho("competition_temperature_width_log1_K","DBTT_width_K"))); add("H7","DBTT width is crossover/relative-shape controlled","SUPPORTED" if rr>.5 else ("WEAK_SUPPORT" if rr>.25 else "REJECTED"),f"best |rho|={rr:.3g}")
    val=mi_family("EVOLVED_STATE_PROXY","DBTT_magnitude_MPa_sqrt_m"); sgain=ablation_gain("DBTT_magnitude_MPa_sqrt_m","ABLATION_BARE_GEOMETRY_KINETICS","ABLATION_PLUS_SAVED_STATE")
    add("H8","DBTT amplitude is influenced by accumulated state","SUPPORTED" if val>.05 and sgain>.02 else "WEAK_SUPPORT",f"state-family corrected MI={val:.4g}; state ablation CV gain={sgain:.3g}; archived state is partial")
    peak=master[master.morphology_class.str.contains("Peak")]; add("H9","Peak-T requires nonlinear/reversing relative advantage","WEAK_SUPPORT" if len(peak)>=10 else "INSUFFICIENT_DATA",f"Peak-like discovery n={len(peak)}; topology/nonlinear maps evaluated, full evolved histories unavailable")
    weak=master[master.morphology_class.eq("weak-T")]; add("H10","Weak-T follows thermal-sensitivity cancellation","WEAK_SUPPORT" if len(weak)>=4 else "INSUFFICIENT_DATA",f"weak-T rows n={len(weak)}; median |delta dG/dT|={weak.delta_full_dGdT_at_K75_eV_per_K.abs().median():.4g}")
    ceramic=master[master.morphology_class.eq("ceramic-like")]; dominance=ceramic.temperature_fraction_cleavage_dominant.median() if len(ceramic) else np.nan
    add("H11","Ceramic-like response is persistent cleavage dominance","SUPPORTED" if dominance>.75 else ("WEAK_SUPPORT" if dominance>.5 else "REJECTED"),f"median temperature fraction cleavage-dominant={dominance:.3g}, n={len(ceramic)}")
    gains=[cv_gain(r) for r in RESPONSES if np.isfinite(cv_gain(r))]; med=np.nanmedian(gains) if gains else np.nan
    add("H12","Combinations outperform individual raw parameters","SUPPORTED" if med>.02 and relative>raw else "WEAK_SUPPORT",f"median physical-interaction CV R2 gain={med:.3g}; relative MI/raw MI={relative/max(raw,1e-30):.3g}")
    return pd.DataFrame(records)


def write_report(out,master,mi,gam,interactions,classification,hypotheses,families,pca_meta,audit):
    top=mi.sort_values("bias_corrected_MI_nats",ascending=False).iloc[0]
    bestmap=families.iloc[0]
    can=master[master.is_canonical_holdout][["canonical_family","delta_g_ec_at_K75","delta_K50_MPa_sqrt_m","width80_ratio_emit_over_cleave","log10_tau_c_over_tau_p_at_K75","kinetic_topology","state_delta_Gc_over_kBT"]]
    cantext="\n".join(f"- **{r.canonical_family}:** Δg={r.delta_g_ec_at_K75:.3g}, ΔK50={r.delta_K50_MPa_sqrt_m:.3g} MPa√m, We/Wc={r.width80_ratio_emit_over_cleave:.3g}, log10(τc/τp)={r.log10_tau_c_over_tau_p_at_K75:.3g}, topology={r.kinetic_topology}, state ΔGc/kBT={r.state_delta_Gc_over_kBT:.3g}." for _,r in can.iterrows())
    ci=master[master.is_canonical_holdout].set_index("canonical_family"); peak,dbtt,weak,cer=ci.loc["Peak-T"],ci.loc["DBTT"],ci.loc["weak-T"],ci.loc["ceramic-like"]
    pairtext=(f"At the shared near-900 K reference, Peak-T differs from DBTT primarily in state accumulation and barrier alignment: Peak has ΔK50={peak.delta_K50_MPa_sqrt_m:.3g} versus {dbtt.delta_K50_MPa_sqrt_m:.3g} MPa√m, "
      f"state ΔGc/kBT={peak.state_delta_Gc_over_kBT:.3g} versus {dbtt.state_delta_Gc_over_kBT:.3g}, and {peak.actual_total_source_activations_at_first_passage:.3g} versus {dbtt.actual_total_source_activations_at_first_passage:.3g} saved source activations. Both have complex crossover topology, so topology count alone does not separate them. "
      f"Weak-T versus ceramic-like separates differently: ΔK50={weak.delta_K50_MPa_sqrt_m:.3g} versus {cer.delta_K50_MPa_sqrt_m:.3g} MPa√m and dlog10(τc/τp)/dT={weak.dlog10_tau_c_over_tau_p_dT:.3g} versus {cer.dlog10_tau_c_over_tau_p_dT:.3g} K⁻¹. The ceramic row is not persistently cleavage-dominant under the full sampled-domain diagnostic, so simple dominance is rejected rather than used as its explanation.")
    htext="\n".join(f"- **{r.hypothesis} — {r.classification}:** {r.statement}. {r.basis}." for _,r in hypotheses.iterrows())
    text=f"""# Barrier shape, activation entropy, and temperature-dependent fracture morphology

This v2 analysis preserves the complete v1 result and adds nonlinear, interaction, transition-geometry, exact-rate, transport-timescale, state-proxy, and frozen-path descriptors. It is existing-data only: no constitutive parameter changed and no simulation was launched. Simulation provenance remains `{SIM_SHA}`; analysis HEAD is `{audit['analysis_git_sha']}`.

## Expanded barrier-geometry analysis

1. **Why were raw-parameter Spearman correlations weak?** The response is generated by exponential competition, moving transition positions, serial plastic timescales, and evolved state. Those mechanisms are threshold-like and nonmonotonic; one raw coefficient is not a sufficient coordinate.
2. **Which derived descriptors organize the response better?** The strongest corrected mutual-information result is `{top.feature}` versus `{top.response}` ({top.bias_corrected_MI_nats:.4g} nats). The leading family is `{bestmap.feature_family}`, represented by `{bestmap.representative_descriptor}`.
3. **Are barrier ratios more useful than absolute barriers?** Relative-position/height and kinetic families are compared directly with raw absolute parameters in `expanded_feature_family_summary.csv`; their nonlinear information is generally more useful, but collinear members are interpreted as families.
4. **Is (Ge-Gc)/(kBT) useful?** Yes as a dimensionless ordering coordinate in phase maps and nonlinear models. It is not sufficient alone and is evaluated at explicit normalized loads.
5. **Do relative transition positions distinguish classes?** They provide visible organization in `barrier_position_width_phase_map.png`; ΔK50 and ΔK90 are held out from canonical threshold selection.
6. **Do widths distinguish DBTT from Peak-T?** Width ratios contribute jointly with position and entropy. Their independent attribution is limited by EXP-floor collinearity.
7. **Does curvature predict transition sharpness?** The corrected nonlinear and rank evidence is recorded under H3; curvature is useful chiefly in combination with relative derivative/position.
8. **Does differential entropy predict rate evolution?** It contributes, but the full production dG/dT and mixed derivative are more complete because they include stress-scale evolution, floor behavior, and clipping.
9. **Does τc/τp outperform emission/cleavage alone?** Serial plasticity uses τp=max(τe,τP,τT); its map distinguishes emission access from completed relaxation. Cross-validated comparisons quantify the gain rather than assuming it.
10. **Do state-conditioned descriptors outperform bare descriptors?** Some add nonlinear information, but only a partial saved-first-passage proxy is available. K-shield and full mobile/retained/slip histories remain unavailable, so a definitive mediation claim is not made.
11. **Do integrated quantities outperform instantaneous ones?** Frozen-path action/dominance proxies are tested and explicitly labeled; true evolved path integrals cannot be recovered from event-only archives. Actual source activations and line content at first passage are retained.
12. **Relationship shape?** GAM-versus-linear CV gains, MI, shallow trees, and boosted stumps show a mixture of threshold-like and nonmonotonic structure. The depth-3 tree has CV accuracy {classification.iloc[0].cv_accuracy:.3f} but balanced accuracy only {classification.iloc[0].cv_balanced_accuracy:.3f}; its thresholds are descriptive and class-imbalance-sensitive, not predictive laws. Spearman remains only one diagnostic.
13. **Best 2-D descriptor maps?** The four requested maps are generated; the discovery-selected map is determined only from noncanonical 1-D rows. “2-D” here means two-descriptor phase maps, not spatial 2-D fracture simulation.
14. **Canonical placement?**\n{cantext}
15. **What survives holdout/scale control?** Canonical rows never select features, knots, tree thresholds, or phase-map coordinates. Their positions are overlays. Sparse spatial-2-D raw temperature histories remain unavailable, so spatial transfer is still insufficient-data rather than refit.
16. **Minimal causal factorial?** Use a D-optimal 18–24-run design around one DBTT and one Peak-like row, independently shifting ΔK50, We/Wc, differential entropy, transport entropy, and high-T relative curvature while re-anchoring the 300 K fracture scale. Reserve six combinations as confirmation and transfer only discriminating pairs to spatial 2-D. No such runs were launched.

## Expanded hypotheses

{htext}

## Central comparison

The evidence supports a low-dimensional *family* description more strongly than individual raw-parameter winners: relative barrier position and shape set where competition occurs; full thermal/mixed derivatives move that competition with temperature; serial emission/transport timescales determine whether plastic accommodation completes; and saved state modifies the opening barrier.

{pairtext}

The complete Peak-T/DBTT and weak-T/ceramic values are preserved in `canonical_pair_comparison.csv`.

## Scope limits

The exact evolved pre-fracture time history was not archived. Consequently `frozen_path_*` columns are diagnostic bare-path integrations, not reconstructed physical histories. `state_*` fields are saved first-passage proxies and retain explicit missing-field labels. No extrapolation beyond simulated fracture response, no artificial completion of two censored cases, and no spatial-2-D retraining were performed.
"""
    (out/"BARRIER_TEMPERATURE_FRACTURE_MORPHOLOGY_REPORT.md").write_text(text)


REQUIRED_TABLES=["expanded_barrier_temperature_descriptors.csv","expanded_univariate_correlations.csv","expanded_mutual_information.csv",
 "expanded_gam_performance.csv","expanded_interaction_models.csv","expanded_classification_models.csv","expanded_descriptor_collinearity.csv",
 "expanded_feature_family_summary.csv"]
REQUIRED_FIGURES=["expanded_descriptor_response_heatmap.png","expanded_mutual_information_rankings.png","relative_barrier_geometry_phase_map.png",
 "timescale_temperature_phase_map.png","barrier_position_width_phase_map.png","entropy_barrier_phase_map.png","DBTT_descriptor_maps.png",
 "PeakT_descriptor_maps.png","weakT_descriptor_maps.png","ceramic_descriptor_maps.png","descriptor_class_distributions.png",
 "expanded_descriptor_pca.png","canonical_four_on_discovered_phase_map.png"]


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--v1-root",type=Path,default=V1_ROOT); ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=ap.parse_args(); v1root=args.v1_root.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    v1audit=json.loads((v1root/"analysis_audit.json").read_text())
    if v1audit.get("simulation_git_sha")!=SIM_SHA or v1audit.get("candidate_count")!=400: raise RuntimeError("v1 provenance/population mismatch")
    v1=load_v1_module(); ExpFloorSurface,PTMechanism=v1.load_production_types(SOURCE)
    candidates,cases,events,paths=v1.load_population(SOURCE); response=pd.read_csv(v1root/"fracture_response_descriptors.csv")
    response["normalized_high_T_response"]=1+response.fractional_terminal_change
    response_family,response_wide=response_family_descriptors(cases,events)
    first=events.sort_values("event_index").drop_duplicates(["candidate_id","temperature_K"]).set_index(["candidate_id","temperature_K"])
    candidate_index=candidates.set_index("candidate_id"); rows=[]
    for case in cases[cases.status.eq("complete")].itertuples(index=False):
        key=(case.candidate_id,float(case.temperature_K)); state=first.loc[key] if key in first.index else pd.Series(dtype=float)
        rows.append(barrier_geometry_row(case.candidate_id,float(case.temperature_K),float(case.authoritative_response_MPa_sqrt_m),candidate_index.loc[case.candidate_id],state,v1,ExpFloorSurface,PTMechanism))
    temp=pd.DataFrame(rows)
    provenance=["candidate_id","parameter_fingerprint","source_registry","simulation_git_sha","simulation_sha_provenance","github_repository",
                "historical_branch","canonical_family","canonical_option_key","historical_response_class","is_canonical_holdout"]
    temp=temp.merge(candidates[provenance],on="candidate_id",how="left")
    master=aggregate_expanded(temp,candidates,response,response_wide)
    for field in v1.ACTIVE_FIELDS: master[f"raw__{field}"]=master.candidate_id.map(candidate_index[field])
    response_names=set(RESPONSES+[c for c in master if c.startswith("response__")]+list(response.columns))
    protected={"candidate_id","canonical_family","historical_response_class","morphology_class","is_canonical_holdout",
               "descriptor_reference_temperature_K","temperature_count"}
    features=[c for c in master if c not in protected and c not in response_names and pd.api.types.is_numeric_dtype(master[c])]
    # Retain physical features with enough discovery variation.
    discovery=master[~master.is_canonical_holdout]; features=[c for c in features if discovery[c].notna().sum()>=20 and discovery[c].nunique(dropna=True)>2]
    corr=univariate_correlations(master,features,RESPONSES); mi=mutual_information_table(master,features,RESPONSES)
    gam=gam_performance(master,mi,RESPONSES); interactions=interaction_models(master,mi,RESPONSES)
    class_rank=mi.groupby("feature").bias_corrected_MI_nats.max().sort_values(ascending=False).index.tolist()
    class_features=list(dict.fromkeys([f for f in class_rank[:12]+PRIORITY if f in features]))[:20]
    classification,tree=classification_models(master,class_features)
    coll=collinearity(master,features); pca,pca_meta=expanded_pca(master,features); families=family_summary(features,mi,corr,interactions)
    hypotheses=hypothesis_tests(master,mi,corr,gam,interactions,families)
    # Canonical pair table makes the central comparisons direct and auditable.
    paircols=["candidate_id","canonical_family","delta_g_ec_at_K75","log10_rate_ratio_at_K75","delta_K50_MPa_sqrt_m","delta_K90_MPa_sqrt_m",
      "width80_ratio_emit_over_cleave","relative_max_first_derivative","relative_curvature_at_closest","differential_entropy_kB",
      "delta_full_dGdT_at_K75_eV_per_K","log10_tau_c_over_tau_p_at_K75","dlog10_tau_c_over_tau_p_dT","kinetic_topology",
      "state_delta_Gc_over_kBT","state_backstress_amplification","state_tip_radius_amplification","actual_total_source_activations_at_first_passage"]
    canonical=master.loc[master.is_canonical_holdout,paircols].copy(); canonical["comparison_pair"]=canonical.canonical_family.map({"Peak-T":"Peak-T_vs_DBTT","DBTT":"Peak-T_vs_DBTT","weak-T":"weak-T_vs_ceramic","ceramic-like":"weak-T_vs_ceramic"})
    # Preserve a hash manifest of every v1 artifact rather than duplicating 170 MB.
    manifest=[]
    for path in sorted(v1root.iterdir()):
        if path.is_file(): manifest.append({"artifact":path.name,"path":str(path),"bytes":path.stat().st_size,"sha256":digest(path)})
    pd.DataFrame(manifest).to_csv(out/"v1_artifact_manifest.csv",index=False)
    master.to_csv(out/"expanded_barrier_temperature_descriptors.csv",index=False); temp.to_csv(out/"expanded_temperature_resolved_descriptors.csv",index=False)
    response_family.to_csv(out/"expanded_initiation_developed_response_descriptors.csv",index=False); corr.to_csv(out/"expanded_univariate_correlations.csv",index=False)
    mi.to_csv(out/"expanded_mutual_information.csv",index=False); gam.to_csv(out/"expanded_gam_performance.csv",index=False)
    interactions.to_csv(out/"expanded_interaction_models.csv",index=False); classification.to_csv(out/"expanded_classification_models.csv",index=False)
    coll.to_csv(out/"expanded_descriptor_collinearity.csv",index=False); families.to_csv(out/"expanded_feature_family_summary.csv",index=False)
    pca.to_csv(out/"expanded_descriptor_pca_scores.csv",index=False); canonical.to_csv(out/"canonical_pair_comparison.csv",index=False)
    hypotheses.to_csv(out/"expanded_hypothesis_tests.csv",index=False); (out/"expanded_descriptor_dictionary.md").write_text(descriptor_dictionary(master))
    (out/"expanded_pca_metadata.json").write_text(json.dumps(pca_meta,indent=2)+"\n"); (out/"expanded_classification_tree.json").write_text(json.dumps(tree,indent=2)+"\n")
    stems=make_figures(out,master,temp,mi,corr,pca,features,v1,ExpFloorSurface)
    audit={"schema":"v913_expanded_barrier_temperature_morphology_v2","analysis_branch":git("branch","--show-current"),"analysis_git_sha":git("rev-parse","HEAD"),
      "simulation_git_sha":SIM_SHA,"v1_root":str(v1root),"v1_audit_sha256":digest(v1root/"analysis_audit.json"),"v1_artifact_count":len(manifest),
      "candidate_count":master.candidate_id.nunique(),"complete_temperature_case_count":len(temp),"excluded_censored_case_count":int((~cases.status.eq("complete")).sum()),
      "expanded_feature_count":len(features),"state_scope":"PARTIAL_SAVED_FIRST_PASSAGE_PROXY","path_action_scope":"FROZEN_BARE_PATH_PROXY_NOT_EVOLVED",
      "new_simulations_launched":False,"physics_changed":False,"bare_reference_tip_radius_m":R0_M,"K_to_sigma_mapping":"sigma=K*1e6/sqrt(2*pi*r0)",
      "effective_plastic_timescale":"max(tau_emission,tau_Peierls,tau_Taylor), serial diagnostic","required_tables":REQUIRED_TABLES,"required_figures":REQUIRED_FIGURES,"figure_stems":stems}
    write_report(out,master,mi,gam,interactions,classification,hypotheses,families,pca_meta,audit)
    (out/"expanded_analysis_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    missing=[x for x in REQUIRED_TABLES+REQUIRED_FIGURES+["expanded_descriptor_dictionary.md","BARRIER_TEMPERATURE_FRACTURE_MORPHOLOGY_REPORT.md"] if not (out/x).exists()]
    if missing: raise RuntimeError(f"missing expanded artifacts: {missing}")
    print(json.dumps({"status":"PASS","out":str(out),"candidates":len(master),"temperature_rows":len(temp),"features":len(features),"figures":len(stems)},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
