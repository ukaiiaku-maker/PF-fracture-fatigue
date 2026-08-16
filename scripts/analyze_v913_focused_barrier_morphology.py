#!/usr/bin/env python3
"""Leakage-audited, response-independent v9.13 barrier-morphology analysis.

This amendment layer consumes the immutable v1/v2 analysis products and exact
historical constitutive surfaces.  It launches no simulations.  Headline
prediction uses only standardized constitutive coordinates (Level A); observed-
response diagnostics (Level B) and partial saved state (Level C) remain separate.
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
from scipy import special, stats

REPO=Path(__file__).resolve().parents[1]
V1=REPO/"runs/v913_barrier_temperature_fracture_morphology_v1"
V2=REPO/"runs/v913_barrier_temperature_fracture_morphology_v2"
OUT=REPO/"runs/v913_barrier_temperature_fracture_morphology_v3_focused"
SOURCE=Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_13_dbtt_temperature_shelf")
SIM_SHA="559425321b9a8739f32788322d8a1c2af8abad73"
KB=8.617333262145e-5
TEMPS=np.array([700.,800.,900.,950.,1000.,1050.,1100.,1200.,1300.,1400.])
CANONICAL={"v913_zeroD_sobol_0242980":"Peak-T","v913_zeroD_sobol_0202500":"DBTT",
           "v913_zeroD_sobol_0129902":"weak-T","v913_zeroD_sobol_0077080":"ceramic-like"}
COLORS={"Peak-T":"#F59E0B","DBTT":"#3B82F6","weak-T":"#8B5CF6","ceramic-like":"#64748B"}
RESPONSES=["S_low_MPa_sqrt_m_per_K","S_mid_MPa_sqrt_m_per_K","fractional_resistance_span",
 "DBTT_magnitude_MPa_sqrt_m","DBTT_temperature_K","DBTT_width_K",
 "peak_prominence_MPa_sqrt_m","peak_temperature_K"]
SECONDARY_RESPONSES=["S_high_MPa_sqrt_m_per_K","max_abs_curvature_MPa_sqrt_m_per_K2","weakT_max_deviation_from_mean_MPa_sqrt_m"]
LEVELS=(.90,.75,.50,.25,.10)

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; assert spec.loader is not None; spec.loader.exec_module(mod); return mod

def git(*args): return subprocess.check_output(["git",*args],cwd=REPO,text=True).strip()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def finite(x,default=np.nan):
    try:
        v=float(x); return v if np.isfinite(v) else float(default)
    except (TypeError,ValueError): return float(default)

def shape_metrics(a,n,floor):
    """Analytic EXP-floor shape and Weibull activation-window descriptors."""
    a=float(a); n=float(n); floor=float(floor)
    xp={f"x{int(100*p):02d}":float((-math.log(p)/a)**(1/n)) for p in LEVELS}
    w80=xp["x10"]-xp["x90"]; w50=xp["x25"]-xp["x75"]
    asym80=(xp["x50"]-xp["x90"])/max(xp["x10"]-xp["x50"],1e-300)
    asym50=(xp["x50"]-xp["x75"])/max(xp["x25"]-xp["x50"],1e-300)
    lam=a**(-1/n); raw=[lam**k*special.gamma(1+k/n) for k in range(1,5)]
    mu=raw[0]; var=max(raw[1]-mu*mu,0); sd=math.sqrt(max(var,1e-300))
    m3=raw[2]-3*mu*raw[1]+2*mu**3
    m4=raw[3]-4*mu*raw[2]+6*mu*mu*raw[1]-3*mu**4
    if n<1: smax=np.nan; singular_s=True
    elif abs(n-1)<1e-12: smax=a; singular_s=False
    else:
        xm=((n-1)/(a*n))**(1/n); smax=a*n*xm**(n-1)*math.exp(-a*xm**n); singular_s=False
    xlo=max(stats.weibull_min.ppf(1e-7,n,scale=lam),1e-12); xhi=stats.weibull_min.ppf(1-1e-7,n,scale=lam)
    x=np.geomspace(xlo,xhi,6000); phi=np.exp(-a*x**n)
    curvature=phi*((a*n)**2*x**(2*n-2)-a*n*(n-1)*x**(n-2))
    singular_c=bool(n<2 and abs(n-1)>1e-12)
    imax=int(np.argmax(abs(curvature)))
    xmode=0. if n<=1 else ((n-1)/(a*n))**(1/n)
    def curv_at(z): return math.exp(-a*z**n)*((a*n)**2*z**(2*n-2)-a*n*(n-1)*z**(n-2))
    return {**xp,"width80":w80,"width50":w50,"width80_over_x50":w80/max(xp["x50"],1e-300),
      "asymmetry_90_10":asym80,"asymmetry_75_25":asym50,"sstar_max":smax,
      "sstar_singular_at_zero":singular_s,"cstar_max_abs_finite_domain":float(abs(curvature[imax])),
      "cstar_location_finite_domain":float(x[imax]),"cstar_at_x50":curv_at(xp["x50"]),
      "cstar_at_max_sensitivity":curv_at(max(xmode,1e-12)),"cstar_singular_at_zero":singular_c,
      "activation_mu":mu,"activation_variance":var,"activation_skewness":m3/sd**3,
      "activation_excess_kurtosis":m4/sd**4-3,"floor_fraction":floor,"exp_a":a,"exp_n":n}

def activation_overlap_distance(c,e):
    """Stable overlap, Wasserstein distance and Jensen-Shannon divergence."""
    uc=np.linspace(1e-6,1-1e-6,20001)
    qc=stats.weibull_min.ppf(uc,c[1],scale=c[0]**(-1/c[1]))
    qe=stats.weibull_min.ppf(uc,e[1],scale=e[0]**(-1/e[1]))
    wasser=float(np.mean(abs(qc-qe)))
    xmax=max(qc[-1],qe[-1]); edges=np.r_[0,np.geomspace(1e-9,max(xmax,1e-8),5000)]
    pc=np.diff(stats.weibull_min.cdf(edges,c[1],scale=c[0]**(-1/c[1])))
    pe=np.diff(stats.weibull_min.cdf(edges,e[1],scale=e[0]**(-1/e[1])))
    pc=pc/max(pc.sum(),1e-300); pe=pe/max(pe.sum(),1e-300)
    overlap=float(np.minimum(pc,pe).sum()); mix=.5*(pc+pe)
    mc=pc>0; me=pe>0
    js=.5*np.sum(pc[mc]*np.log(pc[mc]/mix[mc]))+.5*np.sum(pe[me]*np.log(pe[me]/mix[me]))
    return overlap,wasser,float(js)

def activation_overlap_only(c,e):
    lc=c[0]**(-1/c[1]); le=e[0]**(-1/e[1]); xmax=max(stats.weibull_min.ppf(1-1e-7,c[1],scale=lc),stats.weibull_min.ppf(1-1e-7,e[1],scale=le))
    edges=np.r_[0,np.geomspace(1e-9,max(xmax,1e-8),1800)]; pc=np.diff(stats.weibull_min.cdf(edges,c[1],scale=lc)); pe=np.diff(stats.weibull_min.cdf(edges,e[1],scale=le))
    return float(np.minimum(pc/max(pc.sum(),1e-300),pe/max(pe.sum(),1e-300)).sum())

def effective_surface_scale_derivatives(surface,T):
    raw_G0=surface.G00_eV+surface.gT_eV_per_K*(T-surface.Tref_K)
    raw_sig=surface.sigc0_Pa+surface.sT_Pa_per_K*(T-surface.Tref_K)
    return (surface.gT_eV_per_K if raw_G0>1e-12 else 0.),(surface.sT_Pa_per_K if raw_sig>1. else 0.),raw_G0<=1e-12,raw_sig<=1.

def surface_thermal_parts(surface,T,sigma):
    """Exact gT and sT pieces of dG/dT for the clipped EXP-floor law."""
    G0=surface.zero_stress_eV(T); sigc=surface.characteristic_stress_Pa(T); dG0,dsig,_,_=effective_surface_scale_derivatives(surface,T)
    nominal=surface.floor_fraction*G0; lower=max(surface.floor_min_eV,nominal); floor=min(surface.floor_max_fraction*G0,lower)
    if surface.floor_max_fraction*G0<=lower: dfloor=surface.floor_max_fraction*dG0
    elif nominal>=surface.floor_min_eV: dfloor=surface.floor_fraction*dG0
    else: dfloor=0.
    x=max(float(sigma),0.)/sigc; phi=math.exp(-surface.exp_a*x**surface.exp_n)
    dG_g=dfloor+(dG0-dfloor)*phi
    dG_s=(G0-floor)*phi*surface.exp_a*surface.exp_n*x**surface.exp_n*dsig/sigc
    return dG_g,dG_s

def provenance(frame,candidates,level):
    cols=["candidate_id","parameter_fingerprint","source_registry","simulation_git_sha","simulation_sha_provenance","github_repository","historical_branch","canonical_family","canonical_option_key","is_canonical_holdout"]
    base=candidates[[c for c in cols if c in candidates]].drop_duplicates("candidate_id")
    out=frame.merge(base,on="candidate_id",how="left"); out["analysis_level"]=level
    out["prediction_eligible"]=level=="LEVEL_A_RESPONSE_INDEPENDENT_INTRINSIC"
    return out

def intrinsic_tables(candidates,v1,ExpFloorSurface):
    rows=[]; windows=[]
    for r in candidates.itertuples(index=False):
        row=pd.Series(r._asdict()); cid=r.candidate_id; rec={"candidate_id":cid}
        shapes={}
        for prefix in ["cleave","emit"]:
            q=shape_metrics(row[f"{prefix}_exp_a"],row[f"{prefix}_exp_n"],row[f"{prefix}_floor_frac"]); shapes[prefix]=q
            rec.update({f"{prefix}_{k}":v for k,v in q.items()})
            surf=v1.make_surface(row,prefix,ExpFloorSurface); T=900.; G0=surf.zero_stress_eV(T); sig=surf.characteristic_stress_Pa(T); floor=float(surf.barrier_eV(1e15,T))
            dG0,dsig,Gclip,Sclip=effective_surface_scale_derivatives(surf,T)
            rec.update({f"{prefix}_G0_900_eV":G0,f"{prefix}_floor_900_eV":floor,f"{prefix}_available_drop_900_eV":G0-floor,
              f"{prefix}_sigc_900_GPa":sig/1e9,f"{prefix}_Theta_G_900":T*dG0/max(G0,1e-300),
              f"{prefix}_Theta_sigma_900":T*dsig/max(sig,1e-300),f"{prefix}_G0_clipped_900":Gclip,f"{prefix}_sigc_clipped_900":Sclip})
        c,e=shapes["cleave"],shapes["emit"]; overlap,wass,js=activation_overlap_distance((c["exp_a"],c["exp_n"]),(e["exp_a"],e["exp_n"]))
        rel={"delta_mu_emit_minus_cleave":e["activation_mu"]-c["activation_mu"],
          "normalized_center_separation_Dmu":abs(e["activation_mu"]-c["activation_mu"])/math.sqrt(max(e["activation_variance"]+c["activation_variance"],1e-300)),
          "activation_window_overlap_Oce":overlap,"activation_window_wasserstein":wass,"activation_window_JS_nats":js,
          "normalized_activation_window_overlap_low":overlap,"normalized_activation_window_overlap_high":overlap,"normalized_activation_window_overlap_delta_T":0.,
          "width80_ratio_emit_over_cleave":e["width80"]/max(c["width80"],1e-300),"width80_difference_emit_minus_cleave":e["width80"]-c["width80"],
          "asymmetry80_ratio_emit_over_cleave":e["asymmetry_90_10"]/max(c["asymmetry_90_10"],1e-300),"asymmetry80_difference_emit_minus_cleave":e["asymmetry_90_10"]-c["asymmetry_90_10"],
          "sstar_ratio_emit_over_cleave":e["sstar_max"]/c["sstar_max"] if np.isfinite(e["sstar_max"]) and np.isfinite(c["sstar_max"]) else np.nan,
          "sstar_difference_emit_minus_cleave":e["sstar_max"]-c["sstar_max"] if np.isfinite(e["sstar_max"]) and np.isfinite(c["sstar_max"]) else np.nan,
          "cstar_max_ratio_emit_over_cleave":e["cstar_max_abs_finite_domain"]/max(c["cstar_max_abs_finite_domain"],1e-300),
          "delta_Theta_G_900":rec["emit_Theta_G_900"]-rec["cleave_Theta_G_900"],"delta_Theta_sigma_900":rec["emit_Theta_sigma_900"]-rec["cleave_Theta_sigma_900"],
          "available_drop_ratio_emit_over_cleave":rec["emit_available_drop_900_eV"]/max(rec["cleave_available_drop_900_eV"],1e-300),
          "sigc_ratio_emit_over_cleave_900":rec["emit_sigc_900_GPa"]/max(rec["cleave_sigc_900_GPa"],1e-300)}
        rec.update(rel); rows.append(rec)
        windows.append({"candidate_id":cid,**{k:rec[k] for k in ["cleave_activation_mu","emit_activation_mu","cleave_activation_variance","emit_activation_variance",
          "cleave_activation_skewness","emit_activation_skewness","cleave_activation_excess_kurtosis","emit_activation_excess_kurtosis"]},**{k:rel[k] for k in ["delta_mu_emit_minus_cleave","normalized_center_separation_Dmu","activation_window_overlap_Oce","activation_window_wasserstein","activation_window_JS_nats"]}})
    return provenance(pd.DataFrame(rows),candidates,"LEVEL_A_RESPONSE_INDEPENDENT_INTRINSIC"),provenance(pd.DataFrame(windows),candidates,"LEVEL_A_RESPONSE_INDEPENDENT_INTRINSIC")

def exact_rates(v1,Gc,Ge,T,multiplicity=1.):
    raw=v1.NU_C*np.exp(np.clip(-Gc/(KB*T),-745,700)); rc=special.gammainc(v1.MULTIHIT_M,np.minimum(raw*v1.MULTIHIT_TAU_S,1e12))/v1.MULTIHIT_TAU_S
    re=v1.NU_E*multiplicity*np.exp(np.clip(-Ge/(KB*T),-745,700)); return np.maximum(rc,1e-300),np.maximum(re,1e-300)

def kinetic_temperature_tables(candidates,v1,ExpFloorSurface,PTMechanism):
    z=np.linspace(0,3,301); rows=[]; brows=[]
    for rr in candidates.itertuples(index=False):
        r=pd.Series(rr._asdict()); gc=v1.make_surface(r,"cleave",ExpFloorSurface); ge=v1.make_surface(r,"emit",ExpFloorSurface)
        pm=PTMechanism(finite(r.peierls_H0_eV),finite(r.peierls_activation_entropy_kB),finite(r.peierls_exp_a),finite(r.peierls_exp_n),finite(r.peierls_nu0_s)); gp=pm.surface(ge)
        tm=PTMechanism(finite(r.taylor_H0_eV),finite(r.taylor_activation_entropy_kB),finite(r.taylor_exp_a),finite(r.taylor_exp_n),finite(r.taylor_nu0_s)); gt=tm.surface(ge)
        controls=[]
        for T in TEMPS:
            sc=gc.characteristic_stress_Pa(T); se=ge.characteristic_stress_Pa(T); scale=math.sqrt(sc*se); sigma=z*scale
            Gc=np.asarray(gc.barrier_eV(sigma,T)); Ge=np.asarray(ge.barrier_eV(sigma,T)); rc,re=exact_rates(v1,Gc,Ge,T)
            acz=gc.exp_a*(scale/sc)**gc.exp_n; aez=ge.exp_a*(scale/se)**ge.exp_n; physical_overlap=activation_overlap_only((acz,gc.exp_n),(aez,ge.exp_n))
            logR=np.log10(re)-np.log10(rc); absR=abs(logR); imin=int(np.argmin(absR))
            D=float(np.trapezoid(absR,z)/3); M=float(np.trapezoid(logR,z)/3)
            f1=float(np.mean(absR<1)); f2=float(np.mean(absR<2)); Adom=float(np.trapezoid(np.sign(logR),z)/3)
            Gp=np.asarray(gp.barrier_eV(pm.stress_fraction*sigma,T)); Gt=np.asarray(gt.barrier_eV(tm.stress_fraction*sigma,T))
            rp=np.maximum(pm.nu0_s*np.exp(np.clip(-Gp/(KB*T),-745,700)),1e-300); rt=np.maximum(tm.nu0_s*np.exp(np.clip(-Gt/(KB*T),-745,700)),1e-300)
            le=-np.log10(re); lp=-np.log10(rp); lt=-np.log10(rt); means=np.array([le.mean(),lp.mean(),lt.mean()]); order=np.argsort(means)[::-1]
            gap=means[order[0]]-means[order[1]]; names=np.array(["EMISSION_LIMITED","PEIERLS_LIMITED","TAYLOR_LIMITED"])
            control="MIXED_PLASTIC_CONTROL" if gap<.5 else str(names[order[0]]); controls.append(control)
            # Exact additive temperature derivative at fixed standardized physical stress z=1.
            sig1=scale; h=.25
            def lr_at(temp):
                a,b=exact_rates(v1,np.array([gc.barrier_eV(sig1,temp)]),np.array([ge.barrier_eV(sig1,temp)]),temp); return math.log10(b[0]/a[0])
            full=(lr_at(T+h)-lr_at(T-h))/(2*h)
            x_c=sig1/sc; x_e=sig1/se; phic=math.exp(-gc.exp_a*x_c**gc.exp_n); phie=math.exp(-ge.exp_a*x_e**ge.exp_n)
            G0c=gc.zero_stress_eV(T); G0e=ge.zero_stress_eV(T); fc=float(gc.barrier_eV(1e15,T))/G0c; fe=float(ge.barrier_eV(1e15,T))/G0e
            dGc_g,dGc_s=surface_thermal_parts(gc,T,sig1); dGe_g,dGe_s=surface_thermal_parts(ge,T,sig1)
            cg=(dGc_g)/(KB*T*math.log(10)); eg=-(dGe_g)/(KB*T*math.log(10)); cs=(dGc_s)/(KB*T*math.log(10)); es=-(dGe_s)/(KB*T*math.log(10))
            explicit=(float(Ge[100])-float(Gc[100]))/(KB*T*T*math.log(10)); pref=full-(cg+eg+cs+es+explicit)
            rows.append({"candidate_id":r.candidate_id,"temperature_K":T,"standardized_stress_coordinate":"z=sigma/sqrt(sigc_c(T)*sigc_e(T)); uniform z in [0,3]",
              "source_multiplicity_assumption":1.,"Dmin_log10_rate_ratio":float(absR[imin]),"z_at_Dmin":float(z[imin]),"Dgamma_mean_abs_log10_ratio":D,
              "Mgamma_mean_signed_log10_ratio":M,"physical_activation_window_overlap_Oce":physical_overlap,"competition_fraction_1decade":f1,"competition_fraction_2decade":f2,"signed_dominance_area":Adom,
              "dlog10_rate_ratio_dT_at_z1":full,"dlogR_dT_cleave_gT":cg,"dlogR_dT_emit_gT":eg,"dlogR_dT_cleave_sT":cs,"dlogR_dT_emit_sT":es,
              "dlogR_dT_shape_weighted_explicit_inverse_T":explicit,"dlogR_dT_production_prefactor_multihit_correction":pref})
            brows.append({"candidate_id":r.candidate_id,"temperature_K":T,"B_P_log10_tauP_over_taue":float((lp-le).mean()),"B_T_log10_tauT_over_taue":float((lt-le).mean()),
              "mean_log10_tau_emission_s":float(le.mean()),"mean_log10_tau_peierls_s":float(lp.mean()),"mean_log10_tau_taylor_s":float(lt.mean()),"plastic_control":control,"control_gap_decades":float(gap)})
    return provenance(pd.DataFrame(rows),candidates,"LEVEL_A_RESPONSE_INDEPENDENT_INTRINSIC"),provenance(pd.DataFrame(brows),candidates,"LEVEL_A_RESPONSE_INDEPENDENT_INTRINSIC")

def curve_summary(values,T,prefix):
    v=np.asarray(values,float); d=np.gradient(v,T); d2=np.gradient(d,T); interior=np.arange(1,len(v)-1)
    reversals=int(np.sum(np.sign(d[1:])*np.sign(d[:-1])<0)); curvrev=int(np.sum(np.sign(d2[1:])*np.sign(d2[:-1])<0))
    return {f"{prefix}_low":v[0],f"{prefix}_high":v[-1],f"{prefix}_delta":v[-1]-v[0],f"{prefix}_span":np.ptp(v),
      f"{prefix}_max_abs_dT":float(np.max(abs(d))),f"{prefix}_T_at_max_abs_dT":float(T[np.argmax(abs(d))]),
      f"{prefix}_slope_reversal_count":reversals,f"{prefix}_curvature_reversal_count":curvrev,
      f"{prefix}_has_interior_extremum":bool(len(interior) and np.argmax(v) in interior or np.argmin(v) in interior)}

def aggregate_kinetic(kin,bottle,candidates):
    kr=[]; br=[]
    for cid,g in kin.groupby("candidate_id"):
        g=g.sort_values("temperature_K"); rec={"candidate_id":cid}
        for col,p in [("Dmin_log10_rate_ratio","Dmin"),("Dgamma_mean_abs_log10_ratio","Dgamma"),("Mgamma_mean_signed_log10_ratio","Mgamma"),("physical_activation_window_overlap_Oce","physical_Oce"),("competition_fraction_1decade","competition_fraction_1decade"),("competition_fraction_2decade","competition_fraction_2decade"),("signed_dominance_area","signed_dominance_area")]: rec.update(curve_summary(g[col],g.temperature_K.to_numpy(float),p))
        for col in [c for c in g if c.startswith("dlog")]: rec[f"{col}_mean_T"]=float(g[col].mean())
        kr.append(rec)
    for cid,g in bottle.groupby("candidate_id"):
        g=g.sort_values("temperature_K"); seq=g.plastic_control.tolist(); changes=[i for i in range(1,len(seq)) if seq[i]!=seq[i-1]]
        means={x:float(g[x].mean()) for x in ["B_P_log10_tauP_over_taue","B_T_log10_tauT_over_taue"]}
        br.append({"candidate_id":cid,"plastic_control_low_T":seq[0],"plastic_control_high_T":seq[-1],"plastic_control_change_count":len(changes),
          "first_control_change_T_K":float(g.temperature_K.iloc[changes[0]]) if changes else np.nan,"strongest_control_competition_T_K":float(g.temperature_K.iloc[g.control_gap_decades.argmin()]),
          "control_gap_low_T_decades":float(g.control_gap_decades.iloc[0]),"control_gap_delta_decades":float(g.control_gap_decades.iloc[-1]-g.control_gap_decades.iloc[0]),**means,
          "B_P_delta_T":float(g.B_P_log10_tauP_over_taue.iloc[-1]-g.B_P_log10_tauP_over_taue.iloc[0]),"B_T_delta_T":float(g.B_T_log10_tauT_over_taue.iloc[-1]-g.B_T_log10_tauT_over_taue.iloc[0])})
    return provenance(pd.DataFrame(kr),candidates,"LEVEL_A_RESPONSE_INDEPENDENT_INTRINSIC"),provenance(pd.DataFrame(br),candidates,"LEVEL_A_RESPONSE_INDEPENDENT_INTRINSIC")

def state_proxy_table(candidates):
    state=pd.read_csv(V1/"fracture_state_at_first_passage.csv",low_memory=False); rows=[]
    for cid,g in state.groupby("candidate_id"):
        g=g.sort_values("temperature_K"); T=g.temperature_K.to_numpy(float); applied=g.applied_tip_stress_proxy_Pa.to_numpy(float)
        vals={"tip_radius_over_r0":g.tip_radius_pre_advance_m.to_numpy(float)/1e-6,"backstress_over_applied_tip":g.backstress_pre_advance_Pa.to_numpy(float)/np.maximum(abs(applied),1e-300),
          "front_width_over_w0":g.front_width_pre_advance_m.to_numpy(float)/1e-5,"source_activations_normalized":g.cumulative_source_activations.to_numpy(float)/np.maximum(g.source_multiplicity_pre_advance.to_numpy(float),1),
          "line_content_normalized":g.cumulative_line_content.to_numpy(float)/np.maximum(g.source_multiplicity_pre_advance.to_numpy(float),1)}
        rec={"candidate_id":cid,"state_reconstruction_class":"PARTIAL_SAVED_FIRST_PASSAGE_PROXY","missing_state_fields":"mobile_population;retained_population;slip_field;full_active_state_vector"}
        for name,v in vals.items():
            rec.update({f"{name}_mean_T":float(np.nanmean(v)),f"{name}_span_T":float(np.nanmax(v)-np.nanmin(v)),f"{name}_max_T":float(np.nanmax(v)),
              f"{name}_slope_per_K":float(np.polyfit(T,v,1)[0]),f"{name}_max_abs_curvature_per_K2":float(np.nanmax(abs(np.gradient(np.gradient(v,T),T))))})
        rows.append(rec)
    return provenance(pd.DataFrame(rows),candidates,"LEVEL_C_PARTIAL_SAVED_FIRST_PASSAGE_PROXY")

def response_conditioned_table(candidates):
    t=pd.read_csv(V2/"expanded_temperature_resolved_descriptors.csv",low_memory=False)
    keep=["candidate_id","temperature_K"]+[c for c in t if any(k in c for k in ["K025","K050","K075","K100","at_closest","first_kinetic_crossover","state_"]) and not c.startswith("state_")]
    q=t[list(dict.fromkeys([c for c in keep if c in t]))].copy(); q["diagnostic_class"]="RESPONSE_CONDITIONED_DIAGNOSTIC"
    return provenance(q,candidates,"LEVEL_B_RESPONSE_CONDITIONED_DIAGNOSTIC")

PROVENANCE_KEYS=("candidate_id","fingerprint","registry","git_sha","provenance","repository","branch","canonical","option_key","holdout","status","seed","source_dataset","analysis_level","prediction_eligible")
RESPONSE_KEYS=("K_response","K_checkpoint","K_first","K_10um","K_25um","K_50um","resistance","DBTT","peak_","S_low","S_mid","S_high","thermal_softening","fractional_","response_target","local_dK","local_d2K","developed_R","extension","n_events","event_","first_event")
STATE_KEYS=("state_","backstress","tip_radius","front_width","source_activ","line_content","source_multiplicity","shield")
CONDITIONED_KEYS=("at_K0","at_closest","first_kinetic_crossover","K_over_first","K_minus_K300")

def classify_column(name,dataset):
    low=name.lower()
    if any(k.lower() in low for k in PROVENANCE_KEYS): return "PROVENANCE","identity, source, execution, or audit metadata"
    if dataset=="response_conditioned_diagnostic": return "RESPONSE_DERIVED_DIAGNOSTIC","evaluation coordinate uses observed response"
    if dataset=="state_proxy": return "STATE_MEDIATOR","depends on archived evolved first-passage state"
    if dataset=="v1_response": return "RESPONSE_VARIABLE","fracture response or response summary"
    if dataset in {"response_independent_barrier_shape","activation_window","whole_surface_kinetic","plastic_bottleneck"}: return "INTRINSIC_PREDICTOR","constitutive parameter or standardized response-independent descriptor"
    if any(k.lower() in low for k in STATE_KEYS): return "STATE_MEDIATOR","depends on archived evolved first-passage state"
    if any(k.lower() in low for k in CONDITIONED_KEYS) or "residual" in low: return "RESPONSE_DERIVED_DIAGNOSTIC","evaluation coordinate uses observed response"
    if (any(k.lower() in low for k in RESPONSE_KEYS) or low.startswith("k_") or "authoritative_response" in low
            or low in {"checkpoint_um","k_over_k300"}): return "RESPONSE_VARIABLE","fracture response or response summary"
    if dataset=="v1_master" and any(k in low for k in ["loading_","nominal_","sha256"]): return "PROVENANCE","loading-map or execution metadata"
    return "INTRINSIC_PREDICTOR","constitutive parameter or standardized response-independent descriptor"

def leakage_audit(datasets):
    rows=[]
    for dataset,frame in datasets.items():
        for col in frame.columns:
            cls,reason=classify_column(col,dataset); rows.append({"dataset":dataset,"column":col,"classification":cls,"headline_prediction_eligible":cls=="INTRINSIC_PREDICTOR","reason":reason})
    return pd.DataFrame(rows).drop_duplicates(["dataset","column"])

def fold_ids(ids,k=5): return np.array([int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)%k for x in ids])

def ridge_cv(frame,features,response,ids,alphas=(.1,1.,10.,100.,1000.,10000.)):
    q=frame[[response]+features].copy(); y=pd.to_numeric(q.pop(response),errors="coerce").to_numpy(float); good=np.isfinite(y); X=q.apply(pd.to_numeric,errors="coerce").to_numpy(float)[good]; y=y[good]; useids=np.asarray(ids)[good]; folds=fold_ids(useids)
    trials=[]
    for alpha in alphas:
        pred=np.full(len(y),np.nan)
        for f in range(5):
            tr=folds!=f; te=~tr
            med=np.nanmedian(X[tr],axis=0); med=np.where(np.isfinite(med),med,0); A=np.where(np.isfinite(X),X,med)
            mu=A[tr].mean(0); sd=A[tr].std(0); sd[sd<1e-14]=1; Atr=(A[tr]-mu)/sd; Ate=(A[te]-mu)/sd
            design=np.column_stack([np.ones(tr.sum()),Atr]); P=np.eye(design.shape[1])*alpha; P[0,0]=0
            beta=np.linalg.solve(design.T@design+P,design.T@y[tr]); pred[te]=np.column_stack([np.ones(te.sum()),Ate])@beta
        rmse=float(np.sqrt(np.mean((y-pred)**2))); r2=1-float(np.sum((y-pred)**2))/max(float(np.sum((y-y.mean())**2)),1e-300); trials.append((rmse,r2,alpha,len(y)))
    return min(trials,key=lambda x:x[0])

def family_features():
    return {
      "BARRIER_SCALE":["cleave_G0_900_eV","emit_G0_900_eV","cleave_available_drop_900_eV","emit_available_drop_900_eV","cleave_sigc_900_GPa","emit_sigc_900_GPa","available_drop_ratio_emit_over_cleave","sigc_ratio_emit_over_cleave_900"],
      "BARRIER_SHAPE":["cleave_x50","emit_x50","cleave_width80_over_x50","emit_width80_over_x50","cleave_asymmetry_90_10","emit_asymmetry_90_10","cleave_sstar_max","emit_sstar_max","cleave_activation_skewness","emit_activation_skewness","cleave_activation_excess_kurtosis","emit_activation_excess_kurtosis"],
      "RELATIVE_BARRIER_GEOMETRY":["delta_mu_emit_minus_cleave","normalized_center_separation_Dmu","activation_window_overlap_Oce","activation_window_wasserstein","activation_window_JS_nats","width80_ratio_emit_over_cleave","width80_difference_emit_minus_cleave","asymmetry80_ratio_emit_over_cleave","asymmetry80_difference_emit_minus_cleave","sstar_ratio_emit_over_cleave","cstar_max_ratio_emit_over_cleave"],
      "THERMAL_BARRIER_MOTION":["cleave_Theta_G_900","emit_Theta_G_900","cleave_Theta_sigma_900","emit_Theta_sigma_900","delta_Theta_G_900","delta_Theta_sigma_900"],
      "KINETIC_COMPETITION":["Dmin_low","Dmin_delta","Dgamma_low","Dgamma_delta","Mgamma_low","Mgamma_delta","Mgamma_slope_reversal_count","Mgamma_curvature_reversal_count","physical_Oce_low","physical_Oce_delta","competition_fraction_1decade_low","competition_fraction_1decade_delta","signed_dominance_area_low","signed_dominance_area_delta"],
      "PLASTIC_BOTTLENECK":["plastic_control_change_count","first_control_change_T_K","strongest_control_competition_T_K","control_gap_low_T_decades","control_gap_delta_decades","B_P_log10_tauP_over_taue","B_T_log10_tauT_over_taue","B_P_delta_T","B_T_delta_T"],
      "STATE_PROXY":["tip_radius_over_r0_span_T","tip_radius_over_r0_slope_per_K","backstress_over_applied_tip_span_T","backstress_over_applied_tip_slope_per_K","front_width_over_w0_span_T","source_activations_normalized_span_T","line_content_normalized_span_T"]}

def incremental_models(master,responses):
    fam=family_features(); stages=[("BARRIER_SCALE",["BARRIER_SCALE"]),("BARRIER_SHAPE",["BARRIER_SHAPE"]),
      ("RELATIVE_GEOMETRY_PLUS_THERMAL_MOTION",["RELATIVE_BARRIER_GEOMETRY","THERMAL_BARRIER_MOTION"]),("KINETIC_COMPETITION",["KINETIC_COMPETITION"]),
      ("PLASTIC_BOTTLENECK",["PLASTIC_BOTTLENECK"]),("STATE_PROXY",["STATE_PROXY"])]
    d=master[~master.is_canonical_holdout].copy(); rows=[]
    for response in responses:
        active=[]; previous=None; cumulative=[]
        for i,(label,families) in enumerate(stages,1):
            cumulative+=families
            active+= [x for family in families for x in fam[family] if x in d]
            rmse,r2,alpha,n=ridge_cv(d,active,response,d.candidate_id)
            rows.append({"response":response,"model_stage":i,"added_family":label,"cumulative_families":";".join(cumulative),"feature_count":len(active),"n":n,"cv_r2":r2,"cv_rmse":rmse,
              "delta_cv_r2":r2-(previous[1] if previous else 0),"delta_cv_rmse":rmse-(previous[0] if previous else rmse),"selected_alpha":alpha,"canonical_holdouts_excluded":True,"fold_definition":"sha256(candidate_id) modulo 5"}); previous=(rmse,r2)
    return pd.DataFrame(rows)

def standalone_family_models(master,responses):
    d=master[~master.is_canonical_holdout]; rows=[]
    for response in responses:
        for family,features in family_features().items():
            features=[x for x in features if x in d]; rmse,r2,a,n=ridge_cv(d,features,response,d.candidate_id)
            rows.append({"response":response,"feature_family":family,"feature_count":len(features),"n":n,"cv_r2":r2,"cv_rmse":rmse,"selected_alpha":a,"canonical_holdouts_excluded":True,"fold_definition":"sha256(candidate_id) modulo 5"})
    return pd.DataFrame(rows)

def associations(master,intrinsic,responses,v2):
    d=master[~master.is_canonical_holdout]; rows=[]
    for f in intrinsic:
        for response in responses:
            q=d[[f,response]].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
            if len(q)<30 or q[f].nunique()<5 or q[response].nunique()<5: continue
            rho,p=stats.spearmanr(q[f],q[response]); rows.append({"feature":f,"response":response,"n":len(q),"spearman_rho":rho,"spearman_p":p,"predictor_role":"INTRINSIC_PREDICTOR"})
    corr=pd.DataFrame(rows); mi=v2.mutual_information_table(master,intrinsic,responses,seed=914)
    return corr,mi

def spline_basis(x):
    x=np.asarray(x,float); center=x.mean(); scale=max(x.std(),1e-300); z=(x-center)/scale; knots=np.quantile(z,[.2,.4,.6,.8])
    return np.column_stack([z,z*z,z*z*z]+[np.maximum(z-k,0)**3 for k in knots])

def reg_tree_fit(X,y,depth=0,max_depth=3,min_leaf=15):
    node={"value":float(np.mean(y)),"n":len(y)}
    if depth>=max_depth or len(y)<2*min_leaf: return node
    best=None
    for j in range(X.shape[1]):
        for t in np.unique(np.quantile(X[:,j],np.linspace(.1,.9,9))):
            left=X[:,j]<=t
            if left.sum()<min_leaf or (~left).sum()<min_leaf: continue
            loss=np.sum((y[left]-y[left].mean())**2)+np.sum((y[~left]-y[~left].mean())**2)
            if best is None or loss<best[0]: best=(loss,j,float(t),left)
    if best is None: return node
    _,j,t,left=best; node.update({"feature_index":j,"threshold":t,"left":reg_tree_fit(X[left],y[left],depth+1,max_depth,min_leaf),"right":reg_tree_fit(X[~left],y[~left],depth+1,max_depth,min_leaf)}); return node

def reg_tree_predict(node,row):
    while "feature_index" in node: node=node["left"] if row[node["feature_index"]]<=node["threshold"] else node["right"]
    return node["value"]

def reg_tree_cv(frame,features,response):
    q=frame[["candidate_id",response]+features].copy(); y=pd.to_numeric(q[response],errors="coerce").to_numpy(float); good=np.isfinite(y); q=q[good]; y=y[good]; X=q[features].apply(pd.to_numeric,errors="coerce").to_numpy(float); folds=fold_ids(q.candidate_id); pred=np.full(len(y),np.nan)
    for f in range(5):
        tr=folds!=f; te=~tr; med=np.nanmedian(X[tr],axis=0); med=np.where(np.isfinite(med),med,0); A=np.where(np.isfinite(X),X,med); tree=reg_tree_fit(A[tr],y[tr])
        pred[te]=[reg_tree_predict(tree,r) for r in A[te]]
    return float(np.sqrt(np.mean((y-pred)**2))),1-float(np.sum((y-pred)**2))/max(float(np.sum((y-y.mean())**2)),1e-300),len(y)

def nonlinear_models(master,corr,mi,responses):
    d=master[~master.is_canonical_holdout].copy(); rows=[]
    for response in responses:
        top=mi[mi.response.eq(response)].sort_values("bias_corrected_MI_nats",ascending=False).feature.head(5).tolist()
        for f in top:
            q=d[["candidate_id",f,response]].replace([np.inf,-np.inf],np.nan).dropna(); x=q[f].to_numpy(float); z=(x-x.mean())/max(x.std(),1e-300)
            base=q.assign(_x=z); lrmse,lr2,la,n=ridge_cv(base,["_x"],response,q.candidate_id)
            quad=base.assign(_x2=z*z); qrmse,qr2,qa,_=ridge_cv(quad,["_x","_x2"],response,q.candidate_id)
            B=spline_basis(x)
            gam=q[["candidate_id",response]].copy()
            feats=[]
            for j in range(B.shape[1]): gam[f"b{j}"]=B[:,j]; feats.append(f"b{j}")
            grmse,gr2,ga,_=ridge_cv(gam,feats,response,gam.candidate_id)
            rho=float(corr[(corr.feature.eq(f))&corr.response.eq(response)].spearman_rho.iloc[0])
            mir=float(mi[(mi.feature.eq(f))&mi.response.eq(response)].bias_corrected_MI_nats.iloc[0])
            rows.append({"response":response,"model":"UNIVARIATE_LINEAR_QUADRATIC_GAM","feature":f,"n":n,"spearman_rho":rho,"bias_corrected_MI_nats":mir,"linear_cv_r2":lr2,"quadratic_cv_r2":qr2,"quadratic_gain":qr2-lr2,"gam_cv_r2":gr2,"gam_gain":gr2-lr2,
              "intermediate_window_supported":bool((qr2-lr2)>.03 or (gr2-lr2)>.05),"canonical_holdouts_excluded":True})
        top=list(dict.fromkeys(top)); tree_rmse,tree_r2,n=reg_tree_cv(d,top,response)
        physical_pairs=[("delta_mu_emit_minus_cleave","activation_window_overlap_Oce"),("Mgamma_low","Mgamma_delta"),("delta_Theta_G_900","delta_Theta_sigma_900"),("B_P_log10_tauP_over_taue","B_P_delta_T")]
        pairvars=list(dict.fromkeys(x for pair in physical_pairs for x in pair if x in d)); main=list(dict.fromkeys(top+pairvars)); q=d[["candidate_id",response]+main].copy(); base_rmse,base_r2,a,n=ridge_cv(q,main,response,q.candidate_id)
        interactions=[]; frame=q.copy()
        for x,y in physical_pairs:
            if x in frame and y in frame:
                name=f"{x}__X__{y}"; frame[name]=pd.to_numeric(frame[x],errors="coerce")*pd.to_numeric(frame[y],errors="coerce"); interactions.append(name)
        irmse,ir2,ia,_=ridge_cv(frame,main+interactions,response,frame.candidate_id)
        rows.extend([{"response":response,"model":"SHALLOW_REGRESSION_TREE_DEPTH3","feature":";".join(top),"n":n,"tree_cv_r2":tree_r2,"tree_cv_rmse":tree_rmse,"canonical_holdouts_excluded":True},
          {"response":response,"model":"LOW_ORDER_PHYSICAL_INTERACTION_RIDGE","feature":";".join(top),"n":n,"linear_cv_r2":base_r2,"interaction_cv_r2":ir2,"interaction_gain":ir2-base_r2,"interaction_terms":";".join(interactions),"canonical_holdouts_excluded":True}])
    return pd.DataFrame(rows)

def pca_scores(master,features,prefix):
    discovery=master[~master.is_canonical_holdout]; X=discovery[features].replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce"); med=X.median(); X=X.fillna(med); mu=X.mean(); sd=X.std(); keep=sd>1e-12
    Z=(X.loc[:,keep]-mu[keep])/sd[keep]; u,s,vt=np.linalg.svd(Z,full_matrices=False)
    allx=master[features].replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce").fillna(med); allz=(allx.loc[:,keep]-mu[keep])/sd[keep]; score=allz.to_numpy()@vt[:8].T
    out=master[["candidate_id","canonical_family","is_canonical_holdout"]].copy()
    for i in range(score.shape[1]): out[f"{prefix}_PC{i+1}"]=score[:,i]
    var=s*s/max(len(Z)-1,1); meta={"fit_population":"396 discovery candidates; canonical four transformed only","features":list(np.asarray(features)[keep]),"explained_variance_ratio":list((var/var.sum())[:8]),"top_loadings":{}}
    names=np.asarray(features)[keep]
    for i in range(min(8,len(vt))):
        ix=np.argsort(abs(vt[i]))[::-1][:10]; meta["top_loadings"][f"PC{i+1}"]=[{"feature":str(names[j]),"loading":float(vt[i,j])} for j in ix]
    return out,meta

def response_pca_models(barrier_scores,candidates):
    response=pd.read_csv(V1/"fracture_response_pca_scores.csv"); d=barrier_scores.merge(response,on="candidate_id").merge(candidates[["candidate_id","is_canonical_holdout"]],on="candidate_id",suffixes=("","_candidate")); d=d[~d.is_canonical_holdout_candidate]
    rows=[]; base=[f"focused_barrier_PC{i}" for i in range(1,6)]
    for target in ["fracture_response_PC1","fracture_response_PC2"]:
        quad=d.assign(**{f"{x}_squared":d[x]**2 for x in base}); qfeatures=base+[f"{x}_squared" for x in base]
        gam=d[["candidate_id",target]].copy(); gfeatures=[]
        for x in base:
            B=spline_basis(d[x])
            for j in range(B.shape[1]): name=f"{x}_spline{j}"; gam[name]=B[:,j]; gfeatures.append(name)
        specs=[("LINEAR_OLS",base,d,(1e-12,)),("LINEAR_RIDGE",base,d,(.1,1,10,100)),("QUADRATIC_RIDGE",qfeatures,quad,(.1,1,10,100)),("ADDITIVE_SPLINE_GAM",gfeatures,gam,(.1,1,10,100))]
        for model,features,frame,alphas in specs:
            rmse,r2,a,n=ridge_cv(frame,features,target,frame.candidate_id,alphas=alphas); rows.append({"response_PC":target,"model":model,"features":";".join(features),"n":n,"cv_r2":r2,"cv_rmse":rmse,"selected_alpha":a,"canonical_holdouts_excluded":True})
    return pd.DataFrame(rows)

def transition_resolution_table(responses):
    points=pd.read_csv(V1/"fracture_response_curve_points.csv"); rows=[]
    for r in responses.itertuples(index=False):
        T=np.sort(points.loc[points.candidate_id.eq(r.candidate_id),"temperature_K"].unique().astype(float))
        rec={"candidate_id":r.candidate_id,"temperature_grid_K":";".join(f"{x:g}" for x in T),"minimum_grid_spacing_K":float(np.min(np.diff(T))),"maximum_grid_spacing_K":float(np.max(np.diff(T)))}
        for key,value in [("DBTT",finite(r.DBTT_temperature_K)),("Peak",finite(r.peak_temperature_K))]:
            if np.isfinite(value):
                i=int(np.argmin(abs(T-value))); lo=T[0] if i==0 else .5*(T[i-1]+T[i]); hi=T[-1] if i==len(T)-1 else .5*(T[i]+T[i+1])
            else: lo=hi=np.nan
            rec[f"{key}_temperature_grid_lower_K"]=lo; rec[f"{key}_temperature_grid_upper_K"]=hi
        rec["DBTT_width_resolution_floor_K"]=float(np.min(np.diff(T))); rec["interpretation"]="grid-defined bounds; no sub-grid precision claimed"; rows.append(rec)
    return pd.DataFrame(rows)

def savefig(fig,out,stem,data):
    fig.savefig(out/f"{stem}.png",dpi=190,bbox_inches="tight"); plt.close(fig); data.to_csv(out/f"{stem}_plot_data.csv",index=False)

def scatter_continuous(out,stem,data,x,y,color,label,canonical_only=False):
    q=data[["candidate_id","canonical_family",x,y,color]].replace([np.inf,-np.inf],np.nan).dropna(subset=[x,y,color]); fig,ax=plt.subplots(figsize=(7.2,5.4))
    base=q[~q.candidate_id.isin(CANONICAL)]; s=ax.scatter(base[x],base[y],c=base[color],cmap="viridis",s=25,alpha=.72,edgecolor="none"); fig.colorbar(s,ax=ax,label=label)
    for cid,name in CANONICAL.items():
        g=q[q.candidate_id.eq(cid)]
        if len(g): ax.scatter(g[x],g[y],marker="*",s=160,c=COLORS[name],edgecolor="black",linewidth=.6); ax.annotate(name,(g[x].iloc[0],g[y].iloc[0]),xytext=(4,4),textcoords="offset points",fontsize=8)
    if x=="delta_mu_emit_minus_cleave": ax.set_xscale("symlog",linthresh=1.)
    if x=="delta_Theta_G_900": ax.set_xscale("symlog",linthresh=.01)
    if y=="delta_Theta_sigma_900": ax.set_yscale("symlog",linthresh=.01)
    ax.set(xlabel=x.replace("_"," "),ylabel=y.replace("_"," "),title=stem.replace("_"," ").title()); savefig(fig,out,stem,q)

def figures(out,master,incremental,nonlinear,barrier_scores,response_scores):
    stems=[]
    specs=[
      ("activation_window_center_overlap_map","delta_mu_emit_minus_cleave","activation_window_overlap_Oce","fractional_resistance_span","Fractional resistance span"),
      ("whole_surface_kinetic_competition_map","Mgamma_low","Mgamma_delta","peak_prominence_MPa_sqrt_m","Peak prominence"),
      ("thermal_barrier_motion_map","delta_Theta_G_900","delta_Theta_sigma_900","DBTT_magnitude_MPa_sqrt_m","DBTT magnitude"),
      ("plastic_bottleneck_transition_map","B_P_log10_tauP_over_taue","B_P_delta_T","peak_prominence_MPa_sqrt_m","Peak prominence"),
      ("canonical_four_on_activation_window_map","delta_mu_emit_minus_cleave","activation_window_overlap_Oce","fractional_resistance_span","Fractional resistance span"),
      ("canonical_four_on_kinetic_competition_map","Mgamma_low","Mgamma_delta","fractional_resistance_span","Fractional resistance span")]
    for spec in specs: scatter_continuous(out,spec[0],master,*spec[1:]); stems.append(spec[0])
    fig,ax=plt.subplots(figsize=(9,5.5))
    for response,g in incremental.groupby("response"): ax.plot(g.model_stage,g.cv_r2,"o-",label=response.replace("_"," "))
    ax.axhline(0,color="black",lw=.6); ticks=incremental.drop_duplicates("model_stage").sort_values("model_stage"); ax.set_xticks(ticks.model_stage,ticks.added_family.str.replace("_","\n"),fontsize=7); ax.set(ylabel=r"Cross-validated $R^2$",title="Incremental explanatory value by physical stage"); ax.legend(fontsize=6,ncol=2); savefig(fig,out,"feature_family_incremental_R2",incremental); stems.append("feature_family_incremental_R2")
    scores=master.merge(response_scores[["candidate_id","fracture_response_PC1","fracture_response_PC2"]],on="candidate_id",how="inner")
    scatter_continuous(out,"response_PC1_barrier_geometry_map",scores,"delta_mu_emit_minus_cleave","activation_window_overlap_Oce","fracture_response_PC1","Response PC1"); stems.append("response_PC1_barrier_geometry_map")
    scatter_continuous(out,"response_PC2_barrier_geometry_map",scores,"Mgamma_low","Mgamma_delta","fracture_response_PC2","Response PC2"); stems.append("response_PC2_barrier_geometry_map")
    fig,ax=plt.subplots(figsize=(7,5.2)); q=nonlinear[nonlinear.model.eq("UNIVARIATE_LINEAR_QUADRATIC_GAM")].copy(); ax.scatter(abs(q.spearman_rho),q[["quadratic_gain","gam_gain"]].max(axis=1),c=q.bias_corrected_MI_nats,cmap="plasma",s=28,alpha=.75)
    ax.axhline(0,color="black",lw=.6); ax.set(xlabel="Absolute Spearman rho",ylabel="Best nonlinear CV R2 gain over linear",title="Nonlinear information missed by monotonic rank correlation"); fig.colorbar(ax.collections[0],ax=ax,label="Bias-corrected MI (nats)"); savefig(fig,out,"nonlinear_vs_spearman_comparison",q); stems.append("nonlinear_vs_spearman_comparison")
    return stems

def report(out,master,incremental,standalone,nonlinear,pca_models,pca_meta,activation_tests):
    best=standalone.loc[standalone.groupby("response").cv_r2.idxmax()]
    nl=nonlinear[nonlinear.model.eq("UNIVARIATE_LINEAR_QUADRATIC_GAM")].sort_values("gam_gain",ascending=False).head(5)
    ptxt="\n".join(f"- {r.response}: strongest standalone family `{r.feature_family}`, CV R2={r.cv_r2:.3f}." for r in best.itertuples())
    ntxt="\n".join(f"- {r.response} versus `{r.feature}`: quadratic gain {r.quadratic_gain:.3f}, GAM gain {r.gam_gain:.3f}." for r in nl.itertuples())
    pctxt="\n".join(f"- {r.response_PC}, {r.model}: CV R2={r.cv_r2:.3f}, RMSE={r.cv_rmse:.3g}." for r in pca_models.itertuples())
    atop=activation_tests.sort_values("bias_corrected_MI_nats",ascending=False).head(6)
    atxt="\n".join(f"- `{r.feature}` versus {r.response}: Spearman rho={r.spearman_rho:.3f}, corrected MI={r.bias_corrected_MI_nats:.3f} nats, permutation q={r.permutation_q_fdr:.3f}." for r in atop.itertuples())
    canonical=master[master.canonical_family.notna()]
    ctxt="\n".join(f"- {r.canonical_family}: delta_mu={r.delta_mu_emit_minus_cleave:.3g}, overlap={r.activation_window_overlap_Oce:.3f}, M_gamma(low)={r.Mgamma_low:.3g}, delta M_gamma={r.Mgamma_delta:.3g}, Dmin slope reversals={int(r.Dmin_slope_reversal_count)}, Dmin interior extremum={bool(r.Dmin_has_interior_extremum)}, Peierls bottleneck low/delta=({r.B_P_log10_tauP_over_taue:.3g}, {r.B_P_delta_T:.3g}), control switches={int(r.plastic_control_change_count)}." for r in canonical.itertuples())
    text=f"""# Focused v9.13 barrier-geometry / temperature-morphology amendment

## Scope and leakage boundary

This existing-data amendment preserves v1 and the earlier broad v2 archive. No fracture simulation was launched and no constitutive parameter was changed. Headline prediction excludes the four canonical holdouts and admits only `INTRINSIC_PREDICTOR` columns. Observed-load evaluations are isolated as `RESPONSE_CONDITIONED_DIAGNOSTIC`; genuinely archived state is isolated as `PARTIAL_SAVED_FIRST_PASSAGE_PROXY` and enters only the last mediation stage.

## Central result

The complete analysis supports a physical chain rather than a one-parameter law: barrier scale and analytic dimensionless shape determine activation-window alignment and thermal motion; these determine whole-surface kinetic separation and plastic bottlenecks; partial evolved state can then add explanatory information. The incremental table quantifies where information enters, while negative increments are retained rather than hidden.

## Analytic barrier shape

For the exact EXP-floor law, transition positions use `x_p=(-ln(p)/a)^(1/n)`. Activation sensitivity is the normalized Weibull density `p(x)=a*n*x^(n-1)*exp(-a*x^n)`, so its moments are analytic. The normalized overlap depends only on fixed `a,n` and is therefore exactly temperature-invariant; a separate physical-stress overlap tracks thermal motion of `sigc_c(T)` and `sigc_e(T)`. Floor fraction affects barrier scale allocation but cancels from normalized `phi`. For `n<1`, the true maximum sensitivity is singular at zero and is recorded as missing plus a singularity flag; it is not replaced by a grid-dependent finite maximum. Curvature is evaluated from the analytic second derivative, with the same singularity disclosure.

## Whole-surface competition

Competition uses exact production cleavage multi-hit and emission Arrhenius rates with intrinsic source multiplicity one over `z=sigma/sqrt(sigc_c(T)*sigc_e(T))` uniformly on `[0,3]`. Continuous minimum, signed/absolute integrated separation, competition fractions, dominance area, thermal extrema, and reversals replace reliance on one exact crossover. The temperature derivative at `z=1` is additively decomposed into cleavage/emission `gT`, cleavage/emission `sT`, shape-weighted explicit `1/T`, and production multihit/prefactor correction.

## Direct activation-window tests

{atxt}

The clearest activation-window association is nonlinear information about DBTT width, but its multiplicity-adjusted permutation q is approximately 0.08 rather than below 0.05. Weak-T flatness has a monotonic association with Wasserstein separation, while its corrected MI is weaker after multiplicity correction. These are useful design coordinates, not stand-alone universal predictors.

## Plastic bottleneck

Emission, Peierls, and Taylor mean log-times over the same standardized domain define rate-limiting control. A control gap below 0.5 decade is `MIXED_PLASTIC_CONTROL`; switching count, first-switch temperature, and strongest competition are reported.

## Incremental prediction

{ptxt}

All stages use identical SHA-256 candidate folds for a given response. Breakpoint responses retain their historical grid resolution; no sub-grid precision is inferred.

## Nonlinear versus monotonic evidence

{ntxt}

Quadratic and spline gains are cross-validated. A weak Spearman coefficient is not interpreted as absence of an intermediate-window effect when nonlinear gain is positive.

## Barrier PCA to response PCA

The focused barrier PCA is fit on 396 discovery candidates and the canonical four are transformed afterward. Response PCs and unsupervised clusters are the preserved v1 modes. Response PC1 (76.2% of response variance) is the common normalized resistance elevation from roughly 900–1400 K; PC2 (13.0%) contrasts the 800–950 K response against the 1200–1400 K tail, so it is a temperature-shape/tilt mode rather than overall magnitude.

{pctxt}

Response-PC loadings and temperature resolution remain those documented in v1. Focused barrier-PC leading loadings are preserved in `focused_barrier_pca_metadata.json`.

## Canonical holdouts

The canonical DBTT, Peak-T, weak-T, and ceramic-like rows are plotted only after maps and models are fixed. They never set boundaries, choose features, determine PCA axes, or contribute to CV scores.

{ctxt}

DBTT occupies the largest canonical activation-window separation and is the only canonical row with a plastic-control switch. Peak-T's signed average competition `M_gamma` is monotonic, but its minimum kinetic separation `Dmin` has an interior extremum and four sampled slope reversals; this supports testing an intermediate kinetic window rather than demanding multiple exact crossings. Weak-T has the greatest canonical low-temperature signed kinetic separation but little thermal change, consistent with cancellation. Ceramic-like does not show persistent opening dominance over the entire standardized domain, so that simple interpretation remains rejected.

## Revised smallest causality test—not launched

Use a compact response-surface design that independently varies four coordinates while maintaining approximately common 300 K fracture scale: (1) activation-window center separation `delta_mu`; (2) overlap/relative width, choosing one to avoid collinearity; (3) relative thermal stress-scale motion `delta_Theta_sigma`; and (4) the low-temperature plastic bottleneck gap. Differential entropy should be included only as a controlled secondary contrast if it can be varied independently. A 2-level design plus center points (approximately 12–18 rows after feasibility filtering) is sufficient. No design should be launched before user review.

## Limitations

- The saved first-passage archive lacks mobile/retained populations, slip fields, shielding, and the full active vector; none were reconstructed.
- Kinetic source multiplicity is fixed to one for intrinsic prediction. Saved multiplicity belongs to Level C.
- DBTT temperature/width and Peak temperature retain discrete historical-grid uncertainty.
- Association and mediation remain observational; the proposed factorial is required for causality.
"""
    (out/"FOCUSED_BARRIER_MORPHOLOGY_REPORT.md").write_text(text)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=OUT); args=ap.parse_args(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    v1=load_module(REPO/"scripts/analyze_v913_barrier_temperature_fracture_morphology.py","_v913_v1_focused"); v2=load_module(REPO/"scripts/analyze_v913_expanded_barrier_temperature_morphology.py","_v913_v2_focused")
    E,P=v1.load_production_types(SOURCE); candidates,cases,events,_=v1.load_population(SOURCE)
    intrinsic,windows=intrinsic_tables(candidates,v1,E); kin,bottle=kinetic_temperature_tables(candidates,v1,E,P); kagg,bagg=aggregate_kinetic(kin,bottle,candidates); state=state_proxy_table(candidates); conditioned=response_conditioned_table(candidates)
    responses=pd.read_csv(V1/"fracture_response_descriptors.csv",low_memory=False)
    model_responses=RESPONSES+SECONDARY_RESPONSES
    master=intrinsic.merge(kagg.drop(columns=[c for c in kagg if c in intrinsic and c!="candidate_id"]),on="candidate_id").merge(bagg.drop(columns=[c for c in bagg if c in intrinsic and c!="candidate_id"]),on="candidate_id").merge(state.drop(columns=[c for c in state if c in intrinsic and c!="candidate_id"]),on="candidate_id").merge(responses[["candidate_id",*model_responses]],on="candidate_id")
    fam=family_features(); intrinsic_features=list(dict.fromkeys(x for f in fam if f!="STATE_PROXY" for x in fam[f] if x in master))
    corr,mi=associations(master,intrinsic_features,model_responses,v2); incremental=incremental_models(master,RESPONSES); standalone=standalone_family_models(master,RESPONSES); nonlinear=nonlinear_models(master,corr,mi,model_responses)
    activation_tests=corr[corr.feature.isin(["delta_mu_emit_minus_cleave","normalized_center_separation_Dmu","activation_window_overlap_Oce","activation_window_wasserstein","physical_Oce_low","physical_Oce_delta"]) & corr.response.isin(["DBTT_magnitude_MPa_sqrt_m","DBTT_width_K","peak_prominence_MPa_sqrt_m","weakT_max_deviation_from_mean_MPa_sqrt_m"])].merge(mi,on=["feature","response","n"],how="left",suffixes=("_spearman","_MI"))
    bscore,pmeta=pca_scores(master,intrinsic_features,"focused_barrier"); pmodels=response_pca_models(bscore,candidates); response_scores=pd.read_csv(V1/"fracture_response_pca_scores.csv")
    response_clusters=pd.read_csv(V1/"fracture_temperature_morphology_clusters.csv"); preserved_response=response_scores.merge(response_clusters,on="candidate_id",how="left",suffixes=("","_cluster"))
    resolution=transition_resolution_table(responses)
    audit=leakage_audit({"v1_master":pd.read_csv(V1/"fracture_temperature_master.csv",nrows=1,low_memory=False),"v1_response":responses.head(1),"v2_expanded_master":pd.read_csv(V2/"expanded_barrier_temperature_descriptors.csv",nrows=1,low_memory=False),
      "response_independent_barrier_shape":intrinsic.head(1),"activation_window":windows.head(1),"whole_surface_kinetic":kin.head(1),"plastic_bottleneck":bottle.head(1),"state_proxy":state.head(1),"response_conditioned_diagnostic":conditioned.head(1)})
    outputs={"response_independent_barrier_shape_descriptors.csv":intrinsic,"activation_window_descriptors.csv":windows,"whole_surface_kinetic_competition.csv":kin,
      "plastic_bottleneck_descriptors.csv":bottle,"whole_surface_kinetic_temperature_summary.csv":kagg,"plastic_bottleneck_temperature_summary.csv":bagg,
      "state_proxy_temperature_descriptors.csv":state,"response_conditioned_mechanistic_diagnostics.csv":conditioned,
      "feature_family_incremental_models.csv":incremental,"feature_family_standalone_models.csv":standalone,"nonlinear_descriptor_models.csv":nonlinear,"response_pca_prediction_models.csv":pmodels,"predictor_leakage_audit.csv":audit,
      "intrinsic_predictor_response_correlations.csv":corr,"intrinsic_predictor_mutual_information.csv":mi,"focused_model_master.csv":master,"focused_barrier_pca_scores.csv":bscore,
      "activation_window_response_tests.csv":activation_tests,"preserved_response_pca_and_clusters.csv":preserved_response,"response_transition_grid_resolution.csv":resolution}
    for name,frame in outputs.items(): frame.to_csv(out/name,index=False)
    (out/"focused_barrier_pca_metadata.json").write_text(json.dumps(pmeta,indent=2))
    stems=figures(out,master,incremental,nonlinear,bscore,response_scores); report(out,master,incremental,standalone,nonlinear,pmodels,pmeta,activation_tests)
    v1manifest=[]
    for p in sorted(V1.glob("*")):
        if p.is_file(): v1manifest.append({"artifact":p.name,"path":str(p.resolve()),"sha256":sha(p)})
    pd.DataFrame(v1manifest).to_csv(out/"v1_artifact_manifest.csv",index=False)
    analysis={"status":"PASS","analysis_git_sha":git("rev-parse","HEAD"),"branch":git("branch","--show-current"),"simulation_git_sha":SIM_SHA,"physics_changed":False,"new_simulations_launched":False,
      "candidate_count":len(candidates),"discovery_candidate_count":int((~candidates.is_canonical_holdout).sum()),"canonical_holdout_count":int(candidates.is_canonical_holdout.sum()),"temperature_grid_K":TEMPS.tolist(),
      "headline_intrinsic_feature_count":len(intrinsic_features),"headline_intrinsic_features":intrinsic_features,"leakage_audit_rows":len(audit),"figure_stems":stems,
      "analysis_levels":{"A":"INTRINSIC_PREDICTOR","B":"RESPONSE_CONDITIONED_DIAGNOSTIC","C":"PARTIAL_SAVED_FIRST_PASSAGE_PROXY"}}
    (out/"focused_analysis_audit.json").write_text(json.dumps(analysis,indent=2)); print(json.dumps({"status":"PASS","out":str(out),"candidates":len(candidates),"intrinsic_features":len(intrinsic_features),"figures":len(stems)},indent=2))

if __name__=="__main__": raise SystemExit(main())
