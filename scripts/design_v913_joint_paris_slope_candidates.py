#!/usr/bin/env python3
"""Orthogonal 30-row derivative-focused joint fracture/fatigue design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.analyze_v913_joint_paris_slope_physics import (
        exp_floor_from_deltaK,
        finite,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` invocation.
    from analyze_v913_joint_paris_slope_physics import exp_floor_from_deltaK, finite


ACTIVE_FIELDS = (
    "Tref_K", "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n", "cleave_floor_frac",
    "emit_G00_eV", "emit_gT_eV_per_K", "emit_sigc0_GPa", "emit_sT_GPa_per_K",
    "emit_exp_a", "emit_exp_n", "emit_floor_frac", "peierls_H0_eV",
    "peierls_activation_entropy_kB", "peierls_exp_a", "peierls_exp_n",
    "peierls_nu0_s", "taylor_H0_eV", "taylor_activation_entropy_kB",
    "taylor_exp_a", "taylor_exp_n", "taylor_nu0_s", "rho_source0_m2",
    "taylor_corr_rho_c_m2", "taylor_corr_scale", "c_blunt",
)
PARENT_META = {
    "v913_zeroD_sobol_0202500": ("DBTT", 26.28653661187115, 21.02530765128298),
    "v913_zeroD_sobol_0242980": ("Peak-T", 26.530904648171045, 21.289546465050222),
    "v913_zeroD_sobol_0129902": ("weak-T", 16.66940640184987, 12.702935563752424),
    "v913_zeroD_sobol_0077080": ("ceramic-like", 15.40986559233958, 12.259477791864454),
    "v913_prospective_peakt_07_f4_minus": ("joint-balance", 26.886204839617815, 21.574654720315415),
}
PARENT_ORDER = tuple(PARENT_META)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-registry", type=Path, required=True)
    parser.add_argument("--prospective-registry", type=Path, required=True)
    parser.add_argument("--historical-ranked", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def fingerprint(row: pd.Series) -> str:
    payload = {field: float(row[field]) for field in ACTIVE_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def historical_bounds(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    rows = []
    for field in ACTIVE_FIELDS:
        column = f"x_raw__{field}"
        if column not in source:
            raise RuntimeError(f"historical population lacks {column}")
        values = pd.to_numeric(source[column], errors="coerce").dropna()
        rows.append({
            "parameter": field,
            "historical_min": values.min(),
            "historical_p05": values.quantile(.05),
            "historical_p50": values.quantile(.50),
            "historical_p95": values.quantile(.95),
            "historical_max": values.max(),
            "robust_scale": max(values.quantile(.95)-values.quantile(.05), 1e-15),
        })
    return pd.DataFrame(rows).set_index("parameter")


def parent_rows(canonical_path: Path, prospective_path: Path) -> pd.DataFrame:
    canonical = pd.read_csv(canonical_path)
    prospective = pd.read_csv(prospective_path)
    frames = [canonical[canonical.candidate_id.isin(PARENT_ORDER[:4])],
              prospective[prospective.candidate_id.eq(PARENT_ORDER[4])]]
    parents = pd.concat(frames, ignore_index=True, sort=False)
    if set(parents.candidate_id) != set(PARENT_ORDER):
        raise RuntimeError(f"missing parent rows: {set(PARENT_ORDER)-set(parents.candidate_id)}")
    if any(field not in parents for field in ACTIVE_FIELDS):
        raise RuntimeError("a parent registry lacks active fields")
    return parents.set_index("candidate_id").loc[list(PARENT_ORDER)].reset_index()


def extend_bounds_with_observed_parents(bounds: pd.DataFrame, parents: pd.DataFrame) -> pd.DataFrame:
    """Include later qualified canonical rows in the observed envelope."""
    out = bounds.copy()
    for field in ACTIVE_FIELDS:
        values = pd.to_numeric(parents[field], errors="coerce").dropna()
        out.loc[field, "historical_min"] = min(out.loc[field, "historical_min"], values.min())
        out.loc[field, "historical_max"] = max(out.loc[field, "historical_max"], values.max())
    return out


def _actual_300(row: pd.Series) -> tuple[float, float]:
    tref = finite(row.Tref_K)
    g0 = finite(row.cleave_G00_eV) + finite(row.cleave_gT_eV_per_K) * (300.0-tref)
    sigc = finite(row.cleave_sigc0_GPa) + finite(row.cleave_sT_GPa_per_K) * (300.0-tref)
    return g0, sigc


def _set_actual_300(row: pd.Series, g0: float | None = None, sigc: float | None = None) -> pd.Series:
    out = row.copy(); tref = finite(out.Tref_K)
    if g0 is not None:
        out["cleave_G00_eV"] = g0 - finite(out.cleave_gT_eV_per_K) * (300.0-tref)
    if sigc is not None:
        out["cleave_sigc0_GPa"] = sigc - finite(out.cleave_sT_GPa_per_K) * (300.0-tref)
    return out


def _p2_z_at_fixed_value_and_slope(ff: float, nnew: float, z0: float, n0: float) -> float:
    def weight(z: float) -> float:
        e = math.exp(-z)
        return (1.0-ff)*e/(ff+(1.0-ff)*e)
    target = n0*z0*weight(z0)
    grid = np.geomspace(1e-8, 100.0, 4000)
    values = np.asarray([nnew*z*weight(float(z))-target for z in grid])
    crossings = np.flatnonzero(values[:-1]*values[1:] <= 0.0)
    if not len(crossings):
        raise RuntimeError("P2 fixed-value/fixed-slope constraint has no positive root")
    roots=[]
    for index in crossings:
        lo,hi=float(grid[index]),float(grid[index+1])
        flo=float(values[index])
        for _ in range(60):
            mid=.5*(lo+hi); fm=nnew*mid*weight(mid)-target
            if flo*fm<=0: hi=mid
            else: lo=mid; flo=fm
        roots.append(.5*(lo+hi))
    return min(roots,key=lambda value:abs(math.log(value/z0)))


def apply_variant(parent: pd.Series, code: str, sign: float, amplitude: float, reference: float, bounds: pd.DataFrame) -> pd.Series:
    out = parent.copy(); dk = 1.04 * reference
    base = exp_floor_from_deltaK(parent, dk, 300.0, "cleave")
    g0_300, sigc_300 = _actual_300(parent)
    sigma = float(np.asarray(base["stress_Pa"])); x = sigma/(sigc_300*1e9)
    n0, a0 = finite(parent.cleave_exp_n), finite(parent.cleave_exp_a)
    z0 = a0*x**n0
    ff = finite(parent.cleave_floor_frac)
    if code == "P1":
        nnew = n0*(1.0+sign*.22*amplitude)
        out["cleave_exp_n"] = nnew
        out["cleave_exp_a"] = z0/max(x**nnew, 1e-300)
    elif code == "P2":
        nnew = n0*(1.0+sign*.25*amplitude)
        znew = _p2_z_at_fixed_value_and_slope(ff,nnew,z0,n0)
        out["cleave_exp_n"] = nnew
        # Use characteristic stress, rather than ``a``, as the compensating
        # coordinate.  This preserves n*z (the local first derivative) while
        # leaving the parent's admissible EXP coefficient untouched.
        new_sigc_300 = sigma / max((znew/a0)**(1.0/nnew), 1e-300) * 1e-9
        target_G = float(np.asarray(base["G_eV"]))
        exponential = math.exp(-znew)
        q = ff+(1.0-ff)*exponential
        new_g0 = target_G/q
        target_dT = float(np.asarray(base["dG_dT_eV_per_K"]))
        sigc_T = finite(parent.cleave_sT_GPa_per_K)
        thermal_stress_term = new_g0*(1.0-ff)*exponential*nnew*znew*sigc_T/new_sigc_300
        out["cleave_gT_eV_per_K"] = (target_dT-thermal_stress_term)/q
        out = _set_actual_300(out, g0=new_g0, sigc=new_sigc_300)
    elif code == "P3":
        factor = 1.0+sign*.35*amplitude
        out["cleave_gT_eV_per_K"] = finite(parent.cleave_gT_eV_per_K)*factor
        out["cleave_sT_GPa_per_K"] = finite(parent.cleave_sT_GPa_per_K)*factor
        out = _set_actual_300(out, g0=g0_300, sigc=sigc_300)
    elif code == "P4":
        scale = float(bounds.loc["peierls_H0_eV", "robust_scale"])
        out["peierls_H0_eV"] = finite(parent.peierls_H0_eV)+sign*.20*amplitude*scale
    else:
        raise ValueError(code)
    return out


def within_bounds(row: pd.Series, bounds: pd.DataFrame) -> bool:
    return all(
        finite(bounds.loc[field, "historical_min"]) <= finite(row[field]) <= finite(bounds.loc[field, "historical_max"])
        for field in ACTIVE_FIELDS
    )


def contracted_variant(parent: pd.Series, code: str, sign: float, reference: float, bounds: pd.DataFrame) -> tuple[pd.Series, float]:
    for amplitude in np.linspace(1.0, .1, 19):
        try:
            candidate = apply_variant(parent, code, sign, float(amplitude), reference, bounds)
        except RuntimeError:
            continue
        if within_bounds(candidate, bounds) and all(math.isfinite(finite(candidate[f])) for f in ACTIVE_FIELDS):
            return candidate, float(amplitude)
    raise RuntimeError(f"no feasible contraction for {parent.candidate_id} {code} {sign}")


def descriptor(row: pd.Series, reference: float) -> dict[str, float]:
    dk = 1.04*reference
    c = exp_floor_from_deltaK(row, dk, 300.0, "cleave")
    return {
        "HCF_reference_deltaK_MPa_sqrt_m": dk,
        "cleavage_G_at_HCF_eV": float(np.asarray(c["G_eV"])),
        "cleavage_dG_dK_at_HCF_eV_per_MPa_sqrt_m": float(np.asarray(c["dG_dK_eV_per_MPa_sqrt_m"])),
        "cleavage_d2G_dK2_at_HCF_eV_per_MPa2_m": float(np.asarray(c["d2G_dK2_eV_per_MPa2_m"])),
        "cleavage_dG_dT_at_HCF_eV_per_K": float(np.asarray(c["dG_dT_eV_per_K"])),
    }


def main() -> int:
    args = parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    parents = parent_rows(args.canonical_registry, args.prospective_registry)
    bounds = extend_bounds_with_observed_parents(historical_bounds(args.historical_ranked), parents)
    rows=[]; audits=[]
    for parent_index, parent in parents.iterrows():
        parent_id=str(parent.candidate_id); family,k300,reference=PARENT_META[parent_id]
        variants=(("P1",-1.0),("P1",1.0),("P2",-1.0),("P2",1.0),
                  ("P3",1.0 if parent_index%2==0 else -1.0),
                  ("P4",-1.0 if parent_index%2==0 else 1.0))
        before=descriptor(parent,reference)
        for variant_index,(axis,sign) in enumerate(variants,1):
            candidate,amplitude=contracted_variant(parent,axis,sign,reference,bounds)
            sign_label="minus" if sign<0 else "plus"
            short=family.lower().replace("-","").replace(" ","")
            cid=f"v913_slope_{short}_{variant_index:02d}_{axis.lower()}_{sign_label}"
            after=descriptor(candidate,reference)
            record={field:float(candidate[field]) for field in ACTIVE_FIELDS}
            record.update({
                "candidate_id":cid,"prospective_candidate_id":cid,
                "parent_candidate_id":parent_id,"parent_family":family,
                "design_axis":axis,"design_sign":int(sign),"design_amplitude_after_contraction":amplitude,
                "design_role":"ORTHOGONAL_SINGLE_AXIS_SLOPE_PHYSICS",
                "parent_K300_MPa_sqrt_m":k300,"parent_fatigue_reference_deltaK_MPa_sqrt_m":reference,
                "K300_exact_status":"PENDING_MONOTONIC_QUALIFICATION",
                "fatigue_specific_refit":False,"parameter_law_changed":False,
            })
            record["parameter_fingerprint"]=fingerprint(pd.Series(record)); rows.append(record)
            distance=math.sqrt(sum(((finite(candidate[f])-finite(parent[f]))/finite(bounds.loc[f,"robust_scale"]))**2 for f in ACTIVE_FIELDS))
            audits.append({
                "candidate_id":cid,"parent_candidate_id":parent_id,"parent_family":family,
                "design_axis":axis,"design_sign":int(sign),"requested_amplitude":1.0,
                "accepted_amplitude":amplitude,"all_parameters_within_historical_bounds":within_bounds(candidate,bounds),
                "robust_parameter_L2_distance":distance,
                **{f"parent_{k}":v for k,v in before.items()},
                **{f"candidate_{k}":v for k,v in after.items()},
                "relative_G_at_HCF_change":(after["cleavage_G_at_HCF_eV"]-before["cleavage_G_at_HCF_eV"])/before["cleavage_G_at_HCF_eV"],
                "relative_dG_dK_at_HCF_change":(after["cleavage_dG_dK_at_HCF_eV_per_MPa_sqrt_m"]-before["cleavage_dG_dK_at_HCF_eV_per_MPa_sqrt_m"])/abs(before["cleavage_dG_dK_at_HCF_eV_per_MPa_sqrt_m"]),
                "relative_d2G_dK2_at_HCF_change":(after["cleavage_d2G_dK2_at_HCF_eV_per_MPa2_m"]-before["cleavage_d2G_dK2_at_HCF_eV_per_MPa2_m"])/max(abs(before["cleavage_d2G_dK2_at_HCF_eV_per_MPa2_m"]),1e-30),
                "relative_dG_dT_at_HCF_change":(after["cleavage_dG_dT_at_HCF_eV_per_K"]-before["cleavage_dG_dT_at_HCF_eV_per_K"])/max(abs(before["cleavage_dG_dT_at_HCF_eV_per_K"]),1e-30),
                "bare_300K_cleavage_surface_preserved_exactly": axis=="P3",
                "cleavage_surface_unchanged":axis=="P4",
                "K300_acceptance_gate":"PENDING_EXACT_RUN",
            })
    registry=pd.DataFrame(rows).sort_values(["parent_family","candidate_id"])
    audit=pd.DataFrame(audits).sort_values(["parent_family","candidate_id"])
    if len(registry)!=30 or registry.parameter_fingerprint.nunique()!=30:
        raise RuntimeError("design must contain 30 unique physical rows")
    if not audit.all_parameters_within_historical_bounds.all():
        raise RuntimeError("design escaped the historical parameter envelope")
    registry.to_csv(args.out/"prospective_slope_design_registry.csv",index=False,float_format="%.17g")
    audit.to_csv(args.out/"prospective_slope_design_audit.csv",index=False)
    bounds.reset_index().to_csv(args.out/"prospective_slope_historical_parameter_bounds.csv",index=False)
    manifest={
        "schema":"v913_joint_paris_slope_candidate_design_v1","candidate_count":30,
        "parent_count":5,"variants_per_parent":6,"design":"ORTHOGONAL_SINGLE_AXIS",
        "axes":{"P1":"cleavage first derivative","P2":"cleavage curvature at controlled G and derivative","P3":"thermal derivative at identical 300 K surface","P4":"plastic bottleneck at identical cleavage surface"},
        "historical_bound_basis":"384_ROW_V913_POPULATION_PLUS_FIVE_OBSERVED_QUALIFIED_PARENT_ANCHORS",
        "all_within_historical_bounds":True,"K300_exact_status":"PENDING_MONOTONIC_QUALIFICATION",
        "fatigue_selected_before_fracture":False,"fatigue_specific_refit":False,"physics_law_changed":False,
    }
    (args.out/"prospective_slope_design_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(f"V913_PARIS_SLOPE_DESIGN_COMPLETE candidates={len(registry)} out={args.out}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
