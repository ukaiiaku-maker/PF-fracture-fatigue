#!/usr/bin/env python3
"""Construct the prospective v9.13 fracture-causality parameter design.

This is a design-only layer.  It evaluates the exact historical constitutive
surfaces, but it never advances a fracture state and never labels an unrun row
with a fracture response.  The output is an immutable, fingerprinted registry
for subsequent 300 K qualification and historical-grid fracture simulations.
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

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


REPO = Path(__file__).resolve().parents[1]
SOURCE = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_13_dbtt_temperature_shelf")
V1_SCRIPT = REPO / "scripts/analyze_v913_barrier_temperature_fracture_morphology.py"
FOCUSED_SCRIPT = REPO / "scripts/analyze_v913_focused_barrier_morphology.py"
FOCUSED = REPO / "runs/v913_barrier_temperature_fracture_morphology_v3_focused"
DEFAULT_OUT = REPO / "runs/v913_joint_fracture_fatigue_causality_v1/design"
SIM_SHA = "559425321b9a8739f32788322d8a1c2af8abad73"
KB = 8.617333262145e-5
T_SHAPE = 900.0
T_BOTTLENECK = 700.0
CANONICAL = {
    "DBTT": "v913_zeroD_sobol_0202500",
    "Peak-T": "v913_zeroD_sobol_0242980",
}
COORDS = (
    "F1_delta_mu",
    "F2_activation_window_overlap",
    "F3_delta_Theta_sigma_900",
    "F4_lowT_plastic_bottleneck",
)
VARY_FIELDS = (
    "cleave_exp_a", "cleave_exp_n", "emit_exp_a", "emit_exp_n",
    "emit_sT_GPa_per_K", "peierls_H0_eV",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def stable_json_sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def finite(value, default=np.nan) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float(default)
    except (TypeError, ValueError):
        return float(default)


def parameter_fingerprint(row: pd.Series, active_fields: list[str]) -> str:
    return stable_json_sha({field: finite(row[field]) for field in active_fields})


def overlap_fast(ca: float, cn: float, ea: float, en: float) -> float:
    """Deterministic overlap of the two normalized Weibull sensitivities."""
    # Identical quadrature definition to the focused-v3 descriptor.  Only the
    # upper endpoint of its quantile vector enters the overlap calculation.
    qc_hi = stats.weibull_min.ppf(1 - 1e-6, cn, scale=ca ** (-1 / cn))
    qe_hi = stats.weibull_min.ppf(1 - 1e-6, en, scale=ea ** (-1 / en))
    xmax = max(qc_hi, qe_hi)
    edges = np.r_[0.0, np.geomspace(1e-9, max(xmax, 1e-8), 5000)]
    pc = np.diff(stats.weibull_min.cdf(edges, cn, scale=ca ** (-1 / cn)))
    pe = np.diff(stats.weibull_min.cdf(edges, en, scale=ea ** (-1 / en)))
    pc /= max(pc.sum(), 1e-300)
    pe /= max(pe.sum(), 1e-300)
    return float(np.minimum(pc, pe).sum())


def design_coordinates(row: pd.Series, v1, ExpFloorSurface, PTMechanism) -> dict[str, float]:
    """Evaluate the four prospective coordinates without evolving state."""
    ca, cn = finite(row.cleave_exp_a), finite(row.cleave_exp_n)
    ea, en = finite(row.emit_exp_a), finite(row.emit_exp_n)
    mu_c = ca ** (-1 / cn) * special.gamma(1 + 1 / cn)
    mu_e = ea ** (-1 / en) * special.gamma(1 + 1 / en)
    gc = v1.make_surface(row, "cleave", ExpFloorSurface)
    ge = v1.make_surface(row, "emit", ExpFloorSurface)
    sc = gc.characteristic_stress_Pa(T_SHAPE)
    se = ge.characteristic_stress_Pa(T_SHAPE)
    raw_sc = gc.sigc0_Pa + gc.sT_Pa_per_K * (T_SHAPE - gc.Tref_K)
    raw_se = ge.sigc0_Pa + ge.sT_Pa_per_K * (T_SHAPE - ge.Tref_K)
    dsc = gc.sT_Pa_per_K if raw_sc > 1 else 0.0
    dse = ge.sT_Pa_per_K if raw_se > 1 else 0.0
    theta = T_SHAPE * dse / max(se, 1e-300) - T_SHAPE * dsc / max(sc, 1e-300)

    pm = PTMechanism(
        finite(row.peierls_H0_eV), finite(row.peierls_activation_entropy_kB),
        finite(row.peierls_exp_a), finite(row.peierls_exp_n), finite(row.peierls_nu0_s),
    )
    gp = pm.surface(ge)
    sc0 = gc.characteristic_stress_Pa(T_BOTTLENECK)
    se0 = ge.characteristic_stress_Pa(T_BOTTLENECK)
    sigma = np.linspace(0, 3, 301) * math.sqrt(sc0 * se0)
    Ge = np.asarray(ge.barrier_eV(sigma, T_BOTTLENECK), float)
    Gp = np.asarray(gp.barrier_eV(pm.stress_fraction * sigma, T_BOTTLENECK), float)
    re = np.maximum(v1.NU_E * np.exp(np.clip(-Ge / (KB * T_BOTTLENECK), -745, 700)), 1e-300)
    rp = np.maximum(pm.nu0_s * np.exp(np.clip(-Gp / (KB * T_BOTTLENECK), -745, 700)), 1e-300)
    bp = float(np.mean(-np.log10(rp) + np.log10(re)))
    return {
        COORDS[0]: float(mu_e - mu_c),
        COORDS[1]: overlap_fast(ca, cn, ea, en),
        COORDS[2]: float(theta),
        COORDS[3]: bp,
    }


def coordinate_audit(focused: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    options = {
        "activation_window_overlap_Oce": "F2_activation_window_overlap",
        "width80_ratio_emit_over_cleave": "F2_width80_ratio",
        "width80_difference_emit_minus_cleave": "F2_width80_difference",
    }
    rows = []
    for col, label in options.items():
        q = focused[["delta_mu_emit_minus_cleave", col]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "F2_candidate": label,
            "source_descriptor": col,
            "n": len(q),
            "pearson_with_F1": q.iloc[:, 0].corr(q.iloc[:, 1]),
            "spearman_with_F1": q.iloc[:, 0].corr(q.iloc[:, 1], method="spearman"),
        })
    out = pd.DataFrame(rows)
    # Pearson measures the near-linear inverse-design degeneracy; Spearman is
    # retained as a second audit rather than silently blended into selection.
    out["selection_score"] = out.pearson_with_F1.abs()
    selected = str(out.sort_values(["selection_score", "F2_candidate"]).iloc[0].F2_candidate)
    out["selected_for_F2"] = out.F2_candidate.eq(selected)
    out["selection_rule"] = "minimum absolute Pearson correlation with F1; Spearman disclosed"
    return out, selected


def robust_parameter_stats(candidates: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for field in fields:
        x = pd.to_numeric(candidates[field], errors="coerce").dropna()
        rows.append({
            "parameter": field, "historical_min": x.min(), "historical_p05": x.quantile(.05),
            "historical_p50": x.quantile(.50), "historical_p95": x.quantile(.95),
            "historical_max": x.max(), "robust_scale": max(x.quantile(.95) - x.quantile(.05), 1e-12),
        })
    return pd.DataFrame(rows).set_index("parameter")


def coordinate_stats(focused: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        COORDS[0]: "delta_mu_emit_minus_cleave",
        COORDS[1]: "activation_window_overlap_Oce",
        COORDS[2]: "delta_Theta_sigma_900",
        COORDS[3]: "F4_lowT_plastic_bottleneck",
    }
    rows = []
    for coord, field in mapping.items():
        x = pd.to_numeric(focused[field], errors="coerce").dropna()
        rows.append({
            "coordinate": coord, "source_descriptor": field, "min": x.min(), "p05": x.quantile(.05),
            "p25": x.quantile(.25), "p50": x.quantile(.50), "p75": x.quantile(.75),
            "p95": x.quantile(.95), "max": x.max(),
            "robust_scale": max(x.quantile(.75) - x.quantile(.25), .05 * (x.quantile(.95) - x.quantile(.05)), 1e-8),
        })
    return pd.DataFrame(rows).set_index("coordinate")


def with_low_temperature_bottleneck(focused: pd.DataFrame, plastic: pd.DataFrame) -> pd.DataFrame:
    low = (plastic.sort_values("temperature_K").drop_duplicates("candidate_id")
           [["candidate_id", "temperature_K", "B_P_log10_tauP_over_taue"]]
           .rename(columns={"temperature_K": "F4_reference_temperature_K",
                            "B_P_log10_tauP_over_taue": "F4_lowT_plastic_bottleneck"}))
    out = focused.drop(columns=["F4_lowT_plastic_bottleneck", "F4_reference_temperature_K"], errors="ignore").merge(low, on="candidate_id", how="left")
    if out.F4_lowT_plastic_bottleneck.isna().any():
        raise RuntimeError("missing low-temperature plastic bottleneck for qualified candidate")
    return out


def target_design(family: str, center: dict[str, float], cs: pd.DataFrame) -> list[tuple[str, dict[str, float]]]:
    c = np.array([center[k] for k in COORDS], float)
    med = cs.loc[list(COORDS), "p50"].to_numpy(float)
    if family == "DBTT":
        moderate = c + .30 * (med - c)
        strong = c + .60 * (med - c)
        codes = [
            ("F1_MOD", (0,), moderate), ("F1_STRONG", (0,), strong),
            ("F2_MOD", (1,), moderate), ("F2_STRONG", (1,), strong),
            ("F3_MOD", (2,), moderate), ("F3_STRONG", (2,), strong),
            ("F4_MOD", (3,), moderate), ("F4_STRONG", (3,), strong),
            ("ALL_MOD", (0, 1, 2, 3), moderate),
            ("F1F2_STRONG", (0, 1), strong),
            ("F3F4_STRONG", (2, 3), strong),
            ("ALL_STRONG", (0, 1, 2, 3), strong),
        ]
    else:
        step = .35 * np.maximum(np.abs(c - med), .25 * (cs.loc[list(COORDS), "p75"].to_numpy(float) - cs.loc[list(COORDS), "p25"].to_numpy(float)))
        minus, plus = c - step, c + step
        codes = []
        for i, tag in enumerate(("F1", "F2", "F3", "F4")):
            codes.extend([(f"{tag}_MINUS", (i,), minus), (f"{tag}_PLUS", (i,), plus)])
        patterns = [
            ("F1MIN_F2PLUS", {0: minus[0], 1: plus[1]}),
            ("F1PLUS_F2MIN", {0: plus[0], 1: minus[1]}),
            ("F3MIN_F4PLUS", {2: minus[2], 3: plus[3]}),
            ("F3PLUS_F4MIN", {2: plus[2], 3: minus[3]}),
        ]
        for name, changes in patterns:
            q = c.copy()
            for i, value in changes.items(): q[i] = value
            codes.append((name, tuple(changes), q))
    out = []
    for name, indices, values in codes:
        q = c.copy()
        for i in indices: q[i] = values[i]
        # Targets outside the observed envelope remain requested and are allowed
        # to fail closed in the audit rather than being clipped silently.
        out.append((name, dict(zip(COORDS, q))))
    return out


def fit_target(parent: pd.Series, target: dict[str, float], pstats: pd.DataFrame, cstats: pd.DataFrame,
               v1, ExpFloorSurface, PTMechanism) -> tuple[pd.Series, dict[str, float], optimize.OptimizeResult]:
    x0 = np.array([finite(parent[f]) for f in VARY_FIELDS], float)
    lo = pstats.loc[list(VARY_FIELDS), "historical_min"].to_numpy(float)
    hi = pstats.loc[list(VARY_FIELDS), "historical_max"].to_numpy(float)
    pscale = pstats.loc[list(VARY_FIELDS), "robust_scale"].to_numpy(float)
    t = np.array([target[k] for k in COORDS], float)
    cscale = cstats.loc[list(COORDS), "robust_scale"].to_numpy(float)

    def unpack(x):
        row = parent.copy()
        for f, value in zip(VARY_FIELDS, x): row[f] = float(value)
        return row

    def residual(x):
        coords = design_coordinates(unpack(x), v1, ExpFloorSurface, PTMechanism)
        rcoord = (np.array([coords[k] for k in COORDS]) - t) / cscale
        rdist = .025 * (x - x0) / pscale
        return np.r_[rcoord, rdist]

    result = optimize.least_squares(
        residual, np.clip(x0, lo, hi), bounds=(lo, hi), xtol=2e-11, ftol=2e-11,
        gtol=2e-11, max_nfev=700, x_scale=np.maximum(pscale, 1e-8),
    )
    row = unpack(result.x)
    return row, design_coordinates(row, v1, ExpFloorSurface, PTMechanism), result


def target_quality(target: dict[str, float], achieved: dict[str, float], cstats: pd.DataFrame) -> tuple[float, float]:
    scale = cstats.loc[list(COORDS), "robust_scale"].to_numpy(float)
    r = (np.array([achieved[k] for k in COORDS]) - np.array([target[k] for k in COORDS])) / scale
    return float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))


def prospective_design(candidates: pd.DataFrame, focused: pd.DataFrame, v1, ExpFloorSurface, PTMechanism):
    pstats = robust_parameter_stats(candidates, VARY_FIELDS)
    cstats = coordinate_stats(focused)
    registry, audits, changes = [], [], []
    for family, parent_id in CANONICAL.items():
        parent = candidates[candidates.candidate_id.eq(parent_id)].iloc[0].copy()
        center = design_coordinates(parent, v1, ExpFloorSurface, PTMechanism)
        control_id = f"v913_prospective_{family.lower().replace('-', '')}_CENTER"
        control = parent.copy(); control["prospective_candidate_id"] = control_id
        control["design_family"] = family; control["design_role"] = "EXACT_CANONICAL_CENTER_CONTROL"
        control["parent_candidate_id"] = parent_id; control["target_code"] = "CENTER"
        control["parent_parameter_fingerprint"] = parameter_fingerprint(parent, v1.ACTIVE_FIELDS)
        control["parameter_fingerprint"] = parameter_fingerprint(control, v1.ACTIVE_FIELDS)
        control["design_fingerprint"] = stable_json_sha({"role": "center", "parent": parent_id, "parameter_fingerprint": control.parameter_fingerprint})
        for k, value in center.items(): control[f"requested__{k}"] = value; control[f"achieved__{k}"] = value
        registry.append(control)

        for number, (code, requested) in enumerate(target_design(family, center, cstats), 1):
            original = requested.copy(); accepted = None
            # If a target is incompatible with the bounded six-parameter inverse
            # map, retain the failed attempt and contract only the changed target
            # vector toward the center until a feasible primary row exists.
            for attempt, contraction in enumerate((1.0, .8, .6, .4, .25), 1):
                target = {k: center[k] + contraction * (original[k] - center[k]) for k in COORDS}
                fitted, achieved, result = fit_target(parent, target, pstats, cstats, v1, ExpFloorSurface, PTMechanism)
                rms, max_abs = target_quality(target, achieved, cstats)
                within = all(cstats.loc[k, "min"] <= achieved[k] <= cstats.loc[k, "max"] for k in COORDS)
                feasible = bool(result.success and np.isfinite(rms) and rms <= .08 and max_abs <= .16 and within)
                audit_id = f"{family}:{code}:attempt{attempt}"
                audits.append({
                    "design_attempt_id": audit_id, "design_family": family, "parent_candidate_id": parent_id,
                    "target_code": code, "attempt": attempt, "target_contraction": contraction,
                    "feasibility_status": "FEASIBLE_PRIMARY" if feasible else "INFEASIBLE_RETAINED",
                    "optimizer_success": bool(result.success), "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message), "optimizer_nfev": int(result.nfev),
                    "design_residual_rms_robust_units": rms, "design_residual_max_abs_robust_units": max_abs,
                    "K300_pre_anchor_MPa_sqrt_m": np.nan, "K300_post_anchor_MPa_sqrt_m": np.nan,
                    "K300_parent_MPa_sqrt_m": np.nan,
                    "K300_qualification_status": "NOT_RUN_CANONICAL_300K_NOT_IN_HISTORICAL_GRID",
                    **{f"original_requested__{k}": original[k] for k in COORDS},
                    **{f"requested__{k}": target[k] for k in COORDS},
                    **{f"achieved__{k}": achieved[k] for k in COORDS},
                })
                if feasible:
                    accepted = (fitted, achieved, target, audit_id, rms, max_abs)
                    break
            if accepted is None:
                raise RuntimeError(f"no feasible contracted target for {family} {code}; see in-memory audit")
            fitted, achieved, target, audit_id, rms, max_abs = accepted
            cid = f"v913_prospective_{family.lower().replace('-', '')}_{number:02d}_{code.lower()}"
            fitted["prospective_candidate_id"] = cid; fitted["design_family"] = family
            fitted["design_role"] = "FEASIBLE_PRIMARY"; fitted["parent_candidate_id"] = parent_id
            fitted["parent_parameter_fingerprint"] = parameter_fingerprint(parent, v1.ACTIVE_FIELDS)
            fitted["target_code"] = code; fitted["accepted_design_attempt_id"] = audit_id
            fitted["parameter_fingerprint"] = parameter_fingerprint(fitted, v1.ACTIVE_FIELDS)
            fp_payload = {"candidate": cid, "parent": parent_id, "target": target,
                          "parameter_fingerprint": fitted.parameter_fingerprint}
            fitted["design_fingerprint"] = stable_json_sha(fp_payload)
            fitted["design_residual_rms_robust_units"] = rms
            fitted["design_residual_max_abs_robust_units"] = max_abs
            for k in COORDS: fitted[f"requested__{k}"] = target[k]; fitted[f"achieved__{k}"] = achieved[k]
            registry.append(fitted)
            for field in VARY_FIELDS:
                s = pstats.loc[field]; old, new = finite(parent[field]), finite(fitted[field])
                changes.append({
                    "prospective_candidate_id": cid, "design_family": family, "parent_candidate_id": parent_id,
                    "target_code": code, "parameter": field, "historical_min": s.historical_min,
                    "historical_p05": s.historical_p05, "historical_p50": s.historical_p50,
                    "historical_p95": s.historical_p95, "historical_max": s.historical_max,
                    "canonical_value": old, "prospective_value": new, "absolute_change": new - old,
                    "robust_standardized_change": (new - old) / s.robust_scale,
                    "outside_historical_observed_range": bool(new < s.historical_min or new > s.historical_max),
                    "outside_historical_p05_p95": bool(new < s.historical_p05 or new > s.historical_p95),
                    "inherited_center_outside_p05_p95": bool(old < s.historical_p05 or old > s.historical_p95),
                    "prospective_outside_but_center_inside_p05_p95": bool(
                        (new < s.historical_p05 or new > s.historical_p95)
                        and s.historical_p05 <= old <= s.historical_p95),
                })
    registry = pd.DataFrame(registry)
    # Preserve all exact active fields first; remove inherited retrospective labels
    # that could be mistaken for prospective response evidence.
    base = ["prospective_candidate_id", "design_family", "design_role", "parent_candidate_id", "target_code",
            "accepted_design_attempt_id", "parent_parameter_fingerprint", "parameter_fingerprint", "design_fingerprint",
            "design_residual_rms_robust_units", "design_residual_max_abs_robust_units"]
    coordcols = [f"{p}__{c}" for p in ("requested", "achieved") for c in COORDS]
    cols = [c for c in base + v1.ACTIVE_FIELDS + coordcols if c in registry]
    registry = registry[cols].copy()
    registry["simulation_status"] = "DESIGNED_NOT_RUN"
    registry["simulation_git_sha"] = SIM_SHA
    registry["design_analysis_git_sha"] = git("rev-parse", "HEAD")
    registry["historical_temperature_grid_K"] = "700;800;900;950;1000;1050;1100;1200;1300;1400"
    # Robust L2 parameter distance for transparent locality auditing.
    for i, row in registry.iterrows():
        parent = candidates[candidates.candidate_id.eq(row.parent_candidate_id)].iloc[0]
        z = [(finite(row[f]) - finite(parent[f])) / pstats.loc[f, "robust_scale"] for f in VARY_FIELDS]
        registry.loc[i, "parameter_distance_robust_L2"] = float(np.linalg.norm(z))
        registry.loc[i, "parameter_distance_robust_max_abs"] = float(np.max(np.abs(z)))
    return registry, pd.DataFrame(audits), pd.DataFrame(changes), pstats.reset_index(), cstats.reset_index()


def anchor_plan(registry: pd.DataFrame, candidates: pd.DataFrame, pstats_all: pd.DataFrame) -> pd.DataFrame:
    """Specify, but deliberately do not apply, a minimal K300 nuisance anchor."""
    stats = pstats_all.set_index("parameter")
    fields = ("cleave_sigc0_GPa", "cleave_sT_GPa_per_K", "emit_sigc0_GPa", "emit_sT_GPa_per_K")
    rows = []
    for r in registry.itertuples(index=False):
        if r.design_role != "FEASIBLE_PRIMARY": continue
        low, high = 0.0, np.inf
        for field in fields:
            value = finite(getattr(r, field))
            q = stats.loc[field]
            if value > 0:
                low = max(low, q.historical_min / value)
                high = min(high, q.historical_max / value)
            elif value < 0:
                low = max(low, q.historical_max / value)
                high = min(high, q.historical_min / value)
            elif not (q.historical_min <= 0 <= q.historical_max):
                low, high = 1.0, 0.0
        rows.append({
            "prospective_candidate_id": r.prospective_candidate_id,
            "parent_candidate_id": r.parent_candidate_id,
            "anchor_status": "PROPOSED_NOT_APPLIED_REQUIRES_REAL_300K_QUALIFICATION",
            "anchor_nuisance_coordinate": "common_stress_scale_lambda",
            "anchor_transformation": "multiply cleave/emit sigc0 and sT by one common lambda",
            "why_minimal": "one scalar preserves normalized activation shape, F1, F2, and F3 exactly before state evolution",
            "lambda_formula_after_pre_anchor_run": "K300_parent_real/K300_pre_anchor_real (initial linear-stress estimate only)",
            "historical_envelope_lambda_min": low,
            "historical_envelope_lambda_max": high,
            "lambda_applied": np.nan,
            "K300_parent_real_MPa_sqrt_m": np.nan,
            "K300_pre_anchor_real_MPa_sqrt_m": np.nan,
            "K300_post_anchor_real_MPa_sqrt_m": np.nan,
            "post_anchor_claim_permitted": False,
            "analytic_change_F1": 0.0,
            "analytic_change_F2": 0.0,
            "analytic_change_F3": 0.0,
            "analytic_change_F4_frozen_intrinsic": 0.0,
            "required_validation": "rerun exact 300 K fracture qualification; reject/redesign if target coordinates materially change",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    v1 = load_module(V1_SCRIPT, "_v913_design_v1")
    focused_mod = load_module(FOCUSED_SCRIPT, "_v913_design_focused")
    ExpFloorSurface, PTMechanism = v1.load_production_types(SOURCE)
    candidates, _, _, paths = v1.load_population(SOURCE)
    focused = pd.read_csv(FOCUSED / "focused_model_master.csv", low_memory=False)
    plastic = pd.read_csv(FOCUSED / "plastic_bottleneck_descriptors.csv", low_memory=False)
    focused = with_low_temperature_bottleneck(focused, plastic)
    collinearity, selected = coordinate_audit(focused)
    if selected != "F2_activation_window_overlap":
        raise RuntimeError(f"F2 audit unexpectedly selected {selected}")
    registry, audit, changes, pstats, cstats = prospective_design(
        candidates, focused, v1, ExpFloorSurface, PTMechanism,
    )
    # Include all fields touched by the optional common-scale anchor in the bounds table.
    anchor_fields = tuple(dict.fromkeys((*VARY_FIELDS, "cleave_sigc0_GPa", "cleave_sT_GPa_per_K", "emit_sigc0_GPa")))
    all_stats = robust_parameter_stats(candidates, anchor_fields).reset_index()
    anchors = anchor_plan(registry, candidates, all_stats)
    outputs = {
        "prospective_fracture_candidate_registry.csv": registry,
        "prospective_candidate_design_audit.csv": audit,
        "prospective_parameter_bounds_and_changes.csv": changes,
        "prospective_historical_parameter_bounds.csv": all_stats,
        "prospective_coordinate_historical_bounds.csv": cstats,
        "prospective_coordinate_collinearity_audit.csv": collinearity,
        "prospective_K300_anchor_plan.csv": anchors,
    }
    for name, frame in outputs.items(): frame.to_csv(out / name, index=False)
    manifest = {
        "schema": "v913_prospective_fracture_causality_design_v1",
        "design_only": True,
        "simulations_launched": False,
        "physics_changed": False,
        "analysis_branch": git("branch", "--show-current"),
        "analysis_git_sha": git("rev-parse", "HEAD"),
        "historical_simulation_git_sha": SIM_SHA,
        "source_repository": str(SOURCE),
        "source_registry_files": {k: str(v) for k, v in paths.items()},
        "source_input_sha256": {
            "focused_model_master": file_sha(FOCUSED / "focused_model_master.csv"),
            "plastic_bottleneck_descriptors": file_sha(FOCUSED / "plastic_bottleneck_descriptors.csv"),
            "broad_candidate_pool_features": file_sha(paths["broad_features"]),
            "weak_ceramic_candidate_pool_features": file_sha(paths["wc_features"]),
        },
        "active_parameter_fields": list(v1.ACTIVE_FIELDS),
        "focused_input": str(FOCUSED / "focused_model_master.csv"),
        "F2_selected": selected,
        "canonical_control_count": int(registry.design_role.eq("EXACT_CANONICAL_CENTER_CONTROL").sum()),
        "feasible_primary_count": int(registry.design_role.eq("FEASIBLE_PRIMARY").sum()),
        "infeasible_attempt_count_retained": int(audit.feasibility_status.eq("INFEASIBLE_RETAINED").sum()),
        "DBTT_primary_count": int(((registry.design_family == "DBTT") & (registry.design_role == "FEASIBLE_PRIMARY")).sum()),
        "PeakT_primary_count": int(((registry.design_family == "Peak-T") & (registry.design_role == "FEASIBLE_PRIMARY")).sum()),
        "K300_statement": "No canonical 300 K value exists on the historical grid for these centers; no pre/post-anchor K300 is claimed before real qualification.",
        "files": {name: {"rows": len(frame)} for name, frame in outputs.items()},
    }
    (out / "prospective_design_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("canonical_control_count", "feasible_primary_count", "infeasible_attempt_count_retained", "F2_selected")}, indent=2))


if __name__ == "__main__":
    main()
