"""Stage 2: admissibility-checked P40 derivation under three frozen radius
scenarios, with an explicit rejection registry and a minimax selection.

BOUND PROVENANCE (decided before selection, recorded here in full)
-----------------------------------------------------------------
inverse_fatigue_barrier_design_v10230.ExpFloorBounds defaults floor_fraction
to (0.001, 0.80). That 0.80 is a prior module default with no stated
production basis. The production validation limit is
MaterialManifest.ExpFloorBarrier.floor_max_fraction = 0.95, and the campaign
admissibility rule is likewise "floor <= 0.95 G0". The 0.95 limit is
therefore the bound actually sanctioned for this campaign.

At the 0.80 module default the optimum parks exactly on the bound
(floor_fraction = 0.800000), which is inadmissible under the campaign rule
"no parameter at an optimization bound unless separately justified". At the
production limit of 0.95 the optimum is strictly interior (0.8725) and
materially better (cost 0.0128 vs 0.0743; slope RMS 0.0192 vs 0.0737).

Both are recorded: the bound-parked row goes to the rejection registry with
its exact reason. This is a widening to the documented production limit, not
a narrowing invented after inspection.

DUAL FAMILY NOT REQUIRED: the single EXP-floor family satisfies every
analytical gate with all five parameters interior, so no new production
barrier class is added.

Every prediction here is made under a transfer model that is NOT qualified
(B1_TRANSFER_NOT_QUALIFIED_FOR_CANDIDATE_SELECTION). Outputs are labelled
NOT_A_QUALIFIED_PHYSICAL_PREDICTION.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from a_native_manifest import load_a_native_manifest
from derive_p40_candidates import (
    observed_radius_law, load_targets, measure_reduced_deficit, controls_for,
    dense_target, evaluate, score, DESIGN_K, PILOT_K, FULL_GRID, R_LOAD, NU0_S,
    R0, ALPHA_R_FROZEN,
)
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    ExpFloorBounds, project_exp_floor, invert_cooperative_rate_to_barrier,
    waveform_factor_C, barrier_from_vector, B1Controls, b1_event_conditioned_emission,
)

OUT = Path(__file__).resolve().parents[1]
PARAM_NAMES = ["cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a", "cleave_exp_n", "cleave_floor_frac"]
BOUNDS_PRODUCTION = ExpFloorBounds(floor_fraction=(0.001, 0.95))
BOUNDS_MODULE_DEFAULT = ExpFloorBounds()
B1_K = [13.5, 15.0, 16.5, 18.0, 21.0]
WEIGHTS = dict(barrier=1.0, derivative=0.5, rate=0.0, growth=4.0, slope=4.0, parameter=0.01)
STARTS = [[0.479, 2.0, 0.42, 0.911, 0.79], [1.0, 2.0, 0.4, 1.0, 0.90],
          [1.8, 2.1, 0.42, 2.9, 0.78], [0.6, 3.0, 0.6, 1.2, 0.70],
          [0.45, 2.08, 0.21, 1.58, 0.87]]


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_target_arrays(extra, radius_law, target_rates, target_slopes, D):
    controls_ref = controls_for(extra, R0)
    sig, tr, tg, ts, Kg = [], [], [], [], []
    ks = sorted(target_slopes)
    for K in DESIGN_K:
        r = float(radius_law(K))
        sig.append(K * 1.0e6 / math.sqrt(2.0 * math.pi * r))
        g = dense_target(K, target_rates) / D
        m = float(np.interp(math.log(K), np.log(ks), [target_slopes[k] for k in ks]))
        tr.append(controls_ref.frequency_Hz * g / (controls_ref.event_length_m * waveform_factor_C(m, R_LOAD)))
        tg.append(g); ts.append(m); Kg.append(K)
    sig = np.array(sig); tr = np.array(tr)
    tb = invert_cooperative_rate_to_barrier(
        tr, hits=extra["physics__cleavage_hits"], tau_s=extra["physics__cleavage_correlation_time_s"],
        attempt_frequency_s=NU0_S, temperature_K=300.0)
    return sig, tb, tr, np.array(tg), np.array(ts), np.array(Kg)


def fit(manifest, extra, radius_law, target_rates, target_slopes, D, bounds):
    sig, tb, tr, tg, ts, Kg = build_target_arrays(extra, radius_law, target_rates, target_slopes, D)
    cf = controls_for(extra, float(radius_law(18.0)))
    best = None
    for st in STARTS:
        st = list(np.clip(st, bounds.lower(), bounds.upper()))
        try:
            res = project_exp_floor(sigma_grid_Pa=sig, target_barrier_eV=tb, target_rate_s=tr,
                                    K_grid_MPa_sqrt_m=Kg, target_growth=tg, target_slope=ts,
                                    R=R_LOAD, template=manifest.cleavage, controls=cf,
                                    bounds=bounds, weights=WEIGHTS, start=st)
        except Exception:  # noqa: BLE001
            continue
        if best is None or res["cost"] < best["cost"]:
            best = res
    if best is None:
        raise SystemExit("no successful projection")
    return best


def admissibility(vector, bounds, manifest, extra, rows) -> dict:
    lo, hi = bounds.lower(), bounds.upper()
    v = np.asarray(vector, dtype=float)
    span = np.maximum(hi - lo, 1e-30)
    rel_margin = np.minimum(v - lo, hi - v) / span
    at_bound = {PARAM_NAMES[i]: float(rel_margin[i]) for i in range(5) if rel_margin[i] < 1e-4}
    G0, ff = float(v[0]), float(v[4])
    floor = ff * G0
    barrier = barrier_from_vector(v, manifest.cleavage)
    sig_probe = np.linspace(0.5e9, 30.0e9, 400)
    G = barrier.values_eV(sig_probe, 300.0)
    monotone = bool(np.all(np.diff(G) <= 1e-15))
    win = [r for r in rows if 13.5 - 1e-9 <= r["Kmax_MPa_sqrt_m"] <= 21.0 + 1e-9]
    rate_monotone = bool(all(b["predicted_physical_da_dN"] > a["predicted_physical_da_dN"]
                             for a, b in zip(win, win[1:])))
    max_sat = max(r["cooperative_saturation_fraction"] for r in win)
    checks = dict(
        all_parameters_interior=(len(at_bound) == 0), parameters_at_bound=at_bound,
        relative_bound_margins={PARAM_NAMES[i]: float(rel_margin[i]) for i in range(5)},
        G0_finite_positive=bool(np.isfinite(G0) and G0 > 0),
        sigc_finite_positive=bool(np.isfinite(v[1]) and v[1] > 0),
        floor_le_0p95_G0=bool(floor <= 0.95 * G0 + 1e-15),
        floor_ge_production_minimum=bool(floor >= manifest.cleavage.floor_min_eV),
        floor_eV=floor, G0_minus_floor_eV=G0 - floor,
        barrier_monotone_nonincreasing_in_stress=monotone,
        predicted_rate_monotone_increasing=rate_monotone,
        max_cooperative_saturation_fraction=max_sat,
        cooperative_saturation_below_10pct_of_window=bool(max_sat < 0.10),
        all_rates_finite_positive=bool(all(np.isfinite(r["predicted_physical_da_dN"])
                                           and r["predicted_physical_da_dN"] > 0 for r in rows)),
    )
    checks["admissible"] = bool(
        checks["all_parameters_interior"] and checks["G0_finite_positive"]
        and checks["sigc_finite_positive"] and checks["floor_le_0p95_G0"]
        and checks["floor_ge_production_minimum"] and checks["barrier_monotone_nonincreasing_in_stress"]
        and checks["predicted_rate_monotone_increasing"] and checks["all_rates_finite_positive"])
    return checks


def b1_radius_law(manifest, extra, vector, label):
    """Candidate-specific raw B1 radius, PCHIP over B1_K."""
    barrier = barrier_from_vector(vector, manifest.cleavage)
    controls = B1Controls(n_phase=80, n_bins=80, r0_m=R0,
                          blunting_length_m=extra["physics__blunting_length_m"],
                          cleavage_hits=extra["physics__cleavage_hits"],
                          cleavage_tau_s=extra["physics__cleavage_correlation_time_s"],
                          hazard_seed=1720, burn_events=30, sample_events=80)
    row = dict(rho_source0_m2=extra["rho_source0_m2"])
    pts = []
    for K in B1_K:
        t0 = time.time()
        res = b1_event_conditioned_emission(manifest, row, Kmax_MPa_sqrt_m=K, R=R_LOAD,
                                            frequency_Hz=1000.0, temperature_K=300.0,
                                            controls=controls, cleavage_barrier=barrier)
        pts.append(dict(Kmax_MPa_sqrt_m=K, r_eff_m=res["r_eff_m"], delta_r_m=res["r_eff_m"] - R0,
                        converged=res["fixed_point_converged"], b1_da_dN=res["da_dN"],
                        wall_seconds=time.time() - t0))
        print(f"    [{label}] B1 K={K}: r_eff={res['r_eff_m']:.6e} conv={res['fixed_point_converged']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    ks = np.array([p["Kmax_MPa_sqrt_m"] for p in pts])
    raw = PchipInterpolator(ks, np.array([p["r_eff_m"] for p in pts]), extrapolate=True)
    scaled = PchipInterpolator(ks, R0 + ALPHA_R_FROZEN * np.array([p["delta_r_m"] for p in pts]),
                               extrapolate=True)
    return raw, scaled, pts


def main() -> None:
    manifest, extra = load_a_native_manifest()
    target_rates, target_slopes = load_targets()
    obs_law = observed_radius_law()
    deficit = measure_reduced_deficit(manifest, extra, obs_law)
    D = deficit[18.0]["deficit"]

    rejections = []

    # --- rejected: module-default bound parks floor_fraction --------------
    rej = fit(manifest, extra, obs_law, target_rates, target_slopes, D, BOUNDS_MODULE_DEFAULT)
    rej_v = [float(x) for x in rej["vector"]]
    _, rej_rows = evaluate(manifest.cleavage, rej_v, extra, obs_law, target_rates, target_slopes, D, FULL_GRID)
    rej_checks = admissibility(rej_v, BOUNDS_MODULE_DEFAULT, manifest, extra, rej_rows)
    rejections.append(dict(
        candidate_id="P40_SINGLE_EXP_MODULE_DEFAULT_BOUND_V0",
        bounds_source="inverse_fatigue_barrier_design_v10230.ExpFloorBounds default floor_fraction<=0.80",
        vector=dict(zip(PARAM_NAMES, rej_v)), cost=rej["cost"],
        log_growth_RMS=rej["log_growth_RMS"], slope_RMS=rej["slope_RMS"],
        rejection_reason=("floor_fraction parked exactly on the 0.80 module-default upper bound; "
                          "campaign rule forbids a parameter at an optimization bound without "
                          "separate justification. The 0.80 default has no production basis: the "
                          "production limit (MaterialManifest.floor_max_fraction and the campaign "
                          "admissibility rule floor<=0.95*G0) is 0.95."),
        admissibility=rej_checks))

    # --- accepted family: production bound --------------------------------
    seed = fit(manifest, extra, obs_law, target_rates, target_slopes, D, BOUNDS_PRODUCTION)
    seed_v = [float(x) for x in seed["vector"]]
    print(f"seed candidate (OBSERVED envelope): {dict(zip(PARAM_NAMES, np.round(seed_v,6)))}", flush=True)

    print("  running candidate-specific B1 (sensitivity model, not qualified)...", flush=True)
    raw_law, scaled_law, b1_pts = b1_radius_law(manifest, extra, seed_v, "seed")

    scenarios = {"OBSERVED_A_NATIVE_ENVELOPE": obs_law, "RAW_B1": raw_law, "RADIUS_SCALED_B1": scaled_law}

    candidates = {}
    for cand_id, fit_scenario in (("P40_RAW_B1_ENVELOPE_V1", "RAW_B1"),
                                  ("P40_RADIUS_SCALED_ENVELOPE_V1", "RADIUS_SCALED_B1")):
        law = scenarios[fit_scenario]
        D_scn = measure_reduced_deficit(manifest, extra, law)[18.0]["deficit"]
        res = fit(manifest, extra, law, target_rates, target_slopes, D_scn, BOUNDS_PRODUCTION)
        v = [float(x) for x in res["vector"]]
        per_scenario, scores = {}, {}
        for sname, slaw in scenarios.items():
            D_eval = measure_reduced_deficit(manifest, extra, slaw)[18.0]["deficit"]
            _, rows = evaluate(manifest.cleavage, v, extra, slaw, target_rates, target_slopes, D_eval, FULL_GRID)
            per_scenario[sname] = rows
            scores[sname] = score(rows)
        adm = admissibility(v, BOUNDS_PRODUCTION, manifest, extra, per_scenario[fit_scenario])
        worst = max(scores.values(), key=lambda s: s["J"])
        candidates[cand_id] = dict(
            candidate_id=cand_id, fitted_under_scenario=fit_scenario,
            D_used=D_scn, vector=dict(zip(PARAM_NAMES, v)),
            row_sha256=sha256_obj(dict(zip(PARAM_NAMES, v))),
            fit_diagnostics={k: res[k] for k in ("cost", "success", "nfev", "barrier_RMS_eV",
                                                 "derivative_RMS_eV", "log_growth_RMS",
                                                 "slope_RMS", "minimum_saturation_margin")},
            admissibility=adm, scores_by_scenario=scores,
            minimax_J=worst["J"], minimax_scenario=max(scores, key=lambda s: scores[s]["J"]),
            eligible=bool(adm["admissible"] and all(s["E_rate"] <= 1.0 and s["E_slope"] <= 1.0
                                                    and s["E_max"] <= 1.5 for s in scores.values())),
            predictions_by_scenario=per_scenario)
        print(f"  {cand_id}: minimax J={worst['J']:.4f} eligible={candidates[cand_id]['eligible']} "
              f"admissible={adm['admissible']}", flush=True)

    eligible = {k: v for k, v in candidates.items() if v["eligible"]}
    selected = min(eligible, key=lambda k: eligible[k]["minimax_J"]) if eligible else None

    payload = dict(
        schema="v10.2.30_prospective_paris_p40_stage2_v1",
        prediction_status="NOT_A_QUALIFIED_PHYSICAL_PREDICTION",
        transfer_model_status="B1_TRANSFER_NOT_QUALIFIED_FOR_CANDIDATE_SELECTION",
        bound_provenance=dict(
            module_default_floor_fraction_max=0.80,
            production_limit_floor_fraction_max=0.95,
            production_limit_sources=["MaterialManifest.ExpFloorBarrier.floor_max_fraction=0.95",
                                      "campaign admissibility rule: floor <= 0.95 * G0"],
            decision=("use the production limit 0.95; the 0.80 module default parks the optimum "
                      "exactly on the bound and is recorded as a rejection")),
        dual_exponential_family_required=False,
        dual_family_justification=("the single EXP-floor family satisfies every analytical gate with "
                                   "all five parameters strictly interior under the production bound, "
                                   "so no new production barrier class is added"),
        reduced_model_deficit={str(k): v for k, v in deficit.items()},
        alpha_r_frozen=ALPHA_R_FROZEN,
        b1_radius_points_seed_candidate=b1_pts,
        radius_scenarios=list(scenarios),
        rejection_registry=rejections,
        candidates={k: {kk: vv for kk, vv in v.items() if kk != "predictions_by_scenario"}
                    for k, v in candidates.items()},
        selected_candidate=selected,
        selection_rule="minimax J across the three frozen radius scenarios, eligible candidates only",
    )
    (OUT / "p40_stage2_selection.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")

    rows_out = []
    for cid, c in candidates.items():
        for sname, rows in c["predictions_by_scenario"].items():
            for r in rows:
                rows_out.append(dict(candidate_id=cid, scenario=sname, **r))
    with (OUT / "p40_stage2_predictions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows_out)

    print("\n" + json.dumps(dict(
        selected=selected,
        rejected=[r["candidate_id"] for r in rejections],
        candidates={k: dict(minimax_J=v["minimax_J"], eligible=v["eligible"],
                            admissible=v["admissibility"]["admissible"],
                            vector=v["vector"]) for k, v in candidates.items()}),
        indent=2, default=str))


if __name__ == "__main__":
    main()
