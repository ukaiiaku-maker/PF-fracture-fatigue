"""Fresh prospective P40 candidate derivation under frozen radius scenarios.

B1 is NOT transfer-qualified (see b1_exploratory_sweep_decision.json and
b1_rate_semantics_audit_K18.json). It is retained here only as a sensitivity
model. The directly observed A_NATIVE terminal radii are used as a
provisional stress-transfer ENVELOPE, never as an exact candidate-independent
radius law.

Radius scenarios (all frozen before any candidate fit):
  OBSERVED_A_NATIVE_ENVELOPE : monotone (PCHIP) interpolation of the observed
                               A_NATIVE terminal tip_radius_m at K=12,15,18,24.
                               Candidate-INDEPENDENT -- a provisional envelope.
  RAW_B1                     : candidate-specific raw B1 r_eff(K).
  RADIUS_SCALED_B1           : candidate-specific B1 blunting increment scaled
                               by the frozen K=18 same-seed alpha_r = 5.638634665572309.

Reduced-model normalization
---------------------------
The A0/B1 reduced forward model under-predicts the qualified A_NATIVE rate by
a nearly K-independent factor (log10 deficit 0.978 / 0.922 / 0.957 decade at
K=15/18/24). Designing directly against the physical g_target inside the
reduced model would therefore produce a barrier that is far too weak. Instead
the reduced-model design target is

    g_reduced_target(K) = g_target(K) / D

with D a single frozen scalar deficit measured on A_NATIVE at K=18 under the
same radius scenario used for the fit. D is measured, not fitted to any
candidate, and its K-variation (<=0.06 decade over 15-24) is recorded as a
design uncertainty. This is the "provisional transfer envelope" use of the
observed A_NATIVE data.

Outputs are predictions under an unqualified transfer model. They are NOT
qualified physical predictions and are labelled as such.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from a_native_manifest import load_a_native_manifest
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    RenewalControls, ExpFloorBounds, cycle_growth_and_slope, project_exp_floor,
    invert_cooperative_rate_to_barrier, waveform_factor_C, barrier_from_vector,
)

OUT = Path(__file__).resolve().parents[1]
R0 = 1.0e-6
ALPHA_R_FROZEN = 5.638634665572309   # same-seed, K=18, radius-only (diagnostic-derived)
NU0_S = 1.0e12
R_LOAD = 0.1

OBSERVED_RADIUS = {12.0: 1.0105318630661543e-06, 15.0: 1.041289045198725e-06,
                   18.0: 1.1004741824397272e-06, 24.0: 1.343158213754612e-06}
OBSERVED_RATE = {12.0: 2.448159884833225e-08, 15.0: 3.003481684450034e-07,
                 18.0: 4.473410023231299e-07, 24.0: 5.580641987406302e-07}

DESIGN_K = [13.5, 15.0, 16.5, 18.0, 19.5, 21.0]   # scoring window 13.5-21
PILOT_K = [13.5, 18.0, 21.0]
FULL_GRID = [12.0, 12.75, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 24.3]


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def observed_radius_law():
    ks = sorted(OBSERVED_RADIUS)
    return PchipInterpolator(np.array(ks), np.array([OBSERVED_RADIUS[k] for k in ks]), extrapolate=True)


def controls_for(extra, radius_m):
    return RenewalControls(hits=extra["physics__cleavage_hits"],
                           tau_s=extra["physics__cleavage_correlation_time_s"],
                           event_length_m=5.0e-6, frequency_Hz=1000.0,
                           temperature_K=300.0, n_phase=4096, radius_m=radius_m)


def load_targets():
    """P40 target rates and local slopes on the design grid (STANDARD_WINDOW)."""
    rates, slopes = {}, {}
    with (OUT / "target_rate_profiles.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["profile_id"] == "P40_STANDARD_WINDOW":
                rates[float(r["Kmax_MPa_sqrt_m"])] = float(r["g_target_m_per_cycle"])
    with (OUT / "target_slope_profiles.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["profile_id"] == "P40_STANDARD_WINDOW":
                slopes[float(r["Kmax_MPa_sqrt_m"])] = float(r["m_target"])
    return rates, slopes


def dense_target(K, rates):
    """Log-linear interpolation of the frozen target onto arbitrary K."""
    ks = np.array(sorted(rates))
    vs = np.log(np.array([rates[k] for k in ks]))
    return float(np.exp(np.interp(math.log(K), np.log(ks), vs)))


def measure_reduced_deficit(manifest, extra, radius_law) -> dict:
    """Measured A_NATIVE reduced-model deficit D(K) = g_physical / g_A0."""
    out = {}
    for K, g_phys in OBSERVED_RATE.items():
        r = float(radius_law(K))
        g_a0 = cycle_growth_and_slope(manifest.cleavage, K, R_LOAD, controls_for(extra, r))["da_dN"]
        out[K] = dict(radius_m=r, g_a0=g_a0, g_phys=g_phys,
                      deficit=g_phys / g_a0, log10_deficit=math.log10(g_phys / g_a0))
    return out


def fit_candidate(manifest, extra, radius_law, target_rates, target_slopes, D, label):
    """Project the P40 reduced-model target onto the production EXP-floor family."""
    controls_ref = controls_for(extra, R0)
    sigma_grid, tgt_barrier, tgt_rate, tgt_growth, tgt_slope, Kg = [], [], [], [], [], []
    for K in DESIGN_K:
        r = float(radius_law(K))
        sigma = K * 1.0e6 / math.sqrt(2.0 * math.pi * r)
        g_red = dense_target(K, target_rates) / D
        m_loc = float(np.interp(math.log(K), np.log(sorted(target_slopes)),
                                [target_slopes[k] for k in sorted(target_slopes)]))
        # locally-power-law instantaneous rate at the reference stress
        C = waveform_factor_C(max(m_loc, 1.0e-3), R_LOAD)
        rate_inst = controls_ref.frequency_Hz * g_red / (controls_ref.event_length_m * C)
        sigma_grid.append(sigma); tgt_rate.append(rate_inst)
        tgt_growth.append(g_red); tgt_slope.append(m_loc); Kg.append(K)
    sigma_grid = np.array(sigma_grid); tgt_rate = np.array(tgt_rate)
    tgt_barrier = invert_cooperative_rate_to_barrier(
        tgt_rate, hits=extra["physics__cleavage_hits"],
        tau_s=extra["physics__cleavage_correlation_time_s"],
        attempt_frequency_s=NU0_S, temperature_K=300.0)

    # radius law enters the forward evaluation through per-K controls;
    # project_exp_floor uses a single controls object, so evaluate with the
    # design-centre radius and re-score per-K afterwards.
    controls_fit = controls_for(extra, float(radius_law(18.0)))
    best = None
    starts = [
        [manifest.cleavage.G00_eV, manifest.cleavage.sigc0_Pa / 1e9, manifest.cleavage.alpha,
         manifest.cleavage.exponent, manifest.cleavage.floor_fraction],
        [1.2, 3.0, 0.4, 2.9, 0.12], [0.9, 2.5, 0.5, 2.0, 0.05],
        [1.6, 2.0, 0.3, 3.5, 0.20], [0.6, 4.0, 0.8, 1.5, 0.02],
    ]
    for start in starts:
        try:
            res = project_exp_floor(
                sigma_grid_Pa=sigma_grid, target_barrier_eV=tgt_barrier, target_rate_s=tgt_rate,
                K_grid_MPa_sqrt_m=np.array(Kg), target_growth=np.array(tgt_growth),
                target_slope=np.array(tgt_slope), R=R_LOAD, template=manifest.cleavage,
                controls=controls_fit, bounds=ExpFloorBounds(),
                weights=dict(barrier=1.0, derivative=0.5, rate=0.0, growth=4.0, slope=4.0, parameter=0.01),
                start=start)
        except Exception as exc:  # noqa: BLE001
            continue
        if best is None or res["cost"] < best["cost"]:
            best = res
    if best is None:
        raise SystemExit(f"{label}: no successful EXP-floor projection")
    best["target_barrier_eV"] = tgt_barrier.tolist()
    best["sigma_grid_Pa"] = sigma_grid.tolist()
    best["design_K"] = Kg
    best["target_growth_reduced"] = tgt_growth
    best["target_slope"] = tgt_slope
    best["D_used"] = D
    return best


def evaluate(manifest_cleavage_template, vector, extra, radius_law, target_rates, target_slopes, D, K_list):
    """Forward-evaluate a candidate vector under a given radius scenario."""
    barrier = barrier_from_vector(vector, manifest_cleavage_template)
    rows = []
    for K in K_list:
        r = float(radius_law(K))
        res = cycle_growth_and_slope(barrier, K, R_LOAD, controls_for(extra, r))
        g_pred_phys = res["da_dN"] * D      # map reduced-model rate back to physical units
        g_tgt = dense_target(K, target_rates)
        rows.append(dict(Kmax_MPa_sqrt_m=K, radius_m=r,
                         reduced_da_dN=res["da_dN"], predicted_physical_da_dN=g_pred_phys,
                         target_da_dN=g_tgt,
                         log10_rate_error=math.log10(g_pred_phys / g_tgt) if g_pred_phys > 0 else float("nan"),
                         local_slope=res["local_slope"],
                         target_slope=float(np.interp(math.log(K), np.log(sorted(target_slopes)),
                                                      [target_slopes[k] for k in sorted(target_slopes)])),
                         cooperative_saturation_fraction=res["cooperative_saturation_fraction"]))
    for row in rows:
        row["slope_error"] = row["local_slope"] - row["target_slope"]
    return barrier, rows


def score(rows) -> dict:
    win = [r for r in rows if 13.5 - 1e-9 <= r["Kmax_MPa_sqrt_m"] <= 21.0 + 1e-9]
    lr = np.array([r["log10_rate_error"] for r in win])
    se = np.array([r["slope_error"] for r in win])
    E_rate = float(np.sqrt(np.mean(lr ** 2)) / 0.05)
    E_slope = float(np.sqrt(np.mean(se ** 2)) / 0.25)
    E_max = float(np.max(np.abs(se)) / 0.50)
    return dict(E_rate=E_rate, E_slope=E_slope, E_max=E_max,
                J=E_rate + E_slope + 0.5 * E_max,
                rms_log10_rate_error=float(np.sqrt(np.mean(lr ** 2))),
                rms_slope_error=float(np.sqrt(np.mean(se ** 2))),
                max_abs_slope_error=float(np.max(np.abs(se))))


def main() -> None:
    manifest, extra = load_a_native_manifest()
    target_rates, target_slopes = load_targets()
    obs_law = observed_radius_law()

    deficit = measure_reduced_deficit(manifest, extra, obs_law)
    logs = [v["log10_deficit"] for k, v in deficit.items() if k >= 15.0]
    D = deficit[18.0]["deficit"]
    deficit_summary = dict(
        D_frozen_scalar=D, D_source="A_NATIVE K=18 under OBSERVED_A_NATIVE_ENVELOPE radius",
        log10_deficit_by_K={k: v["log10_deficit"] for k, v in deficit.items()},
        log10_deficit_spread_K15_24=float(max(logs) - min(logs)),
        note=("measured on A_NATIVE only; never fitted to a candidate; its K-variation is a "
              "recorded design uncertainty, not a tuned parameter"))

    fit = fit_candidate(manifest, extra, obs_law, target_rates, target_slopes, D, "P40_OBSERVED_ENVELOPE")
    vector = [float(x) for x in fit["vector"]]

    barrier, rows_obs = evaluate(manifest.cleavage, vector, extra, obs_law,
                                 target_rates, target_slopes, D, FULL_GRID)
    sc = score(rows_obs)

    payload = dict(
        schema="v10.2.30_prospective_paris_p40_derivation_stage1_v1",
        status="STAGE1_OBSERVED_ENVELOPE_ONLY_B1_SCENARIOS_PENDING",
        transfer_model_status="B1_TRANSFER_NOT_QUALIFIED_FOR_CANDIDATE_SELECTION",
        prediction_status="NOT_A_QUALIFIED_PHYSICAL_PREDICTION",
        reduced_model_deficit=deficit_summary,
        radius_scenario="OBSERVED_A_NATIVE_ENVELOPE",
        alpha_r_frozen_for_scaled_scenario=ALPHA_R_FROZEN,
        candidate_vector=dict(cleave_G00_eV=vector[0], cleave_sigc0_GPa=vector[1],
                              cleave_exp_a=vector[2], cleave_exp_n=vector[3],
                              cleave_floor_frac=vector[4]),
        a_native_vector=dict(cleave_G00_eV=manifest.cleavage.G00_eV,
                             cleave_sigc0_GPa=manifest.cleavage.sigc0_Pa / 1e9,
                             cleave_exp_a=manifest.cleavage.alpha,
                             cleave_exp_n=manifest.cleavage.exponent,
                             cleave_floor_frac=manifest.cleavage.floor_fraction),
        fit_diagnostics={k: fit[k] for k in ("cost", "success", "nfev", "barrier_RMS_eV",
                                             "derivative_RMS_eV", "log_growth_RMS", "slope_RMS",
                                             "minimum_saturation_margin")},
        design_K=DESIGN_K, pilot_K=PILOT_K, full_grid=FULL_GRID,
        score_13p5_to_21=sc,
        predictions=rows_obs,
    )
    (OUT / "p40_derivation_stage1.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")

    with (OUT / "p40_stage1_predictions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_obs[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows_obs)

    print(json.dumps(dict(D_frozen=D,
                          log10_deficit_by_K=deficit_summary["log10_deficit_by_K"],
                          deficit_spread=deficit_summary["log10_deficit_spread_K15_24"],
                          candidate=payload["candidate_vector"],
                          a_native=payload["a_native_vector"],
                          fit=payload["fit_diagnostics"],
                          score=sc), indent=2, default=str))
    print("\nper-K predictions (13.5-21 scoring window):")
    for r in rows_obs:
        if 13.5 <= r["Kmax_MPa_sqrt_m"] <= 21.0:
            print(f"  K={r['Kmax_MPa_sqrt_m']:5.2f} r={r['radius_m']:.4e} "
                  f"g_pred={r['predicted_physical_da_dN']:.4e} g_tgt={r['target_da_dN']:.4e} "
                  f"dlog10={r['log10_rate_error']:+.4f} m={r['local_slope']:.3f} "
                  f"m_tgt={r['target_slope']:.3f} dm={r['slope_error']:+.3f}")


if __name__ == "__main__":
    main()
