"""Consume b1_raw_a_native_qualification_records.json and apply the amended
protocol's decision rule:

  1. Check sample-doubling convergence (80 vs 160 sample_events) at each K:
     |da_dN_160 - da_dN_80| / da_dN_80 and same for r_eff_m.
  2. B1_RAW_A_NATIVE_QUALIFICATION: compare raw B1 (seed-bank mean, converged
     sample level) r_eff_m and da_dN against physical values at all 4 K.
  3. If raw B1 already reproduces physical radius trend and rate within a
     generous band, classify B1_TRANSFER_QUALIFIED_WITHOUT_CALIBRATION.
  4. Otherwise, fit exactly one scalar alpha_r at K=18 ONLY using the radius
     observable (never the K=18 rate, since that already defines g_star):
       alpha_r = (r_phys(18) - r0) / (r_B1(18) - r0)
     Apply r_TRANSFER(K) = r0 + alpha_r*(r_B1(K)-r0) to the K=12,15,24
     holdouts and re-evaluate both radius agreement AND rate agreement
     (by feeding r_TRANSFER(K) into the A0 growth formula) at those holdouts.
  5. If holdout rate agreement is within tolerance, classify
     B1_TRANSFER_QUALIFIED_WITH_ONE_RADIUS_INCREMENT_SCALE.
     Otherwise classify B1_TRANSFER_NOT_QUALIFIED and record why (this is
     expected to fail if the true rate gap is not radius-driven).

No candidate scoring or physical launch happens here -- this only decides
which TRANSFER_AWARE construction path Section 5/6 may use.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from a_native_manifest import load_a_native_manifest
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import RenewalControls, cycle_growth_and_slope

OUT = Path(__file__).resolve().parents[1]
R0 = 1.0e-6
RATE_TOL_DECADE = 0.30   # generous: within a factor of ~2 (log10(2)=0.301)
RADIUS_TOL_REL = 0.30


def growth_at_radius(manifest, extra, K, radius_m):
    controls = RenewalControls(hits=extra["physics__cleavage_hits"],
                                tau_s=extra["physics__cleavage_correlation_time_s"],
                                event_length_m=5.0e-6, frequency_Hz=1000.0,
                                temperature_K=300.0, n_phase=4096, radius_m=radius_m)
    return cycle_growth_and_slope(manifest.cleavage, K, 0.1, controls)["da_dN"]


def main() -> None:
    data = json.loads((OUT / "b1_raw_a_native_qualification_records.json").read_text())
    records = data["records"]
    manifest, extra = load_a_native_manifest()

    by_k_sample = defaultdict(list)
    for rec in records:
        by_k_sample[(rec["Kmax_MPa_sqrt_m"], rec["sample_events_requested"])].append(rec)

    convergence = []
    per_k_converged = {}
    for K in sorted({r["Kmax_MPa_sqrt_m"] for r in records}):
        r80 = by_k_sample[(K, 80)]
        r160 = by_k_sample[(K, 160)]
        mean_r_80 = float(np.mean([r["r_eff_m"] for r in r80]))
        mean_r_160 = float(np.mean([r["r_eff_m"] for r in r160]))
        mean_g_80 = float(np.mean([r["da_dN"] for r in r80]))
        mean_g_160 = float(np.mean([r["da_dN"] for r in r160]))
        rel_r = abs(mean_r_160 - mean_r_80) / mean_r_80 if mean_r_80 else float("nan")
        rel_g = abs(mean_g_160 - mean_g_80) / mean_g_80 if mean_g_80 else float("nan")
        converged = rel_r < 0.15 and rel_g < 0.30
        per_k_converged[K] = dict(mean_r_eff_m=mean_r_160, mean_da_dN=mean_g_160,
                                   rel_change_r=rel_r, rel_change_g=rel_g, converged=converged,
                                   std_r_eff_m=float(np.std([r["r_eff_m"] for r in r160])),
                                   std_da_dN=float(np.std([r["da_dN"] for r in r160])))
        convergence.append(dict(Kmax_MPa_sqrt_m=K, **per_k_converged[K]))

    physical = {r["Kmax_MPa_sqrt_m"]: (r["physical_tip_radius_m"], r["physical_da_dN"]) for r in records}

    # Step: raw B1 qualification (no calibration)
    raw_eval = {}
    for K, c in per_k_converged.items():
        r_phys, g_phys = physical[K]
        raw_eval[K] = dict(
            r_eff_B1=c["mean_r_eff_m"], r_phys=r_phys,
            radius_rel_err=abs(c["mean_r_eff_m"] - r_phys) / r_phys,
            g_B1=c["mean_da_dN"], g_phys=g_phys,
            rate_log10_err=abs(np.log10(c["mean_da_dN"]) - np.log10(g_phys)),
        )
    raw_qualified = all(v["radius_rel_err"] < RADIUS_TOL_REL and v["rate_log10_err"] < RATE_TOL_DECADE
                        for v in raw_eval.values())

    classification = None
    calibrated_eval = None
    alpha_r = None
    if raw_qualified:
        classification = "B1_TRANSFER_QUALIFIED_WITHOUT_CALIBRATION"
    else:
        # calibrate alpha_r at K=18 using the RADIUS only
        r_b1_18 = per_k_converged[18.0]["mean_r_eff_m"]
        r_phys_18 = physical[18.0][0]
        alpha_r = (r_phys_18 - R0) / (r_b1_18 - R0) if (r_b1_18 - R0) != 0 else float("nan")

        calibrated_eval = {}
        for K in [12.0, 15.0, 24.0]:
            r_b1_k = per_k_converged[K]["mean_r_eff_m"]
            r_transfer = R0 + alpha_r * (r_b1_k - R0)
            r_phys_k, g_phys_k = physical[K]
            g_transfer = growth_at_radius(manifest, extra, K, r_transfer)
            calibrated_eval[K] = dict(
                r_B1=r_b1_k, r_transfer=r_transfer, r_phys=r_phys_k,
                radius_rel_err=abs(r_transfer - r_phys_k) / r_phys_k,
                g_transfer=g_transfer, g_phys=g_phys_k,
                rate_log10_err=abs(np.log10(g_transfer) - np.log10(g_phys_k)),
            )
        holdout_qualified = all(v["rate_log10_err"] < RATE_TOL_DECADE for v in calibrated_eval.values())
        classification = ("B1_TRANSFER_QUALIFIED_WITH_ONE_RADIUS_INCREMENT_SCALE"
                          if holdout_qualified else "B1_TRANSFER_NOT_QUALIFIED")

    decision = dict(
        schema="v10.2.30_prospective_paris_b1_qualification_decision_v1",
        convergence=convergence,
        raw_evaluation_all_4_points=raw_eval,
        raw_qualified_without_calibration=raw_qualified,
        alpha_r_calibrated_at_K18_via_radius_only=alpha_r,
        holdout_evaluation_K_12_15_24=calibrated_eval,
        classification=classification,
        rate_tolerance_log10_decade=RATE_TOL_DECADE,
        radius_tolerance_relative=RADIUS_TOL_REL,
        note=("K=18 rate is never used to calibrate alpha_r (it defines g_star; "
              "using it would be tautological). alpha_r is fit on the K=18 RADIUS "
              "only, per the amended protocol; K=12,15,24 are genuine holdouts."),
    )
    (OUT / "b1_qualification_decision.json").write_text(json.dumps(decision, indent=2, default=str) + "\n")
    print(json.dumps({k: decision[k] for k in
                      ("raw_qualified_without_calibration", "alpha_r_calibrated_at_K18_via_radius_only",
                       "classification")}, indent=2, default=str))
    print(json.dumps(raw_eval, indent=2, default=str))
    if calibrated_eval:
        print(json.dumps(calibrated_eval, indent=2, default=str))


if __name__ == "__main__":
    main()
