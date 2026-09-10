"""Postmortem for the completed 24-record B1 reduced-model exploratory sweep.

This supersedes decide_b1_qualification.py, which is NOT run and whose
classification is NOT accepted (its gates were defective: convergence not
enforced, holdout gated on rate only, total-radius relative error dominated
by r0, ensemble-vs-single-seed mixing, hardcoded physical literals).

Scientific classifications emitted here:
  B1_EXPLORATORY_SWEEP_V1_COMPLETE
  B1_NUMERICALLY_UNRESOLVED_AT_K12
  B1_CONVERGED_BUT_RATE_TRANSFER_FAILED_AT_K15_K18_K24
  B1_TRANSFER_NOT_QUALIFIED_FOR_CANDIDATE_SELECTION
  A0_RATE_EQUIVALENT_RADIUS_NONPHYSICAL_OR_UNDEFINED

Primary transfer comparison is SAME-SEED (B1 seed 1720 vs physical seed
1720). Seeds 2024/3031 supply stochastic uncertainty only. Physical rows are
loaded at runtime from the source-qualified CSV with file- and row-hash
verification; hardcoded values act only as regression expectations.

No production physical trajectory is run here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from a_native_manifest import load_a_native_manifest
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    RenewalControls, cycle_growth_and_slope,
)

OUT = Path(__file__).resolve().parents[1]
RECORDS = OUT / "b1_raw_a_native_qualification_records.json"
PHYS_CSV = Path("/Volumes/Data/working-papers/fracture_and_fatigue/1A_PT03_PT08_R_developed_points.csv")
PHYS_CSV_SHA256 = "d3d7ec3b03816f23ce0d27cf1cc36a93609c4d1a3fe190e97fadbd6481ee7c94"

R0 = 1.0e-6
PRIMARY_SEED = 1720
CONVERGED_SAMPLE_LEVEL = 160
K_GRID = [12.0, 15.0, 18.0, 24.0]

# Regression expectations (NOT the source of truth; the CSV is).
EXPECTED_PHYSICAL = {
    12.0: dict(tip_radius_m=1.0105318630661543e-06, da_dN=2.448159884833225e-08),
    15.0: dict(tip_radius_m=1.041289045198725e-06, da_dN=3.003481684450034e-07),
    18.0: dict(tip_radius_m=1.1004741824397272e-06, da_dN=4.473410023231299e-07),
    24.0: dict(tip_radius_m=1.343158213754612e-06, da_dN=5.580641987406302e-07),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_physical_rows() -> tuple[dict, dict]:
    """Load and verify the qualified A_NATIVE physical rows at runtime."""
    actual_sha = sha256_file(PHYS_CSV)
    if actual_sha != PHYS_CSV_SHA256:
        raise SystemExit(f"physical source CSV hash changed: {actual_sha} != {PHYS_CSV_SHA256}")
    with PHYS_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    out, audit = {}, {}
    for K in K_GRID:
        matches = [r for r in rows if r["option"] == "A_NATIVE" and r["R"] == "0.1"
                   and float(r["Kmax_MPa_sqrt_m"]) == K and r["seed"] == str(PRIMARY_SEED)
                   and r["n_bins"] == "80" and float(r["temperature_K"]) == 300.0
                   and float(r["frequency_Hz"]) == 1000.0]
        if len(matches) != 1:
            raise SystemExit(f"expected exactly one qualified physical row at K={K}, found {len(matches)}")
        row = matches[0]
        if row["terminal_classification"] not in ("PHYSICAL_TARGET_REACHED", "REUSED_PHYSICAL_TARGET_REACHED"):
            raise SystemExit(f"K={K} physical row not terminal-qualified: {row['terminal_classification']}")
        if row["stable_growth"] != "True" or row["stationarity_classification"] != "STABLE":
            raise SystemExit(f"K={K} physical row not stable/stationary")
        rec = dict(da_dN=float(row["developed_da_dN"]), tip_radius_m=float(row["tip_radius_m"]),
                   event_count=int(row["event_count"]), cycles=float(row["cycles"]),
                   final_extension_um=float(row["final_extension_um"]),
                   stationarity_ratio=float(row["stationarity_ratio"]),
                   terminal_classification=row["terminal_classification"])
        # regression check against the hardcoded expectations
        exp = EXPECTED_PHYSICAL[K]
        for field, expected in exp.items():
            if abs(rec[field] - expected) / expected > 1e-12:
                raise SystemExit(f"K={K} {field} regression mismatch: {rec[field]} != {expected}")
        out[K] = rec
        audit[K] = dict(job_id=row["job_id"], row_sha256=sha256_obj(row),
                        branch=row["branch"], job_launch_head=row["head"],
                        analysis_head=row["analysis_head"],
                        production_solver_hash=row["production_solver_hash"],
                        common_physics_hash=row["common_physics_hash"],
                        composite_hash=row["composite_hash"],
                        terminal_classification=row["terminal_classification"],
                        stationarity_classification=row["stationarity_classification"])
    return out, audit


def adjacent_log_slope(K_lo, K_hi, g_lo, g_hi) -> float:
    if not (np.isfinite(g_lo) and np.isfinite(g_hi) and g_lo > 0 and g_hi > 0):
        return float("nan")
    return (math.log10(g_hi) - math.log10(g_lo)) / (math.log10(K_hi) - math.log10(K_lo))


def a0_root_diagnostic(manifest, extra, K, target) -> dict:
    """Determine whether a positive-radius A0 root exists for the physical rate.

    A NaN here is NOT a root-finder failure: as r -> 0 the A0 opening-only
    rate approaches a finite ceiling, and if the physical rate exceeds that
    ceiling no positive radius can reproduce it.
    """
    base = dict(hits=extra["physics__cleavage_hits"], tau_s=extra["physics__cleavage_correlation_time_s"],
                event_length_m=5.0e-6, frequency_Hz=1000.0, temperature_K=300.0, n_phase=4096)

    def g_at(r_m):
        return cycle_growth_and_slope(manifest.cleavage, K, 0.1,
                                      RenewalControls(**base, radius_m=r_m))["da_dN"]

    ceiling = g_at(1.0e-9)  # r -> 0 limit of the A0 opening-only rate
    g_at_r0 = g_at(R0)
    if target > ceiling:
        return dict(Kmax_MPa_sqrt_m=K, physical_da_dN=target, a0_ceiling_da_dN=ceiling,
                    a0_da_dN_at_r0=g_at_r0, root_exists=False, root_radius_m=float("nan"),
                    root_below_r0=None,
                    classification="A0_RATE_EQUIVALENT_RADIUS_UNDEFINED_PHYSICAL_RATE_EXCEEDS_A0_CEILING")
    from scipy.optimize import brentq
    try:
        root = brentq(lambda r: g_at(r) - target, 1.0e-9, 1.0e-4, xtol=1e-16, rtol=1e-12, maxiter=300)
    except ValueError:
        return dict(Kmax_MPa_sqrt_m=K, physical_da_dN=target, a0_ceiling_da_dN=ceiling,
                    a0_da_dN_at_r0=g_at_r0, root_exists=False, root_radius_m=float("nan"),
                    root_below_r0=None,
                    classification="A0_RATE_EQUIVALENT_RADIUS_UNDEFINED_NO_BRACKET")
    return dict(Kmax_MPa_sqrt_m=K, physical_da_dN=target, a0_ceiling_da_dN=ceiling,
                a0_da_dN_at_r0=g_at_r0, root_exists=True, root_radius_m=root,
                root_below_r0=bool(root < R0),
                classification=("A0_RATE_EQUIVALENT_RADIUS_NONPHYSICAL_BELOW_R0" if root < R0
                                else "A0_RATE_EQUIVALENT_RADIUS_PHYSICALLY_ADMISSIBLE"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    sweep = json.loads(RECORDS.read_text())
    records = sweep["records"]
    if len(records) != 24:
        raise SystemExit(f"expected 24 sweep records, found {len(records)}")
    manifest, extra = load_a_native_manifest()
    physical, phys_audit = load_physical_rows()

    def get(K, seed, n_sample):
        hits = [r for r in records if r["Kmax_MPa_sqrt_m"] == K and r["seed"] == seed
                and r["sample_events_requested"] == n_sample]
        if len(hits) != 1:
            raise SystemExit(f"non-unique sweep record for K={K} seed={seed} n={n_sample}")
        return hits[0]

    seeds = sweep["seed_bank"]

    # ---- 1. flat summary -------------------------------------------------
    summary = []
    for r in records:
        K = r["Kmax_MPa_sqrt_m"]
        finite = bool(np.isfinite(r["da_dN"]))
        summary.append(dict(
            Kmax_MPa_sqrt_m=K, seed=r["seed"], sample_events_requested=r["sample_events_requested"],
            fixed_point_converged=r["fixed_point_converged"], cycles=r["cycles"], events=r["events"],
            sample_events_achieved=r["sample_events"], da_dN=r["da_dN"], da_dN_finite=finite,
            r_eff_m=r["r_eff_m"], delta_r_m=r["r_eff_m"] - R0,
            mean_event_length_m=r["mean_event_length_m"],
            net_source_linked_slip=r["net_source_linked_slip"],
            emission_backstress_density_m2=r["emission_backstress_density_m2"],
            backstress_Pa=r["backstress_Pa"], wall_seconds=r["wall_seconds"],
            physical_da_dN=physical[K]["da_dN"], physical_tip_radius_m=physical[K]["tip_radius_m"],
            terminal_condition=("MAXIMUM_CYCLE_EXHAUSTION_BEFORE_SAMPLE_TARGET"
                                if not r["fixed_point_converged"] else "SAMPLE_TARGET_REACHED"),
        ))
    write_csv(OUT / "b1_exploratory_sweep_summary.csv", summary)

    # ---- 2. sample-doubling convergence (per K, per seed) ----------------
    convergence = []
    for K in K_GRID:
        for seed in seeds:
            a, b = get(K, seed, 80), get(K, seed, 160)
            both_conv = bool(a["fixed_point_converged"] and b["fixed_point_converged"])
            dr_a, dr_b = a["r_eff_m"] - R0, b["r_eff_m"] - R0
            rel_dr = abs(dr_b - dr_a) / abs(dr_a) if dr_a else float("nan")
            if np.isfinite(a["da_dN"]) and np.isfinite(b["da_dN"]) and a["da_dN"] > 0:
                rel_g = abs(b["da_dN"] - a["da_dN"]) / a["da_dN"]
                dlog10_g = abs(math.log10(b["da_dN"]) - math.log10(a["da_dN"]))
            else:
                rel_g = dlog10_g = float("nan")
            convergence.append(dict(
                Kmax_MPa_sqrt_m=K, seed=seed,
                converged_80=a["fixed_point_converged"], converged_160=b["fixed_point_converged"],
                both_levels_converged=both_conv,
                delta_r_80_m=dr_a, delta_r_160_m=dr_b, rel_change_delta_r=rel_dr,
                da_dN_80=a["da_dN"], da_dN_160=b["da_dN"], rel_change_da_dN=rel_g,
                dlog10_da_dN=dlog10_g,
                sample_doubling_usable=bool(both_conv and np.isfinite(rel_g)),
            ))
    write_csv(OUT / "b1_sample_doubling_convergence.csv", convergence)

    # ---- 3. same-seed transfer (PRIMARY) ---------------------------------
    same_seed = []
    for K in K_GRID:
        r = get(K, PRIMARY_SEED, CONVERGED_SAMPLE_LEVEL)
        phys = physical[K]
        dr_b1 = r["r_eff_m"] - R0
        dr_ph = phys["tip_radius_m"] - R0
        finite = bool(np.isfinite(r["da_dN"]) and r["fixed_point_converged"])
        same_seed.append(dict(
            Kmax_MPa_sqrt_m=K, seed=PRIMARY_SEED, sample_events_requested=CONVERGED_SAMPLE_LEVEL,
            b1_converged=r["fixed_point_converged"],
            b1_da_dN=r["da_dN"], physical_da_dN=phys["da_dN"],
            physical_over_b1_rate_ratio=(phys["da_dN"] / r["da_dN"]) if finite else float("nan"),
            rate_log10_error_decade=(abs(math.log10(phys["da_dN"]) - math.log10(r["da_dN"]))
                                     if finite else float("nan")),
            b1_r_eff_m=r["r_eff_m"], physical_tip_radius_m=phys["tip_radius_m"],
            b1_delta_r_m=dr_b1, physical_delta_r_m=dr_ph,
            physical_over_b1_delta_r_ratio=(dr_ph / dr_b1) if dr_b1 else float("nan"),
            total_radius_rel_error_SECONDARY=abs(r["r_eff_m"] - phys["tip_radius_m"]) / phys["tip_radius_m"],
            physical_event_count=phys["event_count"], b1_events=r["events"],
            b1_sample_events_achieved=r["sample_events"],
            radius_semantics_b1="mean_post_event_after_burn_in",
            radius_semantics_physical="single_summary_tip_radius_m_semantics_unconfirmed",
            semantics_matched=False,
        ))
    write_csv(OUT / "b1_same_seed_transfer.csv", same_seed)

    # ---- 4. ensemble sensitivity (seeds 2024/3031 = uncertainty only) ----
    ensemble = []
    for K in K_GRID:
        vals_g, vals_r = [], []
        for seed in seeds:
            r = get(K, seed, CONVERGED_SAMPLE_LEVEL)
            vals_r.append(r["r_eff_m"])
            if np.isfinite(r["da_dN"]):
                vals_g.append(r["da_dN"])
        ensemble.append(dict(
            Kmax_MPa_sqrt_m=K, n_seeds=len(seeds), n_finite_rate_seeds=len(vals_g),
            mean_r_eff_m=float(np.mean(vals_r)), std_r_eff_m=float(np.std(vals_r, ddof=1)),
            mean_delta_r_m=float(np.mean(vals_r) - R0),
            mean_da_dN=float(np.mean(vals_g)) if vals_g else float("nan"),
            std_da_dN=float(np.std(vals_g, ddof=1)) if len(vals_g) > 1 else float("nan"),
            rel_std_da_dN=(float(np.std(vals_g, ddof=1) / np.mean(vals_g)) if len(vals_g) > 1 else float("nan")),
            role="STOCHASTIC_UNCERTAINTY_ONLY_NOT_PRIMARY_COMPARISON",
        ))
    write_csv(OUT / "b1_ensemble_sensitivity.csv", ensemble)

    # ---- 5. local slope comparison ---------------------------------------
    slope_rows = []
    for K_lo, K_hi in [(12.0, 15.0), (15.0, 18.0), (18.0, 24.0)]:
        b_lo = get(K_lo, PRIMARY_SEED, CONVERGED_SAMPLE_LEVEL)
        b_hi = get(K_hi, PRIMARY_SEED, CONVERGED_SAMPLE_LEVEL)
        m_phys = adjacent_log_slope(K_lo, K_hi, physical[K_lo]["da_dN"], physical[K_hi]["da_dN"])
        m_b1 = adjacent_log_slope(K_lo, K_hi, b_lo["da_dN"], b_hi["da_dN"])
        ens_lo = [get(K_lo, s, CONVERGED_SAMPLE_LEVEL)["da_dN"] for s in seeds]
        ens_hi = [get(K_hi, s, CONVERGED_SAMPLE_LEVEL)["da_dN"] for s in seeds]
        ens_slopes = [adjacent_log_slope(K_lo, K_hi, a, b) for a, b in zip(ens_lo, ens_hi)]
        ens_slopes = [s for s in ens_slopes if np.isfinite(s)]
        slope_rows.append(dict(
            interval=f"m_{K_lo:g}_{K_hi:g}", K_lo=K_lo, K_hi=K_hi,
            K_mid_MPa_sqrt_m=math.sqrt(K_lo * K_hi),
            physical_slope=m_phys, b1_same_seed_slope=m_b1,
            slope_error=(m_b1 - m_phys) if np.isfinite(m_b1) else float("nan"),
            b1_ensemble_slope_mean=float(np.mean(ens_slopes)) if ens_slopes else float("nan"),
            b1_ensemble_slope_std=float(np.std(ens_slopes, ddof=1)) if len(ens_slopes) > 1 else float("nan"),
            usable=bool(np.isfinite(m_b1)),
            note=("K=12 leg unusable: B1 numerically unresolved" if K_lo == 12.0 else ""),
        ))
    write_csv(OUT / "b1_local_slope_comparison.csv", slope_rows)

    # ---- 6. radius-increment comparison + alpha_r diagnostic -------------
    r18_same = get(18.0, PRIMARY_SEED, CONVERGED_SAMPLE_LEVEL)["r_eff_m"]
    alpha_same = (physical[18.0]["tip_radius_m"] - R0) / (r18_same - R0)
    r18_mean = float(np.mean([get(18.0, s, CONVERGED_SAMPLE_LEVEL)["r_eff_m"] for s in seeds]))
    alpha_mean = (physical[18.0]["tip_radius_m"] - R0) / (r18_mean - R0)

    radius_rows = []
    for K in K_GRID:
        r = get(K, PRIMARY_SEED, CONVERGED_SAMPLE_LEVEL)
        dr_b1 = r["r_eff_m"] - R0
        dr_ph = physical[K]["tip_radius_m"] - R0
        r_scaled = R0 + alpha_same * dr_b1
        # what rate would the A0 opening model give at the scaled radius?
        base = dict(hits=extra["physics__cleavage_hits"],
                    tau_s=extra["physics__cleavage_correlation_time_s"],
                    event_length_m=5.0e-6, frequency_Hz=1000.0, temperature_K=300.0, n_phase=4096)
        g_scaled = cycle_growth_and_slope(manifest.cleavage, K, 0.1,
                                          RenewalControls(**base, radius_m=r_scaled))["da_dN"]
        radius_rows.append(dict(
            Kmax_MPa_sqrt_m=K, b1_delta_r_m=dr_b1, physical_delta_r_m=dr_ph,
            physical_over_b1_delta_r_ratio=dr_ph / dr_b1 if dr_b1 else float("nan"),
            alpha_r_applied=alpha_same, radius_scaled_m=r_scaled,
            radius_scaled_vs_physical_rel_error=abs(r_scaled - physical[K]["tip_radius_m"]) / physical[K]["tip_radius_m"],
            a0_rate_at_scaled_radius=g_scaled, physical_da_dN=physical[K]["da_dN"],
            scaled_rate_log10_error_decade=abs(math.log10(g_scaled) - math.log10(physical[K]["da_dN"])),
            is_calibration_point=bool(K == 18.0),
            role=("ALPHA_R_CALIBRATION_POINT_RADIUS_ONLY" if K == 18.0 else "HOLDOUT"),
        ))
    write_csv(OUT / "b1_radius_increment_comparison.csv", radius_rows)

    # ---- 7. A0 root diagnostic ------------------------------------------
    a0_rows = [a0_root_diagnostic(manifest, extra, K, physical[K]["da_dN"]) for K in K_GRID]
    write_csv(OUT / "b1_a0_root_diagnostic.csv", a0_rows)

    # ---- 8. decision ------------------------------------------------------
    k12_records = [r for r in records if r["Kmax_MPa_sqrt_m"] == 12.0]
    k12_unresolved = all(not r["fixed_point_converged"] for r in k12_records)
    resolved_K = [K for K in (15.0, 18.0, 24.0)
                  if get(K, PRIMARY_SEED, CONVERGED_SAMPLE_LEVEL)["fixed_point_converged"]]
    rate_failures = {K: next(s for s in same_seed if s["Kmax_MPa_sqrt_m"] == K)["rate_log10_error_decade"]
                     for K in resolved_K}
    rate_transfer_failed = all(v > 0.30 for v in rate_failures.values())

    classifications = ["B1_EXPLORATORY_SWEEP_V1_COMPLETE"]
    if k12_unresolved:
        classifications.append("B1_NUMERICALLY_UNRESOLVED_AT_K12")
    if rate_transfer_failed:
        classifications.append("B1_CONVERGED_BUT_RATE_TRANSFER_FAILED_AT_K15_K18_K24")
    classifications.append("B1_TRANSFER_NOT_QUALIFIED_FOR_CANDIDATE_SELECTION")
    a0_class = ("A0_RATE_EQUIVALENT_RADIUS_NONPHYSICAL_OR_UNDEFINED"
                if all(row["classification"] != "A0_RATE_EQUIVALENT_RADIUS_PHYSICALLY_ADMISSIBLE"
                       for row in a0_rows) else "A0_RATE_EQUIVALENT_RADIUS_PARTIALLY_ADMISSIBLE")
    classifications.append(a0_class)

    decision = dict(
        schema="v10.2.30_prospective_paris_b1_exploratory_postmortem_v1",
        supersedes="decide_b1_qualification.py (never run; defective gates)",
        sweep_records_sha256=sha256_file(RECORDS),
        sweep_record_count=len(records),
        physical_source_csv=str(PHYS_CSV),
        physical_source_csv_sha256=PHYS_CSV_SHA256,
        physical_row_audit=phys_audit,
        primary_comparison="SAME_SEED_1720_B1_vs_SAME_SEED_1720_PHYSICAL",
        ensemble_seeds_role="STOCHASTIC_UNCERTAINTY_ONLY",
        classifications=classifications,
        k12_terminal_condition=dict(
            cycles=k12_records[0]["cycles"], maximum_cycles=1_000_000,
            events_reached=k12_records[0]["events"],
            sample_events_achieved=k12_records[0]["sample_events"],
            fixed_point_converged=False,
            interpretation=("maximum-cycle exhaustion before the requested sampled-event "
                            "count; da_dN is NaN and is NOT a zero rate, NOT a physical "
                            "censor, and NOT a converged result"),
        ),
        same_seed_transfer=same_seed,
        local_slopes=slope_rows,
        alpha_r_same_seed_K18_radius_only=alpha_same,
        alpha_r_three_seed_mean_K18_radius_only=alpha_mean,
        alpha_r_usage="DIAGNOSTIC_ONLY_NEVER_A_QUALIFIED_TRANSFER",
        a0_root_diagnostic=a0_rows,
        radius_only_impossibility=(
            "Increasing r_eff lowers the opening stress sigma = (K - K_shield)/sqrt(2*pi*r), "
            "which lowers the predicted rate. B1 already predicts rates ~8-10x BELOW the "
            "physical rates at K=15/18/24. Scaling B1's blunting increment up to match the "
            "larger physical radius therefore moves the predicted rate further DOWN, away "
            "from the physical rate. A one-scalar radius-increment correction cannot repair "
            "the rate transfer: the missing transfer is not primarily a tip-radius amplitude "
            "correction."
        ),
        outstanding_semantics_gap=(
            "B1 r_eff_m is a mean post-event radius after burn-in; the physical table exposes "
            "a single tip_radius_m whose exact statistic (terminal vs developed-window mean) "
            "is not confirmed from a surviving producer definition. All radius comparisons "
            "here are therefore labeled semantics_matched=False."
        ),
        next_step=("One bounded same-seed rate-semantics audit at K=18 (no fitted parameter), "
                   "then the fresh-P40 envelope fallback."),
    )
    (OUT / "b1_exploratory_sweep_decision.json").write_text(
        json.dumps(decision, indent=2, default=str) + "\n")

    md = [
        "# B1 Exploratory Sweep Postmortem",
        "",
        f"Sweep records SHA-256: `{decision['sweep_records_sha256']}` (24 records).",
        "",
        "## Classifications",
        "",
    ] + [f"- `{c}`" for c in classifications] + [
        "",
        "## Same-seed transfer (seed 1720 B1 vs seed 1720 physical, 160-sample level)",
        "",
        "| Kmax | B1 da/dN | physical da/dN | phys/B1 ratio | log10 err (dec) | B1 dr (m) | phys dr (m) | phys/B1 dr |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in same_seed:
        md.append("| {K:g} | {b:.4e} | {p:.4e} | {r} | {e} | {db:.4e} | {dp:.4e} | {dr} |".format(
            K=s["Kmax_MPa_sqrt_m"], b=s["b1_da_dN"], p=s["physical_da_dN"],
            r=("NaN" if not np.isfinite(s["physical_over_b1_rate_ratio"]) else f"{s['physical_over_b1_rate_ratio']:.3f}"),
            e=("NaN" if not np.isfinite(s["rate_log10_error_decade"]) else f"{s['rate_log10_error_decade']:.4f}"),
            db=s["b1_delta_r_m"], dp=s["physical_delta_r_m"],
            dr=("NaN" if not np.isfinite(s["physical_over_b1_delta_r_ratio"]) else f"{s['physical_over_b1_delta_r_ratio']:.3f}")))
    md += [
        "",
        "## Adjacent log-log slopes",
        "",
        "| interval | physical | B1 same-seed | error |",
        "|---|---|---|---|",
    ]
    for s in slope_rows:
        md.append("| {i} | {p} | {b} | {e} |".format(
            i=s["interval"],
            p=("NaN" if not np.isfinite(s["physical_slope"]) else f"{s['physical_slope']:.4f}"),
            b=("NaN" if not np.isfinite(s["b1_same_seed_slope"]) else f"{s['b1_same_seed_slope']:.4f}"),
            e=("NaN" if not np.isfinite(s["slope_error"]) else f"{s['slope_error']:+.4f}")))
    md += [
        "",
        "## K=12",
        "",
        f"- cycles = {k12_records[0]['cycles']} (== maximum_cycles = 1,000,000)",
        f"- events reached = {k12_records[0]['events']}, sampled = {k12_records[0]['sample_events']}",
        "- `fixed_point_converged = false` -> `da_dN = NaN`",
        "- **NaN is maximum-cycle exhaustion. It is not zero, not a physical censor, "
        "and not a converged rate.**",
        "",
        "## Radius-only impossibility",
        "",
        decision["radius_only_impossibility"],
        "",
        f"- alpha_r (same-seed, K=18, radius only) = {alpha_same:.4f}",
        f"- alpha_r (three-seed mean, K=18, radius only) = {alpha_mean:.4f}",
        "- Both are **diagnostic only** and never enter candidate scoring.",
        "",
        "## A0 rate-equivalent radius",
        "",
        "| Kmax | A0 ceiling (r->0) | physical | root? | root (m) | classification |",
        "|---|---|---|---|---|---|",
    ]
    for row in a0_rows:
        md.append("| {K:g} | {c:.4e} | {p:.4e} | {e} | {r} | `{cl}` |".format(
            K=row["Kmax_MPa_sqrt_m"], c=row["a0_ceiling_da_dN"], p=row["physical_da_dN"],
            e=row["root_exists"],
            r=("n/a" if not row["root_exists"] else f"{row['root_radius_m']:.4e}"),
            cl=row["classification"]))
    md += [
        "",
        "At K=15/18/24 the physical rate **exceeds the limiting A0 opening-only rate as "
        "r -> 0**, so no positive radius can reproduce it. This is a physical statement "
        "about the reduced model's ceiling, not a root-finder failure.",
        "",
        "## Outstanding semantics gap",
        "",
        decision["outstanding_semantics_gap"],
        "",
    ]
    (OUT / "b1_exploratory_sweep_decision.md").write_text("\n".join(md) + "\n")

    print(json.dumps(dict(classifications=classifications,
                          alpha_r_same_seed=alpha_same, alpha_r_three_seed_mean=alpha_mean,
                          rate_log10_errors={s["Kmax_MPa_sqrt_m"]: s["rate_log10_error_decade"]
                                             for s in same_seed}), indent=2, default=str))


if __name__ == "__main__":
    main()
