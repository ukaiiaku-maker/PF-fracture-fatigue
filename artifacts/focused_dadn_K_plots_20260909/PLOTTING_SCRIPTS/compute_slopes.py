"""Compute global log-log Paris fits and adjacent local logarithmic secants
for every developed curve in all_included_curve_points.csv. Every number
here is derived directly and only from that exported table -- this script
is the reproducibility check referenced in the README.

Local secant (primary local-slope measure), per the mission specification:
  m_local,i = [log10(g_{i+1}) - log10(g_i)] / [log10(K_{i+1}) - log10(K_i)]
  K_mid,i = sqrt(K_i * K_{i+1})

Global fit: ordinary least squares of log10(da/dN) vs log10(Kmax), reporting
slope, intercept, and R^2.

single-slope adequacy classification: PARIS_ADEQUATE requires BOTH
  R^2 >= 0.95, AND
  max(m_local) - min(m_local) <= max(0.5, 0.25*|m_global|)
otherwise CURVED_NON_PARIS_RESPONSE.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from collections import defaultdict

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "SOURCE_DATA" / "all_included_curve_points.csv"
OUT = PKG / "SOURCE_DATA"


def _group_key(r: dict) -> tuple:
    # Group by the physically distinct curve: parameterization id (protocol
    # base, stripped of the finite/baseline distinction is NOT done here --
    # zero-cohesion and finite branches of the same protocol are genuinely
    # different curves and must not be merged), R (or condition), and seed.
    pid = r["internal_parameterization_id"]
    cat = r["parameterization_category"]
    return (pid, cat, r["R"], r["seed"], r["frequency_Hz"])


def load_rows():
    with SRC.open() as fh:
        return list(csv.DictReader(fh))


def global_fit(logK, logG):
    n = len(logK)
    mean_x = sum(logK) / n
    mean_y = sum(logG) / n
    sxx = sum((x - mean_x) ** 2 for x in logK)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(logK, logG))
    slope = sxy / sxx if sxx > 0 else float("nan")
    intercept = mean_y - slope * mean_x
    pred = [slope * x + intercept for x in logK]
    ss_res = sum((y - p) ** 2 for y, p in zip(logG, pred))
    ss_tot = sum((y - mean_y) ** 2 for y in logG)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def local_secants(K, G):
    out = []
    for i in range(len(K) - 1):
        K_mid = math.sqrt(K[i] * K[i + 1])
        m = (math.log10(G[i + 1]) - math.log10(G[i])) / (math.log10(K[i + 1]) - math.log10(K[i]))
        out.append((K_mid, m))
    return out


def main():
    rows = load_rows()
    groups = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)

    global_rows, local_rows = [], []
    for key, grp in groups.items():
        grp = sorted(grp, key=lambda r: float(r["Kmax_MPa_sqrt_m"]))
        K = [float(r["Kmax_MPa_sqrt_m"]) for r in grp]
        G = [float(r["da_dN_m_per_cycle"]) for r in grp]
        label = grp[0]["visible_curve_label"]
        n = len(K)
        if n < 2:
            continue
        logK = [math.log10(k) for k in K]
        logG = [math.log10(g) for g in G]
        m_global, b_global, r2 = global_fit(logK, logG)

        secants = local_secants(K, G)
        for K_mid, m_loc in secants:
            local_rows.append(dict(visible_curve_label=label, internal_parameterization_id=key[0],
                                    K_mid_MPa_sqrt_m=round(K_mid, 4), m_local=round(m_loc, 4)))

        m_locals = [m for _, m in secants]
        m_range = (max(m_locals) - min(m_locals)) if m_locals else float("nan")
        threshold = max(0.5, 0.25 * abs(m_global)) if not math.isnan(m_global) else float("nan")
        adequate = (n >= 3) and (r2 >= 0.95) and (m_range <= threshold)
        classification = "PARIS_ADEQUATE" if adequate else "CURVED_NON_PARIS_RESPONSE"
        if n < 3:
            classification = "INSUFFICIENT_POINTS_FOR_CLASSIFICATION"

        global_rows.append(dict(
            visible_curve_label=label, internal_parameterization_id=key[0],
            parameterization_category=key[1], R=key[2], seed=key[3], frequency_Hz=key[4],
            n_points=n, K_min_MPa_sqrt_m=round(min(K), 3), K_max_MPa_sqrt_m=round(max(K), 3),
            global_slope_m=round(m_global, 4), global_intercept=round(b_global, 4), R_squared=round(r2, 5),
            local_slope_min=round(min(m_locals), 4) if m_locals else "",
            local_slope_max=round(max(m_locals), 4) if m_locals else "",
            local_slope_range=round(m_range, 4) if m_locals else "",
            single_slope_adequate_threshold=round(threshold, 4) if not math.isnan(threshold) else "",
            classification=classification,
        ))

    gfields = list(global_rows[0].keys())
    with (OUT / "global_slope_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=gfields, lineterminator="\n")
        w.writeheader(); w.writerows(global_rows)

    lfields = list(local_rows[0].keys())
    with (OUT / "local_secant_slopes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=lfields, lineterminator="\n")
        w.writeheader(); w.writerows(local_rows)

    print(f"Wrote global_slope_summary.csv: {len(global_rows)} curves")
    print(f"Wrote local_secant_slopes.csv: {len(local_rows)} secants")
    for g in global_rows:
        print(f"  {g['visible_curve_label']:55s} n={g['n_points']} m_g={g['global_slope_m']:8.3f} "
              f"R2={g['R_squared']:.4f} local[{g['local_slope_min']},{g['local_slope_max']}] {g['classification']}")


if __name__ == "__main__":
    main()
