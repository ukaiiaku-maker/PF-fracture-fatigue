"""PX4.1 section 6: fit the log-log Kmax-dependence slope m of
S_h_developed (S_h ~ Kmax^m) for every seed=1720 base protocol and its
seed=1001723 confirmation counterpart, restricted to the 3 Kmax points
(15, 18, 21 MPa*sqrt(m)) common to BOTH seeds so the comparison is over
an identical support -- the base protocol also has 12 and 24.3 points
that the confirmation grid does not, and including them would bias the
seed-1720 slope against a different Kmax range than seed-1001723 actually
covers.

Classifies each protocol axis REBONDING_SEED_ROBUST (sign(m) matches
between seeds AND |delta_m| <= 0.25) or REBONDING_SEED_SENSITIVE,
per the review's own gate.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

CONFIRM_OF = {
    "D1": "D1_confirm", "D2": "D2_confirm", "D3": "D3_confirm",
    "D5": "D5_confirm", "D6_conditional_persistent": "D6_confirm",
}
COMMON_KMAX_Pa_sqrt_m = {15.0e6, 18.0e6, 21.0e6}
DELTA_M_ROBUST_TOL = 0.25


def _fit_log_log_slope(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares slope+intercept of ln|S_h| vs ln(Kmax)."""
    xs = [math.log(k) for k, _ in points]
    ys = [math.log(abs(s)) for _, s in points]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    m = cov / var
    b = mean_y - m * mean_x
    return m, b


def main() -> None:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_rate_table.csv")))
    by_seed_protocol: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        kmax = float(r["Kmax_Pa_sqrt_m"])
        if kmax in COMMON_KMAX_Pa_sqrt_m:
            by_seed_protocol[(r["seed"], r["protocol"])].append((kmax, float(r["S_h_developed"])))

    out_rows = []
    for base_protocol, confirm_protocol in CONFIRM_OF.items():
        base_pts = sorted(by_seed_protocol[("1720", base_protocol)])
        confirm_pts = sorted(by_seed_protocol[("1001723", confirm_protocol)])
        if len(base_pts) != 3 or len(confirm_pts) != 3:
            raise RuntimeError(f"{base_protocol}: expected 3+3 common-Kmax points, found {len(base_pts)}+{len(confirm_pts)}")

        m_base, _ = _fit_log_log_slope(base_pts)
        m_confirm, _ = _fit_log_log_slope(confirm_pts)
        delta_m = m_confirm - m_base
        sign_match = (m_base > 0) == (m_confirm > 0)
        classification = (
            "REBONDING_SEED_ROBUST" if sign_match and abs(delta_m) <= DELTA_M_ROBUST_TOL
            else "REBONDING_SEED_SENSITIVE"
        )
        out_rows.append({
            "base_protocol": base_protocol, "confirm_protocol": confirm_protocol,
            "common_Kmax_points_Pa_sqrt_m": ";".join(str(k) for k, _ in base_pts),
            "seed_1720_slope_m": m_base, "seed_1001723_slope_m": m_confirm,
            "delta_m": delta_m, "sign_match": sign_match,
            "delta_m_tolerance": DELTA_M_ROBUST_TOL, "classification": classification,
        })

    with (ARTIFACTS_DIR / "px4_stage1_slope_table.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote px4_stage1_slope_table.csv: {len(out_rows)} protocol axes")
    for row in out_rows:
        print(f"  {row['base_protocol']:28s} m_1720={row['seed_1720_slope_m']:+.4f}  "
              f"m_1001723={row['seed_1001723_slope_m']:+.4f}  delta_m={row['delta_m']:+.4f}  {row['classification']}")


if __name__ == "__main__":
    main()
