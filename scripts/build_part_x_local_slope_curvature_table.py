"""Final closure D: local slopes and curvature of S_h_developed vs Kmax,
per protocol, seed=1720 -- the one genuinely missing table from the
required Part X artifact set (px4_stage1_slope_table.csv already covers
the single global 3-point log-log fit used for seed-robustness; this
table adds the finer-grained LOCAL secant slope between each consecutive
pair of the full 5-point grid, plus a discrete curvature (second
difference of S_h in log10(Kmax)) at each interior point).

Reads only the tracked px4_stage1_rate_table.csv.
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


def main() -> None:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_rate_table.csv")))
    by_protocol: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r["seed"] != "1720":
            continue
        by_protocol[r["protocol"]].append((float(r["Kmax_Pa_sqrt_m"]), float(r["S_h_developed"])))

    out_rows = []
    for protocol, pts in by_protocol.items():
        pts.sort()
        if len(pts) < 2:
            continue
        x = [math.log10(K) for K, _ in pts]
        y = [S for _, S in pts]

        local_slopes = []
        for i in range(len(pts) - 1):
            m = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            local_slopes.append(m)

        for i in range(len(pts) - 1):
            Kmax_lo, Kmax_hi = pts[i][0], pts[i + 1][0]
            curvature = None
            if 0 < i < len(local_slopes):
                curvature = local_slopes[i] - local_slopes[i - 1]
            out_rows.append({
                "protocol": protocol, "Kmax_lo_Pa_sqrt_m": Kmax_lo, "Kmax_hi_Pa_sqrt_m": Kmax_hi,
                "S_h_lo": pts[i][1], "S_h_hi": pts[i + 1][1], "local_secant_slope_m": local_slopes[i],
                "curvature_delta_slope_at_Kmax_hi": curvature,
            })

    with (ARTIFACTS_DIR / "px4_local_slope_curvature_table.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote px4_local_slope_curvature_table.csv: {len(out_rows)} intervals across {len(by_protocol)} protocols")
    for r in out_rows:
        curv_str = f"{r['curvature_delta_slope_at_Kmax_hi']:+.3f}" if r["curvature_delta_slope_at_Kmax_hi"] is not None else "N/A"
        print(f"  {r['protocol']:28s} [{r['Kmax_lo_Pa_sqrt_m']/1e6:5.1f}-{r['Kmax_hi_Pa_sqrt_m']/1e6:5.1f}]MPa  "
              f"local_m={r['local_secant_slope_m']:+.4f}  curvature={curv_str}")


if __name__ == "__main__":
    main()
