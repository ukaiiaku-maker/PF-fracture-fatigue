"""PX4 developed-campaign analysis (mission section 8).

Applies the ALREADY-QUALIFIED stationarity/developed-growth gate
(arrhenius_fracture.crack_rebonding_developed_confirmation_v10230.
stable_growth_gate -- reused verbatim, same constants/formula, from the
non-Part-X developed-confirmation study) to all 44 completed PX4
developed trajectories, pairs matched finite/zero cohesion trajectories,
and computes each pair's developed g/S_h from the STABLE/DEVELOPED
window's own da/dN (not the whole 30-event trajectory), per mission
section 8's explicit developed-target/stationarity requirement ("exclude
first 20 um as transient... assess the last 50 um... Use the already
qualified implementation of this gate. Do not invent a more permissive
gate.").

Pairing is by matching (protocol, row_name, Kmax, R, frequency_Hz,
minimum_load_hold_s, chemistry_factor, K_rebond_max_target) across
cohesion -- not by file/row adjacency, since the D3/D6 rows were appended
to developed_job_registry.csv out of the original construction order.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

from arrhenius_fracture.crack_rebonding_developed_confirmation_v10230 import (
    stable_growth_gate, effective_horizon_censored,
)

DEVELOPED_MAX_ACCEPTED_EVENTS = 30
DEVELOPED_MAX_PROJECTED_EXTENSION_m = 150.0e-6


def _load_developed_results() -> dict[str, dict]:
    key_to_result = {}
    for p in glob.glob(str(RUN_ROOT / "*/result.json")):
        r = json.loads(Path(p).read_text())
        if r.get("schema") == "v10230_part_x_developed_job_result_v1":
            key_to_result[r["job"]["canonical_job_key"]] = r
    return key_to_result


def _pair_key(row: dict) -> tuple:
    return (
        row["protocol"], row["row_name"], float(row["Kmax_Pa_sqrt_m"]), float(row["R"]),
        float(row["frequency_Hz"]), float(row["minimum_load_hold_s"]), float(row["chemistry_factor"]),
    )


def main() -> None:
    key_to_result = _load_developed_results()
    with (ARTIFACTS_DIR / "developed_job_registry.csv").open() as f:
        rows = list(csv.DictReader(f))
    authorized = [r for r in rows if r["status"] == "AUTHORIZED_PX4"]

    by_pair: dict[tuple, dict[str, dict]] = {}
    for row in authorized:
        by_pair.setdefault(_pair_key(row), {})[row["cohesion"]] = row

    pair_analyses = []
    for key, members in by_pair.items():
        if "finite" not in members or "zero" not in members:
            raise RuntimeError(f"incomplete pair for {key}: {list(members.keys())}")
        finite_row, zero_row = members["finite"], members["zero"]
        rf = key_to_result.get(finite_row["canonical_job_key"])
        rz = key_to_result.get(zero_row["canonical_job_key"])
        if rf is None or rz is None:
            raise RuntimeError(f"missing completed developed result for pair {key}")
        traj_f, traj_z = rf["trajectory"], rz["trajectory"]
        freq_Hz = key[4]

        gate_f = stable_growth_gate(traj_f["events"], frequency_Hz=freq_Hz)
        gate_z = stable_growth_gate(traj_z["events"], frequency_Hz=freq_Hz)
        horizon_f = effective_horizon_censored(
            traj_f, max_accepted_events=DEVELOPED_MAX_ACCEPTED_EVENTS,
            max_projected_extension_m=DEVELOPED_MAX_PROJECTED_EXTENSION_m,
        )
        horizon_z = effective_horizon_censored(
            traj_z, max_accepted_events=DEVELOPED_MAX_ACCEPTED_EVENTS,
            max_projected_extension_m=DEVELOPED_MAX_PROJECTED_EXTENSION_m,
        )

        both_stable = gate_f["stable_growth_provisional"] and gate_z["stable_growth_provisional"]
        S_h_developed = None
        if both_stable:
            da_dN_f, da_dN_z = gate_f["developed_da_dN_m_per_cycle"], gate_z["developed_da_dN_m_per_cycle"]
            if da_dN_f and da_dN_z and da_dN_f > 0.0 and da_dN_z > 0.0:
                S_h_developed = math.log10(da_dN_f / da_dN_z)

        pair_analyses.append({
            "protocol": key[0], "row_name": key[1], "Kmax_Pa_sqrt_m": key[2], "R": key[3],
            "frequency_Hz": key[4], "minimum_load_hold_s": key[5], "chemistry_factor": key[6],
            "finite_canonical_job_key": finite_row["canonical_job_key"], "zero_canonical_job_key": zero_row["canonical_job_key"],
            "finite_stable_growth": gate_f["stable_growth_provisional"], "zero_stable_growth": gate_z["stable_growth_provisional"],
            "finite_late_to_early_ratio": gate_f["late_to_early_rate_ratio"], "zero_late_to_early_ratio": gate_z["late_to_early_rate_ratio"],
            "finite_developed_da_dN_m_per_cycle": gate_f["developed_da_dN_m_per_cycle"],
            "zero_developed_da_dN_m_per_cycle": gate_z["developed_da_dN_m_per_cycle"],
            "finite_horizon": horizon_f, "zero_horizon": horizon_z,
            "both_stable_growth": both_stable, "S_h_developed": S_h_developed,
            "finite_n_events": traj_f["n_accepted_events"], "zero_n_events": traj_z["n_accepted_events"],
            "finite_final_extension_um": gate_f["final_extension_um"], "zero_final_extension_um": gate_z["final_extension_um"],
        })

    out = {"schema": "v10230_part_x_px4_developed_pair_analysis_v1", "pairs": pair_analyses}
    (ARTIFACTS_DIR / "px4_developed_pair_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    with (ARTIFACTS_DIR / "px4_developed_pair_analysis.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[k for k in pair_analyses[0].keys() if k not in ("finite_horizon", "zero_horizon")], lineterminator="\n")
        writer.writeheader()
        for row in pair_analyses:
            writer.writerow({k: v for k, v in row.items() if k not in ("finite_horizon", "zero_horizon")})

    print(f"Wrote px4_developed_pair_analysis.{{json,csv}}: {len(pair_analyses)} matched pairs")
    n_both_stable = sum(1 for p in pair_analyses if p["both_stable_growth"])
    print(f"  {n_both_stable}/{len(pair_analyses)} pairs have BOTH members stable_growth_provisional=True")
    for row in pair_analyses:
        S_h_str = f"{row['S_h_developed']:.4f}" if row["S_h_developed"] is not None else "N/A"
        print(f"  {row['protocol']:30s} Kmax={row['Kmax_Pa_sqrt_m']/1e6:>5.1f}MPa R={row['R']:>6} f={row['frequency_Hz']:>10.3f} "
              f"both_stable={row['both_stable_growth']!s:5} S_h_developed={S_h_str}")


if __name__ == "__main__":
    main()
