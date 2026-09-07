"""PX4 developed-campaign analysis (mission section 8).

Applies the ALREADY-QUALIFIED stationarity/developed-growth gate
(arrhenius_fracture.crack_rebonding_developed_confirmation_v10230.
stable_growth_gate -- reused verbatim, same constants/formula, from the
non-Part-X developed-confirmation study) to every completed PX4 developed
trajectory, pairs matched finite/zero cohesion trajectories, and computes
each pair's developed g/S_h from the STABLE/DEVELOPED window's own da/dN
(not the whole 30-event trajectory), per mission section 8's explicit
developed-target/stationarity requirement ("exclude first 20 um as
transient... assess the last 50 um... Use the already qualified
implementation of this gate. Do not invent a more permissive gate.").

Canonical PAIRING key (PX4.1 fix -- the original key was not seed-safe):
seed + protocol + row_name + rebonding_config_hash + Kmax + R +
frequency_Hz + minimum_load_hold_s + chemistry_factor + control_mode +
physical_producer_sha. control_mode is always "dynamic" for this script
(PX4 has no static control legs -- those are PX5's). physical_producer_sha
stands in for physical_source_bundle_sha256 here: every admitted PX4 row
shares one producer commit whose full source-bundle hash is recorded
separately in px4_producer_freeze.json, so the sha alone already
disambiguates any future producer change without needing to look that
file up per row.

K_rebond_max_target_Pa_sqrt_m is DELIBERATELY EXCLUDED from the pairing
key itself: by construction a finite-cohesion row's K_target (e.g.
900000.0) and its matched zero-cohesion row's K_target (always 0.0,
zero-cohesion has no cohesive strength to target) can never be equal, so
requiring key equality across cohesion on that field would make every
pair unmatchable. It IS recorded per-row in the output and used as an
extra assertion: within one base group there must be exactly one "finite"
and one "zero" row (never two distinct finite K_target variants sharing
one zero baseline, as mission section 7.6's cohesive-strength SCREEN axis
allows for -- PX4 has no such case among its currently authorized rows;
this assertion will fail loudly, not silently mispair, the day it does).

Seed-safe by construction: seed is the FIRST element of the pairing key,
so a second seed's rows group into entirely separate pairs, never
overwriting seed 1720's dictionary entries.
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
DEVELOPED_MAX_CUMULATIVE_CYCLES = 1.0e12


def _load_developed_results() -> dict[str, dict]:
    key_to_result = {}
    for p in glob.glob(str(RUN_ROOT / "*/result.json")):
        r = json.loads(Path(p).read_text())
        if r.get("schema") == "v10230_part_x_developed_job_result_v1":
            key_to_result[r["job"]["canonical_job_key"]] = r
    return key_to_result


def canonical_analysis_key(row: dict, *, control_mode: str = "dynamic") -> tuple:
    """The seed-safe canonical key for grouping analysis rows -- covers
    everything that must be identical between a pair's finite and zero
    members, and deliberately excludes K_rebond_max_target_Pa_sqrt_m
    (which, by construction, never matches across cohesion -- see module
    docstring)."""
    return (
        int(row["seed"]), row["protocol"], row["row_name"], row["config_hash"],
        float(row["Kmax_Pa_sqrt_m"]), float(row["R"]), float(row["frequency_Hz"]),
        float(row["minimum_load_hold_s"]), float(row["chemistry_factor"]),
        control_mode, row["physical_producer_sha"],
    )


def _cycle_horizon_check(traj: dict) -> dict:
    events = traj["events"]
    max_event_cycles = max((float(e["cumulative_cycles"]) for e in events), default=0.0)
    return {
        "max_event_cumulative_cycles": max_event_cycles,
        "terminal_cumulative_cycles": float(traj["cumulative_cycles"]),
        "hit_cycle_horizon": (
            traj["censor_reason"] == "complete_physical_cycle_censor"
            or max_event_cycles >= DEVELOPED_MAX_CUMULATIVE_CYCLES
        ),
    }


def main() -> None:
    key_to_result = _load_developed_results()
    with (ARTIFACTS_DIR / "developed_job_registry.csv").open() as f:
        rows = list(csv.DictReader(f))
    authorized = [r for r in rows if r["status"] == "AUTHORIZED_PX4"]

    by_pair: dict[tuple, dict[str, dict]] = {}
    for row in authorized:
        key = canonical_analysis_key(row)
        slot = by_pair.setdefault(key, {})
        if row["cohesion"] in slot and slot[row["cohesion"]]["canonical_job_key"] != row["canonical_job_key"]:
            raise RuntimeError(
                f"canonical key collision: {key} already has a different '{row['cohesion']}' row "
                f"({slot[row['cohesion']]['canonical_job_key'][:12]} vs {row['canonical_job_key'][:12]})"
            )
        slot[row["cohesion"]] = row

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
        seed, protocol, row_name, config_hash, Kmax, R, freq_Hz, hold_s, chem, control_mode, producer_sha = key
        K_target_finite = float(finite_row["K_rebond_max_target_Pa_sqrt_m"])
        K_target_zero = float(zero_row["K_rebond_max_target_Pa_sqrt_m"])

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
        cycle_f, cycle_z = _cycle_horizon_check(traj_f), _cycle_horizon_check(traj_z)

        both_stable = gate_f["stable_growth_provisional"] and gate_z["stable_growth_provisional"]
        S_h_developed = None
        if both_stable:
            da_dN_f, da_dN_z = gate_f["developed_da_dN_m_per_cycle"], gate_z["developed_da_dN_m_per_cycle"]
            if da_dN_f and da_dN_z and da_dN_f > 0.0 and da_dN_z > 0.0:
                S_h_developed = math.log10(da_dN_f / da_dN_z)

        pair_analyses.append({
            "seed": seed, "protocol": protocol, "row_name": row_name, "config_hash": config_hash,
            "Kmax_Pa_sqrt_m": Kmax, "R": R, "frequency_Hz": freq_Hz, "minimum_load_hold_s": hold_s,
            "chemistry_factor": chem,
            "finite_K_rebond_max_target_Pa_sqrt_m": K_target_finite, "zero_K_rebond_max_target_Pa_sqrt_m": K_target_zero,
            "control_mode": control_mode, "physical_producer_sha": producer_sha,
            "finite_canonical_job_key": finite_row["canonical_job_key"], "zero_canonical_job_key": zero_row["canonical_job_key"],
            "finite_stable_growth": gate_f["stable_growth_provisional"], "zero_stable_growth": gate_z["stable_growth_provisional"],
            "finite_late_to_early_ratio": gate_f["late_to_early_rate_ratio"], "zero_late_to_early_ratio": gate_z["late_to_early_rate_ratio"],
            "finite_developed_da_dN_m_per_cycle": gate_f["developed_da_dN_m_per_cycle"],
            "zero_developed_da_dN_m_per_cycle": gate_z["developed_da_dN_m_per_cycle"],
            "finite_horizon": horizon_f, "zero_horizon": horizon_z,
            "finite_cycle_check": cycle_f, "zero_cycle_check": cycle_z,
            "both_stable_growth": both_stable, "S_h_developed": S_h_developed,
            "finite_n_events": traj_f["n_accepted_events"], "zero_n_events": traj_z["n_accepted_events"],
            "finite_final_extension_um": gate_f["final_extension_um"], "zero_final_extension_um": gate_z["final_extension_um"],
        })

    out = {"schema": "v10230_part_x_px4_developed_pair_analysis_v2", "pairs": pair_analyses}
    (ARTIFACTS_DIR / "px4_developed_pair_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    flat_fields = [
        k for k in pair_analyses[0].keys()
        if k not in ("finite_horizon", "zero_horizon", "finite_cycle_check", "zero_cycle_check")
    ]
    with (ARTIFACTS_DIR / "px4_developed_pair_analysis.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_fields, lineterminator="\n")
        writer.writeheader()
        for row in pair_analyses:
            writer.writerow({k: row[k] for k in flat_fields})

    print(f"Wrote px4_developed_pair_analysis.{{json,csv}}: {len(pair_analyses)} matched pairs")
    n_both_stable = sum(1 for p in pair_analyses if p["both_stable_growth"])
    n_cycle_horizon_hit = sum(1 for p in pair_analyses if p["finite_cycle_check"]["hit_cycle_horizon"] or p["zero_cycle_check"]["hit_cycle_horizon"])
    print(f"  {n_both_stable}/{len(pair_analyses)} pairs have BOTH members stable_growth_provisional=True")
    print(f"  {n_cycle_horizon_hit}/{len(pair_analyses)} pairs have a member that hit the cycle horizon")
    for row in pair_analyses:
        S_h_str = f"{row['S_h_developed']:.4f}" if row["S_h_developed"] is not None else "N/A"
        print(f"  seed={row['seed']} {row['protocol']:30s} Kmax={row['Kmax_Pa_sqrt_m']/1e6:>5.1f}MPa R={row['R']:>6} f={row['frequency_Hz']:>10.3f} "
              f"both_stable={row['both_stable_growth']!s:5} S_h_developed={S_h_str}")


if __name__ == "__main__":
    main()
