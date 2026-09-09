"""Build the focused da/dN-vs-K curve database from qualified, developed,
stationary sources ONLY. No physical simulation is run here; this script
reads two already-computed, committed/archived tables:

  - 1A_PT03_PT08_R_developed_points.csv (codex/v10.2.30-R-ratio-nominal-deltaK
    @ 4b554006d60539d7dec659f4bfe4bf298728780a): A_NATIVE/PT03/PT08 x
    R in {-0.95, 0.1, 0.5} x Kmax in {12,15,18,24} MPa*sqrt(m), all
    PHYSICAL_TARGET_REACHED (or REUSED_PHYSICAL_TARGET_REACHED, a legitimate
    reuse status, not a censor), single seed 1720.

  - px4_stage1_rate_table.csv (codex/v10.2.30-crack-rebonding-part-x @
    30db009ff7172728a6bdc885886f21b94cbec225): D1/D2/D3/D5/D6 rebonding
    protocols and their matched zero-cohesion baselines, seeds 1720 and
    1001723 (confirm), all both_stable_growth=True.

Writes: all_included_curve_points.csv, curve_source_provenance.csv,
plot_selection_manifest.csv (this script) -- global_slope_summary.csv and
local_secant_slopes.csv are written by compute_slopes.py, which consumes
this script's output.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "SOURCE_DATA" / "RAW_INPUTS"
OUT = PKG / "SOURCE_DATA"

R_STUDY_BRANCH = "codex/v10.2.30-R-ratio-nominal-deltaK"
R_STUDY_COMMIT = "4b554006d60539d7dec659f4bfe4bf298728780a"
PARTX_BRANCH = "codex/v10.2.30-crack-rebonding-part-x"
PARTX_COMMIT = "30db009ff7172728a6bdc885886f21b94cbec225"

OPTION_DISPLAY = {
    "A_NATIVE": "A_NATIVE",
    "A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5": "PT03",
    "A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6": "PT08",
}

COLUMNS = [
    "visible_curve_label", "internal_parameterization_id", "campaign", "branch", "commit",
    "source_table", "physical_source_bundle_hash", "Kmax_MPa_sqrt_m", "R", "DeltaK_applied_MPa_sqrt_m",
    "tensile_only_range_MPa_sqrt_m", "temperature_K", "frequency_Hz", "seed", "da_dN_m_per_cycle",
    "developed_window_definition", "stationarity_result", "censor_exclusion_state",
    "parameterization_category", "notes",
]


def _row(**kw) -> dict:
    row = {c: kw.get(c, "") for c in COLUMNS}
    return row


def load_r_pt_study() -> list[dict]:
    rows = []
    with (SRC / "1A_PT03_PT08_R_developed_points.csv").open() as fh:
        for r in csv.DictReader(fh):
            option = r["option"]
            display = OPTION_DISPLAY.get(option, option)
            R = float(r["R"])
            Kmax = float(r["Kmax_MPa_sqrt_m"])
            deltaK_full = (1.0 - R) * Kmax
            tensile_only = Kmax if R < 0 else (Kmax - float(r["Kmin_MPa_sqrt_m"]))
            category = "baseline" if display == "A_NATIVE" else "PT substitution"
            rows.append(_row(
                visible_curve_label=f"{display}, R={R:g}",
                internal_parameterization_id=option,
                campaign="A_NATIVE/PT03/PT08 R-ratio and nominal-DeltaK study",
                branch=r.get("branch", R_STUDY_BRANCH), commit=r.get("head", R_STUDY_COMMIT),
                source_table="1A_PT03_PT08_R_developed_points.csv",
                physical_source_bundle_hash=r.get("composite_hash", ""),
                Kmax_MPa_sqrt_m=Kmax, R=R, DeltaK_applied_MPa_sqrt_m=deltaK_full,
                tensile_only_range_MPa_sqrt_m=tensile_only,
                temperature_K=r.get("temperature_K", ""), frequency_Hz=r.get("frequency_Hz", ""),
                seed=r.get("seed", ""), da_dN_m_per_cycle=float(r["developed_da_dN"]),
                developed_window_definition="stationary-window developed rate "
                                             f"(stationarity_ratio={r.get('stationarity_ratio','')})",
                stationarity_result=r.get("stationarity_classification", ""),
                censor_exclusion_state=r.get("terminal_classification", ""),
                parameterization_category=category,
                notes=f"stage={r.get('stage','')}; job_id={r.get('job_id','')}",
            ))
    return rows


# D-protocol metadata: (display_condition, category)
PROTOCOL_INFO = {
    "D1": ("COMPETING_REVERSIBLE, R=-0.95, 1000 Hz", "rebonding dynamic"),
    "D1_confirm": ("COMPETING_REVERSIBLE, R=-0.95, 1000 Hz (2nd seed)", "rebonding dynamic"),
    "D2": ("COMPETING_REVERSIBLE, R=-0.50, 1000 Hz", "rebonding dynamic"),
    "D2_confirm": ("COMPETING_REVERSIBLE, R=-0.50, 1000 Hz (2nd seed)", "rebonding dynamic"),
    "D3": ("COMPETING_REVERSIBLE, R=-0.50, 316.228 Hz", "rebonding dynamic"),
    "D3_confirm": ("COMPETING_REVERSIBLE, R=-0.50, 316.228 Hz (2nd seed)", "rebonding dynamic"),
    "D5": ("PASSIVATION_LIMITED, R=-0.50, 1000 Hz", "rebonding dynamic"),
    "D5_confirm": ("PASSIVATION_LIMITED, R=-0.50, 1000 Hz (2nd seed)", "rebonding dynamic"),
    "D6_conditional_persistent": ("COMPETING_PERSISTENT, R=-0.50, 316.228 Hz", "rebonding dynamic"),
    "D6_confirm": ("COMPETING_PERSISTENT, R=-0.50, 316.228 Hz (2nd seed)", "rebonding dynamic"),
}


def load_part_x() -> list[dict]:
    with (SRC / "px4_producer_freeze.json").open() as fh:
        px4_bundle_hash = json.load(fh)["physical_source_bundle_sha256"]
    rows = []
    with (SRC / "px4_stage1_rate_table.csv").open() as fh:
        for r in csv.DictReader(fh):
            protocol = r["protocol"]
            cond, category = PROTOCOL_INFO.get(protocol, (protocol, "rebonding dynamic"))
            Kmax_MPa = float(r["Kmax_Pa_sqrt_m"]) / 1.0e6
            R = float(r["R"])
            deltaK_full = (1.0 - R) * Kmax_MPa
            tensile_only = Kmax_MPa if R < 0 else deltaK_full
            # D6_confirm's protocol name is a shortened alias of D6_conditional_persistent
            # (the primary run) in the source table; normalize so both seeds group under
            # one visible_curve_label base instead of splitting into "D6" vs "D6_conditional_persistent".
            base_id = "D6_conditional_persistent" if protocol == "D6_confirm" else protocol.replace("_confirm", "")
            common = dict(
                internal_parameterization_id=f"{protocol}/{r['row_name']}",
                campaign="Crack-rebonding Part X, PX4 stage-1 developed rate study",
                branch=PARTX_BRANCH, commit=PARTX_COMMIT,
                source_table="px4_stage1_rate_table.csv",
                physical_source_bundle_hash=px4_bundle_hash,
                Kmax_MPa_sqrt_m=Kmax_MPa, R=R, DeltaK_applied_MPa_sqrt_m=deltaK_full,
                tensile_only_range_MPa_sqrt_m=tensile_only,
                temperature_K=300.0, frequency_Hz=float(r["frequency_Hz"]), seed=r["seed"],
                developed_window_definition="PX4 stage-1 stationary developed window "
                                             f"(both_stable_growth={r['both_stable_growth']})",
                stationarity_result=r["both_stable_growth"],
                censor_exclusion_state="DEVELOPED_STABLE" if r["both_stable_growth"] == "True" else "EXCLUDED",
            )
            rows.append(_row(
                visible_curve_label=f"{base_id} ({cond})",
                da_dN_m_per_cycle=float(r["finite_developed_da_dN_m_per_cycle"]),
                parameterization_category=category,
                notes=f"finite (cohesive) branch; S_h_developed={r.get('S_h_developed','')}",
                **common,
            ))
            rows.append(_row(
                visible_curve_label=f"{base_id} zero-cohesion baseline",
                da_dN_m_per_cycle=float(r["zero_developed_da_dN_m_per_cycle"]),
                parameterization_category="zero-cohesion control",
                notes=f"matched zero-cohesion baseline for {base_id} at identical Kmax/R/frequency/seed",
                **common,
            ))
    return rows


def main() -> None:
    rows = load_r_pt_study() + load_part_x()
    with (OUT / "all_included_curve_points.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote all_included_curve_points.csv: {len(rows)} rows")

    # curve_source_provenance.csv -- one row per DISTINCT curve (visible_curve_label), summarizing branch/commit/table
    seen = {}
    for r in rows:
        key = r["visible_curve_label"]
        if key not in seen:
            seen[key] = dict(visible_curve_label=key, internal_parameterization_id=r["internal_parameterization_id"],
                              campaign=r["campaign"], branch=r["branch"], commit=r["commit"],
                              source_table=r["source_table"],
                              physical_source_bundle_hash=r["physical_source_bundle_hash"],
                              parameterization_category=r["parameterization_category"],
                              n_points=0)
        seen[key]["n_points"] += 1
    prov_fields = ["visible_curve_label", "internal_parameterization_id", "campaign", "branch", "commit",
                   "source_table", "physical_source_bundle_hash", "parameterization_category", "n_points"]
    with (OUT / "curve_source_provenance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=prov_fields, lineterminator="\n")
        w.writeheader()
        w.writerows(seen.values())
    print(f"Wrote curve_source_provenance.csv: {len(seen)} distinct curves")


if __name__ == "__main__":
    main()
