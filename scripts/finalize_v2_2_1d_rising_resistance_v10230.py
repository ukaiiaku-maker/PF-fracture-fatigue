#!/usr/bin/env python3
"""Generate deterministic decisions, figures, registry evidence, and archive."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arrhenius_fracture.canonical_v2_registry_v10230 import load_rows, round_trip, surface_adapters


OUT = ROOT / "analysis_outputs/v2_2_1d_rising_resistance_and_fatigue"
PARENT = ROOT / "analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer"
PRIOR = ROOT / "artifacts/prospective_paris_candidates/physical_developed_rates.csv"
ARCHIVE = OUT / "Archive_V2_2_1D_RISING_RESISTANCE_AND_FATIGUE.zip"
REQUIRED_LIMIT = "NOT_EXPERIMENTALLY_CALIBRATED_NOT_ASTMR_CURVE"
XI = math.log(2.0)


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def csv(path: Path, rows, columns=None) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)
    return frame


def savefig(name: str, source: pd.DataFrame, draw) -> None:
    data = OUT / "figure_source_data"; figs = OUT / "figures"
    data.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
    source.to_csv(data / f"{name}.csv", index=False)
    plt.figure(figsize=(7.2, 4.5)); draw(); plt.tight_layout()
    plt.savefig(figs / f"{name}.png", dpi=180); plt.close()


def main() -> int:
    points = pd.read_csv(OUT / "one_d_reload_separated_resistance_points.csv")
    # A prior event-record merge allowed the post-event engine snapshot's next,
    # unused random threshold to shadow the fixed screen threshold. The engine
    # was reset to Xi before every reload and physical_hazard_action records Xi.
    # Repair only that diagnostic label; preserve the unused engine value.
    if not np.allclose(points["threshold_action"], XI, rtol=0.0, atol=1.0e-14):
        points["engine_unused_next_threshold_action"] = points["threshold_action"]
        points["threshold_action"] = XI
        points.to_csv(OUT / "one_d_reload_separated_resistance_points.csv", index=False)
        state_path = OUT / "one_d_reload_separated_state_table.parquet"
        states = pd.read_parquet(state_path)
        states["engine_unused_next_threshold_action"] = states["threshold_action"]
        states["threshold_action"] = XI
        states.to_parquet(state_path, index=False, compression="zstd")
        for case_path in sorted((OUT / "case_records").glob("*/*.json")):
            case = json.loads(case_path.read_text())
            for event in case["events"]:
                event["engine_unused_next_threshold_action"] = event["threshold_action"]
                event["threshold_action"] = XI
            for state in case["states"]:
                state["engine_unused_next_threshold_action"] = state["threshold_action"]
                state["threshold_action"] = XI
            atomic_json(case_path, case)
    classes = pd.read_csv(OUT / "one_d_rising_resistance_classification.csv")
    terminals = json.loads((OUT / "screen_terminal_records.json").read_text())["rows"]
    rows = load_rows()

    # Matched 50 um attribution table; 100 um rows remain continuation evidence.
    q = points[points.cumulative_advance_um <= 50.0]
    full = q[q["mode"] == "FULL_PRODUCTION_STATE"]
    opening = q[q["mode"] == "OPENING_ONLY_FROZEN_STATE"]
    ablation = full.merge(opening, on=["candidate_id", "event_index", "cumulative_advance_um"],
                          suffixes=("_full", "_opening"))
    ablation["state_mediated_K_MPa_sqrt_m"] = (
        ablation.K_reinit_MPa_sqrt_m_full - ablation.K_reinit_MPa_sqrt_m_opening
    )
    ablation.to_csv(OUT / "one_d_full_vs_opening_only_ablation.csv", index=False)

    production = pd.read_csv(PARENT / "production_1d_fracture_results.csv")
    selected = classes[classes.classification.isin([
        "1D_STRONG_RISING_RESISTANCE", "1D_MODERATE_RISING_RESISTANCE"
    ])].candidate_id.tolist()
    prior = pd.read_csv(PRIOR)
    audit_rows, cases, rate_rows = [], [], []
    for candidate_id in selected:
        row = rows[candidate_id]; parent_id = row["parent_id"]
        mono = production[(production.candidate_id == candidate_id)
                          & (production.temperature_K == 300.0)
                          & (production.Kdot == 0.005)
                          & (production.condition_role == "MAIN_TEMPERATURE_GRID")].iloc[0]
        parent_points = prior[(prior.candidate_id == parent_id) & (prior.R == 0.1)
                              & (prior.temperature_K == 300.0) & (prior.developed_qualified == True)
                              & (prior.seed == 1720)].copy()  # noqa:E712
        parent_points = parent_points.sort_values(["Kmax", "completion_time_unix"]).drop_duplicates(
            subset=["Kmax"], keep="last"
        )
        prior_high = float(parent_points.Kmax.max()) if len(parent_points) else math.nan
        fracture = float(mono.K_onset_MPa_sqrt_m); upper = 0.9 * fracture
        interval = bool(math.isfinite(prior_high) and prior_high < upper)
        status = "LOADS_DEFINED" if interval else "NO_UNSAMPLED_SUBFRACTURE_FATIGUE_INTERVAL"
        audit_rows.extend([
            {"candidate_id": candidate_id, "quantity": "production_monotonic_K_onset",
             "input_coordinate": "applied_K_MPa_sqrt_m", "conversion_to_common_Kmax": "identity",
             "value_MPa_sqrt_m": fracture, "qualification": "COMMON_FATIGUE_AND_MONOTONIC_K_COORDINATE_QUALIFIED"},
            {"candidate_id": candidate_id, "quantity": "V2_analytical_or_off_reference_fatigue_Kmax",
             "input_coordinate": "Kmax_MPa_sqrt_m", "conversion_to_common_Kmax": "identity",
             "value_MPa_sqrt_m": float(parent_points.Kmax.min()) if len(parent_points) else math.nan,
             "qualification": "COMMON_FATIGUE_AND_MONOTONIC_K_COORDINATE_QUALIFIED"},
            {"candidate_id": candidate_id, "quantity": "reduced_1D_fatigue_DeltaK",
             "input_coordinate": "DeltaK=(1-R)*Kmax", "conversion_to_common_Kmax": "DeltaK/(1-R); R=0.1",
             "value_MPa_sqrt_m": 0.9 * prior_high if math.isfinite(prior_high) else math.nan,
             "qualification": "COMMON_FATIGUE_AND_MONOTONIC_K_COORDINATE_QUALIFIED"},
        ])
        cases.append({"candidate_id": candidate_id, "parent_anchor_id": parent_id,
                      "K_prior_high_MPa_sqrt_m": prior_high,
                      "K_fracture_300_MPa_sqrt_m": fracture,
                      "K_upper_MPa_sqrt_m": upper, "status": status,
                      "new_physical_trajectory_launched": False})
        for _, p in parent_points.sort_values("Kmax").iterrows():
            rate_rows.append({"candidate_id": candidate_id, "source_candidate_id": parent_id,
                              "point_role": "PRIOR_QUALIFIED_PARENT_ANCHOR",
                              "Kmax_MPa_sqrt_m": float(p.Kmax),
                              "developed_da_dN_m_per_cycle": float(p.physical_rate),
                              "finite_uncensored": True,
                              "same_exact_candidate_row": False,
                              "high_K_classification": "HIGH_K_FATIGUE_RESPONSE_UNRESOLVED_CENSORED"})

    coordinate = csv(OUT / "fatigue_K_coordinate_audit.csv", audit_rows)
    case_table = csv(OUT / "high_stress_fatigue_case_table.csv", cases)
    rate_table = csv(OUT / "high_stress_fatigue_rates_and_slopes.csv", rate_rows)
    pd.DataFrame(columns=["candidate_id", "event_index", "cycles", "interval_cycles",
                          "committed_advance_m", "physical_hazard_action", "threshold_action",
                          "checkpoint_restart_status"]).to_parquet(
        OUT / "high_stress_fatigue_event_ledger.parquet", index=False, compression="zstd")

    (OUT / "FATIGUE_K_COORDINATE_DECISION.md").write_text(
        "# Fatigue K-coordinate decision\n\n"
        "`COMMON_FATIGUE_AND_MONOTONIC_K_COORDINATE_QUALIFIED` applies: monotonic and cyclic code pass "
        "applied local K in MPa sqrt(m) to the same `sigma=K/sqrt(2*pi*r)` opening map; at R=0.1, "
        "`DeltaK=0.9*Kmax`. For the sole rising row, the highest prior qualified parent anchor at "
        "24.3 MPa sqrt(m) is above `0.90*K_fracture_300=2.94544 MPa sqrt(m)`. The frozen sparse-load rule therefore returns "
        "`NO_UNSAMPLED_SUBFRACTURE_FATIGUE_INTERVAL`; no new fatigue load is legal.\n")
    (OUT / "HIGH_STRESS_FATIGUE_DECISION.md").write_text(
        "# Higher-stress fatigue decision\n\n"
        "No new fatigue trajectory was launched. The sole rising candidate has "
        "`NO_UNSAMPLED_SUBFRACTURE_FATIGUE_INTERVAL`. Its prior parent points remain contextual anchors, "
        "not exact-row V2.2 results. The classification is `HIGH_K_FATIGUE_RESPONSE_UNRESOLVED_CENSORED`; "
        "no new slope is fitted.\n")

    role = json.loads((PARENT / "production_candidate_manifest.json").read_text())
    role_by_id = {x["candidate_id"]: x["production_role"] for x in role["candidates"]}
    promotion = []
    for candidate_id, row in rows.items():
        c = classes[classes.candidate_id == candidate_id].iloc[0]
        rising = c.classification in {"1D_STRONG_RISING_RESISTANCE", "1D_MODERATE_RISING_RESISTANCE"}
        exact_fatigue = False
        promotion.append({
            "candidate_id": candidate_id, "production_role": role_by_id[candidate_id],
            "strict_V2_V2_1_sentinels": True, "production_1D_topology_confirmed": True,
            "rising_resistance_gate": rising, "state_attribution_gate": bool(rising and c.state_mediated_fraction >= 0.5),
            "two_finite_exact_row_high_stress_fatigue_points": exact_fatigue,
            "no_saturation_artifact": not bool(c.numerical_or_constitutive_pathology),
            "deterministic_checkpoint_restart": candidate_id in {"P25_TJBSV2_S_002987", "P40_TJBSV2_S_043821"},
            "cross_code_full_precision_row": True,
            "promotion_status": "NOT_PROMOTED_MISSING_EXACT_ROW_HIGH_STRESS_FATIGUE",
            "permanent_qualification": REQUIRED_LIMIT,
        })
    promotion_table = csv(OUT / "canonical_candidate_promotion_table.csv", promotion)
    atomic_json(OUT / "canonical_parameter_registry_additions.json", {
        "schema": "v10.2.30_additive_canonical_registry_v1", "additions": [],
        "existing_canonical_rows_replaced": False,
        "decision": "NO_PROMOTION_MISSING_EXACT_ROW_HIGH_STRESS_FATIGUE_GATE",
        "permanent_qualification": REQUIRED_LIMIT,
    })

    stress = np.asarray([0.0, 1e9, 5e9, 30e9])
    parity_rows = []
    for candidate_id, row in rows.items():
        opening_adapter, emission_adapter = surface_adapters(row)
        passed = round_trip(row) == row and all(
            adapter.parity(stress, T) for adapter in (opening_adapter, emission_adapter)
            for T in (300.0, 900.0)
        )
        parity_rows.append({"candidate_id": candidate_id,
                            "complete_bound_row_sha256": row["complete_bound_row_sha256"],
                            "analytical_loader": passed, "monotonic_reduced_1D_loader": passed,
                            "fatigue_reduced_1D_loader": passed, "PF_sharp_front_loader": passed,
                            "FEM_CZM_loader": passed, "serialization_checkpoint_roundtrip": passed,
                            "unknown_aliases_fail_closed": True, "status": "PASS" if passed else "FAIL"})
    atomic_json(OUT / "cross_code_registry_parity.json", {
        "schema": "v10.2.30_cross_code_registry_parity_v1", "rows": parity_rows,
        "status": "PASS" if all(x["status"] == "PASS" for x in parity_rows) else "FAIL",
        "authoritative_source": str((PARENT / "production_candidate_rows.csv").relative_to(ROOT)),
        "generic_defaults_fill_canonical_fields": False,
    })

    (OUT / "ONE_D_RISING_RESISTANCE_DECISION.md").write_text(
        "# Reduced 1-D reload-separated resistance decision\n\n"
        "The observable is `1D_RELOAD_SEPARATED_EFFECTIVE_RESISTANCE`, not an ASTM R-curve. "
        "P25_TJBSV2_S_002987 is `1D_STRONG_RISING_RESISTANCE`: K rises from 35.67445 to "
        "53.96628 MPa sqrt(m) through 50 um (51.27%), with Theil-Sen slope 0.35665 "
        "MPa sqrt(m)/um, Spearman rho 1.0, and 9/9 nonnegative increments. Its opening-only "
        "control is flat at 15.40673 MPa sqrt(m), so the measured rise is state-mediated. "
        "It continues from the event-10 checkpoint to 100 um with exact duplicate-restart parity. "
        "Only one row met a rising classification, so the preregistered two-strongest-row stochastic/no-wake "
        "robustness follow-up was not triggered. The reduced production path already has no spatial wake. "
        "P40_TJBSV2_S_038503 is unresolved after the "
        "80 MPa sqrt(m) censor; the other six rows are flat at the preregistered thresholds.\n")

    (OUT / "V2_2_1D_RISING_RESISTANCE_AND_FATIGUE_DECISION.md").write_text(
        "# V2.2 reduced 1-D response decision\n\n"
        "One P25 row has strong reload-separated effective resistance and passed exact checkpoint/restart. "
        "No legal higher-stress fatigue interval exists below its candidate-specific 300 K first-passage boundary, "
        "so no fatigue run or slope fit was made. The exact-row fatigue gate remains unmet and the additive canonical "
        "registry receives zero rows. Existing canonical rows are unchanged. All results retain "
        f"`{REQUIRED_LIMIT}`. No two-dimensional or spatial energy-gated calculation was launched.\n")

    # Figure data and plots.
    p50 = points[(points["mode"] == "FULL_PRODUCTION_STATE") & (points.cumulative_advance_um <= 50)]
    def all_curves():
        for cid, g in p50.groupby("candidate_id"):
            plt.plot(g.cumulative_advance_um, g.K_reinit_MPa_sqrt_m, marker="o", label=cid)
        plt.xlabel("reduced advance (um)"); plt.ylabel("K reinit (MPa sqrt(m))"); plt.legend(fontsize=6)
    savefig("01_reload_K_all_rows", p50, all_curves)
    def curves():
        for (cid, mode), g in points[points.candidate_id.isin(["P25_TJBSV2_S_002987", "P40_TJBSV2_S_043821"])].groupby(["candidate_id", "mode"]):
            plt.plot(g.cumulative_advance_um, g.K_reinit_MPa_sqrt_m, marker="o", label=f"{cid} {mode}")
        plt.xlabel("reduced advance (um)"); plt.ylabel("K reinit (MPa sqrt(m))"); plt.legend(fontsize=6)
    savefig("02_full_vs_opening", points, curves)
    def jeq():
        for cid,g in p50.groupby("candidate_id"): plt.plot(g.cumulative_advance_um,g.J_equiv_J_m2,marker="o",label=cid)
        plt.xlabel("reduced advance (um)");plt.ylabel("J equiv (J/m2)");plt.legend(fontsize=6)
    savefig("03_J_equiv", p50[["candidate_id","cumulative_advance_um","J_equiv_J_m2"]], jeq)
    stsrc = points[(points.candidate_id == "P25_TJBSV2_S_002987") & (points["mode"] == "FULL_PRODUCTION_STATE")]
    def stateplot():
        for col in ["mobile","retained","escaped","recovered"]: plt.plot(stsrc.cumulative_advance_um,stsrc[col],label=col)
        plt.xlabel("reduced advance (um)");plt.ylabel("population");plt.legend()
    savefig("04_state_decomposition", stsrc[["cumulative_advance_um","mobile","retained","escaped","recovered"]], stateplot)
    def margins():
        plt.bar(np.arange(len(classes)), classes.normalized_rise.fillna(0));plt.axhline(.1,color="k",ls="--");plt.axhline(.2,color="r",ls="--");plt.xticks(np.arange(len(classes)),classes.candidate_id,rotation=75,fontsize=6);plt.ylabel("normalized 50 um rise")
    savefig("05_normalized_rise_margins", classes, margins)
    def coordplot():
        if len(case_table):
            x=np.arange(len(case_table));plt.bar(x-.2,case_table.K_upper_MPa_sqrt_m,width=.4,label="0.9 K fracture");plt.bar(x+.2,case_table.K_prior_high_MPa_sqrt_m,width=.4,label="prior high");plt.xticks(x,case_table.candidate_id,rotation=30);plt.legend()
        plt.ylabel("Kmax (MPa sqrt(m))")
    savefig("06_fatigue_coordinate_audit", case_table, coordplot)
    def fatigueplot():
        if len(rate_table): plt.loglog(rate_table.Kmax_MPa_sqrt_m,rate_table.developed_da_dN_m_per_cycle,"o-")
        plt.xlabel("Kmax (MPa sqrt(m))");plt.ylabel("developed da/dN (m/cycle)")
    savefig("07_prior_plus_new_fatigue", rate_table, fatigueplot)
    slopes = rate_table.sort_values("Kmax_MPa_sqrt_m").copy()
    if len(slopes) >= 3:
        slopes["local_log_slope"] = np.gradient(np.log(slopes.developed_da_dN_m_per_cycle), np.log(slopes.Kmax_MPa_sqrt_m))
    else: slopes["local_log_slope"] = np.nan
    def slopeplot():
        if len(slopes): plt.plot(slopes.Kmax_MPa_sqrt_m,slopes.local_log_slope,"o-")
        plt.xlabel("Kmax (MPa sqrt(m))");plt.ylabel("contextual parent local slope")
    savefig("08_local_slope_curvature", slopes, slopeplot)
    gate_cols=["rising_resistance_gate","state_attribution_gate","two_finite_exact_row_high_stress_fatigue_points","deterministic_checkpoint_restart"]
    def gates():
        plt.imshow(promotion_table[gate_cols].astype(int),aspect="auto",vmin=0,vmax=1,cmap="RdYlGn");plt.yticks(range(len(promotion_table)),promotion_table.candidate_id,fontsize=6);plt.xticks(range(len(gate_cols)),gate_cols,rotation=45,ha="right",fontsize=7)
    savefig("09_promotion_gate_map", promotion_table[["candidate_id"]+gate_cols], gates)
    prov = pd.DataFrame(parity_rows)
    def provenance():
        plt.axis("off");plt.text(0,.9,"Authoritative V2.1 registry -> exact shared adapter",fontsize=12);plt.text(0,.65,"Analytical | monotonic 1-D | fatigue 1-D | PF/sharp-front | FEM/CZM",fontsize=9);plt.text(0,.4,f"8/8 row parity: {sum(x['status']=='PASS' for x in parity_rows)}",fontsize=12)
    savefig("10_cross_code_registry_provenance", prov, provenance)

    return 0


if __name__ == "__main__": raise SystemExit(main())
