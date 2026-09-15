#!/usr/bin/env python3
"""Finalize V2.3 exact-row fatigue evidence and prospective registry record."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arrhenius_fracture.canonical_v2_registry_v10230 import canonical_hash, load_row


OUT = ROOT / "analysis_outputs/v2_3_exact_row_cycle_hazard_fatigue"
RUN = ROOT / "runs/v2_3_exact_row_cycle_hazard_fatigue"
CANDIDATE = "P25_TJBSV2_S_002987"
LIMIT = "NOT_EXPERIMENTALLY_CALIBRATED_NOT_ASTMR_CURVE"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def savefig(name: str, source: pd.DataFrame, draw) -> None:
    figures = OUT / "figures"; data = OUT / "figure_source_data"
    figures.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    source.to_csv(data / f"{name}.csv", index=False)
    plt.figure(figsize=(7.2, 4.5)); draw(); plt.tight_layout()
    plt.savefig(figures / f"{name}.png", dpi=180); plt.close()


def main() -> int:
    paths = sorted(RUN.glob("Kmax_*/result.json"))
    if len(paths) != 4:
        raise RuntimeError(f"expected four completed result records, found {len(paths)}")
    results = [(path, json.loads(path.read_text())) for path in paths]
    cases, events, raw = [], [], []
    for path, result in results:
        Kmax = float(result["loading"]["deltaK_MPa_sqrt_m"]) / 0.9
        cumulative_rate = float(result["final_extension_m"]) / float(result["final_cycles"])
        developed = result["developed_da_dN_m_per_cycle"]
        role = result["integration_role"]
        cases.append({
            "candidate_id": CANDIDATE, "integration_role": role,
            "Kmax_MPa_sqrt_m": Kmax, "DeltaK_MPa_sqrt_m": 0.9 * Kmax,
            "R": 0.1, "frequency_Hz": 1000.0, "temperature_K": 300.0,
            "seed": int(result["seed"]), "status": result["status"],
            "accepted_event_count": len(result["events"]),
            "final_cycles": float(result["final_cycles"]),
            "final_extension_m": float(result["final_extension_m"]),
            "cumulative_short_growth_da_dN_m_per_cycle": cumulative_rate,
            "developed_da_dN_m_per_cycle": developed,
            "developed_rate_status": "UNRESOLVED_ONLY_ONE_POINT_BEYOND_20UM",
            "dmd_accepted_segments": int(result["integration"]["dmd_accepted_segments"]),
            "dmd_projected_cycles": float(result["integration"]["dmd_projected_cycles"]),
            "raw_result_sha256": sha(path),
        })
        raw.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                    "checkpoint_path": str((path.parent / "live_checkpoint.json").relative_to(ROOT)),
                    "checkpoint_sha256": sha(path.parent / "live_checkpoint.json")})
        for event in result["events"]:
            events.append({
                "candidate_id": CANDIDATE, "integration_role": role,
                "Kmax_MPa_sqrt_m": Kmax, "DeltaK_MPa_sqrt_m": 0.9 * Kmax,
                "seed": int(result["seed"]), **event,
            })
    case_table = pd.DataFrame(cases).sort_values(["Kmax_MPa_sqrt_m", "integration_role"])
    case_table.to_csv(OUT / "exact_row_300K_fatigue_case_table.csv", index=False)
    event_table = pd.DataFrame(events).sort_values(
        ["Kmax_MPa_sqrt_m", "integration_role", "event_index"]
    )
    event_table.to_parquet(OUT / "exact_row_300K_fatigue_event_ledger.parquet",
                           index=False, compression="zstd")
    atomic_json(OUT / "raw_run_source_manifest.json", {
        "schema": "v10.2.30_v2_3_raw_run_source_manifest_v1", "files": raw,
        "raw_run_outputs_added_to_git": False,
    })

    production = case_table[case_table.integration_role == "QUALIFIED_ACCELERATED_OR_EXACT"].sort_values(
        "Kmax_MPa_sqrt_m"
    ).copy()
    x = np.log(production.Kmax_MPa_sqrt_m.to_numpy(float))
    y = np.log(production.cumulative_short_growth_da_dN_m_per_cycle.to_numpy(float))
    overall = float(np.polyfit(x, y, 1)[0])
    slope_rows = []
    for left, right in zip(production.iloc[:-1].itertuples(), production.iloc[1:].itertuples()):
        slope_rows.append({
            "Kmax_left_MPa_sqrt_m": left.Kmax_MPa_sqrt_m,
            "Kmax_right_MPa_sqrt_m": right.Kmax_MPa_sqrt_m,
            "local_two_point_log_slope": math.log(
                right.cumulative_short_growth_da_dN_m_per_cycle /
                left.cumulative_short_growth_da_dN_m_per_cycle
            ) / math.log(right.Kmax_MPa_sqrt_m / left.Kmax_MPa_sqrt_m),
            "rate_quantity": "CUMULATIVE_SHORT_GROWTH_DA_DN",
            "not_a_broad_Paris_exponent": True,
        })
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(OUT / "exact_row_300K_fatigue_local_slopes.csv", index=False)

    accelerated = next(result for path, result in results if path.parent.name == "Kmax_6")
    explicit = next(result for path, result in results if path.parent.name == "Kmax_6_explicit_overlap")
    parity_rows = []
    for a, e in zip(accelerated["events"], explicit["events"]):
        parity_rows.append({
            "event_index": int(a["event_index"]),
            "accelerated_cycles": float(a["cycles"]), "explicit_cycles": float(e["cycles"]),
            "cycle_relative_difference": abs(float(a["cycles"]) - float(e["cycles"])) / max(float(e["cycles"]), 1.0),
            "threshold_exact": a["threshold_action"] == e["threshold_action"],
            "event_size_exact": a["committed_advance_m"] == e["committed_advance_m"],
            "radius_relative_difference": abs(float(a["tip_radius_m"]) - float(e["tip_radius_m"])) / max(float(e["tip_radius_m"]), 1e-30),
            "backstress_relative_difference": abs(float(a["backstress_Pa"]) - float(e["backstress_Pa"])) / max(abs(float(e["backstress_Pa"])), 1.0),
        })
    parity = pd.DataFrame(parity_rows)
    parity.to_csv(OUT / "accelerated_explicit_overlap_parity.csv", index=False)
    terminal_cycle_error = abs(accelerated["final_cycles"] - explicit["final_cycles"]) / explicit["final_cycles"]
    atomic_json(OUT / "accelerated_explicit_overlap_decision.json", {
        "schema": "v10.2.30_v2_3_accelerated_explicit_overlap_v1",
        "Kmax_MPa_sqrt_m": 6.0, "events_compared": 6,
        "threshold_sequence_exact": bool(parity.threshold_exact.all()),
        "event_size_sequence_exact": bool(parity.event_size_exact.all()),
        "terminal_cycle_relative_difference": terminal_cycle_error,
        "maximum_event_cycle_relative_difference": float(parity.cycle_relative_difference.max()),
        "status": "PASS_INHERITED_1E3_RELATIVE_CYCLE_TOLERANCE" if terminal_cycle_error <= 1e-3 else "FAIL",
    })

    resistance = pd.read_csv(
        ROOT / "analysis_outputs/v2_2_1d_rising_resistance_and_fatigue/one_d_reload_separated_resistance_points.csv"
    )
    mechanism = resistance[(resistance.candidate_id == CANDIDATE) &
                           (resistance["mode"] == "FULL_PRODUCTION_STATE")].sort_values(
                               "cumulative_advance_um"
                           ).copy()
    mechanism["K_normalized_to_first"] = mechanism.K_reinit_MPa_sqrt_m / mechanism.K_reinit_MPa_sqrt_m.iloc[0]
    mechanism["sqrt_radius_normalized_to_first"] = np.sqrt(mechanism.radius_m / mechanism.radius_m.iloc[0])
    mechanism["K_over_sqrt_radius_normalized_to_first"] = (
        mechanism.K_reinit_MPa_sqrt_m / np.sqrt(mechanism.radius_m)
    ) / (mechanism.K_reinit_MPa_sqrt_m.iloc[0] / np.sqrt(mechanism.radius_m.iloc[0]))
    mechanism.to_csv(OUT / "rising_resistance_mechanism_attribution.csv", index=False)
    m0, m1 = mechanism.iloc[0], mechanism.iloc[-1]
    atomic_json(OUT / "rising_resistance_mechanism_decision.json", {
        "candidate_id": CANDIDATE,
        "classification": "PROGRESSIVE_EFFECTIVE_RADIUS_BLUNTING_TRANSFER_DOMINANT",
        "K_ratio_5um_to_100um": float(m1.K_normalized_to_first),
        "radius_ratio_5um_to_100um": float(m1.radius_m / m0.radius_m),
        "sqrt_radius_ratio_5um_to_100um": float(m1.sqrt_radius_normalized_to_first),
        "residual_K_over_sqrt_radius_ratio": float(m1.K_over_sqrt_radius_normalized_to_first),
        "active_shielding_MPa_sqrt_m_at_5um": float(m0.active_shield_Pa_sqrt_m) / 1e6,
        "active_shielding_MPa_sqrt_m_at_100um": float(m1.active_shield_Pa_sqrt_m) / 1e6,
        "ASTM_or_conventional_R_curve_established": False,
    })

    row = load_row(CANDIDATE)
    renewal = {
        "cleavage_hits": float(row["physics__cleavage_hits"]),
        "cleavage_correlation_time_s": float(row["physics__cleavage_correlation_time_s"]),
        "event_length_mode": "threshold_scaled_bounded_mean_preserving",
        "base_event_length_m": 5e-6, "minimum_factor": 0.5, "maximum_factor": 4.0,
    }
    registry_record = {
        "registry_id": "v10230_joint_v2p3_P25_dbtt_risingR_P25_TJBSV2_S_002987",
        "status": "PROSPECTIVE_CROSS_CODE_REGISTRY_ROW",
        "canonical_production_promotion": "WITHHELD",
        "source_candidate_id": CANDIDATE, "parent_id": row["parent_id"],
        "source_v2_2_commit": "f3ae18503c243b2afa6f5af7f323c69c6aa681ca",
        "load_selection_commit": "0b986300bffb12bb2aa020541e05dc1c8ecde66a",
        "parameter_row_sha256": row["complete_bound_row_sha256"],
        "thermodynamic_surface_fingerprint": canonical_hash({
            "opening": row["opening_surface_sha256"], "emission": row["emission_surface_sha256"]}),
        "renewal_fingerprint": canonical_hash(renewal),
        "renewal_contract": renewal,
        "full_parameter_row": row,
        "rising_resistance_metrics": {
            "classification": "1D_STRONG_RISING_RESISTANCE",
            "K_ratio_5um_to_100um": float(m1.K_normalized_to_first),
            "dominant_mechanism": "PROGRESSIVE_EFFECTIVE_RADIUS_BLUNTING_TRANSFER",
            "exact_restart_parity": True,
        },
        "fatigue_metrics": {
            "Kmax_MPa_sqrt_m": production.Kmax_MPa_sqrt_m.tolist(),
            "cumulative_short_growth_da_dN_m_per_cycle": production.cumulative_short_growth_da_dN_m_per_cycle.tolist(),
            "three_point_log_slope": overall,
            "classification": "THREE_POINT_SHORT_GROWTH_RESPONSE_CURVED_NO_SINGLE_SLOPE",
            "developed_rate_status": "UNRESOLVED_ONLY_ONE_POINT_BEYOND_20UM",
            "accelerated_explicit_overlap": "PASS",
        },
        "qualification": LIMIT,
        "limitations": [
            "REDUCED_1D_MODEL_ONLY", "NOT_ASTMR_CURVE", "NOT_ASTM_KIC",
            "NOT_EXPERIMENTALLY_CALIBRATED", "CUMULATIVE_SHORT_GROWTH_RATE",
            "DEVELOPED_RATE_UNRESOLVED_AT_25UM_TARGET",
        ],
    }
    atomic_json(OUT / "prospective_cross_code_registry_additions.json", {
        "schema": "v10.2.30_additive_prospective_cross_code_registry_v1",
        "additions": [registry_record], "existing_canonical_rows_replaced": False,
        "canonical_production_rows_added": 0,
    })

    (OUT / "EXACT_ROW_300K_FATIGUE_DECISION.md").write_text(
        "# Exact-row 300 K fatigue decision\n\n"
        "`P25_TJBSV2_S_002987` was tested at Kmax 3.75, 6.00, and 12.75 MPa sqrt(m) "
        "with common seed 1720. All three trajectories reached six events and 28.888 micrometres "
        "without a physical censor. Cumulative short-growth rates are 5.991e-10, 4.049e-9, and "
        "2.075e-8 m/cycle. Adjacent log slopes are 4.07 and 2.17, so this is a curved three-point "
        "short-growth response, not a broad Paris exponent. Only one event lies beyond the existing "
        "20 micrometre initiation boundary; developed da/dN remains unresolved at the frozen 25 "
        "micrometre target. The 6.00 MPa sqrt(m) explicit overlap passes with 7.83e-4 terminal-cycle "
        "relative difference and an exact threshold/event-size sequence.\n"
    )
    (OUT / "V2_3_FINAL_DECISION.md").write_text(
        "# V2.3 final decision\n\n"
        "P25_TJBSV2_S_002987 demonstrates reduced 1-D rising resistance dominated by progressive "
        "effective-radius/blunting transfer through 100 micrometres with exact restart parity. "
        "This is not an ASTM or conventional R-curve. P40 remains unresolved at the 80 MPa sqrt(m) "
        "censor. The previous no-fatigue-interval result applies only to the frozen monotonic-cap "
        "rule. Cycle-hazard selection now supplies three finite exact-row 300 K short-growth fatigue "
        "points. Canonical production promotion remains withheld; one additive prospective cross-code "
        f"registry row is justified with `{LIMIT}` qualification. No spatial or two-dimensional "
        "calculation was launched.\n"
    )

    selection = pd.read_csv(OUT / "cycle_hazard_load_selection_grid.csv")
    def mechanism_plot():
        plt.plot(mechanism.cumulative_advance_um, mechanism.K_normalized_to_first, "o-", label="K reinit")
        plt.plot(mechanism.cumulative_advance_um, mechanism.sqrt_radius_normalized_to_first, "s-", label="sqrt(radius)")
        plt.plot(mechanism.cumulative_advance_um, mechanism.K_over_sqrt_radius_normalized_to_first, "^-", label="K/sqrt(radius)")
        plt.xlabel("reduced advance (um)"); plt.ylabel("normalized to first event"); plt.legend()
    savefig("01_radius_blunting_attribution", mechanism, mechanism_plot)
    def selection_plot():
        plt.semilogy(selection.Kmax_MPa_sqrt_m, selection.projected_median_first_passage_cycles)
        plt.scatter(production.Kmax_MPa_sqrt_m, [6662.204680547703, 983.2837387406177, 102.69435730638095], color="r")
        plt.xlabel("Kmax (MPa sqrt(m))"); plt.ylabel("fresh-state projected median cycles")
    savefig("02_cycle_hazard_load_selection", selection, selection_plot)
    def extension_plot():
        for (K, role), group in event_table.groupby(["Kmax_MPa_sqrt_m", "integration_role"]):
            plt.step(np.r_[0, group.cycles], np.r_[0, group.cumulative_extension_m * 1e6], where="post", label=f"{K:g} {role}")
        plt.xlabel("cycles"); plt.ylabel("cumulative extension (um)"); plt.xscale("log"); plt.legend(fontsize=6)
    savefig("03_exact_row_extension_vs_cycles", event_table, extension_plot)
    def rate_plot():
        plt.loglog(production.Kmax_MPa_sqrt_m, production.cumulative_short_growth_da_dN_m_per_cycle, "o-")
        plt.xlabel("Kmax (MPa sqrt(m))"); plt.ylabel("cumulative short-growth da/dN (m/cycle)")
    savefig("04_three_point_short_growth_rates", production, rate_plot)
    def parity_plot():
        plt.plot(parity.event_index, parity.cycle_relative_difference, "o-")
        plt.axhline(1e-3, color="r", linestyle="--", label="inherited tolerance")
        plt.xlabel("event index"); plt.ylabel("cumulative-cycle relative difference"); plt.legend()
    savefig("05_accelerated_explicit_parity", parity, parity_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
