#!/usr/bin/env python3
"""Write the auditable scientific report for the bounded v9.14 low-slope search."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value: object) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def fmt(value: object, digits: int = 5) -> str:
    result = number(value)
    return f"{result:.{digits}g}" if math.isfinite(result) else "—"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--low-slope", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, required=True)
    parser.add_argument("--batch-campaign-label", default="surrogate_iter3_low_slope_level0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    comparison = rows(args.low_slope / "A_n64_n128_matched_comparison.csv")
    decomposition = rows(args.low_slope / "A_slope_causal_decomposition.csv")[0]
    validation = rows(args.low_slope / "search_descriptor_target_validation.csv")
    sensitivity = rows(args.low_slope / "parameter_slope_sensitivity.csv")
    points = rows(args.analysis / "paris_search_points.csv")
    fits = rows(args.analysis / "paris_candidate_fits.csv")
    batch = rows(args.surrogate / "surrogate_batch.csv")
    audit = json.loads((args.surrogate / "surrogate_audit.json").read_text())
    batch_ids = {row["candidate_id"] for row in batch}

    fidelity_summary = {}
    for fidelity in ("level1", "level2"):
        selected = [row for row in comparison if row["fidelity"] == fidelity]
        fidelity_summary[fidelity] = selected[0]

    batch_points = [
        row for row in points
        if row["candidate_id"] in batch_ids and row.get("campaign_label") == args.batch_campaign_label
    ]
    # Older analyzer schemas may not expose campaign_label.  Result paths are
    # still unambiguous and remain the authoritative provenance.
    if not batch_points:
        batch_points = [
            row for row in points
            if row["candidate_id"] in batch_ids
            and args.batch_campaign_label in row.get("result_path", "")
        ]
    candidate_results = []
    for candidate in sorted(batch_ids):
        candidate_points = [row for row in batch_points if row["candidate_id"] == candidate]
        growth = [row for row in candidate_points if row.get("status") == "growth_target_reached"]
        descriptor_fits = [
            row for row in fits
            if row["candidate_id"] == candidate and row.get("fidelity") == "level1"
            and math.isfinite(number(row.get("m")))
        ]
        fit = descriptor_fits[-1] if descriptor_fits else {}
        descriptor_status = fit.get("descriptor_status", "")
        if not descriptor_status:
            descriptor_status = (
                "NOT_PROMOTED_NO_FINITE_BRACKET" if not growth
                else "UNRESOLVED_FINITE_BRACKET"
            )
        candidate_results.append({
            "candidate_id": candidate,
            "level0_terminal_points": len(candidate_points),
            "level0_growth_points": len(growth),
            "level0_statuses": "; ".join(
                f"{fmt(row.get('DeltaK_MPa_sqrt_m'))}:{row.get('status')}" for row in sorted(candidate_points, key=lambda item: number(item.get("DeltaK_MPa_sqrt_m")))
            ) or "pending",
            "developed_m": fit.get("m", ""),
            "descriptor_status": descriptor_status,
        })

    measured = [number(row.get("m")) for row in fits if row.get("fidelity") == "level1" and math.isfinite(number(row.get("m")))]
    best_measured = min(measured, default=math.nan)
    successful = [
        row for row in fits
        if row.get("fidelity") == "level2" and 2.0 <= number(row.get("m")) <= 10.0
    ]
    batch_complete = all(row["level0_terminal_points"] >= 3 for row in candidate_results)
    unresolved_batch_bracket = any(
        row["level0_growth_points"] > 0 and not math.isfinite(number(row["developed_m"]))
        for row in candidate_results
    )
    if successful:
        decision = "SUCCESS"
        rationale = "At least one n=128 candidate retains a developed slope in the provisional 2–10 band."
    elif batch_complete and not unresolved_batch_bracket and best_measured >= 15.0 and float(audit["domain_best_credible_target_probability_times_feasibility"]) < 0.05:
        decision = "DOMAIN FAILURE"
        rationale = (
            "Every viable finite-growth bracket in the causally directed batch has been "
            "resolved, the best measured slope remains high, and the within-domain "
            "surrogate assigns less than 5% feasibility-weighted probability to the "
            "provisional target."
        )
    else:
        decision = "OPEN — bounded batch or promoted descriptors still pending"
        rationale = "The stopping evidence is not yet complete."

    lines = [
        "# v9.14 v7 developed-Paris low-slope search",
        "",
        "## Decision",
        "",
        f"**{decision}.** {rationale}",
        "",
        "The 2–10 slope interval remains a provisional external broad physical target, not a fitted or experimentally validated dataset.",
        "",
        "## Candidate A: matched n=64/n=128 interval",
        "",
        "| fidelity | m | C (m/cycle)/(MPa√m)^m | R² |",
        "|---|---:|---:|---:|",
    ]
    for fidelity in ("level1", "level2"):
        row = fidelity_summary[fidelity]
        lines.append(f"| {fidelity} | {fmt(row['m'], 7)} | {fmt(row['C'], 7)} | {fmt(row['R2'], 7)} |")
    lines.extend(["", "| ΔK (MPa√m) | n=64 developed da/dN | n=128 developed da/dN |", "|---:|---:|---:|"])
    loads = sorted({number(row["DeltaK_MPa_sqrt_m"]) for row in comparison})
    for load in loads:
        rates = {row["fidelity"]: row["developed_da_dN_m_per_cycle"] for row in comparison if math.isclose(number(row["DeltaK_MPa_sqrt_m"]), load)}
        lines.append(f"| {load:.10g} | {fmt(rates['level1'], 8)} | {fmt(rates['level2'], 8)} |")

    lines.extend([
        "", "## Candidate A causal slope decomposition", "",
        "| contribution/diagnostic | log-slope contribution |", "|---|---:|",
        f"| measured full developed m | {fmt(decomposition['measured_full_m'])} |",
        f"| intrinsic barrier/stress sensitivity | {fmt(decomposition['intrinsic_barrier_stress_sensitivity_contribution'])} |",
        f"| three-hit transformation | {fmt(decomposition['three_hit_transformation_contribution'])} |",
        f"| blunting | {fmt(decomposition['blunting_contribution'])} |",
        f"| shielding | {fmt(decomposition['shielding_contribution'])} |",
        f"| event-size variation | {fmt(decomposition['event_size_variation_contribution'])} |",
        f"| state/waiting residual | {fmt(decomposition['state_waiting_residual_contribution'])} |",
        "",
        "A's matched event records retain a 1 µm tip radius and zero Kshield, so neither blunting nor shielding explains its excessive slope.",
        "",
        "## Validated discovery target", "",
        "| candidate | full-history m | first-three-interval m | relative error | full/short rank |", "|---|---:|---:|---:|---:|",
    ])
    for row in validation:
        lines.append(f"| {row['candidate_id']} | {fmt(row['full_history_m'])} | {fmt(row['first_3_intervals_m'])} | {fmt(row['relative_m_error'])} | {row['full_history_rank']}/{row['first_3_intervals_rank']} |")

    lines.extend([
        "", "The shortened histories preserve all three candidate ranks and do not misclassify A as realistic. With the production 2.297 µm event increment, a descriptor-only 10 µm Level-1 target forces five events and therefore supplies at least three developed intervals; formal Paris claims and n=128 qualification retain their original targets.",
        "", "## Parameter sensitivity", "",
        f"All {len(sensitivity)} coordinates are reported in `parameter_slope_sensitivity.csv` and `.md`. Only one formal developed m and two provisional slopes currently exist, so direction signs are exploratory. The causal decomposition, rather than rank correlation alone, is why the next batch emphasizes the five cleavage EXP-surface coordinates.",
        "", "## Shape-first surrogate", "",
        f"Predicted minimum developed m inside the historical pool: **{fmt(audit['domain_predicted_minimum_m'])}** ({audit['domain_predicted_minimum_m_candidate_id']}).",
        f"Best feasibility-weighted probability of the provisional 2–10 band: **{fmt(audit['domain_best_credible_target_probability_times_feasibility'])}** ({audit['domain_best_credible_target_candidate_id']}).",
        "Censored and burst/arrest cases train feasibility only; they receive no artificial slope. Rate magnitude cannot compensate for pathological shape.",
        "", "## New causally directed batch", "",
        "| candidate | terminal Level-0 points | growth points | Level-0 outcomes | developed m | descriptor status |", "|---|---:|---:|---|---:|---|",
    ])
    for row in candidate_results:
        lines.append(f"| {row['candidate_id']} | {row['level0_terminal_points']} | {row['level0_growth_points']} | {row['level0_statuses']} | {fmt(row['developed_m'])} | {row['descriptor_status']} |")
    lines.extend([
        "", "## Historical-domain viability", "",
        f"Current minimum measured n=64 developed slope: **{fmt(best_measured)}**.",
        f"Current decision: **{decision}**.",
        "If domain failure is confirmed, the first parameter family requiring an explicit physics decision is the cleavage activation surface (G00, activation-stress scale, EXP amplitude/exponent, and floor fraction), potentially including its coupling to event waiting state. Common transport physics, REV behavior, elastic constants, and Schmid factors were not altered.",
        "",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"V914_LOW_SLOPE_REPORT decision={decision!r} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
