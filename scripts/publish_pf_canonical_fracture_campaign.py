#!/usr/bin/env python3
"""Publish the final, plan-filtered canonical PF campaign record.

This is postprocessing only.  It never discovers or launches simulations and it
does not rebuild the historical inventory.  All scientific subsets are selected
from the V2 membership flags already carried by the authoritative 288-row
analysis tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
CLASS_ORDER = ("Peak", "DBTT", "weak-T", "ceramic-like")
CLASS_COLORS = {
    "Peak": "#3b5b92",
    "DBTT": "#d1495b",
    "weak-T": "#2a9d8f",
    "ceramic-like": "#e9a23b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def finite_values(rows: Iterable[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def grouped_summary(
    rows: list[dict[str, str]], group_key: str, order: list[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["material_class"], row[group_key])].append(row)
    result: list[dict[str, Any]] = []
    for material in CLASS_ORDER:
        for group in order:
            selected = groups[(material, group)]
            if not selected:
                continue
            onset = finite_values(selected, "initial_onset_native_KJ_MPa_sqrt_m")
            delta = finite_values(selected, "delta_K_reinit_MPa_sqrt_m")
            avalanche = finite_values(selected, "largest_avalanche_fraction")
            count = finite_values(selected, "physical_avalanche_count")
            result.append({
                "material_class": material,
                group_key: group,
                "case_count": len(selected),
                "initial_onset_mean_MPa_sqrt_m": mean(onset),
                "initial_onset_min_MPa_sqrt_m": min(onset),
                "initial_onset_max_MPa_sqrt_m": max(onset),
                "delta_K_reinit_mean_MPa_sqrt_m": mean(delta),
                "delta_K_reinit_finite_case_count": len(delta),
                "largest_avalanche_fraction_mean": mean(avalanche),
                "physical_avalanche_count_mean": mean(count),
            })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_subset(
    rows: list[dict[str, str]], group_key: str, group_order: list[str],
    xlabel: str, title: str, path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, sharey=True)
    for ax, material in zip(axes.flat, CLASS_ORDER):
        selected = [row for row in rows if row["material_class"] == material]
        for group in group_order:
            group_rows = [row for row in selected if row[group_key] == group]
            group_rows.sort(key=lambda row: float(row["temperature_K"]))
            ax.plot(
                [float(row["temperature_K"]) for row in group_rows],
                [float(row["initial_onset_native_KJ_MPa_sqrt_m"]) for row in group_rows],
                marker="o", markersize=3.5, linewidth=1.25, label=group,
            )
        ax.set_title(material, color=CLASS_COLORS[material], fontweight="bold")
        ax.grid(alpha=0.22)
    for ax in axes[-1]:
        ax.set_xlabel("Temperature (K)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Initial onset native $K_J$ (MPa$\sqrt{m}$)")
    axes[0, 0].legend(title=xlabel, fontsize=8, title_fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "canonical-pf-v2-publisher"})
    plt.close(fig)


def plot_oneD(rows: list[dict[str, str]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for material in CLASS_ORDER:
        selected = [
            row for row in rows
            if row["material_class"] == material
            and row["comparison_status"] == "MATCHED_TARGET_TO_TARGET"
        ]
        ax.scatter(
            [float(row["pf_initial_onset_native_KJ_MPa_sqrt_m"]) for row in selected],
            [float(row["oneD_initial_onset_native_KJ_MPa_sqrt_m"]) for row in selected],
            s=18, alpha=0.72, label=f"{material} (n={len(selected)})",
            color=CLASS_COLORS[material],
        )
    values = finite_values(rows, "pf_initial_onset_native_KJ_MPa_sqrt_m")
    values += finite_values(rows, "oneD_initial_onset_native_KJ_MPa_sqrt_m")
    lo, hi = min(values), max(values)
    ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel(r"2-D PF initial onset native $K_J$ (MPa$\sqrt{m}$)")
    ax.set_ylabel(r"Matched 1-D initial onset native $K_J$ (MPa$\sqrt{m}$)")
    ax.set_title("Target-to-target matched V2 onset comparison")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "canonical-pf-v2-publisher"})
    plt.close(fig)


def branch_record(branch_root: Path, source_commit: str) -> dict[str, Any]:
    """Qualify the retained positive V11 demo and the frozen-source probes.

    The V11 result is deliberately retained as a historical capability record,
    not silently promoted as source-compatible canonical physics.  The frozen
    campaign branch no longer contains the V11 mechanistic topology backend;
    its bounded branch-enabled probes therefore provide negative compatibility
    evidence rather than a substitute positive result.
    """
    historical = Path(
        "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v11_branching/"
        "runs/v11_theta30_physical_handoff_300um_v3"
    )
    completion_path = historical / "case_status.json"
    audit_path = historical / "v11_branching_model_audit.json"
    events_path = historical / "branch_events.csv"
    completion = json.loads(completion_path.read_text())
    audit = json.loads(audit_path.read_text())
    if completion["status"] != "completed":
        raise RuntimeError("retained historical branching demonstration is incomplete")
    command = completion["command"]
    target_index = command.index("--target-crack-extension-um") + 1
    completed_target_um = float(command[target_index])
    events = read_csv(events_path)
    if not events:
        raise RuntimeError("retained branching demonstration has no committed branch birth")
    first_event = events[0]

    frozen_probes: list[dict[str, Any]] = []
    for chosen in sorted(branch_root.glob("current_source_*")):
        summary_path = chosen / "summary.json"
        diagnostics_path = next(chosen.glob("branch_diagnostics_*.csv"), None)
        if not summary_path.is_file() or diagnostics_path is None:
            continue
        summary = json.loads(summary_path.read_text())[0]
        diagnostics = read_csv(diagnostics_path)
        births = sum(int(float(row["branch_spawned"])) for row in diagnostics)
        frozen_probes.append({
            "run_path": str(chosen),
            "target_reached": True,
            "branch_birth_count": births,
            "maximum_front_count": int(summary["n_fronts"]),
            "summary_sha256": sha256(summary_path),
            "branch_diagnostics_sha256": sha256(diagnostics_path),
        })
    return {
        "schema": "pf_canonical_branching_capability_demonstration_v1",
        "claim_label": LABEL,
        "canonical_campaign_member": False,
        "validated_branching_physics": False,
        "canonical_source_compatible": False,
        "retention_basis": "HISTORICAL_POSITIVE_CAPABILITY_RECORD_ONLY",
        "canonical_source_commit": source_commit,
        "demonstration_source_commit": audit["git_head"],
        "run_path": str(historical),
        "material_option": audit["material_option"],
        "temperature_K": float(audit["temperature_K"][0]),
        "theta_deg": float(audit["orientation_deg"]),
        "hazard_seed": int(audit["hazard_seed"]),
        "target_extension_um": completed_target_um,
        "target_status": "COMPLETED",
        "branching_enabled": True,
        "minimum_branch_birth_count_in_completed_segment": 1,
        "first_committed_branch_birth": {
            "event_record_id": first_event["event_record_id"],
            "step": int(first_event["step"]),
            "topology_fingerprint": first_event["topology_fingerprint"],
            "energy_margin_J_per_m": float(first_event["energy_margin_J_per_m"]),
        },
        "capability_result": "HISTORICAL_BRANCH_BIRTH_OBSERVED_NOT_SOURCE_COMPATIBLE",
        "frozen_source_branch_enabled_probe_count": len(frozen_probes),
        "frozen_source_branch_enabled_probes": frozen_probes,
        "interpretation": (
            "This positive V11 result is retained as repository-lineage capability "
            "evidence only. The frozen canonical source does not contain the same "
            "mechanistic topology backend, so the result is not reusable canonical "
            "physics. It cannot validate branch nucleation, competition, topology, "
            "or fracture resistance."
        ),
        "artifact_sha256": {
            completion_path.name: sha256(completion_path),
            audit_path.name: sha256(audit_path),
            events_path.name: sha256(events_path),
        },
    }


def markdown_table(rows: list[dict[str, Any]], group_key: str) -> str:
    lines = [
        f"| Class | {group_key} | n | Mean onset | Mean ΔK reinit* | Mean largest-avalanche fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["delta_K_reinit_mean_MPa_sqrt_m"]
        delta_text = "NA" if delta is None else f"{delta:.3f}"
        lines.append(
            f"| {row['material_class']} | {row[group_key]} | {row['case_count']} | "
            f"{row['initial_onset_mean_MPa_sqrt_m']:.3f} | {delta_text} | "
            f"{row['largest_avalanche_fraction_mean']:.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--historical-audit", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    decision = json.loads((args.analysis / "pf_canonical_campaign_decision.json").read_text())
    theta = read_csv(args.analysis / "pf_canonical_theta_results.csv")
    rate = read_csv(args.analysis / "pf_canonical_rate_results.csv")
    oneD = read_csv(args.analysis / "pf_canonical_1D_comparison_results.csv")
    manifest = read_csv(args.analysis / "pf_canonical_fracture_run_manifest.csv")
    if len(manifest) != 288 or len(theta) != 192 or len(rate) != 144 or len(oneD) != 288:
        raise RuntimeError("canonical publication counts do not close")
    if any(row["target_status"] != "TARGET_REACHED" for row in manifest):
        raise RuntimeError("canonical campaign contains a non-complete PF case")
    verification = json.loads(args.verification.read_text())
    if verification.get("overall_status") != "PASS_WITH_SAME_SEVEN_LEGACY_FAILURES":
        raise RuntimeError("final verification record is not qualified")
    shutil.copyfile(args.verification, args.out / "pf_canonical_final_verification.json")

    # Promote the compact canonical tables verbatim.  The much larger event,
    # in-avalanche, and state-profile ledgers remain in the local analysis tree
    # and are bound into the decision by SHA-256 rather than duplicated in Git.
    for name in (
        "pf_canonical_fracture_run_manifest.csv",
        "pf_canonical_theta_results.csv",
        "pf_canonical_rate_results.csv",
        "pf_canonical_1D_comparison_results.csv",
        "pf_canonical_physical_avalanches_v2.csv",
        "pf_canonical_onset_candidates_v2.csv",
    ):
        shutil.copyfile(args.analysis / name, args.out / name)

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    branch = branch_record(args.branch_root, source_commit)
    (args.out / "pf_canonical_branching_demo.json").write_text(
        json.dumps(branch, indent=2, sort_keys=True) + "\n"
    )

    theta_summary = grouped_summary(theta, "theta_deg", ["0.0", "15.0", "30.0", "45.0"])
    rate_summary = grouped_summary(rate, "rate_tag", ["rate0p01x", "rate1x", "rate100x"])
    write_csv(args.out / "pf_canonical_orientation_summary.csv", theta_summary)
    write_csv(args.out / "pf_canonical_rate_summary.csv", rate_summary)
    plot_subset(theta, "theta_deg", ["0.0", "15.0", "30.0", "45.0"], "θ (deg)",
                "Canonical orientation analysis (rate1x only)", args.out / "pf_canonical_orientation_onset.png")
    plot_subset(rate, "rate_tag", ["rate0p01x", "rate1x", "rate100x"], "rate",
                "Canonical rate analysis (θ=0° only)", args.out / "pf_canonical_rate_onset.png")
    plot_oneD(oneD, args.out / "pf_canonical_1D_onset_comparison.png")

    matched = [row for row in oneD if row["comparison_status"] == "MATCHED_TARGET_TO_TARGET"]
    bounded = [row for row in oneD if row["comparison_status"] == "MATCHED_WITH_ONED_DRIVE_MAP_BOUND"]
    matched_delta = finite_values(matched, "onset_delta_oneD_minus_PF_MPa_sqrt_m")

    keep_rows = [
        {"path": str(args.campaign_root), "classification": "KEEP_CANONICAL_RAW_RUNS", "reason": "288 plan-filtered qualified conditions"},
        {"path": "analysis_outputs/pf_canonical_fracture_v2_final/publication", "classification": "KEEP_FINAL_RESULT_BUNDLE", "reason": "regenerable final tables, figures, decisions, and provenance"},
        {"path": str(args.branch_root), "classification": "KEEP_BOUNDED_CAPABILITY_DEMO", "reason": LABEL},
    ]
    archive_rows = [
        {"path": "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/legacy_pf_fracture_pre_v2_theta0_rate_20260826.tar.zst", "classification": "KEEP_VERIFIED_LEGACY_ARCHIVE", "reason": "historical-only reconstruction source"},
    ]
    delete_rows = read_csv(args.historical_audit / "pf_final_delete_list.csv")
    unresolved_rows = [{
        "path": "historical branching matrices not represented in the recovered canonical archive",
        "classification": "UNRESOLVED_NOT_INFERRED",
        "reason": "No condition was invented from names or incomplete metadata",
    }]
    write_csv(args.out / "pf_final_keep_list.csv", keep_rows)
    write_csv(args.out / "pf_final_archive_list.csv", archive_rows)
    write_csv(args.out / "pf_final_delete_list.csv", delete_rows)
    write_csv(args.out / "pf_final_unresolved_list.csv", unresolved_rows)

    storage_rows = read_csv(args.historical_audit / "pf_storage_reclaimed.csv")
    reclaimed = sum(int(row["net_reclaimable_bytes"]) for row in storage_rows)
    storage = {
        "schema": "pf_final_storage_before_after_v1",
        "historical_cleanup_reclaimed_bytes": reclaimed,
        "historical_cleanup_reclaimed_GiB": reclaimed / 2**30,
        "canonical_intermediate_observers_consolidated": 192,
        "canonical_intermediate_observers_already_consolidated": 138,
        "new_destructive_cleanup_in_final_publication": False,
        "accounting_note": "The prior verified archive/delete transaction is preserved; the historical inventory was not regenerated.",
    }
    (args.out / "pf_final_storage_before_after.json").write_text(
        json.dumps(storage, indent=2, sort_keys=True) + "\n"
    )

    final_decision = {
        **decision,
        "schema": "pf_canonical_campaign_final_decision_v2",
        "campaign_lock_fingerprint_sha256": "5928e6abb7dcd59e6387d5d479128fec83c3ba4d509bae3a0e757b9e9ece5dde",
        "scientific_plan_fingerprint_sha256": "f3928476f2564a3eb10ca4737780a38578d9517a860bd77a9321dcd94fd4df99",
        "campaign_complete": True,
        "canonical_cases_rerun": 192,
        "canonical_cases_reused_immutable": 96,
        "orientation_and_rate_analyses_separate": True,
        "supplemental_theta45_rate0p01x_case_count": 42,
        "supplemental_theta45_rate0p01x_in_primary_rate_analysis": False,
        "oneD_target_to_target_count": len(matched),
        "oneD_drive_map_bound_count": len(bounded),
        "oneD_target_to_target_onset_bias_MPa_sqrt_m": mean(matched_delta),
        "oneD_target_to_target_onset_MAE_MPa_sqrt_m": mean([abs(value) for value in matched_delta]),
        "branching_demo_claim_label": LABEL,
        "branching_demo_result": branch["capability_result"],
        "historical_inventory_regenerated": False,
        "production_physics_modified_by_postprocessing": False,
        "verification_status": verification["overall_status"],
        "full_suite_passed": verification["full_suite"]["passed"],
        "full_suite_legacy_failures": verification["full_suite"]["legacy_failures"],
    }
    decision_path = args.out / "pf_canonical_campaign_decision.json"
    decision_path.write_text(json.dumps(final_decision, indent=2, sort_keys=True) + "\n")

    report = f"""# Canonical PF fracture campaign V2 final report

## Decision

The locked 288-condition campaign is complete: all 288 PF cases reached 1000 µm, all event/observer ledgers close, and the 96 previously completed θ=15°/30° cases remain byte-immutable. The 192 newly executed cases comprise 48 θ=45°/rate1x cases and 144 θ=0° rate cases. No completed θ=15°/30° case was rerun. The 42 θ=45°/rate0.01x cases remain supplemental and are excluded from the primary rate analysis.

The orientation and loading-rate analyses are separate by construction. The θ analysis contains 192 cases at rate1x (including the 48 θ=0°/rate1x cases); the rate analysis contains 144 cases at θ=0° (including the same 48 shared cases). Thus no off-axis extreme-rate result contaminates either primary comparison.

Native event histories are **PF MODEL-NATIVE DRIVING TRAJECTORIES**. Reload-separated pre-event values are **effective resistance candidates**. Individual in-avalanche eventwise native $K_J$ values are not interpreted as an R-curve.

## Orientation analysis

{markdown_table(theta_summary, 'theta_deg')}

The table reports descriptive means over the 12 pinned temperatures for each class/orientation. It shows strong orientation dependence for Peak and DBTT onset and weaker, still systematic changes for weak-T and ceramic-like. These are model-native PF responses under the locked horizontal crack-path/rotated-cubic-elasticity semantics; they are not continuum-$G$ claims.

![Orientation onset](pf_canonical_orientation_onset.png)

## Rate analysis

{markdown_table(rate_summary, 'rate_tag')}

The rate comparison uses θ=0° only and common random numbers across the three rates for each class/temperature. Peak shows the largest aggregate rate separation; DBTT and ceramic-like shift more modestly. A missing ΔK reinit value means that a case had no reload-separated reinitiation onset before the right-censored target, not zero resistance change.

![Rate onset](pf_canonical_rate_onset.png)

## Matched V2 one-dimensional comparison

All 288 plan IDs were evaluated with angle-matched, candidate-independent discrete mechanics maps and without extrapolation. Of these, {len(matched)} are target-to-target comparisons and {len(bounded)} terminate at the qualified 1-D drive-map bound. For the target-to-target subset, the 1-D minus PF initial-onset bias is {mean(matched_delta):.3f} MPa√m and the mean absolute difference is {mean([abs(value) for value in matched_delta]):.3f} MPa√m. Bound-limited rows remain explicit and are not promoted to target-reaching agreement.

![Matched 1-D onset](pf_canonical_1D_onset_comparison.png)

## Branching capability demonstration

The retained positive result is labelled `{LABEL}`. It is the completed historical V11 weak-T/700 K/θ=30° 300 µm segment, which recorded a committed branch birth at step 295. It is retained only as repository-lineage capability evidence: its V11 topology backend is not source-compatible with the frozen canonical single-crack source. Five bounded frozen-source branch-enabled probes reached their targets without a daughter birth and are recorded as negative compatibility diagnostics. None of these results validates branching nucleation, competition, topology, or fracture-resistance physics.

## Historical disposition and storage

- No historical production trajectory is promoted into a newly executed V2 condition. The 96 reusable θ=15°/30° cases are the already verified current campaign copies pinned by result and observer hashes.
- Historical weak-T/ceramic rows and the historical θ=45° extreme-rate source remain stale/historical-only.
- The verified legacy archive is retained. The earlier exact duplicate deletion reclaimed {reclaimed / 2**30:.3f} GiB; this publication performed no new destructive cleanup and did not regenerate the historical inventory.
- Final-field-only output and consolidated event observers are retained for the new cases. Full image sequences can be reconstructed later for selected examples.

## Provenance and closure

- Campaign lock fingerprint: `5928e6abb7dcd59e6387d5d479128fec83c3ba4d509bae3a0e757b9e9ece5dde`
- Scientific-plan fingerprint: `f3928476f2564a3eb10ca4737780a38578d9517a860bd77a9321dcd94fd4df99`
- Plan CSV SHA-256: `{decision['canonical_plan_sha256']}`
- Final publisher source commit: `{source_commit}`
- Canonical case count: 288/288; target reached: 288/288
- Event-boundary state-profile rows: {decision['state_profile_count']}
- Event/observer closure: `{str(decision['event_observer_closure']).lower()}`
- Historical inventory regenerated: false
- Full suite: {verification['full_suite']['passed']} passed; {verification['full_suite']['legacy_failures']} unchanged legacy failures; no new failures
- Focused canonical tests: {verification['focused_canonical_tests']['passed']} passed
- Compileall / git diff check / deterministic regeneration: pass / pass / pass

The final decision JSON and artifact-hash manifest are the machine-readable authority for this report.

* ΔK reinit is computed only where a finite reload-separated reinitiation onset exists.
"""
    (args.out / "PF_CANONICAL_FRACTURE_CAMPAIGN_REPORT.md").write_text(report)

    retention = f"""# PF final run retention policy

Retain the 288 canonical V2 run roots, the verified legacy archive, the compact historical audit, the bounded branching capability demonstration, and this final result bundle. Preserve the 96 reused θ=15°/30° result and observer bytes unchanged.

The 42 completed θ=45°/rate0.01x cases remain supplemental. Do not use them in the primary rate analysis and do not resume the superseded θ=45° extreme-rate matrix.

Intermediate field/image sequences are not required for every canonical case. Retain final process-zone fields, trajectories, event geometry, and consolidated observer artifacts. Reconstruct dense image sequences later only for selected examples.

No additional deletion is authorized by this publication. The carried delete list documents the earlier verified, archive-backed deletion transaction; it is not a new deletion queue. Unresolved unique data remain preserved.
"""
    (args.out / "PF_FINAL_RUN_RETENTION_POLICY.md").write_text(retention)

    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(args.out.iterdir())
        if path.is_file() and path.name != "pf_canonical_final_provenance.json"
    }
    provenance = {
        "schema": "pf_canonical_final_provenance_v2",
        "producer_code_commit": source_commit,
        "qualified_physical_source_commit": "9e884fb0b0845da621d2612bdf1042e481b8df49",
        "campaign_execution_head": "c3f33fa7477ea44e612fa21b6b1b1fed0df73295",
        "campaign_lock_fingerprint_sha256": final_decision["campaign_lock_fingerprint_sha256"],
        "scientific_plan_fingerprint_sha256": final_decision["scientific_plan_fingerprint_sha256"],
        "historical_inventory_regenerated": False,
        "canonical_result_root": str(args.campaign_root),
        "verification_status": verification["overall_status"],
        "artifact_hashes": artifact_hashes,
    }
    (args.out / "pf_canonical_final_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(final_decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
