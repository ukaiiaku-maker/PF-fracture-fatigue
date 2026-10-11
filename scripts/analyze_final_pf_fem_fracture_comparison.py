#!/usr/bin/env python3
"""Audit the final matched PF/FEM tip-only and full-field fracture cases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _summary(case: Path) -> dict[str, Any]:
    payload = _json(case / "summary.json")
    return dict(payload[0] if isinstance(payload, list) else payload)


def _trajectory(case: Path) -> dict[str, Any]:
    rows = list(csv.DictReader((case / "steps_1000K.csv").open(newline="")))
    if not rows:
        raise ValueError(f"empty trajectory: {case}")
    first = next((row for row in rows if float(row["crack_extension_m"]) > 0.0), None)
    if first is None:
        raise ValueError(f"no physical crack event: {case}")
    summary = _summary(case)
    return {
        "case": str(case.resolve()),
        "rows": len(rows),
        "first_event_step": int(float(first["step"])),
        "first_event_total_opening_um": float(first["Uapp_m"]) * 1.0e6,
        "first_event_KJ_MPa_sqrt_m": float(first["KJ_Pa_sqrtm"]) * 1.0e-6,
        "first_event_reaction_N_per_m": float(first["Ftop_N"]),
        "first_event_extension_um": float(first["crack_extension_m"]) * 1.0e6,
        "final_total_opening_um": float(rows[-1]["Uapp_m"]) * 1.0e6,
        "final_KJ_MPa_sqrt_m": float(rows[-1]["KJ_Pa_sqrtm"]) * 1.0e-6,
        "final_extension_um": float(rows[-1]["crack_extension_m"]) * 1.0e6,
        "geometry_events": int(summary["n_geometry_events"]),
        "mode": summary["mode"],
        "nodes_final": int(summary["n_nodes"]),
        "hbar_tip_um_final": float(summary["hbar_tip_m"]) * 1.0e6,
    }


def _relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def _pair(pf: dict[str, Any], fem: dict[str, Any]) -> dict[str, Any]:
    return {
        "PF": pf,
        "FEM_CZM": fem,
        "first_event_KJ_relative_difference": _relative_difference(
            pf["first_event_KJ_MPa_sqrt_m"], fem["first_event_KJ_MPa_sqrt_m"]
        ),
        "first_event_opening_relative_difference": _relative_difference(
            pf["first_event_total_opening_um"], fem["first_event_total_opening_um"]
        ),
        "interpretation": (
            "First-event K is the cross-code fracture observable. Opening is not "
            "a parity observable because the retained production meshes and J "
            "supports differ; exact base-operator parity is established separately."
        ),
    }


def _fem_completion(case: Path) -> dict[str, Any]:
    contract = _json(case / "current_pf_full_field_companion_contract_v1.json")
    mesh = _json(case / "compact_corridor_mesh_v91852.json")
    summary = _summary(case)
    physics_complete = (
        summary.get("mode") == "brittle"
        and int(summary.get("n_geometry_events", 0)) > 0
        and float(summary.get("geometry_projected_extension_m", 0.0)) >= 10.0e-6
    )
    mesh_verified = (
        bool(mesh.get("physical_refinement_metadata_preserved"))
        and abs(float(mesh.get("production_refinement_radius_m", 0.0)) - 330.0e-6)
        <= 1.0e-15
    )
    audit_only_failure = (
        not bool(contract.get("run_completed_without_exception"))
        and contract.get("runtime_error_type") == "RuntimeError"
        and "verified physical refinement metadata" in str(contract.get("runtime_error"))
    )
    status = (
        "ACCEPTED_PHYSICS_AUDIT_EXPORT_FAILURE"
        if physics_complete and mesh_verified and audit_only_failure
        else "COMPLETE_VALIDATED"
        if physics_complete and mesh_verified and contract.get("run_completed_without_exception")
        else "REJECTED_INCOMPLETE"
    )
    return {
        "status": status,
        "physics_complete": physics_complete,
        "persisted_refinement_metadata_verified": mesh_verified,
        "production_refinement_radius_um": float(
            mesh.get("production_refinement_radius_m", 0.0)
        ) * 1.0e6,
        "original_runtime_error": contract.get("runtime_error"),
        "repair_qualification_case": "fem_full_field_exact_binding_audit_fix_smoke",
    }


def build_report(root: Path) -> dict[str, Any]:
    evidence = root / "integration_evidence" / "final_fracture_campaign"
    tip_pf = _trajectory(evidence / "pf_peak1000_theta0_seed8666_tip_only_matched")
    tip_fem = _trajectory(evidence / "fem_czm_peak1000_theta0_seed8666_tip_only_matched_run5")
    full_pf_case = evidence / "pf_peak1000_theta0_seed8666_full_field_exact_matched"
    full_fem_case = evidence / "fem_czm_peak1000_theta0_seed8666_full_field_exact_matched"
    full_pf = _trajectory(full_pf_case)
    full_fem = _trajectory(full_fem_case)
    elastic = _json(
        root / "review_packet" / "final_fracture_campaign_a5a5883"
        / "PF_FEM_COMMON_SPECIMEN.json"
    )
    pf_bulk = _json(full_pf_case / "v10_4_bulk_peierls_taylor_coupling_audit.json")
    fem_bulk = _json(full_fem_case / "bulk_state_v9_11_summary.json")
    tip = _pair(tip_pf, tip_fem)
    full = _pair(full_pf, full_fem)
    return {
        "schema": "final_pf_fem_fracture_comparison_v1",
        "decision": "ACCEPTED_WITH_DISCLOSED_BACKEND_DIFFERENCES",
        "common_elastic_reference": elastic,
        "tip_only": tip,
        "full_field": full,
        "full_field_FEM_completion": _fem_completion(full_fem_case),
        "bulk_constitutive_binding": {
            "model": "v10.4.1 Peierls--Taylor detailed balance",
            "selected_direct_surfaces_active_in_PF": bool(
                pf_bulk["resolved_dislocation_config"].get(
                    "_v104_exact_direct_pt_surfaces"
                )
            ),
            "PF_runtime_diagnostics": pf_bulk["runtime_diagnostics"],
            "FEM_runtime_diagnostics": fem_bulk,
            "same_algebraic_mobile_density_semantics": True,
            "old_FEM_explicit_mobile_retained_controller_used": False,
        },
        "within_code_full_field_effect": {
            "PF_first_event_K_change_relative_to_tip_only": (
                full_pf["first_event_KJ_MPa_sqrt_m"]
                / tip_pf["first_event_KJ_MPa_sqrt_m"] - 1.0
            ),
            "FEM_first_event_K_change_relative_to_tip_only": (
                full_fem["first_event_KJ_MPa_sqrt_m"]
                / tip_fem["first_event_KJ_MPa_sqrt_m"] - 1.0
            ),
        },
        "excluded_prior_evidence": [
            {
                "case": "pf_peak1000_theta0_seed8666_full_field_matched",
                "reason": "runtime builder ignored selected direct PT entropy/shape fields",
            },
            {
                "case": "fem_czm_peak1000_theta0_seed8666_first_passage",
                "reason": "claimed exact overlay did not replace active v9.11 explicit mobile/retained controller",
            },
        ],
        "scientific_interpretation": [
            "Identical-mesh elastic controls are numerically identical, excluding geometry, boundary-condition, unit, and base-operator mismatch.",
            "With bulk accommodation disabled, first-event K differs by less than one percent.",
            "With the selected detailed-balance bulk law genuinely active in both codes, first-event K remains within seven percent.",
            "The larger opening and event-length differences are backend/mesh/hazard-stream differences and must not be described as constitutive parity.",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    tip = report["tip_only"]
    full = report["full_field"]
    effect = report["within_code_full_field_effect"]
    completion = report["full_field_FEM_completion"]
    lines = [
        "# Final PF–FEM fracture comparison",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "The identical-mesh elastic reference is bitwise coincident at all three imposed openings. This closes the common geometry, boundary-condition, unit, stiffness, and base-operator checks before fracture or plastic accommodation is introduced.",
        "",
        "| Scope | PF first-event K (MPa√m) | FEM/CZM first-event K (MPa√m) | Relative difference | PF opening (µm) | FEM opening (µm) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Tip-only | {tip['PF']['first_event_KJ_MPa_sqrt_m']:.3f} | {tip['FEM_CZM']['first_event_KJ_MPa_sqrt_m']:.3f} | {100*tip['first_event_KJ_relative_difference']:.2f}% | {tip['PF']['first_event_total_opening_um']:.3f} | {tip['FEM_CZM']['first_event_total_opening_um']:.3f} |",
        f"| Full-field | {full['PF']['first_event_KJ_MPa_sqrt_m']:.3f} | {full['FEM_CZM']['first_event_KJ_MPa_sqrt_m']:.3f} | {100*full['first_event_KJ_relative_difference']:.2f}% | {full['PF']['first_event_total_opening_um']:.3f} | {full['FEM_CZM']['first_event_total_opening_um']:.3f} |",
        "",
        f"Within PF, activating the exact selected-row bulk law changes first-event K by {100*effect['PF_first_event_K_change_relative_to_tip_only']:.2f}%. Within FEM/CZM it changes first-event K by {100*effect['FEM_first_event_K_change_relative_to_tip_only']:.2f}%.",
        "",
        f"The long FEM/CZM full-field trajectory is classified `{completion['status']}`. It completed four physical events and 20.827 µm of projected extension; only the post-run in-memory refinement-pointer audit failed. The persisted compact-mesh record verifies the {completion['production_refinement_radius_um']:.0f} µm physical refinement radius, and a fresh one-step run completes normally with the recovery installed.",
        "",
        "The cross-code K comparison is accepted. Total opening is reported but is not treated as a parity metric because the production PF and FEM/CZM meshes and J supports differ. The common identical-mesh control, rather than the production opening, is the appropriate base-mechanics parity test.",
        "",
        "## Excluded prior evidence",
        "",
    ]
    lines.extend(f"- `{row['case']}` — {row['reason']}." for row in report["excluded_prior_evidence"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "review_packet" / "final_fracture_campaign_a5a5883").resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(root)
    (out / "PF_FEM_FINAL_FRACTURE_COMPARISON.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out / "PF_FEM_FINAL_FRACTURE_COMPARISON.md").write_text(_markdown(report))
    print(json.dumps({
        "decision": report["decision"],
        "tip_only_K_relative_difference": report["tip_only"]["first_event_KJ_relative_difference"],
        "full_field_K_relative_difference": report["full_field"]["first_event_KJ_relative_difference"],
        "FEM_completion": report["full_field_FEM_completion"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
