"""Final closure E: the Part X figure manifest -- maps all 19 required
figures to a canonical existing or newly generated file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
DEV_CONF_DIR = "artifacts/crack_rebonding_developed_confirmation"
PART_X = "artifacts/crack_rebonding_part_x_v1"

REQUIREMENTS = [
    dict(n=1, description="P/C/B periodic or phase-resolved state trajectories",
         canonical_file=f"{PART_X}/fig01_pcb_phase_trajectory.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="ANALYTICAL_PREDICTION"),
    dict(n=2, description="Transition actions and realized transition fluxes",
         canonical_file=f"{PART_X}/fig02_transition_actions_fluxes.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="ANALYTICAL_PREDICTION"),
    dict(n=3, description="Contact exposure or contact fraction versus R",
         canonical_file=f"{PART_X}/fig03_screen_contact_exposure_vs_R.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION"),
    dict(n=4, description="Screen-regime S_h versus R",
         canonical_file=f"{PART_X}/fig04_screen_S_h_vs_R.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION_POST_PROCESSED"),
    dict(n=5, description="Frequency response",
         canonical_file=f"{PART_X}/fig05_screen_frequency_response.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION_POST_PROCESSED"),
    dict(n=6, description="Dwell response, invalidated vs corrected",
         canonical_file=f"{PART_X}/fig06_dwell_invalidated_vs_corrected.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION (both invalidated and corrected series explicitly distinguished)"),
    dict(n=7, description="Passivation or chemistry response",
         canonical_file=f"{PART_X}/fig07_screen_passivation_chemistry_response.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION_POST_PROCESSED"),
    dict(n=8, description="Cohesive-strength response",
         canonical_file=f"{PART_X}/fig08_screen_cohesive_strength_response.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION_POST_PROCESSED"),
    dict(n=9, description="Absolute developed da/dN versus Kmax",
         canonical_file=f"{PART_X}/fig09_absolute_developed_da_dN_vs_Kmax.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION"),
    dict(n=10, description="Developed S_h versus Kmax",
         canonical_file=f"{PART_X}/px6_S_h_developed_vs_Kmax_all_protocols.png", status="PRESENT_AND_VERIFIED",
         provenance="PHYSICAL_SIMULATION_POST_PROCESSED (reused from PX6)"),
    dict(n=11, description="Local slopes and curvature",
         canonical_file=f"{PART_X}/fig11_local_slope_curvature.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="POST_PROCESSING"),
    dict(n=12, description="Reduced-frequency developed curves",
         canonical_file=f"{PART_X}/fig12_reduced_frequency_developed_curves.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION"),
    dict(n=13, description="Clean reversible versus passivation-limited curves",
         canonical_file=f"{PART_X}/fig13_reversible_vs_passivation_curves.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION"),
    dict(n=14, description="Reversible versus persistent curves",
         canonical_file=f"{PART_X}/fig14_reversible_vs_persistent_curves.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="PHYSICAL_SIMULATION"),
    dict(n=15, description="Dynamic rebonding versus prescribed static-shield controls",
         canonical_file=f"{PART_X}/px6_px5_static_shield_vs_dynamic.png", status="PRESENT_AND_VERIFIED",
         provenance="PHYSICAL_SIMULATION (dynamic + 2 prescribed static controls, reused from PX6)"),
    dict(n=16, description="Eventwise and rolling waiting-time ratios",
         canonical_file=f"{DEV_CONF_DIR}/eventwise_and_rolling_waiting_time_ratios.png", status="PRESENT_AND_VERIFIED",
         provenance="PHYSICAL_SIMULATION (reused from the pre-Part-X developed_confirmation study -- same "
                     "stable_growth_gate/event-schema primitives PX4 reuses verbatim; not PX4's own 5-protocol "
                     "grid, a predecessor 2-seed confirmation at Kmax=15/18/21)"),
    dict(n=17, description="Available mechanism diagnostics versus Kmax",
         canonical_file=f"{PART_X}/fig17_mechanism_diagnostics_vs_Kmax.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="ANALYTICAL_PREDICTION"),
    dict(n=18, description="Censor, attempt, invalidation, and admissibility map",
         canonical_file=f"{PART_X}/fig18_admissibility_map.png", status="GENERATED_IN_FINAL_CLOSURE",
         provenance="POST_PROCESSING"),
    dict(n=19, description="Seed-robustness comparison",
         canonical_file=f"{PART_X}/px6_seed_robustness_slope_comparison.png", status="PRESENT_AND_VERIFIED",
         provenance="POST_PROCESSING (both seeds shown, reused from PX6)"),
]


def main() -> None:
    rows_out = []
    for req in REQUIREMENTS:
        row = dict(req)
        first_path = row["canonical_file"]
        full_path = REPO_ROOT / first_path
        if not full_path.is_file():
            raise RuntimeError(f"figure manifest requirement #{row['n']} claims {row['status']} but {full_path} does not exist")
        if full_path.stat().st_size == 0:
            raise RuntimeError(f"figure manifest requirement #{row['n']} canonical file is zero bytes: {full_path}")
        rows_out.append(row)

    n_present = sum(1 for r in rows_out if r["status"] == "PRESENT_AND_VERIFIED")
    n_generated = sum(1 for r in rows_out if r["status"] == "GENERATED_IN_FINAL_CLOSURE")
    n_resolved = n_present + n_generated
    canonical_files = [r["canonical_file"] for r in rows_out]
    if len(canonical_files) != len(set(canonical_files)):
        raise RuntimeError("figure manifest has duplicate canonical_file entries across distinct requirements")

    import csv
    with (ARTIFACTS_DIR / "part_x_figure_manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["n", "description", "canonical_file", "status", "provenance"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)

    (ARTIFACTS_DIR / "part_x_figure_manifest.json").write_text(json.dumps({
        "schema": "v10230_part_x_figure_manifest_v1",
        "n_requirements": len(rows_out), "n_present_and_verified": n_present,
        "n_generated_in_final_closure": n_generated, "n_resolved": n_resolved,
        "n_required_total": 19, "all_19_resolved": n_resolved == 19,
        "requirements": rows_out,
    }, indent=2, default=str))

    print(f"Wrote part_x_figure_manifest.{{csv,json}}: {len(rows_out)}/19 requirements resolved "
          f"({n_present} present, {n_generated} generated)")


if __name__ == "__main__":
    main()
