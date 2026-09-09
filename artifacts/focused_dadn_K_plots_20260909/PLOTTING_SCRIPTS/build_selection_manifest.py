"""Write plot_selection_manifest.csv: one row per dataset/curve that was
either INCLUDED in a numbered figure or explicitly CONSIDERED_AND_EXCLUDED,
with the reason. This is a static record of the inclusion/exclusion
decisions made while building this package -- it is not derived by
computation from the other CSVs (those decisions were made by hand against
the mission's exclusion rules), so it is authored directly rather than
generated from all_included_curve_points.csv.
"""
from __future__ import annotations
import csv
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "SOURCE_DATA" / "plot_selection_manifest.csv"

FIELDS = ["item", "status", "figure", "branch_or_source", "reason"]

ROWS = [
    dict(item="A_NATIVE, R in {-0.95, 0.1, 0.5}, Kmax in {12,15,18,24}",
         status="INCLUDED", figure="Figure 1 (A/B), Figure 2, Figure 4A, Figure 5A",
         branch_or_source="codex/v10.2.30-R-ratio-nominal-deltaK @ 4b554006d60539d7dec659f4bfe4bf298728780a",
         reason="Qualified developed rate table, single seed 1720, PHYSICAL_TARGET_REACHED/REUSED_PHYSICAL_TARGET_REACHED terminal status."),
    dict(item="PT03, PT08 substitutions, same R/Kmax grid",
         status="INCLUDED", figure="Figure 2, Figure 5A",
         branch_or_source="codex/v10.2.30-R-ratio-nominal-deltaK @ 4b554006d60539d7dec659f4bfe4bf298728780a",
         reason="Same qualified table as A_NATIVE. Plotted and labeled as Peierls/Taylor transport-mechanism substitutions, NOT distinct calibrated materials (PT_R_INVARIANT classification already established in source campaign's decision doc)."),
    dict(item="Part X D1/D2/D3/D5/D6_conditional_persistent (+ 2nd-seed confirms) finite branch",
         status="INCLUDED", figure="Figure 3 (A/B/C), Figure 3 Sh companion, Figure 4B",
         branch_or_source="codex/v10.2.30-crack-rebonding-part-x @ 30db009ff7172728a6bdc885886f21b94cbec225 (accepted terminal HEAD)",
         reason="PX4 stage-1 developed rate table, all rows both_stable_growth=True. Covers reversible/persistent/passivation-limited rebonding as required."),
    dict(item="Part X matched zero-cohesion baselines for D1/D2/D3/D5/D6_conditional_persistent",
         status="INCLUDED", figure="Figure 3 (A/B/C), Figure 3 Sh companion, Figure 4B",
         branch_or_source="codex/v10.2.30-crack-rebonding-part-x @ 30db009ff7172728a6bdc885886f21b94cbec225",
         reason="Same-row matched baseline (identical Kmax/R/frequency/seed) needed to compute the shielding metric S_h and to show the with/without-rebonding contrast."),
    dict(item="Part X zero-cohesion baseline at R=-0.5 as an addition to Figure 1",
         status="CONSIDERED_AND_EXCLUDED", figure="(would have been Figure 1)",
         branch_or_source="codex/v10.2.30-crack-rebonding-part-x @ 30db009ff7172728a6bdc885886f21b94cbec225",
         reason="Visible label is 'A_NATIVE'-like but provenance hash scheme differs entirely from the R-ratio study (physical_producer_sha/config_hash/material_row_hash/physical_source_bundle_sha256 vs. production_solver_hash/common_physics_hash/composite_hash), so shared material fingerprint, temperature, frequency, event-length semantics, developed-window definition, and physical source bundle could NOT be proven equal. Mission requires this proof before combining; not combined. Recorded as an unavailable comparison."),
    dict(item="Peak/DBTT/ceramic-like/weak-T per-point developed da/dN(K) curves",
         status="CONSIDERED_AND_EXCLUDED", figure="(would have been Figure 5A/5B as full curves)",
         branch_or_source="unresolved -- searched codex/v10.2.30-physical-slope-transfer, codex/v10.2.30-A-native-TP-panel, codex/v10.2.30-joint-fracture-fatigue-archetype-atlas; no committed or archived per-point table located",
         reason="Only the summary global-slope values (Peak~100.5, DBTT~133.5, ceramic-like~71.7; weak-T no qualified fit) were locatable, from a distinct v9.13/v9.14 30-parameterization study's family-median summary, not a per-point developed table. Drawing a full curve would require fabricating intermediate points. Represented in Figure 5B as reference slope-guide LINES only, anchored at an arbitrary offset and explicitly labeled as such, using only the verified slope value -- not as an implied full developed curve."),
    dict(item="physical-slope-transfer target-M~4 row",
         status="CONSIDERED_AND_EXCLUDED", figure="(would have been Figure 5A)",
         branch_or_source="codex/v10.2.30-physical-slope-transfer",
         reason="Branch has no committed artifacts beyond artifacts/v10_2_27_kernel_registry.json; grep/find across the filesystem for 'physical_slope_transfer' found nothing. No committed or portable per-point table exists. Labeled UNAVAILABLE directly on Figure 5A rather than omitted silently."),
    dict(item="Invalidated dwell-duration-weighting trajectories (Part X)",
         status="CONSIDERED_AND_EXCLUDED", figure="(none)",
         branch_or_source="codex/v10.2.30-crack-rebonding-part-x",
         reason="Explicitly invalidated (duration-weighting bug) in the source campaign; mission prohibits including these in a developed-curve comparison."),
    dict(item="PX3 screening-stage rate points for protocols where a developed PX4 result exists",
         status="CONSIDERED_AND_EXCLUDED", figure="(none)",
         branch_or_source="codex/v10.2.30-crack-rebonding-part-x",
         reason="PX4 stage-1 is the developed, qualified result for D1/D2/D3/D5/D6; PX3 is a short screening trajectory superseded by PX4 for these protocols and is excluded per the mission's data-provenance rules."),
    dict(item="Closure-corrected DeltaK_eff (any resolved opening/contact criterion)",
         status="CONSIDERED_AND_EXCLUDED", figure="(none)",
         branch_or_source="n/a",
         reason="The model has no validated resolved opening/contact criterion (Part X uses a signed-K contact surrogate that does not resolve opposing crack-face contact; the R-ratio study omits DeltaK_eff without a validated opening criterion per its own decision doc). No DeltaK_eff quantity appears anywhere in this package -- verified as an explicit validation checkpoint."),
    dict(item="Manuscript Figures 1-7, FEM/CZM R-curves, Cramer's V, AUC, S-N, field maps, identifiability plots, monotonic fracture-temperature curves",
         status="CONSIDERED_AND_EXCLUDED", figure="(none)",
         branch_or_source="n/a (out of scope by mission definition)",
         reason="Explicitly out of scope for this focused da/dN(K) package per the mission statement; not assembled, not copied into this directory."),
]


def main() -> None:
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(ROWS)
    print(f"Wrote plot_selection_manifest.csv: {len(ROWS)} rows")


if __name__ == "__main__":
    main()
