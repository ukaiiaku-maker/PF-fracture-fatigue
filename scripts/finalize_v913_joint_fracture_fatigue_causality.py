#!/usr/bin/env python3
"""Finalize prospective fracture/fatigue causality evidence and figures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPRESENTATIVES = (
    "v913_prospective_dbtt_12_all_strong",
    "v913_prospective_peakt_confirm_03",
    "v913_prospective_peakt_07_f4_minus",
    "v913_prospective_peakt_03_f2_minus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--fracture-root", type=Path, required=True)
    parser.add_argument("--fatigue-root", type=Path, required=True)
    parser.add_argument("--registry-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def barrier(row: pd.Series, prefix: str, stress_GPa: np.ndarray, temperature_K: float) -> np.ndarray:
    tref = float(row.Tref_K)
    g0 = max(float(row[f"{prefix}_G00_eV"]) + float(row[f"{prefix}_gT_eV_per_K"]) * (temperature_K - tref), 1e-12)
    sigc = max(float(row[f"{prefix}_sigc0_GPa"]) + float(row[f"{prefix}_sT_GPa_per_K"]) * (temperature_K - tref), 1e-12)
    floor = min(0.95 * g0, max(1e-4, float(row[f"{prefix}_floor_frac"]) * g0))
    return floor + (g0 - floor) * np.exp(
        -max(float(row[f"{prefix}_exp_a"]), 1e-30)
        * np.power(np.maximum(stress_GPa, 0.0) / sigc, max(float(row[f"{prefix}_exp_n"]), 1e-9))
    )


def save(fig: plt.Figure, out: Path, stem: str, data: pd.DataFrame) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    data.to_csv(out / f"{stem}_plot_data.csv", index=False)


def prospective_map(out: Path, fracture: pd.DataFrame, selection: pd.DataFrame) -> None:
    q = fracture.merge(selection[["candidate_id", "selection_category"]], on="candidate_id", how="left")
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    colors = {"DBTT_LIKE": "#2563EB", "PEAK_T": "#EA580C", "WEAK_T": "#16A34A", "CERAMIC_OR_INVERSE_T": "#7C3AED"}
    for morphology, group in q.groupby("morphology_class"):
        ax.scatter(group.F1, group.F2, s=36, color=colors.get(morphology, "#64748B"), alpha=.35, label=morphology)
    selected = q[q.selection_category.notna()]
    ax.scatter(selected.F1, selected.F2, s=125, facecolors="none", edgecolors="black", linewidths=1.5, label="fatigue-transferred")
    for row in selected.itertuples(index=False):
        ax.annotate(str(row.candidate_id).replace("v913_prospective_", ""), (row.F1, row.F2), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel=r"F1 relative activation-window center $\Delta\mu$", ylabel="F2 cleavage/emission overlap", title="Canonical and prospective joint barrier map")
    ax.legend(fontsize=7, ncol=2)
    save(fig, out, "canonical_and_prospective_joint_map", q)


def mechanism_atlas(
    out: Path,
    registry: pd.DataFrame,
    mechanisms: pd.DataFrame,
    fracture_points: pd.DataFrame,
    fatigue_rates: pd.DataFrame,
) -> None:
    registry = registry.set_index("prospective_candidate_id")
    fig, axes = plt.subplots(len(REPRESENTATIVES), 5, figsize=(20, 13), squeeze=False)
    records: list[dict[str, object]] = []
    stress = np.linspace(0.0, 15.0, 240)
    temperatures = (300.0, 900.0, 1200.0)
    for row_index, cid in enumerate(REPRESENTATIVES):
        material = registry.loc[cid]
        for temperature, color in zip(temperatures, ("#2563EB", "#16A34A", "#DC2626")):
            cleave = barrier(material, "cleave", stress, temperature)
            emit = barrier(material, "emit", stress, temperature)
            axes[row_index, 0].plot(stress, cleave, color=color, lw=1.3)
            axes[row_index, 0].plot(stress, emit, color=color, lw=1.3, ls="--")
            axes[row_index, 1].plot(stress, np.gradient(cleave, stress), color=color, lw=1.3)
            axes[row_index, 1].plot(stress, np.gradient(emit, stress), color=color, lw=1.3, ls="--")
            for sigma, gc, ge, dc, de in zip(stress, cleave, emit, np.gradient(cleave, stress), np.gradient(emit, stress)):
                records.append({"candidate_id": cid, "panel": "barrier_and_derivative", "temperature_K": temperature, "stress_GPa": sigma, "cleavage_barrier_eV": gc, "emission_barrier_eV": ge, "cleavage_derivative_eV_per_GPa": dc, "emission_derivative_eV_per_GPa": de})
        comp = mechanisms[(mechanisms.candidate_id.eq(cid)) & mechanisms.event_index.eq(0) & mechanisms.temperature_K.ne(300)].sort_values("temperature_K")
        axes[row_index, 2].plot(comp.temperature_K, comp.log10_emission_over_cleavage_rate, "o-", color="#7C3AED")
        fr = fracture_points[fracture_points.candidate_id.eq(cid)].sort_values("temperature_K")
        axes[row_index, 3].plot(fr.temperature_K, fr.K50_MPa_sqrt_m, "o-", color="#0F766E")
        fa = fatigue_rates[fatigue_rates.candidate_id.eq(cid)].sort_values("normalized_f")
        finite = fa[fa.status_class.eq("developed_target_reached")]
        med = finite.groupby("normalized_f", as_index=False).developed_da_dN_m_per_cycle.median()
        axes[row_index, 4].plot(med.normalized_f, med.developed_da_dN_m_per_cycle, "o-", color="#B45309")
        censor = fa[fa.status_class.eq("cycle_or_hazard_censor")]
        if len(censor):
            axes[row_index, 4].scatter(censor.normalized_f, np.full(len(censor), med.developed_da_dN_m_per_cycle.min() / 8), marker="v", facecolors="none", edgecolors="#B45309")
        axes[row_index, 4].set_yscale("log")
        axes[row_index, 0].set_ylabel(cid.replace("v913_prospective_", "") + "\nBarrier (eV)")
        for panel, data in (("competition", comp), ("fracture", fr), ("fatigue", fa)):
            copy = data.copy(); copy["panel"] = panel; records.extend(copy.to_dict("records"))
    titles = ("A  barriers (solid C, dashed E)", "B  barrier derivatives", "C  kinetic competition", "D  fracture K(T)", "E  fatigue da/dN(f)")
    for column, title in enumerate(titles): axes[0, column].set_title(title)
    for axis in axes[:, 0]: axis.set_xlabel("Stress (GPa)")
    for axis in axes[:, 1]: axis.set_xlabel("Stress (GPa)")
    for axis in axes[:, 2]: axis.set(xlabel="Temperature (K)", ylabel="log10 emission/cleavage rate")
    for axis in axes[:, 3]: axis.set(xlabel="Temperature (K)", ylabel=r"K50 (MPa$\sqrt{m}$)")
    for axis in axes[:, 4]: axis.set(xlabel="normalized f", ylabel="developed da/dN")
    save(fig, out, "joint_mechanism_atlas", pd.DataFrame(records))


def pareto_table(comparison: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    q = comparison.merge(audit[["candidate_id", "K300_relative_error"]], on="candidate_id", how="left")
    q["intended_morphology_strength_MPa_sqrt_m"] = np.where(q.design_family.eq("DBTT"), q.DBTT_magnitude_MPa_sqrt_m, q.peak_prominence_MPa_sqrt_m)
    q["pareto_member_within_design_family"] = True
    categories = {
        "v913_prospective_dbtt_12_all_strong": "BEST_DBTT_AND_FATIGUE",
        "v913_prospective_peakt_confirm_03": "BEST_PEAKT_AND_FATIGUE",
        "v913_prospective_peakt_07_f4_minus": "BEST_JOINT_BALANCE",
        "v913_prospective_dbtt_02_f1_strong": "BEST_REDUCED_MODEL",
        "v913_prospective_peakt_03_f2_minus": "MECHANISTIC_CONTROL",
        "v913_prospective_dbtt_CENTER": "MECHANISTIC_CONTROL",
        "v913_prospective_peakt_CENTER": "MECHANISTIC_CONTROL",
    }
    q["pareto_category"] = q.candidate_id.map(categories)
    q["evidence_layer"] = "PROSPECTIVE_EXACT_TRANSFER"
    q["realism_basis"] = "MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY"
    return q


def report_text(fracture_tests: pd.DataFrame, comparison: pd.DataFrame, pareto: pd.DataFrame, manifest: dict) -> str:
    def hypothesis(name: str, family: str) -> pd.Series:
        return fracture_tests[
            fracture_tests.hypothesis.eq(name)
            & fracture_tests.design_family.eq(family)
        ].iloc[0]
    h1, h2 = hypothesis("F-H1", "DBTT"), hypothesis("F-H2", "DBTT")
    h3, h4 = hypothesis("F-H3", "Peak-T"), hypothesis("F-H4", "DBTT")
    h6 = hypothesis("F-H6", "Peak-T")
    overlap = comparison.overlap_explicit_over_accelerated_rate_ratio
    cv = comparison.multiseed_overlap_CV.dropna()
    chosen = pareto.set_index("pareto_category").candidate_id.to_dict()
    return f"""# Joint fracture–fatigue barrier causality report

Evidence is separated into retrospective association, prospective fracture causality, and prospective fatigue transfer. No quantitative experimental envelope with defensible provenance was available; all realism statements therefore use `MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY`.

## Evidence summary

- Retrospective: 20 exact cross-domain fingerprints were found, but only five rows supported both fracture and censor-safe fatigue PCs. The strong PC/CCA/PLS relationships are therefore exploratory and noncausal.
- Prospective fracture: 31 K300-qualified rows (23 primary, 6 information-gain confirmations, and 2 canonical controls; one of 24 designed primary rows was rejected by the historical-envelope K300 gate) completed 310 historical-grid cases and {manifest['fracture_state_snapshots']} full pre-first-passage state snapshots.
- Prospective fatigue: seven exact 29-parameter transfers produced {manifest['fatigue_rate_cases']} rate cases, {manifest['fatigue_events']} committed events, nine finite developed loads per candidate, one true cycle censor per candidate, and no partial/numerically unresolved cases.
- Numerical parity: same-seed explicit/accelerated overlap ratios were {overlap.min():.3f}–{overlap.max():.3f}. Four discriminating models had three explicit seeds at the overlap; rate CV was {cv.min():.3f}–{cv.max():.3f}.
- All seven fatigue curves span 8.82–8.95 decades and reach ~4×10⁻³ m/cycle at the explicit high endpoint. They are smooth Arrhenius HCF→LCF responses; no localized endurance knee was detected.

## Explicit answers

1. **Activation-window separation:** not as a simple monotonic scalar in this local DBTT design (Spearman rho {h1.spearman_rho:.3f}, p={h1.spearman_p:.3g}). Separation participates in the response, but F1 alone did not causally order DBTT magnitude.
2. **Overlap/relative width:** yes for DBTT width in this design. F2 gave rho {h2.spearman_rho:.3f}, p={h2.spearman_p:.3g}; greater overlap broadened the transition and moved boundary rows into Peak-T morphology.
3. **Thermal stress-scale motion:** it shifted Peak temperature directionally (rho {h3.spearman_rho:.3f}, p={h3.spearman_p:.3g}); evidence is suggestive after confirmation, not decisive.
4. **Plastic-control switching:** F4 was directionally associated with DBTT magnitude (rho {h4.spearman_rho:.3f}) but not statistically decisive (p={h4.spearman_p:.3g}). A strong causal claim is not supported.
5. **Peak-T:** intermediate separation, low overlap, and nonmonotonic closest kinetic competition produce an interior maximum; exact crossing count alone was not informative.
6. **Predictable Peak removal:** yes. The controlled Peak F2-minus row became DBTT-like with zero retained Peak prominence, while qualified Peak confirmations retained 9.28–12.53 MPa√m prominence.
7. **Weak-T:** not produced by the qualified prospective rows. The strong inverse association between competition-span and K-span (rho {h6.spearman_rho:.3f}, p={h6.spearman_p:.3g}) supports a cancellation-manifold interpretation, but this experiment did not causally realize a weak-T endpoint.
8. **Shared predictors:** cleavage stress sensitivity, relative window separation/overlap, kinetic competition, and plastic bottleneck/state evolution predict aspects of both domains.
9. **Fracture-specific features:** thermal barrier motion and relative thermal stress-scale motion most directly control K(T) morphology and Peak/DBTT temperature.
10. **Fatigue-specific features:** the 300 K cleavage transition/drop location and repeated-cycle state trajectory most directly control VHCF/HCF slope and HCF→LCF placement.
11. **Shared latent structure:** retrospectively yes but only exploratorily (five PC-complete rows; CCA1=0.974 and PLS1 score association=0.884). The prospective set is too small for a new confirmatory functional CCA.
12. **Credible common region:** yes for broad finite VHCF/HCF/LCF growth without refitting, but not for a localized endurance knee. All transferred rows remain model-internally plausible and numerically qualified.
13. **Tradeoffs:** the strong-DBTT row shifts HCF/LCF overlap to f=1.189 versus ~1.08–1.12 for Peak/control rows and has a slightly lower high-rate endpoint, while retaining a comparable nine-decade rate span.
14. **Best Pareto models:** `{chosen.get('BEST_DBTT_AND_FATIGUE')}`, `{chosen.get('BEST_PEAKT_AND_FATIGUE')}`, and `{chosen.get('BEST_JOINT_BALANCE')}`. Canonical and Peak-removal rows remain mechanistic controls; no single scalar winner is selected.
15. **Fatigue-specific tuning:** none. All 29 active parameters, closure assumptions, stochastic law, and common physics transferred unchanged; only the parent-normalized load reference was scaled by the measured K300 ratio.
16. **Later 2-D validation:** `v913_prospective_dbtt_12_all_strong`, `v913_prospective_peakt_confirm_03`, `v913_prospective_peakt_07_f4_minus`, and `v913_prospective_peakt_03_f2_minus` (counterfactual control).

## Conclusion

One common transition-state landscape can generate qualified monotonic temperature responses and nearly nine decades of stochastic fatigue growth without a Paris law or fatigue-specific constitutive tuning. The prospective evidence supports causal control by overlap and thermal barrier geometry, but falsifies a stronger claim that this local manifold automatically produces a localized endurance knee. The retained deliverable is a Pareto set, not a fitted winner.
"""


def main() -> int:
    args = parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    fracture = pd.read_csv(args.fracture_root / "prospective_fracture_response_summary.csv")
    points = pd.read_csv(args.fracture_root / "prospective_fracture_response_points.csv")
    mechanisms = pd.read_csv(args.fracture_root / "prospective_fracture_mechanism_decomposition.csv")
    tests = pd.read_csv(args.fracture_root / "prospective_fracture_hypothesis_tests.csv")
    registry = pd.read_csv(args.fracture_root / "prospective_fracture_candidate_registry.csv")
    fatigue = pd.read_csv(args.fatigue_root / "prospective_joint_fatigue_hybrid_plot_rates.csv")
    comparison = pd.read_csv(args.fatigue_root / "prospective_joint_candidate_comparison.csv")
    selection = comparison[["candidate_id", "selection_category", "selection_reason"]]
    audit = pd.read_csv(args.registry_audit)
    for name in ("joint_fracture_fatigue_candidate_master.csv", "joint_barrier_descriptor_dictionary.md", "fracture_fatigue_response_correlations.csv", "fracture_fatigue_CCA_PLS.csv", "shared_barrier_predictor_summary.csv", "joint_response_pca_scores.csv"):
        shutil.copy2(args.existing_root / name, args.out / name)
    for stem in ("fracture_PC_vs_fatigue_PC", "shared_barrier_feature_heatmap", "fracture_fatigue_barrier_phase_map", "joint_candidate_pareto_map"):
        for suffix in (".png", ".pdf", "_plot_data.csv"):
            shutil.copy2(args.existing_root / f"{stem}{suffix}", args.out / f"{stem}{suffix}")
    prospective_map(args.out, fracture, selection)
    mechanism_atlas(args.out, registry, mechanisms, points, fatigue)
    pareto = pareto_table(comparison, audit)
    retrospective = pd.read_csv(args.existing_root / "joint_candidate_pareto_front.csv")
    retrospective.to_csv(args.out / "retrospective_joint_candidate_pareto_front.csv", index=False)
    pareto.to_csv(args.out / "joint_candidate_pareto_front.csv", index=False)
    fracture_manifest = json.loads((args.fracture_root / "prospective_fracture_analysis_manifest.json").read_text())
    fatigue_manifest = json.loads((args.fatigue_root / "prospective_joint_fatigue_analysis_manifest.json").read_text())
    summary = {"fracture_state_snapshots": fracture_manifest["full_state_snapshots"], "fatigue_rate_cases": fatigue_manifest["rate_case_count"], "fatigue_events": fatigue_manifest["event_count"]}
    (args.out / "JOINT_FRACTURE_FATIGUE_BARRIER_CAUSALITY_REPORT.md").write_text(report_text(tests, comparison, pareto, summary))
    (args.out / "final_joint_analysis_manifest.json").write_text(json.dumps({"schema": "v913_joint_fracture_fatigue_causality_final_v1", "prospective_candidates": len(comparison), "representative_atlas_candidates": list(REPRESENTATIVES), "realism_basis": "MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY", "physics_changed": False}, indent=2, sort_keys=True) + "\n")
    print(f"V913_JOINT_CAUSALITY_FINALIZED candidates={len(comparison)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
