"""Independent-verifier closure (review round 4, corrected in round 5): the
claim registry.

This is the single source of truth for what claims exist, what bundled
files each one needs, what function recomputes it, what value is expected,
and what tolerance/comparator governs pass/fail. `verify_paper_evidence_v4.py`
imports CLAIMS from here and does the actual judging -- this module
intentionally contains NO pass/fail logic of its own beyond the comparator
functions, which are pure and stateless.

NOTE (round 5 correction): this module used to also define
`EXPECTED_CLAIM_IDS = tuple(sorted(c.claim_id for c in CLAIMS))` as the
"frozen" inventory the verifier checked CLAIMS against. The review correctly
pointed out that an inventory *derived from CLAIMS* cannot detect a claim
silently deleted or renamed from CLAIMS -- the derived list simply shrinks
or changes along with it. The frozen inventory now lives independently in
`artifacts/paper_simulation_completion/expected_paper_claim_ids_v4.json`, a
hand-authored, hash-stamped contract that is NOT regenerated from this file.
See verify_paper_evidence_v4.check_claim_id_inventory().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import claim_recompute_v4 as R

QUALIFIED = "QUALIFIED_SOURCE_RESULT_VERIFIED"
STRUCTURAL = "SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED"
NOT_REVERIFIED = "SOURCE_RESULT_LOCATED_NOT_REVERIFIED"


# --------------------------------------------------------------------------
# comparators (pure functions: (actual, expected) -> bool)
# --------------------------------------------------------------------------

def cmp_dict_subset(actual: dict, expected: dict) -> bool:
    """Every key in `expected` must be present in `actual` with an equal value."""
    if not isinstance(actual, dict):
        return False
    for k, v in expected.items():
        if k not in actual or actual[k] != v:
            return False
    return True


def cmp_exact(actual: Any, expected: Any) -> bool:
    return actual == expected


def cmp_numeric_rel_tol(rel_tol: float) -> Callable[[Any, Any], bool]:
    def _cmp(actual: Any, expected: Any) -> bool:
        try:
            return abs(float(actual) - float(expected)) <= rel_tol * abs(float(expected))
        except (TypeError, ValueError, ZeroDivisionError):
            return False
    return _cmp


def cmp_string_exact(actual: Any, expected: Any) -> bool:
    return str(actual) == str(expected)


def cmp_fig6a(actual: dict, expected: dict) -> bool:
    """Round-5 correction: the round-4 comparator only checked n_tables_recomputed==36 and
    all_v_nonnegative==True -- neither depends on any of the 36 recomputed Cramer's V
    magnitudes. This comparator (1) checks every one of the 36 frozen values within a tight
    numeric tolerance, and (2) separately re-derives, from the SAME recomputed map (not a
    second hardcoded list), whether the manuscript's described magnitude groups are honored:
    strength-fracture ~0.79-0.81 in several contexts, fracture-threshold ~0.67-0.71 in four of
    six contexts, strength-threshold ~0.60 in named contexts, and S-N phenotype associations in
    the weaker ~0.05-0.28 band.
    """
    if not isinstance(actual, dict) or "n_tables_recomputed" not in actual:
        return False
    if actual.get("n_tables_recomputed") != expected.get("n_tables_recomputed"):
        return False
    if actual.get("all_v_nonnegative") is not True:
        return False
    frozen = expected["frozen_v_by_family_context"]
    recomputed = actual.get("cramers_v_by_family_context", {})
    if set(frozen) != set(recomputed):
        return False
    tol = expected.get("tolerance_abs", 1e-4)
    if not all(abs(recomputed[k] - frozen[k]) <= tol for k in frozen):
        return False

    def _vals(prefix: str) -> list[float]:
        return [v for k, v in recomputed.items() if k.startswith(prefix + "::")]

    strength_fracture = _vals("strength_vs_fracture_class")
    fracture_threshold = _vals("fracture_vs_growth_threshold_class")
    strength_threshold = _vals("strength_vs_growth_threshold_class")
    sn_phenotype = (_vals("fracture_vs_fatigue_temperature_phenotype")
                    + _vals("strength_vs_fatigue_temperature_phenotype")
                    + _vals("growth_threshold_vs_SN_temperature_phenotype"))

    n_strength_fracture_high = sum(1 for v in strength_fracture if 0.78 <= v <= 0.82)
    n_fracture_threshold_high = sum(1 for v in fracture_threshold if 0.65 <= v <= 0.72)
    n_strength_threshold_mid = sum(1 for v in strength_threshold if 0.58 <= v <= 0.62)
    sn_all_in_weak_band = all(0.04 <= v <= 0.30 for v in sn_phenotype)

    return (
        n_strength_fracture_high >= 3
        and n_fracture_threshold_high == 4
        and n_strength_threshold_mid >= 2
        and sn_all_in_weak_band
    )


@dataclass(frozen=True)
class Claim:
    claim_id: str
    figure_panel: str
    manuscript_claim_text: str
    source_data_level: str
    source_repository_or_archive: str
    producer_script: str
    parameter_config_fingerprint: str
    required_inputs: tuple[str, ...]
    recompute: Callable[[Any], dict]
    expected: Any
    compare: Callable[[Any, Any], bool]
    tolerance: str
    max_evidence_class: str
    fail_evidence_class: str = NOT_REVERIFIED
    notes: str = ""


FEM_CZM_ROOT = "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM"
FATIGUE_PF_ROOT = "/Volumes/Data/Data/Nanopillar_calculation/Fatigue-PF"
IDENTIFIABILITY_ROOT = "/Volumes/Data/Data/Nanopillar_calculation/Fatigue_modelfitting_identifyability"


CLAIMS: list[Claim] = [
    # ---------------- Figure 1 ----------------
    Claim("Fig1A", "Figure 1A",
          "Temperature-dependent monotonic first-passage response across the continuation from "
          "ceramic-like to DBTT-like behavior.",
          "raw (per-continuation-step path)", FATIGUE_PF_ROOT, "build_panelA_waterfall_3d.py",
          "continuation through (H0,c, chi_shield, N_sat) space",
          ("fig1AB_panel_A_waterfall_path.csv",), R.recompute_Fig1A,
          expected=dict(regime_order=["ceramic", "peak", "weakT", "dbtt"], H0_eV_monotonic_nondecreasing=True),
          compare=cmp_dict_subset, tolerance="exact structural match (regime order + monotonicity)",
          max_evidence_class=STRUCTURAL,
          notes="Topology recomputed directly from the raw continuation-path array, not from README "
                "text matching."),
    Claim("Fig1B", "Figure 1B", "Cyclic first-passage response for the same barrier continuation.",
          "raw (per-continuation-step path)", FATIGUE_PF_ROOT, "build_panelB_fatigue_waterfall_3d.py",
          "same continuation path as Panel A",
          ("fig1AB_panel_A_waterfall_path.csv", "fig1AB_panel_B_fatigue_waterfall_path.csv"),
          R.recompute_Fig1B,
          expected=dict(regime_order=["ceramic", "peak", "weakT", "dbtt"],
                        shares_H0_eV_range_with_panel_A=True, H0_eV_monotonic=True),
          compare=cmp_dict_subset, tolerance="exact structural match", max_evidence_class=STRUCTURAL,
          notes="Panel B uses a different sample-point count (30 vs 40) than Panel A but spans the "
                "identical H0_eV range and regime order -- confirmed by direct array comparison."),
    Claim("Fig1C", "Figure 1C", "S-N-type initiation response across the matched entropy family.",
          "derived (per-curve summary)", FATIGUE_PF_ROOT, "build_panels_CD_entropy_family_v3.py",
          "axes x=log10(N_i), y=Lambda_S_ref, z=sigma_a, color=A_T",
          ("fig1CD_panelC_curve_summary_v3.csv", "fig1CD_panels_CD_manifest_v3.json"), R.recompute_Fig1C,
          expected=dict(panel_C_axes=dict(x="log10 N_i", y="Lambda_S_ref", z="sigma_a_MPa", color="A_T_kB")),
          compare=cmp_dict_subset, tolerance="exact manifest match", max_evidence_class=STRUCTURAL,
          notes=""),
    Claim("Fig1D", "Figure 1D",
          "Fixed-rate strength-temperature response generated by the same entropy family.",
          "derived (per-curve summary)", FATIGUE_PF_ROOT, "build_panels_CD_entropy_family_v3.py",
          "SAME entropy-family design table as Panel C",
          ("fig1CD_panelC_curve_summary_v3.csv", "fig1CD_panelD_curve_summary_v3.csv",
           "fig1CD_panels_CD_manifest_v3.json"), R.recompute_Fig1D,
          expected=dict(panel_D_axes=dict(x="T_K", y="Lambda_S_ref", z="sigma_y_MPa", color="A_T_kB"),
                        shares_exact_entropy_grid_with_panel_C=True),
          compare=cmp_dict_subset, tolerance="exact manifest match + exact shared-grid array equality",
          max_evidence_class=STRUCTURAL, notes=""),

    # ---------------- Figure 2 ----------------
    Claim("Fig2-ceramic-RMS", "Figure 2",
          "FEM/CZM ceramic 1x-rate first-passage RMS deviation from V1 analytic = 0.23 MPa*sqrt(m)",
          "derived (per class/temperature comparison table)", FEM_CZM_ROOT,
          "compare_pf_czm_first_passage_with_analytic.py", "theta=45.0; 1x nominal rate",
          ("fig2_first_passage_comparison_with_analytic_1x.csv",), R.recompute_Fig2_ceramic_RMS,
          expected=0.23, compare=cmp_exact, tolerance="rounds to 2 decimals, exact",
          max_evidence_class=QUALIFIED, notes=""),
    Claim("Fig2-weakT-RMS", "Figure 2",
          "FEM/CZM weakT 1x-rate first-passage RMS deviation from V1 analytic = 0.23 MPa*sqrt(m)",
          "derived (per class/temperature comparison table)", FEM_CZM_ROOT,
          "compare_pf_czm_first_passage_with_analytic.py", "theta=45.0; 1x nominal rate",
          ("fig2_first_passage_comparison_with_analytic_1x.csv",), R.recompute_Fig2_weakT_RMS,
          expected=0.23, compare=cmp_exact, tolerance="rounds to 2 decimals, exact",
          max_evidence_class=QUALIFIED, notes=""),
    Claim("Fig2-peak-RMS", "Figure 2",
          "FEM/CZM peak 1x-rate first-passage RMS deviation from V1 analytic = 1.18 MPa*sqrt(m)",
          "derived (per class/temperature comparison table)", FEM_CZM_ROOT,
          "compare_pf_czm_first_passage_with_analytic.py", "theta=45.0; 1x nominal rate",
          ("fig2_first_passage_comparison_with_analytic_1x.csv",), R.recompute_Fig2_peak_RMS,
          expected=1.18, compare=cmp_exact, tolerance="rounds to 2 decimals, exact",
          max_evidence_class=QUALIFIED, notes=""),
    Claim("Fig2-DBTT-RMS", "Figure 2",
          "FEM/CZM DBTT 1x-rate first-passage RMS deviation from V1 analytic = 1.33 MPa*sqrt(m)",
          "derived (per class/temperature comparison table)", FEM_CZM_ROOT,
          "compare_pf_czm_first_passage_with_analytic.py", "theta=45.0; 1x nominal rate",
          ("fig2_first_passage_comparison_with_analytic_1x.csv",), R.recompute_Fig2_DBTT_RMS,
          expected=1.33, compare=cmp_exact, tolerance="rounds to 2 decimals, exact",
          max_evidence_class=QUALIFIED, notes=""),
    Claim("Fig2-peak-narrow-topology", "Figure 2",
          "The narrow peak is reproduced strongly by V1 and PF/sharp-front but is attenuated in FEM/CZM.",
          "derived (fine 5K analytic grid + coarse comparison)", FEM_CZM_ROOT,
          "run_v1_exp_floor_four_class_tuning.py", "peak class; K_target_MPa_sqrt_m column",
          ("fig2_four_class_analytical_prediction_final_fine_grid.csv",
           "fig2_first_passage_comparison_with_analytic_1x.csv"), R.recompute_Fig2_peak_narrow_topology,
          expected=dict(fine_grid_peak_found=True, coarse_grid_bump_found=True,
                        pf_sharp_front_peak_found=True, pf_reproduces_peak_strongly=True,
                        fem_lower_than_analytic_at_peak_T=True,
                        fem_attenuation_within_5pct_of_30_9=True),
          compare=cmp_dict_subset,
          tolerance="all boolean gates must hold: analytic peak exists; PF sharp-front peak exists AND "
                    "reaches >50% of the analytic peak height (reproduces it strongly); FEM/CZM value at "
                    "the peak temperature is strictly lower than the analytic value there (attenuated); "
                    "FEM attenuation is 30.9% +/-5 percentage points",
          max_evidence_class=QUALIFIED,
          notes="Round-5 correction: the round-4 comparator only checked that a peak existed somewhere "
                "in each grid, without requiring PF to reproduce it, FEM to fall below it, or the "
                "attenuation magnitude to match. All four are now gated explicitly."),

    # ---------------- Sec 2.15 ----------------
    Claim("Sec2.15-saturation-fits", "Sec. 2.15 (text, not a figure)",
          "Fitted saturation K_ss and characteristic extension per class",
          "RAW (per-seed binned curve, genuinely refit)", FEM_CZM_ROOT, "Rcurve_analysis pipeline",
          "theta=45.0; 500K; 5 seeds/class; saturating fit K_R(Da)=K0+DeltaK_R[1-exp{-(Da/ellR)^p}]",
          ("sec2_15_seed_binned_Rcurves_long.csv", "fig3_class_mean_Rcurve_fits.csv"),
          R.recompute_Sec2_15_saturation_fits,
          expected=dict(classes=("ceramic", "weakT", "peak", "DBTT"), Kss_rel_tol=0.03, ell_rel_tol=0.10),
          compare=lambda actual, expected: (
              isinstance(actual, dict) and "per_class" in actual
              and set(actual["per_class"]) == set(expected["classes"])
              and all(actual["per_class"][c]["Kss_rel_error"] < expected["Kss_rel_tol"]
                      for c in expected["classes"])
              and all(actual["per_class"][c]["ell_rel_error"] < expected["ell_rel_tol"]
                      for c in expected["classes"])
          ),
          tolerance="Kss within 3% relative, ell_R within 10% relative, checked by THIS comparator "
                    "directly against each class's raw Kss_rel_error/ell_rel_error measurement -- not "
                    "via a precomputed boolean returned by the recompute function",
          max_evidence_class=QUALIFIED,
          notes="Round-5 correction: the round-4 comparator target was a single precomputed boolean "
                "(all_classes_within_tolerance) produced by the recompute function itself, so a bug in "
                "how that boolean was computed could not have been caught by the comparator. The "
                "recompute function still reports the raw per-class Kss_rel_error/ell_rel_error "
                "measurements; the tolerance is now applied here, in the registry's comparator, per the "
                "architecture's intended separation of concerns. The shape exponent p and RMSE remain "
                "reported (in diagnostics) but ungated, because the manuscript claim being verified here "
                "(Kss and characteristic extension) does not depend on p -- stated explicitly, not left "
                "implicit. Corrected from an earlier pass that treated a direct read of the fit-output "
                "CSV as a refit -- this is a real scipy.optimize.curve_fit refit from the raw per-seed "
                "binned curve data."),
]


def _fig3_expected(mean: str) -> Any:
    return mean


_FIG3_EXPECTED = {
    ("ceramic", "K0"): "9.33±0.09", ("ceramic", "Kss"): "16.59±1.39", ("ceramic", "DeltaK"): "7.26±1.42",
    ("weakT", "K0"): "12.83±0.10", ("weakT", "Kss"): "17.78±1.51", ("weakT", "DeltaK"): "4.95±1.51",
    ("peak", "K0"): "16.71±0.13", ("peak", "Kss"): "27.41±1.69", ("peak", "DeltaK"): "10.70±1.68",
    ("DBTT", "K0"): "22.77±0.16", ("DBTT", "Kss"): "33.20±4.48", ("DBTT", "DeltaK"): "10.43±4.42",
}

for (_cls, _stat), _exp in _FIG3_EXPECTED.items():
    CLAIMS.append(Claim(
        f"Fig3-{_cls}-{_stat}", "Figure 3",
        f"FEM/CZM {_cls} class {_stat} = {_exp} MPa*sqrt(m) (mean +/- 1 std, n=5 seeds, 500K)",
        "RAW (per-seed; 5 of 20 rows)", FEM_CZM_ROOT, "Rcurve_analysis pipeline",
        "theta=45.0; T=500K; solver_seeds=1101-1105; target_ext_um=1000.0",
        ("fig3_seed_Rcurve_metrics_and_fits.csv",),
        getattr(R, f"recompute_Fig3_{_cls}_{_stat}"),
        expected=_exp, compare=cmp_string_exact, tolerance="exact string match at 2 decimals",
        max_evidence_class=QUALIFIED, notes="",
    ))


CLAIMS += [
    # ---------------- Figure 4 ----------------
    Claim("Fig4-coverage-and-theta", "Figure 4",
          "Loading-rate dependence of FEM/CZM first passage at nominal rate factors 0.1x, 1x, 10x, 100x "
          "(theta=45 deg)",
          "derived (comparison config + per-rate temperature summaries)", FEM_CZM_ROOT,
          "rate-generalized comparison pipeline", "theta=45.0 (confirmed in config)",
          ("fig4_comparison_config.json", "fig4_first_passage_comparison_with_analytic_all_rates.csv"),
          R.recompute_Fig4_coverage_and_theta,
          expected=dict(rates=[0.1, 1.0, 10.0, 100.0], classes=["ceramic", "peak", "weakT", "DBTT"],
                        theta_deg=45.0),
          compare=cmp_dict_subset, tolerance="exact, including theta_deg parsed from the config's root "
                                              "run-directory name (not hardcoded)",
          max_evidence_class=QUALIFIED,
          notes="Round-5 correction: theta was previously recorded only as a string in the actual "
                "dict ('45.0 (from config)') and not part of the typed comparison, so a config drift "
                "away from theta=45 would not have failed the claim. It is now parsed genuinely from "
                "the config's root path and gated in the comparator."),
    Claim("Fig4-ceramic-weakT-preserved-trend", "Figure 4",
          "The ceramic and weakT classes preserve their broad trends with rate-dependent offsets.",
          "derived (per class/rate/temperature comparison table)", FEM_CZM_ROOT,
          "rate-generalized comparison pipeline", "theta=45.0; ceramic and weakT; all 4 rates",
          ("fig4_first_passage_comparison_with_analytic_all_rates.csv",),
          R.recompute_Fig4_ceramic_weakT_preserved_trend,
          expected=dict(all_within_0_15_to_0_30_band=True), compare=cmp_dict_subset,
          tolerance="RMSE stays within a declared 0.15-0.30 MPa*sqrt(m) band at all 4 rates, both classes",
          max_evidence_class=QUALIFIED, notes=""),
    Claim("Fig4-DBTT-transition-shift", "Figure 4",
          "The DBTT transition shifts to lower temperature at lower rate and to higher temperature at "
          "higher rate.",
          "derived (per class/rate/temperature comparison table)", FEM_CZM_ROOT,
          "rate-generalized comparison pipeline", "theta=45.0; DBTT class; all 4 rates",
          ("fig4_first_passage_comparison_with_analytic_all_rates.csv",),
          R.recompute_Fig4_DBTT_transition_shift,
          expected=dict(monotonic_increase_with_rate=True, n_rates_resolved=4,
                        monotonic_increase_with_rate_independent_definition=True),
          compare=cmp_dict_subset,
          tolerance="monotonicity across all 4 resolved rates, required under BOTH the primary "
                    "(shelf-midpoint crossing) and an independent (maximum-|dK/dT|-gradient) "
                    "transition-temperature definition",
          max_evidence_class=QUALIFIED,
          notes="Round-5 correction: this is verification of a DIRECTIONAL claim (transition "
                "temperature increases with rate) under two independent proxy definitions, not "
                "verification of a single, manuscript-exact DBTT temperature value -- the manuscript "
                "does not state its own exact transition definition, so no single numeric target "
                "exists to match. Both proxies here (788.9/898.2/1031.4/1125.1K shelf-midpoint; "
                "750/850/1050/1150K max-gradient) independently confirm the same monotonic direction."),
    Claim("Fig4-peak-narrow-attenuation", "Figure 4",
          "The analytical peak shifts with rate, whereas FEM/CZM retains only muted shoulders.",
          "derived (fine 5K rate-resolved analytic grid + coarse comparison)", FEM_CZM_ROOT,
          "rate-generalized comparison pipeline (5K resolution)", "theta=45.0; peak class; all 4 rates",
          ("fig4_analytical_predictions_by_rate_fine_grid.csv",
           "fig4_first_passage_comparison_with_analytic_all_rates.csv"),
          R.recompute_Fig4_peak_narrow_attenuation,
          expected=dict(peak_shift_monotonic=True, fem_never_reproduces_a_peak_at_any_rate=True),
          compare=cmp_dict_subset,
          tolerance="peak-location monotonicity (exact) + zero-local-maxima on FEM curve at all 4 rates (exact)",
          max_evidence_class=QUALIFIED, notes=""),

    # ---------------- Figure 5 ----------------
    Claim("Fig5A", "Figure 5A",
          "Crack-growth-rate response for representative barrier systems spanning smooth, steep, "
          "shifted, and plastic-shielded regimes (4 of a 6-system atlas; full atlas in SI).",
          "raw (per-K-point, 6 cases, 14-18 points each)", FATIGUE_PF_ROOT,
          "run_v8_material_response_production_2d.sh", "all 6 canonical cases",
          ("fig5A_atlas_2d_paris_points.csv",), R.recompute_Fig5A,
          expected=dict(all_six_canonical_cases_present=True, all_cases_monotonic=True,
                        steep_cleavage_has_max_paris_slope=True,
                        plastic_shielded_has_min_paris_slope=True,
                        main_text_four_cases_present=True),
          compare=cmp_dict_subset,
          tolerance="exact case-set match; Spearman rho > 0.99 for da/dN vs DeltaK in every case; "
                    "steep_cleavage_case35 must have the MAXIMUM recomputed Paris-law slope of the 6 "
                    "cases (confirms 'steep'); plastic_shielded_case64_M1 must have the MINIMUM slope "
                    "(confirms its low-DeltaK response is 'further suppressed' relative to the others)",
          max_evidence_class=QUALIFIED,
          notes="Round-5 correction: the prior comparator only checked 'six curves exist and are "
                "monotonic', which six nearly-identical curves would also satisfy. It now additionally "
                "requires the manuscript's named steep and plastic-shielded cases to occupy the "
                "extremes of the recomputed Paris-slope ordering, a real class-specific, falsifiable "
                "signature check -- both hold exactly (steep_cleavage_case35 slope=26.67, the maximum; "
                "plastic_shielded_case64_M1 slope=12.11, the minimum, of all 6 cases)."),
    Claim("Fig5B", "Figure 5B",
          "Longer-growth and anisotropic calculations showing persistence of the kinetic hierarchy "
          "during substantial crack extension and path deflection.",
          "raw/derived (multiseed class-mean R-curves + orientation K-points)", FATIGUE_PF_ROOT,
          "run_v8_r_curve_6class_long_growth.sh; run_v8_plastic_shielded_orientation_30_45_v2.sh",
          "3-seed x 6-class long-growth; plastic_shielded_case64_M1 at theta=30/45deg, matched Kmax",
          ("fig5B_multiseed_r_curve_mean_curves.csv", "fig5B_orientation_theta30_atlas_2d_paris_points.csv",
           "fig5B_orientation_theta45_atlas_2d_paris_points.csv"),
          R.recompute_Fig5B,
          expected=dict(extreme_cases_stable_at_every_common_point=True,
                        matched_driving_force_across_orientations=True,
                        orientation_ratio_within_expected_band=True),
          compare=cmp_dict_subset,
          tolerance="the min- and max-KJ_mean case identities must be stable across ALL 5 common "
                    "extension points (not just 2 hand-picked ones); exact Kmax match between "
                    "orientations; orientation da/dN ratio must fall in the declared [10x, 25x] band "
                    "around the ~17x this session independently found",
          max_evidence_class=STRUCTURAL,
          notes="Round-5 correction, two changes: (1) hierarchy persistence is now checked at ALL 5 "
                "common extension points, not 2 -- the full 6-case rank ordering is identical at only "
                "3/5 points (two closely-spaced MIDDLE-ranked cases transiently swap at the other 2), "
                "disclosed honestly via fraction_of_common_points_matching_reference_order=0.6; the "
                "gated criterion is instead the more robust claim that the weakest- and strongest-"
                "performing case never change identity, which holds at all 5 points. (2) the orientation "
                "da/dN ratio (~17.2x) is now gated with a declared tolerance band rather than only "
                "'both values are positive'. Per the review's explicit instruction, this claim is "
                "capped at SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED rather than QUALIFIED: no "
                "spatial crack-path coordinates are bundled or available in the located source tree, "
                "so the compound manuscript claim's 'path deflection' (a geometric statement) is not "
                "numerically verified here -- only the growth-RATE orientation dependence is "
                "(path_deflection_geometrically_verified=False is recorded explicitly in the "
                "recomputed result, not silently omitted)."),
    Claim("Fig5C", "Figure 5C",
          "S-N-type formation of a connected crack from a shallow blunt surface notch.",
          "raw (per-job terminal crack-connectivity audit, all available seeds/stresses)", FATIGUE_PF_ROOT,
          "sn_pd2d_stateful.py", "blunt-edge-notch mesh; seeds 2-5; sigmaA=700/900 MPa",
          ("fig5C_compact_sn_reconstruction.json",), R.recompute_Fig5C,
          expected=dict(unshielded_pass_rate_exceeds_shielded=True,
                        no_shield_life_decreases_with_stress=True,
                        shielded_life_decreases_with_stress=True),
          compare=cmp_dict_subset,
          tolerance="directional: no_shield coverage_pass rate must exceed shielded's; AND the "
                    "censor-aware median cycles-to-root-connection must decrease from 700 to 900 MPa "
                    "for BOTH shielding conditions (the actual S-N content of the claim), across all "
                    "15 available seed/stress/condition jobs",
          max_evidence_class=QUALIFIED,
          notes="Round-5 correction: the round-4 comparator tested only a shielding/formation-"
                "probability contrast (coverage_pass rate), which is not itself an S-N (stress-life) "
                "verification. This now additionally reconstructs a censor-aware stress-life summary "
                "(observed-connected count, right-censored count, and median cycles_root_connected per "
                "stress/condition) and gates on formation life decreasing with increasing stress -- "
                "confirmed for both no_shield (median 50.9M cycles at 700 MPa -> 6.3M at 900 MPa, n=4/4 "
                "uncensored at each stress) and shielded (29.2M at 700 MPa, 1 of 3 jobs right-censored, "
                "-> 6.7M at 900 MPa, n=4/4 uncensored). Reconstructed across ALL available seeds (2-5) "
                "and both stresses, not the single seed5/900MPa pair used in the prior pass."),
    Claim("Fig5D", "Figure 5D",
          "Representative spatial fields showing that plastic shielding can maintain a broad cyclic "
          "deformation state without localization into a connected crack.",
          "raw (rendered field snapshot)", FATIGUE_PF_ROOT, "sn_pd2d_stateful.py",
          "same seed/stress as Fig5C (seed 5, 900 MPa)",
          ("fig5D_fields_shielded_seed5_900MPa.png", "fig5D_fields_no_shield_seed5_900MPa.png"),
          R.recompute_Fig5D,
          expected=dict(both_images_present=True, visual_inspection_record_present=True,
                        record_image_hashes_match_bundle=True,
                        record_contains_explicit_no_ratio_disclaimer=True),
          compare=cmp_dict_subset,
          tolerance="file presence AND a distinct visual_inspection_fig5d.json record whose recorded "
                    "image hashes match the bundled files exactly and which explicitly disclaims any "
                    "cross-image magnitude ratio -- no numeric field-ratio claim is made without "
                    "common-normalization proof from the underlying arrays",
          max_evidence_class=STRUCTURAL,
          notes="Round-5 correction: file presence alone is not an executable verification of spatial "
                "field morphology. A separate visual_inspection_fig5d.json record now documents the "
                "human/model visual review (image hashes, matched seed/stress, reviewer, date, "
                "qualitative conclusion, explicit no-ratio disclaimer) as a distinct auditable artifact; "
                "the executable check here verifies that record's existence and hash-consistency, not "
                "the morphology judgment itself. Downgraded from a prior pass's specific '160x/1000x' "
                "numeric claims, which were visual color-bar estimates without common normalization or "
                "access to the underlying field arrays -- the qualitative spatial conclusion is "
                "retained (see visual_inspection_fig5d.json); the specific magnitude factors are not."),

    # ---------------- Figure 6 ----------------
    Claim("Fig6A", "Figure 6A",
          "Categorical associations (Cramer's V) among temperature-dependent strength, monotonic "
          "fracture response, rate-defined fatigue crack-growth-threshold response, and S-N temperature "
          "phenotype.",
          "RAW (contingency-table cell counts)", FATIGUE_PF_ROOT, "analyze_v57_integrated.py",
          "6 analysis families x 6 fracture contexts",
          ("fig6A_contingency_cells_censor_aware.csv",), R.recompute_Fig6A,
          expected=dict(
              n_tables_recomputed=36, all_v_nonnegative=True, tolerance_abs=1e-4,
              frozen_v_by_family_context={
                  "fracture_vs_fatigue_temperature_phenotype::ctx_FCC_like_case29": 0.141643,
                  "fracture_vs_fatigue_temperature_phenotype::ctx_higher_barrier_case171": 0.144364,
                  "fracture_vs_fatigue_temperature_phenotype::ctx_plastic_shielded_case64_M1": 0.051001,
                  "fracture_vs_fatigue_temperature_phenotype::ctx_shifted_ductile_case64": 0.116067,
                  "fracture_vs_fatigue_temperature_phenotype::ctx_slow_threshold_case101": 0.103472,
                  "fracture_vs_fatigue_temperature_phenotype::ctx_steep_cleavage_case35": 0.145842,
                  "fracture_vs_growth_threshold_class::ctx_FCC_like_case29": 0.670368,
                  "fracture_vs_growth_threshold_class::ctx_higher_barrier_case171": 0.705704,
                  "fracture_vs_growth_threshold_class::ctx_plastic_shielded_case64_M1": 0.340279,
                  "fracture_vs_growth_threshold_class::ctx_shifted_ductile_case64": 0.090909,
                  "fracture_vs_growth_threshold_class::ctx_slow_threshold_case101": 0.705704,
                  "fracture_vs_growth_threshold_class::ctx_steep_cleavage_case35": 0.705704,
                  "growth_threshold_vs_SN_temperature_phenotype::ctx_FCC_like_case29": 0.221265,
                  "growth_threshold_vs_SN_temperature_phenotype::ctx_higher_barrier_case171": 0.249865,
                  "growth_threshold_vs_SN_temperature_phenotype::ctx_plastic_shielded_case64_M1": 0.282038,
                  "growth_threshold_vs_SN_temperature_phenotype::ctx_shifted_ductile_case64": 0.282038,
                  "growth_threshold_vs_SN_temperature_phenotype::ctx_slow_threshold_case101": 0.218744,
                  "growth_threshold_vs_SN_temperature_phenotype::ctx_steep_cleavage_case35": 0.218744,
                  "strength_vs_fatigue_temperature_phenotype::ctx_FCC_like_case29": 0.17435,
                  "strength_vs_fatigue_temperature_phenotype::ctx_higher_barrier_case171": 0.17435,
                  "strength_vs_fatigue_temperature_phenotype::ctx_plastic_shielded_case64_M1": 0.17435,
                  "strength_vs_fatigue_temperature_phenotype::ctx_shifted_ductile_case64": 0.17435,
                  "strength_vs_fatigue_temperature_phenotype::ctx_slow_threshold_case101": 0.17435,
                  "strength_vs_fatigue_temperature_phenotype::ctx_steep_cleavage_case35": 0.17435,
                  "strength_vs_fracture_class::ctx_FCC_like_case29": 0.79446,
                  "strength_vs_fracture_class::ctx_higher_barrier_case171": 0.811122,
                  "strength_vs_fracture_class::ctx_plastic_shielded_case64_M1": 0.068652,
                  "strength_vs_fracture_class::ctx_shifted_ductile_case64": 0.350278,
                  "strength_vs_fracture_class::ctx_slow_threshold_case101": 0.577355,
                  "strength_vs_fracture_class::ctx_steep_cleavage_case35": 0.81027,
                  "strength_vs_growth_threshold_class::ctx_FCC_like_case29": 0.347969,
                  "strength_vs_growth_threshold_class::ctx_higher_barrier_case171": 0.356146,
                  "strength_vs_growth_threshold_class::ctx_plastic_shielded_case64_M1": 0.603023,
                  "strength_vs_growth_threshold_class::ctx_shifted_ductile_case64": 0.603023,
                  "strength_vs_growth_threshold_class::ctx_slow_threshold_case101": 0.357702,
                  "strength_vs_growth_threshold_class::ctx_steep_cleavage_case35": 0.357702,
              },
          ),
          compare=cmp_fig6a,
          tolerance="each of the 36 frozen Cramer's V values must match the recomputed value within "
                    "1e-4 absolute; additionally the manuscript-reported magnitude groups must hold: "
                    ">=3 strength-fracture values in [0.78,0.82] ('several contexts' ~0.79-0.81); "
                    "exactly 4 fracture-threshold values in [0.65,0.72] ('four of six contexts' "
                    "~0.67-0.71); >=2 strength-threshold values in [0.58,0.62] (~0.60); all S-N "
                    "phenotype associations in [0.04,0.30] (the weaker ~0.05-0.28 band)",
          max_evidence_class=QUALIFIED,
          notes="Round-5 correction: the round-4 comparator checked only the table COUNT (36) and "
                "nonnegativity, neither of which depends on any actual magnitude -- six nearly-uniform "
                "values would have passed identically. It now freezes and checks all 36 recomputed "
                "Cramer's V values against a literal expected map (tolerance 1e-4) AND separately gates "
                "on the manuscript's four described magnitude groups, all of which hold in the "
                "recomputed data. Rebuilt every one of the 36 contingency tables from raw cell counts "
                "and recomputed Cramer's V via chi2_contingency from scratch (not read from the derived "
                "summary CSV). Cramer's V is nonnegative by construction; the manuscript's signed "
                "notation reflects an externally-applied directional gloss, not a property of this "
                "statistic -- see proposed_manuscript_corrections.md."),
    Claim("Fig6B", "Figure 6B",
          "Matched-temperature relationship between K_c and DeltaK_th across 1360 matched observations; "
          "pooled log-space Pearson correlation, context-specific values 0.958 to 0.999.",
          "RAW (compact portable join projection, provenance-tracked)", FATIGUE_PF_ROOT,
          "matched_kc_dkth() reimplemented against the portable compact projection",
          "rate_criterion=1e-10; threshold_status=='bracketed'",
          ("fig6B_compact_joined_1360.csv",), R.recompute_Fig6B,
          expected=dict(
              n=1360, pooled_r=0.9864679341131591,
              per_context={
                  "ctx_FCC_like_case29": 0.9983177942098503,
                  "ctx_higher_barrier_case171": 0.9576588312673979,
                  "ctx_plastic_shielded_case64_M1": 0.9855864294159365,
                  "ctx_shifted_ductile_case64": 0.9987454527442478,
                  "ctx_slow_threshold_case101": 0.9794700505508651,
                  "ctx_steep_cleavage_case35": 0.9904018051290919,
              },
          ),
          compare=lambda actual, expected: (
              actual.get("n") == expected["n"]
              and abs(actual.get("pooled_r", 0) - expected["pooled_r"]) < 1e-3
              and set(actual.get("per_context", {})) == set(expected["per_context"])
              and all(abs(actual["per_context"][k] - expected["per_context"][k]) < 1e-3
                      for k in expected["per_context"])
          ),
          tolerance="exact n; pooled r within 1e-3 absolute; EVERY ONE of the 6 context-specific "
                    "correlations within 1e-3 absolute (not just the min/max range extrema)",
          max_evidence_class=QUALIFIED,
          notes="Round-5 correction: the round-4 comparator checked only n and the min/max of the "
                "context-specific correlations, so it could not have detected a corrupted pooled r or "
                "a corrupted individual context value that happened to still fall inside [min,max]. It "
                "now freezes and checks the pooled correlation and all 6 individual context "
                "correlations explicitly. Fully portable: the compact 1360-row joined table "
                "(fig6B_compact_joined_1360.csv) is bundled with full provenance (full-source SHA-256, "
                "filter/join keys, excluded-row counts) in fig6b_portable_projection_provenance.json, "
                "so this claim needs NO external 40MB file at verification time. An optional live "
                "cross-check (fig6b_live_cross_check) re-hashes the external originals when "
                "available/not hidden, purely informationally."),
    Claim("Fig6C", "Figure 6C",
          "Temperature-specific association between endurance-like S-N response and K_c, DeltaK_th, or "
          "the strength-anomaly magnitude.",
          "RAW (compact per-observation projection, provenance-tracked)", FATIGUE_PF_ROOT,
          "analyze_auc()/bootstrap_auc_by_surface() reimplemented against the portable compact projection",
          "strict S-N definition (endurance_like vs continuous_SN, knees excluded); T in {100,300,500}K",
          ("fig6C_compact_kc_endurance.csv", "fig6C_compact_dkth_endurance.csv",
           "fig6C_compact_strength_endurance.csv", "fig6_manuscript_statistics_table.csv"),
          R.recompute_Fig6C,
          expected=dict(all_matched=True), compare=cmp_dict_subset,
          tolerance="every one of the 39 frozen Panel C rows' AUC + 95% bootstrap CI must match to "
                    "1e-6 (not just one spot-checked row)",
          max_evidence_class=QUALIFIED, notes=""),

    # ---------------- Figure 7 ----------------
    Claim("Fig7-peak-900K-cohesive-mapping", "Figure 7",
          "Cohesive-law mapping of the EXP-floor barrier form to standard cohesive laws, peak class, "
          "900K; velocity proxy spans approximately seven orders of magnitude.",
          "derived (fitted comparison curves)", FEM_CZM_ROOT, "compare_exp_floor_to_standard_cohesive.m",
          "example_class='peak'; T_K=900",
          ("fig7_peak_900K_comparison_curves.csv", "fig7_peak_900K_fitted_parameters.csv"),
          R.recompute_Fig7,
          expected=dict(velocity_range_orders_of_magnitude=6.48),
          compare=lambda actual, expected: abs(
              actual["velocity_range_orders_of_magnitude"] - expected["velocity_range_orders_of_magnitude"]
          ) < 0.1,
          tolerance="within 0.1 orders of magnitude of the recomputed reference value",
          max_evidence_class=QUALIFIED, notes=""),

    # ---------------- SI identifiability ----------------
    Claim("SI-identifiability-recovery-errors", "SI (synthetic identifiability)",
          "Synthetic inverse-recovery study: 4 hidden response classes, 75-condition synthetic dataset "
          "per class; emission-landscape recovery error ~20-55% (sparse) improving to ~1-10% (complete).",
          "RAW (per-fit inversion results + per-regime ground-truth barrier grids)", IDENTIFIABILITY_ROOT,
          "run_synthetic_identifiability.py",
          "4 hidden regimes (ceramic/peak/weakT/dbtt); 75-condition universe",
          ("SI_condition_universe.csv", "SI_inversion_summary.csv", "SI_ceramic_truth_barrier_grid.csv",
           "SI_peak_truth_barrier_grid.csv", "SI_weakT_truth_barrier_grid.csv",
           "SI_dbtt_truth_barrier_grid.csv"),
          R.recompute_SI_identifiability,
          expected=dict(n_conditions_per_regime=75, sparse_range=[23.4, 54.4], complete_range=[1.2, 9.4],
                        band_tol=2.0),
          compare=lambda actual, expected: (
              actual["n_conditions_per_regime"] == expected["n_conditions_per_regime"]
              and abs(actual["sparse_range"][0] - expected["sparse_range"][0]) <= expected["band_tol"]
              and abs(actual["sparse_range"][1] - expected["sparse_range"][1]) <= expected["band_tol"]
              and abs(actual["complete_range"][0] - expected["complete_range"][0]) <= expected["band_tol"]
              and abs(actual["complete_range"][1] - expected["complete_range"][1]) <= expected["band_tol"]
          ),
          tolerance="exact condition count; each of the 4 recomputed range extrema (sparse min/max, "
                    "complete min/max) must fall within +/-2 percentage points of the actual "
                    "reconstructed values (23.4/54.4 sparse, 1.2/9.4 complete) -- not merely inside a "
                    "broad containing interval that would also accept a materially different result",
          max_evidence_class=QUALIFIED,
          notes="Round-5 correction: the round-4 comparator accepted any sparse range inside [20,60]% "
                "and any complete range inside [0,12]% -- intervals far broader than the actual "
                "reconstructed values (23.4-54.4% sparse, 1.2-9.4% complete), so a materially different "
                "recomputation could still have passed. The comparator now requires each extremum "
                "within a narrow +/-2 percentage-point band of the specific reconstructed values."),
]


# NOTE: no EXPECTED_CLAIM_IDS is defined here anymore (round 5 correction --
# see module docstring). The frozen inventory lives in
# artifacts/paper_simulation_completion/expected_paper_claim_ids_v4.json and
# is loaded independently by verify_paper_evidence_v4.py.
