# Figure Scientific Notes

Detailed, panel-by-panel provenance, statistical definitions, censor/
exclusion treatment, and limitations for every figure in this package.
Manuscript-facing captions are in `figure_captions.md`; machine-readable
provenance (hashes, source repo/commit/branch, producer script) is in
`07_QC_AND_PROVENANCE/figure_manifest.csv` and
`04_SOURCE_DATA/*/*_source_data_hashes.json`.

All main-text and SI figures derive from the committed, portable evidence
bundle at `codex/v10.2.30-paper-evidence-independent-verifier` @
`cbf2cf8bf73c649c1b87acd6a6a94f322f936bcc`,
`artifacts/paper_simulation_completion/source_bundle_figures_2_4/`. No
mutable raw run directory was used for any main-text/SI figure. No physical
simulation was launched to produce any figure in this package.

## Figure 1 (A-D)

Source data level: raw parameter-continuation path (A, B) and derived
per-curve summary (C, D). Evidence class:
`SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED` for all four panels — the
independent verifier confirmed the recovered regime-transition order,
parameter ranges, and shared axes/entropy-family grid, but no
manuscript-stated *number* is reproduced by these panels (none exists in the
portable evidence bundle to reproduce; the manuscript's Fig. 1 caption makes
no specific numerical claim). Panels A/B plot the verified continuation
*topology* (H₀,c vs. χ_shield, colored by regime) rather than a full
temperature- or cycle-resolved response curve, because the full response-
surface array underlying the original "waterfall" renders is not in the
portable bundle (only the defining parameter path is). This is a disclosed,
deliberate scope limitation, not an attempt to fabricate a curve shape not
present in the evidence.

## Figure 2 (A-D)

Source: `fig2_first_passage_comparison_with_analytic_1x.csv` (comparison
table) + `fig2_four_class_analytical_prediction_final_fine_grid.csv` (5K-
resolution analytic curve). RMS values displayed are read from the
independently-verified claim registry (`Fig2-*-RMS` claims), not recomputed
by the plotting script. "Complete"/"incomplete" run status is the `is_complete`
field in the raw comparison table; incomplete PF/sharp-front runs are shown
as open markers per the manuscript's own convention (first passage recorded
before the target crack extension was reached).

## Figure 3 (A-D)

Source: `sec2_15_seed_binned_Rcurves_long.csv` (raw per-seed binned R-curve,
5 seeds × 4 classes) and `fig3_class_mean_Rcurve_fits.csv` (saturating fit).
K₀/K_ss/ΔK_R inset values are the mean ± sample standard deviation (n=5,
n−1 denominator) computed directly from `fig3_seed_Rcurve_metrics_and_fits.csv`,
matching the independently verified `Fig3-*` claim values exactly. Individual
seed curves are plotted at reduced opacity specifically so seed-to-seed
variability is not obscured by the fit line, per the figure-package
instruction.

## Figure 4 (A-E)

Source: `fig4_first_passage_comparison_with_analytic_all_rates.csv` (coarse,
100K-spaced FEM-vs-analytic comparison, all 4 rates) and
`fig4_analytical_predictions_by_rate_fine_grid.csv` (5K-resolution,
rate-resolved analytic grid). θ = 45° is parsed programmatically from
`fig4_comparison_config.json`'s `root` field and asserted, not hardcoded.

Panel E's two DBTT transition-temperature series are the values recorded in
the independently verified `Fig4-DBTT-transition-shift` claim:
- Shelf-midpoint crossing (primary, self-declared proxy — NOT the
  manuscript's own unstated exact transition definition): 788.9 / 898.2 /
  1031.4 / 1125.1 K at 0.1×/1×/10×/100×.
- Maximum-|dK/dT|-gradient temperature (independent cross-check, different
  method, same raw data): 750 / 850 / 1050 / 1150 K.

Both definitions agree on the monotonic direction of the shift, which is
the claim being illustrated — neither is presented as *the* manuscript's
definition of the DBTT temperature, since the manuscript does not state one.

The peak-class panel's "no resolved local maximum" wording is deliberate
(see `manuscript_corrections.md`): the verified claim
(`Fig4-peak-narrow-attenuation`) established that FEM/CZM's own Kc(T) curve
has **zero local maxima on the 100K-sampled grid at any of the four rates**
— a directly checked structural fact — while the ANALYTICAL prediction's
narrow peak location shifts monotonically with rate (860/910/970/1035 K at
0.1×/1×/10×/100×, from the fine 5K grid). The quantitative attenuation
percentage (30.9%) is only available at 1× rate, where the coarse
comparison grid happens to sample a point coincident with the analytic
peak; this is disclosed, not extrapolated to the other rates.

## Figure 5

**Panel A**: source `fig5A_atlas_2d_paris_points.csv`, all 6 systems present
with descriptive labels (internal case IDs archived in
`04_SOURCE_DATA/MAIN_TEXT/fig5A_system_id_to_label_mapping.json`). Four
systems (smooth ductile, steep cleavage, shifted ductile, plastic-shielded)
are shown in the main-text panel per the manuscript's own Sec. 3.3.1
description; the complete 6-system atlas is SI Figure 1. Upper-bound
censored points (`is_censored_upper_bound == True`) are shown as open
triangles and are never connected by a line through non-censored points.

**Panel B**: long-growth hierarchy from `fig5B_multiseed_r_curve_mean_curves.csv`
(all 6 systems, 3-seed multiseed mean). Class-ordering persistence was
checked at all 5 available common-extension points (not 2 hand-picked
ones): the full 6-system rank order is identical at 3 of 5 points, and the
weakest- and strongest-performing systems never change identity at any of
the 5 points — the latter, more robust fact is what is illustrated and
captioned. The orientation-dependence inset uses
`fig5B_orientation_theta{30,45}_atlas_2d_paris_points.csv` (matched nominal
K_max, confirmed to agree to <1e-6 MPa√m). **The geometrical crack-path
deflection itself is not plotted or claimed** — no spatial crack-path
coordinate data were recovered as part of the portable evidence bundle for
this branch; only the growth-RATE orientation dependence (~17×) is
quantified. This matches the claim registry's evidence class for Fig5B:
`SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED` (capped below QUALIFIED
specifically because of the unverified geometric component).

**Panel C**: `fig5C_compact_sn_reconstruction.json`, all 15 available
seed × stress × condition jobs (seeds 2-5, stresses 700/900 MPa) —
reconstructed as a censor-aware stress-life comparison, not a single
representative pair. A job with no finite `cycles_root_connected` value
(i.e., no `summary.json` was ever written because the run never reached
root connection) is plotted as a right-pointing open marker at a fixed,
clearly-out-of-range x-position and is excluded from any median or
central-tendency calculation — it is never treated as an infinite or a
zero lifetime.

**Panel D**: the two rendered field snapshots
(`fig5D_fields_{no_shield,shielded}_seed5_900MPa.png`) at matched seed
(5) and stress (900 MPa). A separate `visual_inspection_fig5d.json` record
(in the independent-verifier branch) documents the qualitative visual
review and explicitly disclaims any cross-image magnitude ratio; this
package's caption and panel title repeat that disclaimer directly.

## Figure 6 (A-C)

**Panel A**: Cramér's V recomputed directly from
`fig6A_contingency_cells_censor_aware.csv` (raw contingency-table cell
counts) via `scipy.stats.chi2_contingency`, reproducing the independently
verified 36-value map exactly (tolerance 1e-4 in the claim registry).
Cramér's V, as implemented, is `sqrt(chi2/(n*(k-1)))` — the square root of a
non-negative quantity — and is therefore mathematically incapable of being
negative. This figure plots only non-negative magnitudes; see
`manuscript_corrections.md` for the required manuscript-text correction.

**Panel B**: `fig6B_compact_joined_1360.csv`, the portable, provenance-
tracked compact join (n=1360 exactly, matching the manuscript). Pooled log-
space Pearson r = 0.9865 (independently recomputed and verified); the six
individual context correlations range 0.958-0.999 (see SI Figure 6 for each
context's own panel and exact r).

**Panel C**: `fig6_manuscript_statistics_table.csv`'s frozen Panel-C rows
(all 39 rows independently reproduced — AUC + 95% bootstrap CI — in the
evidence package's `Fig6C` claim). The strict S-N definition excludes
knee-like intermediate cases (sensitivity analysis with knees included is a
separate, not-yet-plotted dataset in the evidence bundle).

## Figure 7 (A-B)

Source: `fig7_peak_900K_comparison_curves.csv` and
`fig7_peak_900K_fitted_parameters.csv`. This is a deterministic MATLAB fit
(`compare_exp_floor_to_standard_cohesive.m`) of three standard cohesive
forms to the EXP-floor barrier curve at the peak class, 900 K — there is no
new physical simulation anywhere in this figure. The velocity-proxy dynamic-
range annotation (6.5 decades) is computed directly from the bundled curve
data (min/max of `v_EXPfloor_m_per_s`), matching the independently verified
value.

## Supporting Information figures

See `figure_captions.md` for one-line captions. Full provenance for every
SI figure is recorded identically to the main text in
`04_SOURCE_DATA/SUPPORTING_INFORMATION/si_source_data_hashes.json`. No SI
figure duplicates a main-text figure; each either shows the complete
underlying dataset (e.g., SI Fig. 1's full six-system atlas vs. Fig. 5A's
four-system main-text subset) or a diagnostic not otherwise shown (e.g., SI
Fig. 3's fit residuals).

## Part X (crack-rebonding) curated set

These figures are reused, not regenerated with new physics, from
`codex/v10.2.30-crack-rebonding-part-x` @
`30db009ff7172728a6bdc885886f21b94cbec225` (classification
`SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE`). Vector PDF/SVG exports were
produced by re-running the branch's own existing, already-tracked plotting
scripts (`plot_part_x_final_closure_figures.py`,
`plot_part_x_px6_figures.py`, `plot_developed_confirmation.py`) with a
save-format patch only (no change to any plotted data, axis, or physics) in
a detached, read-only worktree; that worktree was discarded after export and
the original branch's history is untouched. Figure 01's analytical
periodic-orbit trajectory is recomputed deterministically from the tracked,
frozen `COMPETING_REVERSIBLE` configuration at generation time — this is a
zero-fitting analytical evaluation, never a new physical simulation.

Curated captions:

1. **P/C/B periodic-orbit state trajectory** (`fig01_pcb_phase_trajectory`) —
   Analytical periodic-orbit trajectory through the persistent/contact/
   broken (P/C/B) state space over one loading cycle, COMPETING_REVERSIBLE
   configuration.
2. **Screen-level rebonding effect vs. R** (`fig04_screen_S_h_vs_R`) —
   Screening-study dependence of the shielding metric S_h on stress ratio R.
3. **Frequency dependence** (`fig05_screen_frequency_response`) and
   **reversible/persistent comparison** (`fig14_reversible_vs_persistent_curves`) —
   Screening-study frequency response, and comparison of reversible vs.
   conditional-persistent rebonding protocols.
4. **Passivation/chemistry dependence** (`fig07_screen_passivation_chemistry_response`).
5. **Cohesive-strength dependence** (`fig08_screen_cohesive_strength_response`).
6. **Developed absolute da/dN vs. K_max** (`fig09_absolute_developed_da_dN_vs_Kmax`).
7. **Developed S_h vs. K_max, all protocols** (`px6_S_h_developed_vs_Kmax_all_protocols`)
   and **local slope/curvature** (`fig11_local_slope_curvature`).
8. **Dynamic vs. prescribed-static shielding attribution**
   (`px6_px5_static_shield_vs_dynamic`).
9. **Seed-robustness comparison** (`px6_seed_robustness_slope_comparison`).
10. **Corrected dwell result** (`fig06_dwell_invalidated_vs_corrected`) — the
    superseded duration-weighting calculation is shown explicitly labeled
    as an INVALIDATED NUMERICAL ARTIFACT (dotted gray, per the shared style
    convention), never presented as physical evidence. **The manuscript
    text and this figure do not report a closure-corrected ΔK_eff.**

Part X limitations (apply to the entire curated set and complete diagnostic
set): `SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT`,
`HAZARD_ONLY_COHESIVE_FEEDBACK`, `TOPOLOGICAL_HEALING_NOT_MODELED`,
`PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED`. This is a separate
future-paper/extended-data package and is not automatically part of the
current manuscript.
