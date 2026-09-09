# Manuscript corrections — applied in this figure package

**Status: two specific wording corrections proposed; no claim requires
narrowing or removal. Both are applied throughout this figure package's
figures and captions (Figures 6A and 5D; see `figure_captions.md` and
`figure_scientific_notes.md`).**

Two further conservative wording refinements (not corrections of an error,
but precision refinements requested during figure-package review) are also
applied throughout this package:

- **Figure 4**: captions and panel titles use "FEM/CZM shows no resolved
  local maximum on the available temperature grid" rather than describing
  FEM/CZM's response as a quantitatively established "muted shoulder" — the
  verified check establishes peak *absence* on the sampled grid, not a
  formally defined and measured shoulder metric.
- **Figure 5B**: captions describe the result as persistence of the broad
  response hierarchy and strong orientation-dependent growth; geometrical
  path deflection is explicitly stated to be qualitative/unverified because
  portable crack-path coordinates were not recovered.

The original evidence-branch text (unmodified) follows below.

---

This file is generated as a required deliverable of the paper-evidence
independent-verifier closure pass regardless of outcome. As of
`paper_claim_evidence_matrix_v4.csv` / `paper_simulation_completion_verification_v4.json`,
all 35 claim-level rows pass the executable verifier
(`verify_paper_evidence_v4.py`, which computes pass/fail itself rather than
reading a pre-declared field): 30 at `QUALIFIED_SOURCE_RESULT_VERIFIED` and
5 at the honest topology-only ceiling `SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED`
(Fig1A-D, Fig5D). `classification = PAPER_SIMULATION_EVIDENCE_COMPLETE`.

## Proposed correction 1: Cramer's V sign convention (Figure 6A)

The manuscript's Sec. describing Fig. 6A uses signed language for
categorical associations, e.g. "the strength-fracture association reaches
Cramer's V ≈ -0.81 in several fracture contexts" and "the fracture-ΔK_th
association reaches ≈ -0.71 in four of the six contexts."

This session rebuilt all 36 of the underlying contingency tables (6 analysis
families × 6 fracture contexts) directly from raw cell counts
(`fig6A_contingency_cells_censor_aware.csv`, bundled) and recomputed every
Cramer's V value from scratch via `scipy.stats.chi2_contingency`, exactly
reproducing `global_class_associations_clean.csv`'s stored values. Cramer's
V as implemented -- `sqrt(chi2 / (n*(k-1)))` -- is a square root of a
non-negative quantity and is therefore **mathematically incapable of being
negative**. Every one of the 36 recomputed values is confirmed non-negative
(`all_v_nonnegative=True`), including the specific strength-fracture and
fracture-ΔK_th cells the manuscript describes with a minus sign.

**Proposed fix:** the minus signs in the manuscript's Cramer's V language
should be removed, with the *direction* of the association (if it is
scientifically important to convey) described in words instead of via a
signed coefficient, e.g. "an inverse association with Cramer's V ≈ 0.81"
rather than "Cramer's V ≈ -0.81." This is a wording correction, not a change
to any reported effect size -- the underlying magnitudes are exactly
verified and unchanged.

## Proposed correction 2: no specific field-magnitude factor for Figure 5D

An earlier pass (prior to this session) stated specific numeric field
comparisons for Fig. 5D (e.g. "~160x larger plastic strain," "~1000x larger
dislocation density") derived by reading values off two PNG color-bar
legends. This session determined that claim was not verifiable to the
standard required here: the two renders were not confirmed to share a common
color-scale normalization, and only the rendered images (not the underlying
numeric field arrays) are available in the located source tree. A specific
multiplicative factor read off two independently-scaled color bars is a
visual estimate, not a verified recomputation.

**Proposed fix:** the manuscript should not state a specific numeric
multiplicative factor for Fig. 5D's field comparison unless the underlying
field arrays and a common normalization are made available for independent
recomputation. The qualitative conclusion -- the shielded calculation shows
a broader, more diffuse deformation and residual-stress field than the
unshielded case at matched conditions -- remains supported (this session
directly viewed both bundled images and confirms the qualitative contrast)
and needs no correction. `Fig5D`'s evidence class is held at
`SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED` rather than
`QUALIFIED_SOURCE_RESULT_VERIFIED` to reflect this honestly.

## What changed since the prior (v3) pass

Beyond the two corrections above, this pass replaced self-declared
verification with an executable one (see `PAPER_EVIDENCE_FINAL_HANDOFF_V4.md`
for the full account) and, in doing so, genuinely resolved several claims
that were previously verified only against a single representative case or
a coarse grid:

- **Fig4-peak-narrow-attenuation**: now confirmed at all 4 rate factors
  (not just 1x) using a fine 5K-resolution, rate-resolved analytic file.
- **Fig5A**: monotonicity, point counts, Paris-law slope and curvature are
  now computed per case, for all 6 canonical cases, from the raw K-point
  array.
- **Fig5B**: class-ordering persistence and orientation-dependence magnitude
  are now recomputed directly from bundled raw multiseed/orientation data
  (rank correlation = 1.0 between early- and late-extension class rankings;
  ~17x da/dN difference between 30deg and 45deg orientations at matched
  driving force).
- **Fig5C**: reconstructed across all 15 available seed/stress/condition
  jobs (seeds 2-5, both stresses), not a single representative pair. The
  unshielded-vs-shielded contrast is a genuine, quantified statistical
  tendency (coverage_pass rate materially higher for unshielded) rather than
  a deterministic split in this finite sample -- reported honestly as such.
- **Fig6A**: every contingency table rebuilt from raw cell counts (see
  correction 1 above).
- **Fig6B**: made fully portable via a compact 1360-row joined table with
  full join/filter provenance, eliminating the prior dependency on a live
  40MB external file for the core claim.
- **Fig6C**: every one of the 39 frozen Panel C rows (not one spot-checked
  row) has its AUC and 95% bootstrap CI independently reproduced from a
  compact per-observation projection.
- **Sec.2.15**: the saturation R-curve parameters are now genuinely refit
  (`scipy.optimize.curve_fit`) from the raw per-seed binned R-curve data,
  not read from the existing fit-output CSV.

## Round-5 addendum (no new manuscript corrections)

A fifth review hardened the executable verifier's comparators and claim
inventory (see `PAPER_EVIDENCE_FINAL_HANDOFF_V4.md`'s "Round-5 corrections"
section) but did not surface any new manuscript wording issue beyond the two
already recorded above. One evidence-class change is worth noting here for
completeness: **Fig5B** is now held at
`SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED` rather than
`QUALIFIED_SOURCE_RESULT_VERIFIED`, because its compound manuscript claim
includes a geometric "path deflection" component that this branch has not
numerically verified (no spatial crack-path coordinates are bundled or
confirmed available). This is a verification-depth disclosure, not a
manuscript-correctness concern: the growth-rate orientation-dependence half
of the claim (~17x da/dN difference between orientations) IS independently
verified, and no manuscript text is being flagged as unsupported.

## If a genuine gap re-emerges in a future pass

Should a future review or re-verification find that any row's
`final_evidence_class` regresses below `SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED`,
this file is the correct place to record the specific proposed replacement
manuscript text -- narrowing a claim to what the surviving evidence
supports, or removing it -- rather than silently omitting the affected
figure/number. No such regression exists at the time of this commit.
