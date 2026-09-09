# Slope Interpretation Notes

Companion to `README.md`. All numbers below are read directly from
`SOURCE_DATA/global_slope_summary.csv` and `SOURCE_DATA/local_secant_slopes.csv`,
both produced by `PLOTTING_SCRIPTS/compute_slopes.py` from
`SOURCE_DATA/all_included_curve_points.csv` alone.

## 1. Global slope classification rule

A curve's single-slope-adequate classification requires **both**:

- global R^2 >= 0.95, **and**
- max(m_local) - min(m_local) <= max(0.5, 0.25*|m_global|)

Otherwise it is `CURVED_NON_PARIS_RESPONSE`. This is a strict dual
threshold: a curve can have a high R^2 from a global log-log fit and still
fail if its local behavior swings widely across the developed window (the
common case here), because R^2 alone does not penalize systematic
curvature that averages out over the fit range.

## 2. A_NATIVE global slope vs. R (n=4 points each, Kmax in {12,15,18,24})

| R | global slope m | R^2 | local slope range |
|---|---|---|---|
| -0.95 | 4.14 | 0.720 | 0.72 -- 11.17 |
| 0.1 | 4.20 | 0.725 | 0.77 -- 11.24 |
| 0.5 | 4.33 | 0.742 | 0.95 -- 11.30 |

Global slope increases mildly and monotonically with R (4.14 to 4.33 across
this grid), consistent with the previously-verified characterization of
**m ~ 4.2-4.6, appreciably curved, depending on resolution/seed**. All
three are classified `CURVED_NON_PARIS_RESPONSE`: the local-slope range
(~10.3-10.5 decades of exponent swing) vastly exceeds the adequacy
threshold (max(0.5, 0.25*4.2) ~ 1.05). PT03 and PT08 track A_NATIVE within
0.02-0.04 of global slope at every R (see `global_slope_summary.csv`),
consistent with the PT_R_INVARIANT finding in Sec. 4 of the README.

**Physical reading**: da/dN rises steeply just above the developed
window's lower Kmax bound (local m up to ~11 near Kmax=12-15) and flattens
toward the top of the window (local m ~0.7-1.0 near Kmax=18-24). A single
Paris exponent is not an adequate description of this response at any R
tested; the curve should be read and reported as curved, not linearized.

## 3. Rebonding-family slopes and the D1/D2, D2/D5 near-degeneracy

Global slopes for the primary-seed (1720) rebonding curves are
substantially higher (~5.0-7.3) than their matched zero-cohesion baselines
(~3.84-3.86), because the finite (cohesive) branch's rate rises faster with
Kmax over the same window before converging toward the baseline at high
Kmax (visible directly in Figure 3 and its S_h companion). The 2nd-seed
confirm runs, fit over a narrower 3-point Kmax window (15-21 instead of
12-24.3), come back with lower global slopes (~1.9-2.3) purely because a
narrower fit window sees less of the steep low-Kmax curvature -- this is a
window-size artifact of the OLS fit, not a physically different material
response, and both seeds still fail the `PARIS_ADEQUATE` test.

A specific finding, checked directly against the raw source rows (not a
plotting artifact): at seed 1720, D1 (COMPETING_REVERSIBLE, R=-0.95) and D2
(COMPETING_REVERSIBLE, R=-0.50) produce developed rates that agree to
better than 1% at every shared Kmax, e.g. at Kmax=12 MPa*sqrt(m):
D1 finite = 1.4862e-9 m/cycle vs. D2 finite = 1.6957e-9 m/cycle (S_h =
-1.190023 vs. -1.190248, a difference of 0.0002 dex). The same
near-identity holds between D2 (clean reversible) and D5
(passivation-limited) at R=-0.50: at Kmax=12, D2 finite = 1.69566e-9 vs. D5
finite = 1.69584e-9 m/cycle. This is why the D1/D2 curves in Figure 3 Panel
A and the D2/D5 curves in Panel B (and their S_h companions) appear as a
single visible line -- the underlying values genuinely coincide at this
seed/condition rather than one curve being hidden by a plotting bug. This
is consistent with the `REBONDING_SEED_ROBUST` classification already
recorded for these protocol pairs in the source campaign's own
`px4_stage1_slope_table.csv`.

## 4. Comparison with the fracture-selected activation-cliff controls

| Control | m (verified) | Status |
|---|---|---|
| Peak | ~100.5 | qualified reference condition |
| DBTT | ~133.5 | qualified reference condition |
| ceramic-like | ~71.7 | qualified reference condition |
| weak-T | -- | **no qualified stable global fit** |

These three controls are 15-30x steeper than the A_NATIVE/PT03/PT08 global
slopes (~4.1-4.4) found in this package, and are drawn in Figure 5 Panel B
purely as 2-point slope-guide lines at an arbitrary vertical offset (no
per-point developed table for these conditions was locatable in this
session -- see `plot_selection_manifest.csv`). **These values describe this
project's own fracture-selected activation-cliff conditions and must not be
read as generic experimental Paris exponents for all BCC metals or all
ceramics.** Broad material-class Paris exponents in the published
literature typically run m~2-4 (many structural metals) up to m~30-100+
only in unusually brittle/cliff-like regimes; the qualitative ordering here
(A_NATIVE moderate and curved; Peak/DBTT/ceramic-like far steeper) is
directionally consistent with that broad expectation, but the specific
numeric values above are project-specific control values, not measurements
of any named real material.

## 5. DeltaK_eff checkpoint

Grep-verified: no closure-corrected DeltaK_eff **quantity** is plotted, or
appears as a reported/tabulated value, anywhere in this package -- not in
`all_included_curve_points.csv`, `global_slope_summary.csv`,
`local_secant_slopes.csv`, `curve_source_provenance.csv`, any figure, or
`make_figures.py`. The string "DeltaK_eff" appears only in prose in
`plot_selection_manifest.csv` and in this document, both explaining
*why* it is excluded (both source campaigns lack a validated resolved
opening/contact criterion: Part X's signed-K contact surrogate does not
resolve opposing crack-face contact, and the R-ratio study's own decision
doc states DeltaK_eff is omitted without a validated opening criterion).
Only nominal Kmax, R, and the derived full-range/tensile-only DeltaK
quantities are used as actual data throughout.
