# Focused da/dN vs. K Comparison Package

Generated 2026-09-09 on branch `codex/v10.2.30-focused-dadn-K-plots`.

## 1. Scope and purpose

This package answers exactly four questions about developed fatigue
crack-growth rate da/dN as a function of stress-intensity factor K, using
only already-qualified, developed (stationary-window) simulation results.
It does **not** assemble manuscript Figures 1-7, does not include monotonic
fracture-temperature curves, FEM/CZM R-curves, Cramer's V, AUC, S-N curves,
field maps, or identifiability plots, and does not run any new physical
simulation. No constitutive or physical-solver source file was touched.

1. **How does developed da/dN(K) change with R?** -- Figure 1.
2. **How much do the A_NATIVE / PT03 / PT08 parameterizations differ?** -- Figure 2.
3. **How do reversible, persistent, and passivation-limited rebonding alter
   the developed curve?** -- Figure 3.
4. **What global and local slopes do these curves imply, and how do they
   compare with broad material-class expectations?** -- Figures 4 and 5.

## 2. Data provenance (authoritative sources)

| Dataset | Branch | Commit | Table |
|---|---|---|---|
| A_NATIVE / PT03 / PT08 x R x Kmax | `codex/v10.2.30-R-ratio-nominal-deltaK` | `4b554006d60539d7dec659f4bfe4bf298728780a` | `1A_PT03_PT08_R_developed_points.csv` |
| Crack-rebonding Part X, PX4 stage-1 | `codex/v10.2.30-crack-rebonding-part-x` | `30db009ff7172728a6bdc885886f21b94cbec225` (accepted terminal HEAD) | `px4_stage1_rate_table.csv` |

Every plotted point traces to one of these two qualified, developed tables.
Full per-curve provenance (branch, commit, source table, physical-source
bundle hash, point count) is in `SOURCE_DATA/curve_source_provenance.csv`;
every individual point is in `SOURCE_DATA/all_included_curve_points.csv`.
The full inclusion/exclusion reasoning for every dataset considered is in
`SOURCE_DATA/plot_selection_manifest.csv`.

Excluded by design (see the manifest for full reasoning): old v7 results,
superseded/exploratory "knee" plots, PX3 screening trajectories (superseded
by the qualified PX4 developed result for the same protocols), the
invalidated dwell-duration-weighting Part X trajectories, any monotonic
fracture simulation, PF/FEM/CZM results, virtual C(T) specimen curves, and
any analytical prediction not explicitly drawn as a dashed prediction.

## 3. Figure 1 -- A_NATIVE R-dependence

Single-seed (1720) A_NATIVE developed da/dN plotted against Kmax (Panel A)
and against applied full-range DeltaK = (1-R)*Kmax (Panel B), for R in
{-0.95, 0.1, 0.5}. Rate increases monotonically with R at fixed Kmax (more
tensile mean stress raises the developed rate), and Panel B shows the three
R curves separate cleanly in applied-DeltaK space rather than collapsing.

**R=-0.5 was deliberately excluded from this figure.** A Part X
zero-cohesion baseline at R=-0.5 carries a visible "A_NATIVE"-like label,
but its provenance hash scheme (`physical_producer_sha` /
`config_hash` / `material_row_hash` / `physical_source_bundle_sha256`) is
from a different tooling generation than the R-ratio study's
(`production_solver_hash` / `common_physics_hash` / `composite_hash`), and
no crosswalk between the two exists. The mission requires proof of a shared
material fingerprint, temperature, frequency, event-length semantics,
developed-window definition, and physical source bundle before combining
curves across campaigns; that proof is not available here, so R=-0.5 is
recorded as an **unavailable comparison**, not silently dropped.

## 4. Figure 2 -- A_NATIVE / PT03 / PT08

PT03 and PT08 are **Peierls/Taylor transport-mechanism substitutions used
to test sensitivity of the developed rate to that mechanism -- they are not
distinct calibrated materials.** The three curves are visually
indistinguishable in the main panel; the ratio companion figure
(`Figure2_ratio_companion`) shows the actual (da/dN)_PT / (da/dN)_A_NATIVE
ratio is within about 3% across the full R x Kmax grid (PT03 up to ~3.1%,
PT08 under 0.2%), consistent with the source campaign's own
`PT_R_INVARIANT` classification (max |log10(PT/native)| = 0.0134 decade).

## 5. Figure 3 -- crack-rebonding parameterizations

PX4 developed points for D1/D2 (COMPETING_REVERSIBLE), D5
(PASSIVATION_LIMITED), D3/D6_conditional_persistent (COMPETING_PERSISTENT
at 316.23 Hz), each plotted against its matched zero-cohesion baseline
(identical Kmax/R/frequency/seed). All rebonding curves sit below their
zero-cohesion baseline (shielding), most pronounced at low Kmax and
vanishing near the top of the developed window -- shown quantitatively by
the companion shielding-metric figure,
S_h = log10[(da/dN)_finite / (da/dN)_zero]. **All Part X curves use a
signed-K contact surrogate that does not resolve opposing crack-face
contact**; this is stated on both Figure 3 panels' captions. The
invalidated dwell-duration-weighting trajectories from Part X are excluded
entirely, per the mission's data-provenance rules.

A genuine, verified finding (not a plotting artifact -- confirmed directly
against the raw source rows): the D1 (R=-0.95) and D2 (R=-0.50) rebonding
curves, and separately the D2 (clean) and D5 (passivation-limited) curves,
are numerically almost identical at this seed/condition (S_h differs by
<0.001 dex in every case checked), so their plotted lines overlap almost
exactly. See `SLOPE_INTERPRETATION.md` Sec. 3 for the numeric check.

## 6. Figure 4 -- local and global slopes

Local slope is computed **only** via the mandated adjacent logarithmic
secant formula:

```
m_local,i = [log10(g_(i+1)) - log10(g_i)] / [log10(K_(i+1)) - log10(K_i)]
K_mid,i   = sqrt(K_i * K_(i+1))
```

Global slope is an ordinary least-squares fit of log10(da/dN) vs.
log10(Kmax) over the full developed window. A curve is classified
`PARIS_ADEQUATE` only if **both** R^2 >= 0.95 **and**
max(m_local) - min(m_local) <= max(0.5, 0.25*|m_global|); otherwise it is
`CURVED_NON_PARIS_RESPONSE`. Every one of this package's 29 independently
fitted curves (`SOURCE_DATA/global_slope_summary.csv`) comes back
`CURVED_NON_PARIS_RESPONSE` -- local slopes near the bottom of the
developed window run as high as ~11-21, collapsing to ~1-3 near the top,
which is far outside single-slope tolerance. This is an expected, honest
result, not a bug: it is exactly the "curved, moderate-slope response"
behavior the mission anticipated for A_NATIVE, and it holds for every
family in this package.

## 7. Figure 5 -- material-class context

Panel A shows A_NATIVE/PT03/PT08 (R=0.1) against fixed-exponent m=2/4/8
reference guides. The "physical-slope-transfer target-M~4 row" mentioned in
the mission is **unavailable**: `codex/v10.2.30-physical-slope-transfer`
has no committed artifact beyond a kernel registry, and no per-point table
was found anywhere on the filesystem under that name. This is stated
directly on the panel rather than omitted silently.

Panel B shows the three canonical fracture-selected controls as slope-guide
lines only (arbitrary vertical offset, 2-point construction from the
verified global-slope value -- no per-point developed table for these was
locatable either):

- **Peak: m ~ 100.5**
- **DBTT: m ~ 133.5**
- **ceramic-like: m ~ 71.7**
- **weak-T: no qualified stable global fit -- no line is drawn for it.**

These three are labeled **FRACTURE-SELECTED ACTIVATION-CLIFF CONTROLS**,
not generic experimental Paris exponents for all BCC metals or ceramics.
The figure's caption states, verbatim, as required:

> These are heuristic comparison bands, not fitted targets and not
> universal material-class constants. Material class cannot be inferred
> from Paris slope alone.

## 8. What does NOT appear in this package

No closure-corrected DeltaK_eff appears anywhere in this package -- the
model has no validated resolved opening/contact criterion (Part X's
signed-K surrogate does not resolve crack-face contact; the R-ratio study
omits DeltaK_eff without a validated opening criterion per its own decision
doc). This was checked as an explicit validation item; see
`SLOPE_INTERPRETATION.md` Sec. 5 and the validation log below.

## 9. Reproducibility

`PLOTTING_SCRIPTS/build_curve_database.py` reads only the two raw source
tables in `SOURCE_DATA/RAW_INPUTS/` and writes
`all_included_curve_points.csv` + `curve_source_provenance.csv`.
`compute_slopes.py` reads only `all_included_curve_points.csv` and writes
`global_slope_summary.csv` + `local_secant_slopes.csv` (every global fit
and local secant is independently recomputed from the exported table, not
copied from any pre-existing slope table). `build_selection_manifest.py`
writes the static inclusion/exclusion reasoning. `make_figures.py` reads
only those exported CSVs and produces every figure. No script performs or
launches any physical simulation.

## Validation checklist (run at completion)

- [x] No physical simulation launched; no constitutive/solver source file changed.
- [x] Every plotted point traces to `SOURCE_DATA/all_included_curve_points.csv`, itself built only from the two named qualified, developed tables.
- [x] No PX3 screening substituted where a developed PX4 result exists; no invalidated/censored/excluded row plotted as ordinary.
- [x] Every R-transformation (DeltaK = (1-R)*Kmax, tensile-only range) verified by direct arithmetic.
- [x] Every local secant and global fit is independently reproducible from the exported CSVs alone (`compute_slopes.py` re-derives them; no numbers are hand-copied from prior campaign tables).
- [x] All 8 figure files visually inspected; no clipped/overlapping titles, axis labels, legends, or tick labels remain (narrow-log-decade axes were reformatted with plain-number ticks after an initial defect was found and fixed).
- [x] No closure-corrected DeltaK_eff quantity is plotted or tabulated anywhere in this package (the string appears only in prose explaining why it is excluded -- see `SLOPE_INTERPRETATION.md` Sec. 5).
- [x] No unsupported material-class generalization; Figure 5's required caption text is present verbatim; Peak/DBTT/ceramic-like are labeled FRACTURE-SELECTED ACTIVATION-CLIFF CONTROLS.
- [x] Output directory contains only this focused package (5 figure groups + combined PDF + source data + plotting scripts + provenance manifest + these notes) -- no manuscript-wide bleed-through.
