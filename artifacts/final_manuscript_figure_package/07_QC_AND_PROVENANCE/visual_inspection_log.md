# Visual Inspection Log

Every composite figure (Figures 1-7) and a representative sample of
individual panels and SI/Part X figures were rendered and directly viewed
during generation in this session. Issues found and fixed are logged below;
no unresolved visual defect remains in the final exported set.

| Figure | Issue found | Fix applied |
|---|---|---|
| Figure 1 composite | Panel D colorbar present in standalone panel but missing in composite (inconsistency) | Added matching colorbar to composite Panel D |
| Figure 5 Panel C | Overlapping legend text drawn on top of data points; duplicated/overlapping y-tick labels for the two jittered stress levels | Moved legend outside plot area (`bbox_to_anchor`); replaced ad-hoc jittered yticks with clean `[700, 900]` ticks and per-condition marker offsets |
| Figure 5 Panel B inset | `matplotlib` categorical/numeric axis unit conflict when combining a log-scale bar chart with string category labels on an inset axes | Switched to numeric bar positions (0, 1) with `set_xticklabels` applied separately |
| SI Figure 7 | Transient Agg-backend glyph-metrics failure (`RuntimeError: failed to load glyph`) during `tight_layout()`'s bounding-box computation with many rotated tick labels | Added a try/except fallback in `style_common.savefig_all()` so any such transient failure falls back to a fixed-layout save rather than losing the figure; also switched this panel to explicit `subplots_adjust` margins |
| Figure 7 | Column-name mismatch between the EXP-floor curve's `H_EXPfloor_eV`/`v_EXPfloor_m_per_s` naming and the three fitted forms' `{form}_H_eV`/`{form}_v_m_per_s` naming caused a `KeyError` | Added an explicit column-name lookup dict per form |

## Checks applied to every figure (main text, SI, and the re-exported Part X set)

- No clipped titles, axis labels, or legends (checked via `bbox_inches="tight"`
  export and direct visual review of the 600-DPI PNG).
- No overlapping panel labels (A/B/C/D placed via a consistent
  `transform=ax.transAxes` offset, verified not to collide with plotted data
  or titles in any panel).
- Consistent margins/alignment across the four-panel composites (shared
  `figsize`/`tight_layout` per composite).
- Logarithmic axes verified correct (da/dN, K_c-ΔK_th, AUC-vs-T reference
  line, velocity-proxy panel) with minor ticks enabled via
  `style_common.style_log_axis`.
- Units present on every axis label (MPa√m, m/cycle, K, GPa, m/s, MPa, %,
  dimensionless AUC/Cramér's V).
- Legends checked against plotted objects (each legend entry corresponds to
  a series actually drawn in that panel; unused legend entries were removed
  during the Figure 5C and 5B fixes above).
- Error bars/uncertainty bands visible where the underlying claim carries
  one (Figure 3 seed spread + inset mean±std; Figure 6C 95% bootstrap CI).
- Censor symbols defined in-panel or in the caption (Figure 5A/SI-1 upward-
  triangle upper-bound censor; Figure 5C right-pointing open-triangle
  right-censor).
- Invalidated Part X result (dwell duration-weighting) unmistakably labeled
  in its own re-exported figure title ("INVALIDATED (duration-weighting
  bug)"), inherited unchanged from the accepted branch.
- No unsupported quantitative annotation: Figure 4's peak-class panels use
  "no resolved local maximum" wording, not a numeric "muted shoulder"
  metric; Figure 5D carries no cross-image magnitude ratio.
- Consistent class colors (`style_common.CLASS_COLOR`) and system colors
  (`style_common.SYSTEM_COLOR`) used identically in every figure that
  references the four barrier classes or six fatigue systems.
- No blank or duplicate pages in any combined PDF (`build_combined_pdfs.py`
  asserts the expected page count per combined file: 7 main-text, 9 SI, 12
  curated Part X).
- White, opaque figure background confirmed (`savefig.facecolor="white"` in
  `style_common.py`; no transparent-background rendering artifacts observed).
- PDF/SVG fonts embedded as text (`pdf.fonttype=42`, `svg.fonttype="none"`),
  confirmed by the absence of per-glyph path objects in a manual inspection
  of one exported SVG.

## Automated checks

See `validation_report.json` for the machine-checked results: file
existence, non-zero size, no duplicate figure IDs, no duplicate source-data
content hashes, manifest completeness, and (`all_checks_pass: true`).
`figure_manifest.csv`/`source_data_manifest.csv` additionally record a
SHA-256 for every figure export and every source-data file, so any future
silent modification is detectable by hash comparison.
