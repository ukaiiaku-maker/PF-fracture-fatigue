# Recovered literature fatigue data — import and suitability review
Date: 20 October 2026

## Decision

The supplied ZIP contains two genuine numerical datasets: one 316LN fatigue workbook
and a Lu et al. RAR containing nine Excel workbooks. All ten workbooks were opened.
The original files are preserved. Data normalization is complete; no barrier fitting
or native fracture/fatigue simulation was performed.

**Use the QSTE340TM data first for the proposed inverse calibration.** Unlike an
isolated growth-rate curve, the recovered records include four constant-amplitude
load ratios, eight overload histories, and eight closure-load series. These give a
basis for testing history-dependent mechanics as well as fitting rate sensitivity.

The 316LN workbook remains useful as a separate material/replicate dataset.
Its specimen drawing and independently exported plastic-zone measurements were not
in the supplied ZIP, so it is not the best first dataset for identifying both a
local barrier and a plasticity contribution.

## 1. Exact recovered content

Input ZIP SHA-256 is in `audit/archive_integrity.json`. Its ZIP CRC test passed.
The nine RAR members were decompressed using the system libarchive reader, and
each extracted file CRC agrees with the RAR entry. This verifies supplied-byte
integrity; it does not assert verification against publisher-provided checksums.

| Input | What is present |
|---|---|
| `ffe14223-sup-0001-supplementary data - fatigue.xlsx` | Non-irradiated and Irradiated sheets; seven specimen histories |
| `data_Lu Yun-chao_20181223.rar` | `fig3`, `fig4`, `fig5`, `fig10`, `fig12`, `fig13`, `fig14`, `fig15`, `fig16` workbooks |

The Apple metadata forks were excluded from the numerical dataset. No experimental
file, outlier, duplicate view, or model-comparison file was deleted.

## 2. Primary import tables

Counts below are observable-series counts, not independent specimen totals.
For example, an a(N) history and its author-derived da/dN curve come from the same
experiment and must not be treated as statistically independent measurements.

| Table under `normalized/` | Series | Paired rows |
|---|---:|---:|
| `qste340tm_crack_histories.csv` (fig3) | 12 | 2,417 |
| `qste340tm_growth_rates.csv` (fig4) | 12 | 2,396 |
| `qste340tm_constant_amplitude.csv` (subset of fig4) | 4 | 760 |
| `qste340tm_overload.csv` (subset of fig4) | 8 | 1,636 |
| `qste340tm_closure_loads.csv` (fig5) | 8 | 1,380 |
| `steel316ln_crack_histories.csv` | 7 | 1,539 |
| `steel316ln_unirradiated_primary.csv` (transparent optional subset) | 3 | 799 |

The subset files are not additional observations. Every main record carries its
source workbook, sheet, row, column-cell references, original condition label and
original row order. Original numerical units remain alongside SI conversions.

- crack lengths: mm -> m, factor 1e-3;
- crack-growth rates: mm/cycle -> m/cycle, factor 1e-3;
- closure loads: kN -> N, factor 1e3;
- stress intensity: MPa sqrt(m) -> Pa sqrt(m), factor 1e6.

**The Lu growth-rate values are in mm/cycle, not m/cycle.** The plotted and exported
SI values apply the 1e-3 conversion. No local crack-tip stress is inferred merely
from the stress-intensity column. Closure load remains a force observable, not
an inferred K_open or imposed closure correction.

## 3. QSTE340TM: contents and immediate use

The workbook labels supply R = 0.1, 0.3, 0.5 and 0.7. Each has constant-amplitude
(CA), OLR=1.5 and OLR=2.0 growth histories/rates. OLR labels are preserved; their
precise force-ratio definition and pulse implementation need the original methods.

The four constant-amplitude curves are:

| R | Points | Reported DeltaK range, MPa sqrt(m) | Converted rate range, m/cycle |
|---:|---:|---:|---:|
| 0.1 | 248 | 17.71736–45.43079 | 2.53077e-08–8.56643e-07 |
| 0.3 | 219 | 16.74102–37.22898 | 3.05216e-08–4.35743e-07 |
| 0.5 | 171 | 15.03523–27.53850 | 2.17485e-08–2.02951e-07 |
| 0.7 | 122 | 11.82010–18.07004 | 1.28061e-08–4.33906e-08 |

For exact extrema, use the generated series inventory rather than rounded prose.
The figure `qste340tm_constant_amplitude.png` uses measured/author-reduced data
only; it contains no fitted model line. The separate R=0.1 overload figure shows
the supplied transient curves without merging them into the CA calibration.

A productive eventual division is to fit a common material against selected CA
conditions, test held-out R values, then use overload histories and measured closure
loads to constrain/check state evolution. Measurements at the same DeltaK need
not have the same growth rate after an overload; those are different histories.

Do not import Wheeler/Alpha/2-PI predictions as experimental data, and do not replace
the existing Arrhenius mechanics with those fitted retardation models. Do not
correct applied DeltaK using the source's closure model and independently apply
the same closure again inside the user's model.

## 4. Duplicate, predicted and unresolved data are explicitly separated

### Repeated experimental curves
The experimental columns of fig10, fig13 and fig14 are exactly the same 247-pair
sequence as fig4 Q:R (R=0.3, OLR=2.0). These are not three further experiments.

Fig15 reproduces the eight overload conditions, but not every export is identical:
- some are exact repeats;
- the R=0.1, OLR=2.0 version omits one fig4 pair;
- the R=0.7, OLR=1.5 version omits three fig4 pairs;
- the R=0.7, OLR=2.0 version contains all 122 fig4 pairs plus 14 additional
  experimental-labeled tail points.

All later figures are preserved in
`lu_comparison_figures_NOT_PRIMARY_CALIBRATION.csv`. The 14 extra points are also
exported separately in `qste340tm_additional_fig15_points_REVIEW.csv`. They are not
silently discarded, nor concatenated as independent observations before reconciliation.
Pair-overlap details are in `audit/duplicate_curve_audit.csv`.

### Model outputs
The later comparison figures include 4,127 explicitly model-predicted rate pairs,
kept separate from primary measurements. Some predictions use measured inputs
(e.g. a measured closure quantity); that does not make the output rate experimental.

Fig12 contains transformed experimental/model coordinates and fitting curves,
not an immediately identifiable DeltaK–da/dN table. Its six columns and headings
are preserved without inventing inverse transformations.

Fig16 contains separate DP780 and 6082T6 reference curves and associated model
predictions. These are not additional QSTE340TM specimens.

### Unequal row lengths and source ordering
Six fig3 M:N rows (182–187) contain cycles but no crack length. They are preserved
in `audit/unpaired_cells.csv` and the raw sheet export; no zero crack length is
imputed. The paired history ends at its last complete observation.

Fig3 orders R=0.3 before R=0.5. Fig4 places R=0.5 before R=0.3. The importer reads
each condition label independently, rather than equating ordinal columns.

Many source curves have nonmonotone DeltaK or reported crack-length increments.
All source order and numerical values are retained with flags. No smoothing,
sorting, physical healing interpretation, isotonic correction or extrapolation
was applied. Cross-figure rows must not be matched by index: their counts and
sampling differ.

## 5. 316LN: useful replicate histories, not yet a ready DeltaK dataset

| Specimen | Source condition | Points | Last observed cycle |
|---|---|---:|---:|
| A15 | Unirradiated | 262 | 626,400 |
| B3 | Unirradiated | 279 | 667,200 |
| B7 | Unirradiated | 258 | 616,800 |
| B42 | Unirradiated; author labels “outlier” | 111 | 264,000 |
| A1 | 1 dpa | 210 | 501,600 |
| B5 | 1 dpa | 192 | 458,400 |
| A6 | 3 dpa | 227 | 542,400 |

These are last recorded observations, not independently verified failure times.
B42 remains in the full table and plotted comparison. The optional primary
unirradiated view follows the source's label transparently rather than deleting it.

The three main unirradiated specimens have many repeated recorded lengths.
Typical reported steps are approximately 0.03333 mm (33.3 micrometres).
These are observations of the tabulation, not an independently calibrated
instrument-resolution estimate. B5 contains two negative reported increments.
Consequently, direct a(N) fitting is preferable to declaring zero/negative
physical growth from adjacent finite differences.

The upload contains no nanoindentation workbook and no numerical plastic-zone-area
series. Irradiated specimens remain separate from the initial unirradiated calibration;
there is no need to implement irradiation physics just to use the controls.

## 6. Metadata verified outside the archive, and what remains missing

Source-derived data and supplementary web verification are intentionally separate.

### QSTE340TM
The author-deposited dataset description confirms that it contains CA/OL experiments
and predicted rates. The publisher confirms the load-ratio/overload/closure scope.
The uploaded workbooks do not give the exact test temperature, frequency,
waveform, geometry, force schedule or overload timing. The article's full methods
were not recoverable in this review. These fields remain null, not 300 K or 1000 Hz.

The existing DeltaK and rate columns are sufficient for data-level plotting,
regression and curve-shape comparison. Absolute barrier energies and stateful
overload/closure predictions need the physical protocol and the material-specific
local-stress/plasticity mapping.

### 316LN
The online paper's section 2.3 specifies sinusoidal 450 +/- 150 N loading at
20 Hz, hence Fmin=300 N, Fmax=600 N and R=0.5. Precracking used 80 Hz and is a
different history. The paper lists outside dimensions 25 x 24 mm and nominal
post-polishing thickness 0.8 mm; those outside dimensions are not substituted
for the CT geometry-factor width W.

The paper describes a grip-hole-center crack-length coordinate. The workbook
simply labels its near-1-mm initial values as “Crack Length (mm)”. Without the
drawing and the numerical origin convention, no notch or load-line offset was
invented. `absolute_CT_coordinate_verified` stays false, and no DeltaK was calculated.

The exact fatigue-test temperature is not identified in the uploaded workbook.
Room-temperature irradiation labels must not become a fatigue temperature panel.

## 7. Consequences for the intended barrier inverse

The strongest first dataset is now QSTE340TM, not 316LN, because the recovered
files provide both applied-DeltaK growth curves and closure/history information.
Use one material vector across its R conditions, with the actual waveform and
evolving state. Do not fit the target exponent to four.

The source's measured closure force is an additional observable, not a direct
measurement of a universal “plasticity percentage”. A full model should match
the appropriate mechanically defined closure observable or state explicitly
that the current reduction lacks it. A minimum plastic contribution imposed
during fitting would remain a prior; it is not extracted automatically from
these files.

A constant-amplitude opening-only fit is a useful reference. It is not enough
to establish that plasticity is negligible. Held-out overload recovery and
closure trends can test that interpretation without forcing a numerical fraction.

The QSTE340TM and 316LN data concern different materials and protocols. No common
barrier vector, common plasticity calibration, or shared temperature law is assumed.

## 8. Reproduction and verification

Run:
```
python verify_import.py
```

The standard-library verifier checks all source-file hashes, 7,732 paired records
against their exact source-cell CSV exports, unit conversion, CA/overload selection,
outlier flags, extra points and missing-metadata safeguards. It needs neither RAR
software nor a spreadsheet application. The original extracted workbooks are
included for direct inspection. Main spreadsheet values were read using
`artifact_tool`, not retyped or digitized from figures.

`MANIFEST_SHA256.json` hashes every delivered file except itself. The whole ZIP
has its own checksum sidecar. No hashes claim that native calculations were run.

## Source references
- Lu, Yang and Chen (2019), “Effect of single overload on fatigue crack growth in
  QSTE340TM steel and retardation model modification.”
  Paper: https://doi.org/10.1016/j.engfracmech.2019.03.029
  Author data: https://data.mendeley.com/datasets/8pcx2mgfd4/1
- Weihrauch, Patel and Patterson (2024), “Effects of Ion Irradiation on Stage II
  Fatigue Crack Growth and Plasticity.”
  https://onlinelibrary.wiley.com/doi/10.1111/ffe.14223
  Methods sections 2.1/2.3 and supporting-information listing.
- Actual uploaded files are identified by hashes in `audit/source_inventory.csv`.
  Source license terms remain those of the publications/data deposits; this import
  does not relicense their measurements.
