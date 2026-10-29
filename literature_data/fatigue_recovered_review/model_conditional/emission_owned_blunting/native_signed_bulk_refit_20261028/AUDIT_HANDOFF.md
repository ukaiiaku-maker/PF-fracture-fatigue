# QSTE340TM signed-bulk 1-D/2-D transfer and refit: audit handoff

> **Audit status (2026-10-28):** The independent audit identified a mismatched
> no-bulk control, incomplete transfer semantics, and insufficient event-window
> length. See `AUDIT_RESPONSE.md` for the executed matched refit, seven-event
> continuations, corrected verdict, and current claim limits. The original claim-1
> magnitude and the “developed” wording below are retained only as the record that
> was audited; they are superseded by that response.

Date: 2026-10-28  
Repository worktree: `PF-fracture-fatigue_worktrees/r10-cohesive-arrhenius-joint-sweep`  
Branch: `codex/qste340tm-restored-native-refit`  
Starting HEAD: `fe3762d44d17681206260b570b122cc0cd16b5d4`

## Audit decision requested

Determine whether the following limited claims are supported:

1. the additive signed-bulk 1-D treatment improves the experimental fit relative
   to its paired no-bulk control;
2. the first native 2-D comparison contained a real parameter-transfer defect;
3. correcting that defect and refitting only the authorized opening surface
   improves the native 2-D comparison; and
4. the remaining slope and load-ratio discrepancies are honestly bounded and do
   not support claiming a complete experimental fit.

Do not treat this handoff as a request to rerun unrelated historical PF/FEM,
fracture-temperature, cohesive, or parameter-search campaigns.

## Important source-state qualification

The working tree is dirty and contains older unrelated project changes and prior
disk-cleanup deletions. The starting HEAD does **not** contain the implementation
or result files listed below. Audit the explicit files and hashes in this handoff;
do not infer that all worktree changes belong to this result, and do not clean or
reset the worktree as part of review.

## Scientific sequence

### A. Paired 1-D fit

The established seven-state reduction was preserved. Three additive coordinates
were introduced: lower-face signed bulk moment, upper-face signed bulk moment,
and opening-effective absorbed tip content. Gross forward passage creates the
antisymmetric face moments. Crack advance sweeps the geometrically crossed
fraction into the tip, and only the outward-opening projection enters the existing
source-slip blunting law. There is no fitted R-specific term or independent bulk
efficacy multiplier.

Results:

| 1-D treatment | all-data log10 RMSE |
|---|---:|
| paired no-bulk control | 0.439567 |
| signed-bulk additive treatment | 0.295233 |

The signed-bulk block slopes are 3.573, 3.386, 3.837, and 3.163 for R=0.1,
0.3, 0.5, and 0.7. The measured slopes are 3.709, 3.378, 3.762, and 2.921.

Primary evidence:

- `resolved_increment_1d/signed_bulk_joint_refit_20261028/fit.json`
- `resolved_increment_1d/signed_bulk_joint_refit_20261028/predictions_signed_bulk.csv`
- `resolved_increment_1d/signed_bulk_joint_refit_20261028/predictions_no_bulk.csv`
- `resolved_increment_1d/signed_bulk_joint_refit_20261028/paired_common_axis_fit.png`

### B. Original native transfer defect

The first 16-case native matrix enabled continuum bulk plasticity and signed GND
tip absorption, but the native runner retained a hard-coded 5.557665 um blunting
memory. The fitted 1-D material specified 17.353325 um. Therefore that matrix is
an incomplete-transfer control, not a same-material validation.

Original incomplete-transfer metrics:

- native RMSE versus experiment: 0.632725 decades;
- native RMSE versus signed-bulk 1-D: 0.424604 decades;
- R-block bias span: 1.34147 decades.

The old results are preserved under `native_signed_bulk_matrix_20261028/`.

### C. Corrected same-parameter transfer

`run_qste340tm_main_result_full_2d.py` now obtains
`mpz_blunting_length_um` from `parameters.blunting_length_m`. Campaign receipts
record 17.353325 um and `bulk_element_plasticity_enabled=true`.

Sixteen fresh fixed-local-K native trajectories were executed. Each contains
three accepted 5 um crack events. Rates are the event length divided by the mean
of the two inter-event waits; no first-event startup secants enter the metrics.

Corrected-transfer metrics:

- native RMSE versus experiment: 0.512131 decades;
- native RMSE versus signed-bulk 1-D: 0.254016 decades;
- R-block bias span: 1.24847 decades;
- slopes: 2.115, 2.420, 2.867, 5.723.

Evidence: `native_signed_bulk_matrix_corrected_transfer_20261028/`.

### D. State-informed opening refit and fresh native validation

The refit changed only the existing smooth sloping-floor opening coordinates.
Plasticity, geometry, mechanics, event reward, signed state, and loading controller
were held fixed. Acquisition used developed native stress histories, so it was
explicitly labelled a frozen-history approximation. The selected vector was then
tested in a fresh 16-case native matrix rather than accepted from the approximation.

The selected opening coordinates are:

- floor: 17.08392369 kBT;
- amplitude: 69.9999999997 kBT;
- stress scale: 271.84929695 MPa;
- exponent: 0.499999999997.

Amplitude and exponent reached the declared acquisition bounds. This is an
important limitation, not evidence of an interior optimum.

Fresh native-validation metrics:

| Metric | Result |
|---|---:|
| developed cases | 16 / 16 |
| log10 RMSE versus experiment | 0.383605 |
| log10 RMSE versus original signed-bulk 1-D prediction | 0.166316 |
| mean log10 bias | -0.041396 |
| R-block bias span | 0.933274 decades |

| R | block bias (decades) | native slope | measured slope |
|---:|---:|---:|---:|
| 0.1 | -0.4353 | 2.474 | 3.709 |
| 0.3 | -0.2691 | 2.068 | 3.378 |
| 0.5 | +0.0408 | 2.341 | 3.762 |
| 0.7 | +0.4980 | 2.429 | 2.921 |

The supported conclusion is improved transfer, vertical agreement, and R collapse.
The native slopes remain too shallow. Do not describe this as a complete fit.

## Model and execution identity

All final native cases use:

- full 2-D FEM/J mechanics;
- `plasticity_scope=tip_plus_bulk`;
- element plastic-strain and scalar-density evolution;
- signed GND reconstruction from the compatible plastic-distortion curl;
- `opening_advection_state_owner=signed_gnd`;
- signed crack-swept tip absorption enabled;
- moving MPZ transport and blunting ledgers;
- fixed-local-K loading control;
- 48 hazard phases and 16 cyclic-mechanics phases;
- exact one-cycle blocks;
- three accepted 5 um events per condition.

The matrix conditions are four delta-K values for each R:

- R=0.1: 18, 24, 32, 40 MPa sqrt(m);
- R=0.3: 17, 23, 30, 36;
- R=0.5: 15.5, 19, 23, 27;
- R=0.7: 12, 14, 16, 18.

Fixed local K is a material/mechanism comparison controller. Available protocol
evidence suggests the experiment used constant load range. Consequently, these
matrices are not an exact reconstruction of experimental machine control.

## Audit priorities

1. **Parameter binding:** confirm that the final native receipt binds the refitted
   opening coordinates and 17.353325 um blunting length.
2. **Bulk activation:** confirm element bulk plasticity is active and absorbed GND
   fields/counts are nonzero, rather than relying only on CLI labels.
3. **Rate extraction:** independently reconstruct cumulative cycles, three event
   times, two inter-event waits, and developed rate for all 16 cases.
4. **No startup mixing:** verify no first-event secant or one-event upper bound is
   used in the reported RMSE and slopes.
5. **Acquisition/validation separation:** verify the frozen-history refit and fresh
   native validation are distinct and the latter supplies the reported result.
6. **Bound contact:** confirm amplitude and exponent are reported as bound-active.
7. **Transfer semantics:** review whether the 1-D forward-passage moment is a
   defensible acquisition coordinate for the spatial signed-GND transaction.
8. **Controller scope:** preserve fixed-K results as mechanism comparisons; do not
   relabel them constant-force experimental simulations.

## Reproduction commands

Use:

```bash
export PYTHONPATH=.
export MPLCONFIGDIR=/private/tmp/qste-mpl
PY=/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python
```

Analyze the final native matrix without solver advancement:

```bash
$PY scripts/analyze_qste340tm_signed_bulk_2d_matrix.py \
  --matrix literature_data/fatigue_recovered_review/model_conditional/emission_owned_blunting/native_signed_bulk_refit_20261028/matrix \
  --fit literature_data/fatigue_recovered_review/model_conditional/emission_owned_blunting/resolved_increment_1d/signed_bulk_joint_refit_20261028 \
  --out /private/tmp/qste-audit-reanalysis
```

Focused regression suite:

```bash
$PY -m pytest -q \
  tests/test_wake_diagnostic_fields_v10021.py \
  tests/test_v10_2_30_reversible_transport.py \
  tests/test_v10_2_30_high_cycle_engine.py \
  tests/test_qste340tm_restored_binding.py \
  tests/test_qste340tm_resolved_sloping_barrier.py \
  tests/test_fixed_deltaK_v1021.py
```

Observed result: 62 passed.

Do not rerun native trajectories merely to verify CSV arithmetic. If a native
replay is required because a concrete source defect is found, use
`run_qste340tm_signed_bulk_2d_matrix.py --fit .../native_signed_bulk_refit_20261028/fit.json`
with a new output root so accepted evidence is not overwritten.

## SHA-256 bindings

Results:

```text
f93696bba83207db098e4bda8f02f6e8fdf70bb4b8bfe1493491e82ab6c1bfdc  native_signed_bulk_refit_20261028/fit.json
25c314873a2dea3addf0ab2be94073c6024d75d40b5366e7122c440fc05e0ac9  native_signed_bulk_refit_20261028/matrix/ANALYSIS.json
087acbde6373625f2ff05a0fe9171b4eadca02efecf51557ee96c7bfad774e14  native_signed_bulk_matrix_corrected_transfer_20261028/ANALYSIS.json
300fe9830d4d47afe23fa6d9d386a0fe81ee83fdfa5c4a52314ba7d3b39e8205  resolved_increment_1d/signed_bulk_joint_refit_20261028/fit.json
```

Relevant source files:

```text
42b7d901b67bf2bfdd65ce1c4029c94ea0a1562d6c7dd65ae7955e601fc75f9c  scripts/qste340tm_signed_bulk_1d.py
1eb4645e09395d34638e0c61608db87929634e02c5a308ce543d1f27d7dc3b50  scripts/refit_qste340tm_signed_bulk_1d.py
22d82fdbdde8d5a093b9108d4a377ff2a691bf46025a0b83047aef89a55cd0a2  scripts/run_qste340tm_main_result_full_2d.py
69f199a0ab13db31248ba6290b2f2673914dd04f53c02dbbdc8db8aa0db5f5a6  scripts/run_qste340tm_signed_bulk_2d_matrix.py
452b95aa85b77de1ee476e3a10cb6cc00754c30b774ce4e9aad140e7ae1528ac  scripts/analyze_qste340tm_signed_bulk_2d_matrix.py
69114e89fe1e8477c92807bc39001674757d6a77b7fff1ce2a720ba3240e2416  scripts/refit_qste340tm_native_opening_from_histories.py
088de0a22a48e9278bcbb1dd599c8c7d95afd1ce7cf03bfc8face52feea9f3ff  arrhenius_fracture/sharp_front.py
7efe8b0ff857392a34407461d7148200b9bc456ea74d7566f933ac8001aefe4e  arrhenius_fracture/unified_mpz.py
```

## Claim limits

Supported: the signed-bulk coordinate improves the 1-D fit; the native transfer
bug was real; correcting it improves agreement; the constrained opening refit
further improves native vertical agreement and R collapse.

Not supported: an exact experimental-controller reproduction; complete agreement
with measured slopes; uniqueness of the fitted opening barrier; proof that bulk
plasticity alone explains the experimental R invariance; or proof that other
cohesive/opening formulations cannot do better.

