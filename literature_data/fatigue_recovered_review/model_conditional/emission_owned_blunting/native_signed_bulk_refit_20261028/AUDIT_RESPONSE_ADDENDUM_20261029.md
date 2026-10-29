# QSTE340TM signed-bulk audit: addendum to AUDIT_RESPONSE.md

Date: 2026-10-29  
Branch: `codex/qste340tm-restored-native-refit` (worktree still dirty; nothing reset or cleaned)

Two sessions worked on the audit follow-up at the same time on 2026-10-28.
`AUDIT_RESPONSE.md` records one session's work. This addendum records the other
session's work and the points where its evidence changes the conclusions.
No hash-bound accepted evidence was rewritten. Paths are relative to
`emission_owned_blunting/` unless shown as `scripts/`.

## What this addendum adds

1. A second equal-effort no-bulk 1-D control, run independently. It shows that
   the final full-data opening profile is not reproducible.
2. All 16 native cases continued to 7 events. `AUDIT_RESPONSE.md` continued only
   3 cases.
3. A provenance note: the 7-event runs use the corrected `--L-pz` driver, so
   they also measure how sensitive the accepted matrix is to that setting.

## 1. One-dimensional control: the final opening step is not robust

| 1-D model, 760 points | log10 RMSE | mechanical SSE | final opening-profile SSE |
|---|---:|---:|---:|
| Unrefit parent, no bulk (original handoff control) | 0.4396 | — | — |
| Equal-effort no-bulk control (this addendum) | 0.3381 | 0.559159 | 0.5708 |
| Matched no-bulk refit (`AUDIT_RESPONSE.md`) | 0.3088 | 0.559149 | 0.4466 |
| Signed bulk (accepted) | 0.2952 | — | 0.4113 |

The two no-bulk controls used the same 20-start bank (seed 20261028), support
points, phases, refinements and bounds. Both picked start `perturb_019`, and
their mechanical optima agree to 1e-5 in SSE. They differ only in the last
step, the 48-phase full-data opening differential-evolution profile
(seed 20261128). That step moved the RMSE by 0.029 decades between two
near-identical inputs. In this addendum's control,
`cleavage_backstress_fraction` ends at its upper bound of 2.0.

**Effect on claim 1.** `AUDIT_RESPONSE.md` reports a signed-bulk benefit of
0.0136 decades (4.4%). That is less than half the 0.029-decade spread produced
by the final step alone. The signed-bulk arm also ran that step only once. So
neither the sign nor the size of the benefit is established. Until the final
profile is repeated with several seeds in both arms, claim 1 should read:
"no resolvable difference under the matched search".

## 2. Native matrix: all 16 cases continued to 7 events (35 µm)

Same material fit (`fit.json`, sha256 `f93696bb…`), same command-line flags,
`--events 7`, written to a new output root. Every case completed.

| Window | RMSE vs experiment | mean bias | R-block bias span | slopes R = 0.1 / 0.3 / 0.5 / 0.7 |
|---|---:|---:|---:|---|
| Accepted matrix, waits 1–2 (driver `L_pz` = 5.56 µm) | 0.384 | −0.041 | 0.933 | 2.47 / 2.07 / 2.34 / 2.43 |
| This run, waits 1–2 (driver `L_pz` = 1 µm) | 0.406 | −0.027 | 0.976 | 2.51 / 1.70 / 2.25 / 2.78 |
| This run, waits 4–6 | 0.516 | +0.112 | 1.259 | 2.48 / 1.68 / 2.05 / 2.00 |
| Experiment slopes | — | — | — | 3.71 / 3.38 / 3.76 / 2.92 |

Late window minus accepted window, log10 rate, for each case:

| R | ΔK cases | late − early (decades) |
|---:|---|---|
| 0.1 | 18, 24, 32, 40 | +0.015, +0.033, −0.144, +0.075 |
| 0.3 | 17, 23, 30, 36 | +0.113, +0.091, +0.152, +0.082 |
| 0.5 | 15.5, 19, 23, 27 | +0.184, +0.208, +0.171, +0.139 |
| 0.7 | 12, 14, 16, 18 | +0.350, +0.301, +0.247, +0.214 |

Findings:

- **The rates are not developed at 35 µm, which is about 2 blunting lengths.**
  For R ≥ 0.3 the waits keep shrinking through wait 6 in every case. For
  example, R = 0.7, ΔK = 12 gives waits of 119, 99, 80, 60, 47 and 39 cycles.
  The acceleration grows with R, so the R-block bias span widens rather than
  collapsing.
- **R = 0.1 is mostly stable**, as `AUDIT_RESPONSE.md` states. The exception is
  ΔK = 32, whose waits run 42, 67, 78, 91, 72 and 65 cycles and are not
  monotone.
- **Most of the claim 3 gain disappears in the late window.** RMSE is
  0.633 → 0.384 in the accepted 3-event window, but 0.633 → 0.516 in the late
  window. The R-block span goes from 1.34 to 0.93 in the accepted window and
  from 1.34 to 1.26 in the late window. The predicted slopes stay 1.2–1.7 below
  experiment at R ≤ 0.5.
- **The results agree with the other session.** It continued 3 cases
  (R0.1/ΔK18, R0.5/ΔK23, R0.7/ΔK18), and they match this run wait for wait.

## 3. Provenance: the `--L-pz` driver change

`scripts/run_qste340tm_main_result_full_2d.py` was edited at 21:14 on
2026-10-28, before the 7-event runs started. The edit replaced the stale
`--L-pz 5.557665e-6` with the 1 µm analytical MPZ length, and receipts now
record `driver_L_pz_m`. That makes the 7-event runs a test of two changes at
once: a longer window and the `L_pz` correction.

- The first event matches the accepted matrix in all 16 cases.
- Events 2 and 3 match in none. Most shift by a few percent; the largest shifts
  are R0.3/ΔK36 (60 → 47 cycles) and R0.7/ΔK18 (31 → 45 cycles).
- With only that change, the early-window slopes move by up to 0.37: R = 0.3
  goes 2.07 → 1.70 and R = 0.7 goes 2.43 → 2.78.

The accepted 0.384 figure therefore depends on a numerical setting that is now
known to be wrong. The corrected-driver early-window value is **0.406**, and
that is the figure to quote for the current code.

## 4. Changes made, by session

### This addendum's session (new files only; no existing file edited)

| Path | sha256 |
|---|---|
| `scripts/refit_qste340tm_no_bulk_1d_control.py` | `a8770b83…9513295` |
| `scripts/analyze_qste340tm_signed_bulk_extended_events.py` | `642e7ad3…25bfa` |
| `resolved_increment_1d/no_bulk_equal_effort_control_20261028/fit.json` | `b78a3425…136ebe7` |
| `resolved_increment_1d/no_bulk_equal_effort_control_20261028/predictions_no_bulk_refit.csv` | `04e4313a…74c3` |
| `resolved_increment_1d/no_bulk_equal_effort_control_20261028/README.md` | — |
| `native_signed_bulk_refit_20261028_extended_7events/MATRIX.json` and 16 case folders | `118c5496…1cb61aeb` |
| `native_signed_bulk_refit_20261028_extended_7events/EXTENDED_ANALYSIS.json` | `32feb283…91b78` |
| `native_signed_bulk_refit_20261028_extended_7events/extended_event_windows.png` | — |
| `native_signed_bulk_refit_20261028_extended_7events/README.md` | — |
| this file | — |

The analysis script checks itself against the accepted matrix: given the
accepted CSVs, it reproduces the accepted early-window metrics exactly.

### The `AUDIT_RESPONSE.md` session (summarised from that file and file timestamps)

| Path | Change | Current sha256 |
|---|---|---|
| `scripts/run_qste340tm_main_result_full_2d.py` | `--L-pz` set from the MPZ length; `driver_L_pz_m` recorded | `3722e958…220e1` |
| `scripts/run_qste340tm_signed_bulk_2d_matrix.py` | supports case subsets and event targets | `cf6730a3…d33` |
| `scripts/refit_qste340tm_native_opening_from_histories.py` | records bounds and bound contact | `7f466433…b7f` |
| `scripts/analyze_qste340tm_signed_bulk_2d_matrix.py` | adds `--out` | `c3d13bf2…62` |
| `scripts/refit_qste340tm_no_bulk_matched.py`, `scripts/compare_qste340tm_matched_1d_controls.py`, `scripts/analyze_qste340tm_extended_event_windows.py` | new | — |
| `resolved_increment_1d/no_bulk_matched_refit_20261028/`, `resolved_increment_1d/MATCHED_CONTROL_COMPARISON.json`, `matched_bulk_no_bulk_comparison.png` | new | — |
| `native_signed_bulk_refit_20261028/BOUND_CONTACT.json`, `extended_events/`, `AUDIT_RESPONSE.md` | new | — |
| `native_signed_bulk_refit_20261028/AUDIT_HANDOFF.md` | status header added | — |

The first four scripts above are bound by hash in the original handoff, so
those bindings no longer match. Accepted evidence (`fit.json`,
`matrix/ANALYSIS.json`, `matrix/native_2d_rates.csv`) is unchanged.

## 5. Revised claim status

| Claim | `AUDIT_RESPONSE.md` | With this addendum |
|---|---|---|
| 1. Signed bulk improves the 1-D fit | Supported as a small (4.4%) improvement | Not resolved: the 0.0136 gap is less than half the 0.029 spread of the final opening step |
| 2. Native transfer defect (5.56 vs 17.35 µm) | Supported | Supported |
| 3. Correction and refit improve native agreement | Supported, with qualifications | Holds only in the 3-event window, and it depends on the stale `L_pz`. Late window: RMSE 0.516, span 1.26 |
| 4. Limitations are honestly bounded | Supported | Supported; also add the undeveloped R ≥ 0.3 rates and the `L_pz` sensitivity |

## 6. Open decisions and next runs

1. **Opening-profile robustness.** Repeat the final 48-phase opening profile
   with at least 5 seeds in both the no-bulk and signed-bulk arms, starting
   from their mechanical optima. Report the minimum and the spread before
   comparing the arms.
2. **`L_pz` baseline.** Accept the corrected driver and re-quote the native
   numbers under it (early window 0.406). Keep the 0.384 figure only as a
   historical value.
3. **Rate development.** For R ≥ 0.3, run at least 10–12 events (about 3–4
   blunting lengths), or until consecutive waits change by less than 5%.
   Then refit the native opening surface on a developed window, not on waits
   1–2.
4. **Backstress transfer.** The 1-D fits drive `cleavage_backstress_fraction`
   to between 1.57 (signed bulk) and its 2.0 bound (no bulk). The native model
   has no such term. Decide whether this is an accepted difference in how the
   two models close the shielding, or whether native needs an equivalent.
5. Fixed-K native results stay labelled as mechanism comparisons. No complete
   fit is claimed.
