# QSTE340TM final audited fracture-fatigue result

Identity: `QSTE340TM_FINAL_SIGNED_BULK_AUDITED_20261029`

This directory is the paper-replot entry point for the preserved QSTE340TM
result. It retains the accepted signed-bulk 1-D fit and the audit-qualified
all-16-case native 2-D comparison. It does not promote the artificial
radius/backstress counterfactual to a fitted material.

## Current result

- Accepted signed-bulk 1-D log10 RMSE: `0.29523348393431903`.
- Signed-bulk improvement over independently optimized no-bulk controls is not
  resolved because the final opening-profile search is not reproducible to the
  smaller claimed difference.
- Current native driver uses `L_pz = 1.0e-6 m`.
- Corrected-driver native waits 1–2 log10 RMSE: `0.4063853423730828`.
- Native waits 4–6 log10 RMSE: `0.5157911772928764`.
- At 35 um extension, R >= 0.3 cases continue accelerating; the native rates are
  not uniformly developed.

The historical `0.3836048489` native RMSE belongs to a three-event calculation
using the stale `L_pz = 5.557665e-6 m`. It is preserved for provenance but is
not the current baseline.

## Paper-ready tables

- `paper_1d_curves_and_state.csv`: measured rates, signed-bulk prediction, both
  independent no-bulk predictions, and the signed-bulk state coordinates for
  all 760 observations.
- `paper_native_2d_event_windows.csv`: all 16 native conditions, seven event
  times, six waits, corrected-driver early and late rates, and rate evolution.
- `paper_barrier_parameters.json`: opening, emission, Peierls, and Taylor
  barrier parameters and sampled state ranges.
- `MANIFEST.json`: SHA-256 and size for every preserved evidence and source file.

## Primary evidence outside this directory

- `../resolved_increment_1d/signed_bulk_joint_refit_20261028/`
- `../resolved_increment_1d/no_bulk_matched_refit_20261028/`
- `../resolved_increment_1d/no_bulk_equal_effort_control_20261028/`
- `../native_signed_bulk_refit_20261028/`
- `../native_signed_bulk_refit_20261028_extended_7events/`

Read `../native_signed_bulk_refit_20261028/AUDIT_RESPONSE_ADDENDUM_20261029.md`
as the controlling audit qualification. `AUDIT_RESPONSE.md` is retained as the
earlier interpretation.

## Rebuild exports

From the repository root:

```bash
python scripts/export_qste340tm_final_result_20261029.py
```

This command reads existing accepted data only. It does not advance a solver or
refit parameters.
