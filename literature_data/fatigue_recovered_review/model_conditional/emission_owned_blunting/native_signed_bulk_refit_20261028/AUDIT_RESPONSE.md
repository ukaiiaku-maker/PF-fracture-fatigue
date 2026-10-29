# QSTE340TM signed-bulk audit response and targeted follow-up

Date: 2026-10-28  
Branch: `codex/qste340tm-restored-native-refit`  
Starting source HEAD: `fe3762d44d17681206260b570b122cc0cd16b5d4`

This response accepts the independent audit. It supersedes the strength of claim
1 and the rate-development language in `AUDIT_HANDOFF.md`; it does not replace or
rewrite the hash-bound accepted calculations.

## Corrected verdict

1. **Signed bulk gives a modest improvement under a fair refit.** The original
   0.439567 no-bulk number was an unrefitted historical-parent evaluation and was
   not a valid attribution control. An independent no-bulk fit was completed with
   the same 20-start bank, seed, 64-point support, six fit phases, 48 final phases,
   four local refinements, opening profile, bounds, and 760-point evaluation used
   by the accepted signed-bulk fit. Its RMSE is 0.308799 decades, compared with
   0.295233 for signed bulk. The reduction is 0.013566 decades, or 4.39%. Claim 1
   is therefore supported only as a small numerical improvement; the former large
   difference cannot be attributed to the bulk coordinate.
2. **The 5.56 versus 17.35 um native blunting-memory transfer defect was real and
   remains supported.**
3. **The corrected/refitted native sequence remains an improvement, with two
   qualifications.** The one-dimensional opening backstress multiplier is not
   passed to the spatial solver, and the three-event rates are not uniformly
   developed. The accepted 0.633 to 0.512 to 0.384 RMSE sequence and 1.34 to 0.93
   decade R-bias contraction are numerically reproduced, but they must not be
   described as a fully same-equation transfer or a converged long-extension rate
   curve.
4. **The stated scientific limitations remain supported.** The result is not a
   complete experimental fit and does not establish that signed bulk alone causes
   the observed weak R dependence.

## Fair one-dimensional control

| R | measured slope | refitted no-bulk slope | signed-bulk slope |
|---:|---:|---:|---:|
| 0.1 | 3.709 | 3.510 | 3.573 |
| 0.3 | 3.378 | 3.551 | 3.386 |
| 0.5 | 3.762 | 3.881 | 3.837 |
| 0.7 | 2.920 | 2.895 | 3.163 |

The fair comparison is saved in:

- `resolved_increment_1d/no_bulk_matched_refit_20261028/fit.json`;
- `resolved_increment_1d/no_bulk_matched_refit_20261028/predictions.csv`;
- `resolved_increment_1d/MATCHED_CONTROL_COMPARISON.json`; and
- `resolved_increment_1d/matched_bulk_no_bulk_comparison.png`.

The comparison does not isolate every coordinate physically: each arm is allowed
to reoptimize its permitted common parameter vector. It answers whether adding
the signed-bulk state improves the best fit found under the matched search, not
whether a single frozen parameter vector changes when the bulk term is toggled.

## Transfer semantics that must remain explicit

The one-dimensional model evaluates opening stress using the fitted cleavage
backstress coefficient (1.568 for the accepted signed-bulk fit). The native model
does not apply that scalar coefficient. It computes signed spatial shielding from
its evolved fields. This is an intentional difference in state closure, but it
means “same parameter” cannot imply algebraically identical shielding.

The 17.353325 um blunting length also has overlapping rather than identical roles:

- in 1-D it controls memory decay and wake projection;
- in native 2-D it controls ledger memory, forest-density width, and the stress
  profile scale.

The numerical value transfers; the complete operator does not. This is a model
correspondence limitation, not a receipt-binding failure.

## Seven-event rate-window check

Three selected cases were continued to seven accepted 5 um events, or 35 um total
extension (2.02 fitted blunting lengths). No case reached the 5000-cycle cap.

| R, delta K | inter-event waits (cycles) | all-six rate (m/cycle) | log10 all-six / accepted-three | log10 last-two / first-two |
|---|---|---:|---:|---:|
| 0.1, 18 | 333, 293, 306, 325, 295, 288 | 1.630e-8 | +0.009 | +0.031 |
| 0.5, 23 | 52, 42, 39, 36, 31, 28 | 1.316e-7 | +0.110 | +0.202 |
| 0.7, 18 | 29, 42, 32, 25, 21, 19 | 1.786e-7 | +0.220 | +0.249 |

The low-R rate is stable at this resolution. The R=0.5 and R=0.7 trajectories
continue to accelerate, so their two-wait values are not settled developed rates.
The accepted 16-case matrix remains valid as a precisely reproduced three-event
comparison, but its slopes and R spread are preliminary with respect to longer
extension.

These continuations also use the corrected driver process-zone length described
below. They therefore test rate-window length and that numerical correction
together; they are not a pure one-variable continuation of the accepted matrix.
The early event histories are nearly unchanged for R=0.1 and R=0.5, while the
R=0.7 event spacing is materially rearranged. No stronger causal separation is
claimed.

Evidence is under `extended_events/`, including `EXTENDED_EVENT_ANALYSIS.json`,
`extended_event_rates.csv`, `interevent_waits.png`, the three raw step histories,
and campaign receipts.

## Source and evidence corrections

- The native driver now passes its actual 1 um analytical process-zone length to
  `--L-pz`; the stale 5.557665 um literal no longer sets the J-domain and deposit
  kernel. The fitted 17.353325 um blunting-memory length remains a separate input.
- The matrix runner supports explicit case subsets and event targets, allowing the
  targeted seven-event continuations without regenerating the 16-case matrix.
- Future native opening fits record optimization bounds and bound contact. The
  accepted fit remains byte-identical; its bound information is captured in the
  hash-bound sidecar `BOUND_CONTACT.json`. Amplitude 70 and exponent 0.5 contact
  their upper bounds, and the 0.5 cap is narrower than the earlier 1.0 lineage.
- The analyzer now accepts `--out`. Reanalysis into
  `/private/tmp/qste-audit-reanalysis-20261028` reproduced the accepted
  `ANALYSIS.json` and `native_2d_rates.csv` byte-for-byte without overwriting them.

Accepted evidence hashes before and after reanalysis:

```text
25c314873a2dea3addf0ab2be94073c6024d75d40b5366e7122c440fc05e0ac9  ANALYSIS.json
c83610e5516793993ae09f6767354bc1f9c1c361b23c128338d78f181fd9385f  native_2d_rates.csv
f93696bba83207db098e4bda8f02f6e8fdf70bb4b8bfe1493491e82ab6c1bfdc  fit.json
```

## Verification

- Focused regression suite: **62 passed**.
- New and changed scripts compile successfully.
- `git diff --check` reports no whitespace errors in the targeted files.
- The accepted native fit hash remains unchanged.

## Final claim boundary

Supported: a small matched-search benefit from the signed-bulk 1-D state; the
original blunting-memory transfer defect; improved native vertical agreement and
R collapse after correction/refit; active bulk absorption; and stable low-R rate
over two blunting lengths.

Not supported: attribution of the old 0.440 to 0.295 gap to signed bulk; identical
1-D/2-D shielding operators; fully developed rates for the whole native matrix;
settled R=0.5 or R=0.7 rates at 35 um; a complete experimental fit; or uniqueness
of the bound-active opening barrier.
