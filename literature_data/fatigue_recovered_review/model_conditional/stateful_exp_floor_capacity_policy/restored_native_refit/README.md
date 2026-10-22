# QSTE340TM restored-runtime joint refit

## Decision

This executed continuation is a **bounded negative**, not a successful
QSTE340TM calibration and not a global negative for the model family.  The
accepted R2/R3/R8/R9 persistent/reversible runtime was bound successfully and
the legacy implicit-one-second Taylor consumer was excluded.  Nine material
vectors completed native history replay; three deliberately active vectors
were stopped by a declared 360 s per-candidate resource bound during validated
exact-cycle fallback.

Among completed and phase-refined cases, jointly changing the authorized
alternative opening barrier and supported EXP_floor/Peierls/Taylor coordinates
does not reproduce both the measured slopes and the four-history R separation.
The completed native state paths are effectively plasticity-inactive, so the
full-state and frozen-state predictions coincide to reported precision.  The
unresolved active lanes prevent a stronger constitutive impossibility claim.

## Executed model identity

- Starting point: `07ca1070`; additive branch
  `codex/qste340tm-restored-native-refit`.
- Final engine class:
  `arrhenius_fracture.reduced_persistent_reference_r2.ReducedPersistentReversibleTipEngine`.
- Restored model: `r2.restored_reduced_persistent_reversible/1`.
- Calibration binding:
  `qste340tm.restored-r2-r9-alternative-gstar-exp-floor/1`.
- Opening owner: production alternative-G* observer with the inherited
  thermodynamic coupling and cooperative completion rule.
- Source: persistent reusable source, 96,388.8 effective sites per system;
  no legacy finite inventory or source refresh.
- Signed state/transport: installed reversible positive/negative population,
  accumulated-slip, return, active/wake, and moving-frame owners.
- Taylor consumer:
  `MEAN_FORWARD_MINUS_ZERO_RATE_OVER_DENSITY_ORDER`; forward-minus-zero-stress
  elementary rate divided by
  `1 + 1.818093050430288*sqrt(rho/7.496119201299166e11 m^-2)`.
  No finite time window and no `1-exp(-taylor_single)` mapping are present.
- Mechanics: steel `G=80 GPa`, `nu=0.30`, `b=0.248 nm`; prescribed reported K
  reduced provider; 80-bin, 50 micrometre active MPZ.
- Explicit retained/mobile recovery is fixed to zero because the accepted
  persistent-source factory requires it.  It was not bypassed or fitted.
- Protocol: 298.15 K, R = 0.1/0.3/0.5/0.7, provisional 10 Hz.

The full post-installer method identities, file hashes, material definition,
geometry, event, numerical options, owning arrays and source invariants are in
`restored_native_refit.json` under each `runtime_receipt`.

## Search and qualification

The search evaluated historical absolute/shape warm starts plus sustained
active, transport-dominated, trapping-dominated, strong-escape, weak-state,
two targeted activity-transition vectors, one Sobol vector and two local
refinements.  Nine completed.  Sustained-active, transport-dominated and
trapping-dominated vectors each exceeded the 360 s candidate bound after the
accepted high-cycle integrator rejected projective steps and entered exact
cycling.  They are `CandidateWallLimit` resource censors, not material
failures.

Coarse selection used two support states per R history and eight phases.
Finalists used three support states per history and 16 phases.  Low/middle/high
saved states were then evaluated with the production one-cycle observer at
8/16/32 phases.  Across those probes:

- absolute finalist: maximum |log10(exact/profiled)| = 0.00419 decades and
  maximum 8-to-32-phase change = 3.41%;
- shape finalist: maximum |log10(exact/profiled)| = 0.00282 decades and
  maximum 8-to-32-phase change = 2.36%.

Both fitting and qualification use the final bound production opening surface,
actual native radius and shielding.  Peierls and Taylor rates are obtained from
the same installed `_mean_rate_transport` method in fitting and qualification,
so the earlier source-backstress stress-input mismatch is not present.

## Final parameterizations and fit

### Absolute-rate finalist

Plasticity coordinates at 298.15 K:

| coordinate | value |
|---|---:|
| emission G0 | 1.480 eV |
| emission critical stress | 2.500 GPa |
| Peierls G0 | 0.520 eV |
| Taylor G0 | 0.400 eV |

Opening coordinates at 298.15 K:

| coordinate | value |
|---|---:|
| floor | 0.46025733 eV |
| broad amplitude | 5.25e-20 eV |
| broad stress scale | 11.4185 GPa |
| broad exponent | 4.99998 |
| guard amplitude | 0.0400833 eV |

Phase-refined overall RMSE is **0.44790 decades**; within-R centered shape RMSE
is **0.33757 decades**.  Predicted broad slopes for R=0.1/0.3/0.5/0.7 are
**0.111, -0.088, -0.126, -0.047**, versus measured
**3.709, 3.378, 3.762, 2.920**.

### Shape finalist

Plasticity coordinates at 298.15 K: emission G0 1.650 eV, emission critical
stress 2.500 GPa, Peierls G0 0.520 eV and Taylor G0 0.400 eV.

Opening coordinates: floor 0.45112588 eV, broad amplitude 0.0274033 eV,
broad stress scale 10.8925 GPa, broad exponent 2.52773 and guard amplitude
0.0497057 eV.

Overall RMSE is **0.56783 decades** and within-R centered shape RMSE is
**0.12496 decades**.  Predicted slopes are
**2.386, 2.348, 2.180, 1.332**.  Thus the shape improves substantially but the
between-history offsets are wrong.

At approximately 45 MPa sqrt(m) fixed Kmax, the measured low-R/high-R rate
ratio is **31.407**.  The absolute finalist gives **1.046** and the shape
finalist **0.528**.  The former nearly removes the separation; the latter
reverses it.

## State and mechanism result

The absolute finalist reaches at most 5.69e-9 mobile lines, 2.14e-14 retained
lines, 4.93e-5 accumulated-slip lines, a 9.97e-15 m radius increase, and
7.63e-13 Pa sqrt(m) shielding magnitude.  The shape finalist is still less
active.  Returned-slip activity is zero in both completed finalists.  Therefore
the coincident full/frozen curves are evidence of negligible state consequence
for these completed materials; an RMSE difference is not interpreted as a
plasticity fraction.

This result does not establish that the resource-censored active regimes would
also be ineffective.  Their exact-cycle exposure must be completed or a
separately qualified acceleration envelope established before making that
claim.

## Experimental-history mapping and sensitivities

R=0.1, 0.3 and 0.7 rate/history blocks have equal row counts and use ordered
index pairing.  R=0.5 has 171 rate rows and 178 history rows and uses normalized
rank interpolation.  No shared measurement identifier was found, so this is an
explicit mapping assumption rather than a measured time coordinate.

Mapping DeltaK monotonically to normalized measured crack length gives a coarse
absolute RMSE of 0.44527 decades.  Removing pre-measurement exposure gives the
same value to better than 3e-9 decades.  These sensitivities do not change the
decision because the completed state response is negligible.

Reported DeltaK already contains the specimen crack-geometry calculation.  The
accepted signed kernel covers only 1.175 mm cumulative local extension, whereas
the experimental histories span 6.36--13.07 mm.  The primary replay therefore
uses a fixed local kernel window with evolving native source/transport/radius/
shielding state; it does **not** claim a fully remapped 6--13 mm native crack
trajectory.  No kernel extrapolation or unqualified periodic reset was added.

## Files

- `restored_native_refit.json`: complete result, search records, runtime
  receipts, exact observer probes and sensitivities.
- `restored_predictions.csv`: all 760 source-bound observations and finalist
  predictions.
- `restored_state_history.csv`: native state/process summaries.
- `fixed_Kmax_contrasts.csv`: fixed-Kmax matched summaries.
- `restored_fit_DeltaK.png`, `restored_fit_Kmax.png`,
  `restored_residuals.png`, `restored_state_histories.png`, and
  `restored_Gstar.png`: generated comparisons.
- `candidate_cache/`: durable complete/rejected vector records.

## Executed-source and input hashes

| item | SHA-256 |
|---|---|
| `arrhenius_fracture/qste340tm_restored_binding.py` | `5472df4adf84dca94739ebbce76ca1f87c9797fb81f550bd0aba7f77d4fcdf53` |
| `scripts/refit_qste340tm_restored_native.py` | `d1cbef4765b9b42ea51e429a1c4a5ba1de6509cdf5983b6f74f187bfab9fa89f` |
| `tests/test_qste340tm_restored_binding.py` | `3df36ef73a1a5bbe7df12928061153d8aaf7fde73813ed05373f6d17be6c7715` |
| constant-amplitude input | `99f869f5fb4963d37791aceff144c56150fd27ab51c59735588446efba4e36cd` |
| crack-history input | `037218fd910439a3db6a49d919cc84af93fbbea3c2587cc3ef8bd8b548f0053d` |
| alternative-G* registry | `006fc26dd19b67531103942db3d28627be58cc311af665716c5daee4336156f5` |
| V3 parent registry | `016af3a9e79fc7489c18e4b9d825b27f4b1b4e830553e61623b5aa9a573af14f` |
| signed kernel family | `5aa74370e6419104684c52bbcf93323f905d2752d1bba59252f9b0b35c77e07c` |

Validation after execution: `python -m py_compile` passed for the binding and
runner; the focused restored-runtime tests report **5 passed**; `git diff
--check` passes.
