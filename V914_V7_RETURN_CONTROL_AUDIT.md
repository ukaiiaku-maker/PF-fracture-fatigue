# v9.14 v7 return-control and irreversible-baseline audit

## Decision

The strict current-v7 no-return control has been implemented and behaves as a
causal A/B control.  It does **not** recover the historical weak-T finite-life
regime when used with the common-physics input used by the current v7 audit.
The historical regime is recovered immediately when the same current v7-IRREV
cycle law is supplied the historical calibrated common-physics input.

This meets the requested stop condition: the loss of the old regime predates
the returned-source-slip cancellation comparison and is caused by a change in
the supplied common physics, not by reversible cancellation.  A paired Paris/R
or frequency campaign would mix two different baselines and is therefore not
run.

## Historical weak-T provenance

Candidate: `v913_zeroD_sobol_0129902`; 300 K; R=0.1; 1000 Hz; seed 1720.

The canonical campaign is from branch `v9.14-cyclic-fatigue-knee-search`, commit
`3fc0d1a3d10a649fc2ff2d9db062154bd0f936ba`, runner
`scripts/run_v914_fatigue_campaign.py`, and engine
`arrhenius_fracture/fatigue_v914.py`.  Its acceleration had two distinct paths:

- validated affine DMD projected mobile, retained, and irreversible source-slip
  active state while integrating cumulative ledgers from independently sampled
  rates;
- a separate periodic/frozen-state skip held the state and per-cycle hazard
  fixed after a periodic-state test.

Thus these results were not all explicit and were not all DMD-only.

| DeltaK (MPa sqrt(m)) | da/dN (m/cycle) | method actually used | validity for a Paris fit |
|---:|---:|---|---|
| 8.892055 | <=6.892e-20 | DMD + periodic/frozen; 3 events; censored at 1e14 | censor only |
| 10.797495 | 4.877e-16 | DMD + periodic/frozen | not exact; provisional |
| 12.702936 | 3.607e-11 | DMD + periodic/frozen | not exact; provisional |
| 13.973229 | 8.441e-8 | validated affine DMD; 221227 projected cycles | not exact; partial explicit control did not reach target |
| 15.243523 | 1.791e-5 | explicit in the canonical driver; zero DMD projection | valid finite-life realization |

Fitting the four completed canonical points gives the previously reported large
slope, `m_old=71.45`.  That is an accelerated mixed-method slope, not an exact
irreversible Paris slope.

Later explicit-cycle controls used branch `codex/v10.2.32-explicit-cycle-LCF`,
runner `scripts/run_v1032_explicit_lcf.py`, and integrator
`scripts/v1032_explicit_cycle_lcf.py` with DMD and multi-cycle projection both
disabled.  Exact commits are retained per row because the refinement campaign
advanced during execution.

| DeltaK | da/dN or partial extension/N | status | commit | fit status |
|---:|---:|---|---|---|
| 13.973229 | 3.447e-9 | 4 events, censored at 5000 cycles | `57dc3fde41b2df93701ac4094a3af317eeb6713d` | partial only |
| 14.227288 | 3.906e-9 | 5 events, censored at 5000 cycles | `85dd24b90b261fecbabfa4475c34de650cf357bc` | partial only |
| 14.481347 | 3.906e-9 | 5 events, censored at 5000 cycles | `85dd24b90b261fecbabfa4475c34de650cf357bc` | partial only |
| 14.608376 | 2.593e-8 | target reached at 4053.51 cycles | `57dc3fde41b2df93701ac4094a3af317eeb6713d` | exact finite-life point |
| 14.735405 | 2.639e-8 | target reached at 3982.44 cycles | `85dd24b90b261fecbabfa4475c34de650cf357bc` | exact finite-life point |
| 14.989464 | 1.988e-4 | target reached at 0.528526 cycles | `85dd24b90b261fecbabfa4475c34de650cf357bc` | exact, but discontinuous realization |
| 15.243523 | 3.630e-7 | target reached at 289.512 cycles | `9019e475ca8425e8d880bf5d92a4b2c2870deb95` | exact finite-life point |

The exact points establish a finite-life regime, but they are sharply
non-monotone around DeltaK=14.99--15.24.  No defensible contiguous exact Paris
slope is inferred from them.  In particular, they do not validate `m=71.45`.

## Strict v7-IRREV implementation

`IntrinsicReverseGlideNoReturnState` inherits the full current v7 state and
overrides only `_cancel_returned_source_slip`.  Physical left-boundary mobile
return is still transported and recorded, but it does not increment
`returned_slip_m2` or the source-slip cancellation ledgers.  Cleavage,
three-hit kinetics, finite-tip stress, intrinsic signed glide, GND stress,
emission, transport/storage, exact adaptation, rate-separated DMD, thresholds,
event localization, and event laws are unchanged.

The exact cycle-map and VHCF runners expose this as
`--return-control REV|IRREV`.  The state model ID is included in run contracts
and checkpoint identity, preventing a REV checkpoint from restarting as IRREV
or conversely.

## Positive-R null and fixed-Kmax mechanism checks

At fixed Kmax=30 MPa sqrt(m), current common physics, 300 K, and 1000 Hz:

- R=+0.1: the complete 12-cycle REV and IRREV result payloads are exactly equal.
  Cycle-12 hazard is 2.23330e-10, radius is 7.03881 um, and physical
  reverse-driven return is zero.
- R=0: the two-cycle REV and IRREV histories are exactly equal.  Cycle-2 hazard
  is 2.91085e-16, radius is 1.61433 um, and physical return is zero.
- R=-0.5: the four-cycle histories are equal to reported precision; physical
  return fraction is 7.16e-48.
- R=-0.95: reverse motion is measurable and creates the intended controlled
  difference.

At R=-0.95 after four exact cycles:

| observable | V7-REV | V7-IRREV |
|---|---:|---:|
| physical mobile return fraction | 1.60983e-4 | 1.60764e-4 |
| cancelled weighted source-slip count | 1.78040 | 0 |
| net weighted source-slip count | 11057.774 | 11058.821 |
| radius (um) | 6.910351 | 6.910910 |
| signed plastic-displacement proxy (um) | 4.958652 | 4.958212 |
| reverse displacement proxy (nm) | 1.30925 | 1.30729 |
| retained line content | 61.2601 | 61.1274 |
| Kshield (MPa sqrt(m)) | 0.548594 | 0.547443 |
| exact cycle hazard | 3.34066e-10 | 3.33148e-10 |
| frozen endpoint hazard | 7.12528e-11 | 7.12464e-11 |

The sign is resolved by the calculation: cancellation slightly resharpens REV,
but its small changes to retained content and shielding leave its exact cycle
hazard only 0.275% above IRREV at this time.  The reverse displacement is only
0.0264% of the signed displacement magnitude.

## Regression diagnosis

The historical canonical and explicit runs used the calibrated transferred-2-D
common physics with SHA-256
`59ebf68b366045c91fb08bfe870703c52e1f4e6b51c3f1c480b28dfc8ef8e9c6`.
The current v7 causal audit used the legacy-1-D common file with SHA-256
`b5d7e54631928c6e7fb5b5d7304e15ec60a9e3e38c7ef5c3d50fc69ea98ec813`.
The candidate row itself was not refit, but the common mappings differ:

| common quantity | historical calibrated | current v7 audit input |
|---|---:|---:|
| emission Schmid factors | [0.0766503, 0.00499120] | [1, 1] |
| shielding orientation factors | [0, 0] | [1, -1] |
| activation-to-line conversion | [0.912409, 0.912409] | [1, 1] |
| coupled moving tip | true | false |
| mobile transport velocity scale | 0 | 1 (dataclass default) |
| encounter efficiency | 9.16025 | 1 (dataclass default) |

This is not a narrow reversibility change.  Most importantly for the observed
positive-R passivation, the source drive changes from 0.07665/0.00499 to 1/1.

At the old exact finite-life point DeltaK=14.608376, R=0.1:

- current v7-IRREV + historical calibrated input: the exact cycle hazard stays
  near 25.454 through 12 cycles and radius stays essentially 1 um;
- current v7-IRREV + current audit input: the rate-separated VHCF engine finds
  no event through 1186 cycles, accumulated hazard is only 3.876e-9 against the
  seed-1720 threshold 0.375369, radius is 2.401 um, and the last exact endpoint
  hazard is 6.230e-13.  Nine DMD blocks were accepted, reaching stride 512; the
  run then stopped fail-closed at its declared 257-map work budget.

The historical exact trajectory had its first event at 0.445858 cycle and
reached 105.087 um by 4053.51 cycles.  Hence current v7-IRREV has lost a
previously exact finite-life regime before return cancellation can act.

Per the requested decision rule, development stops here to resolve which common
physics is authoritative for the controlled reversible extension.  Reverting
to the historical input would also set mobile transport velocity to zero and
would therefore eliminate the very signed transport/return mechanism being
studied; retaining the current input changes several non-return physics fields.
Choosing between them is a physics-contract decision, not a numerical fix.

## Artifacts and change classification

Artifacts (not tracked by Git):

- `runs/v914_v7_return_control_exact_Kmax30/`
- `runs/v914_v7_return_control_exact_reduced/`
- `runs/v914_v7_return_control_displacement/`
- `runs/v914_v7_return_control_old_physics_exact/`
- `runs/v914_v7_irrev_old_finite_load_vhcf/`

Change classification:

- Physics: unchanged; one explicitly labeled no-cancellation control added.
- Numerical representation: unchanged; the selected state model is included in
  VHCF checkpoint identity.
- Diagnostics: return-control selection and causal comparison outputs added.
- Launch tooling: `--return-control REV|IRREV` added to exact and VHCF runners.
