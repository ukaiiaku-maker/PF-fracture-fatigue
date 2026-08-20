# Weak-T v9.14 v7 causal physics audit

## Scope and authority

This audit uses the unmodified adaptive exact v7 cycle map at 300 K and 1000 Hz.
The production trajectory always uses the existing cleavage, emission, transport,
storage, shielding, and blunting laws. Counterfactual cleavage evaluations are
read-only evaluations of the same saved constitutive state; they never feed back
to that state.

The production-resolution artifact is
`runs/v914_v7_causal_physics_weakt_exact128/causal_physics_audit.json`, with its
flat table in `causal_physics_rows.csv` and summary figure in
`causal_physics_summary.png`. It uses 128 MPZ bins, 32 base phase intervals, two
coupled operator substeps, adaptive depth 22, and saved cycles 0, 1, 2, 4, 8,
and 12.

## Result

The weak-T model self-passivates because source emission creates a large
source-slip blunting state during the first cycle. The increased radius lowers
the cleavage stress enough that the raw one-hit rate leaves its saturated
regime. The existing three-hit transformation then adds another 7--10 orders
of suppression at the developed saved states. Shielding is irrelevant at
positive R in these trajectories.

| Kmax (MPa sqrt(m)) | cycle-1 exact H | radius after cycle 1 (um) | cycle-12 radius (um) | cycle-12 full H | initial-radius/full H at cycle 12 | raw/three-hit H at cycle 12 |
|---:|---:|---:|---:|---:|---:|---:|
| 12.00 | 6.952e-13 | 1.326 | 1.369 | 1.252e-13 | 1.595e5 | 2.094e10 |
| 30.00 | 8.714e-8 | 6.626 | 7.039 | 2.149e-10 | 2.332e12 | 1.458e8 |
| 62.60 | 5.498e-7 | 27.624 | 29.549 | 8.667e-10 | 8.312e11 | 5.753e7 |
| 70.77 | 7.519e-7 | 35.039 | 37.531 | 1.099e-9 | 6.822e11 | 4.912e7 |

At Kmax 70.77, the virgin-radius frozen state supports 749.44 hazard units per
cycle. After one exact physical cycle the full saved-state value is 1.507e-8;
the exact first-cycle action is 7.519e-7 because most of it accumulates before
the endpoint is fully blunted. By cycle 12, the raw one-hit frozen action is
5.396e-2 while the production three-hit action is 1.099e-9. Thus neither a
long-cycle projector nor low applied load causes the collapse.

## High-stress cleavage ceiling

For the immutable weak-T cleavage surface at 300 K,

```text
G0(T)                         = 1.5918719428 eV
Gfloor                        = 0.00306779253 eV
nu0                           = 1.0e12 1/s
Gamma_raw,max                 = 8.8810303396e11 1/s
n_hits                        = 3
tau_c                         = 1.0e-6 s
Gamma_c,max                   = 1.0e6 1/s
maximum action / 1000-Hz cycle = 1000
minimum mean waiting time      = 0.001 cycle
```

The ceiling exceeds the action needed for mean waits of 1e3, 1e5, 1e7, 1e9,
1e12, and 1e14 cycles by factors 1e6, 1e8, 1e10, 1e12, 1e15, and 1e17,
respectively. The EXP floor and three-hit saturation ceiling therefore do not
preclude an ordinary-growth regime at sufficiently high instantaneous stress.
The three-hit law is nevertheless decisive after blunting moves the raw rate
into its low-exposure regime.

## Source slip to tip radius

The exact code path is:

```text
one source activation
  -> one line-content unit distributed uniformly across the source-zone bins
  -> weighted slip count = sum(rho_slip * dx * strip_width * exp(-x/strip_width))
  -> r_tip = r0 + c_blunt * b * blunting_slip_fraction * weighted slip count
```

The density has units m^-2 and `dx * strip_width` has units m^2, so the weighted
slip count is dimensionless and the radius increment has units metres. For the
128-bin audit, the six source bins have mean tip weight 0.2060874. With
`c_blunt=1.95072017`, `b=2.74e-10 m`, unit line conversion, and unit blunting
fraction, one activation gives:

```text
crack-tip slip displacement before projection = 0.2740 nm
unweighted geometric increment                = 0.5345 nm
weighted effective CTOD/radius increment      = 0.1102 nm
```

Using the immutable emission EXP characteristic stress at 300 K as the model's
flow-stress reference (`sigma_flow=0.72135 GPa`), with `E=410 GPa` and plane-
strain `E'=444.88 GPa`, the cycle-12 positive-R radius is 7.35--7.40% of
`K^2/(6*pi*sigma_flow^2)` at Kmax 62.6--70.77. The plastic radius increment is
2.34 times the dimensional CTOD reference `K^2/(E'*sigma_flow)`. These are
large-blunting but conventional-scale values, not evidence of a units or
line-count amplification defect.

## Radius versus shielding causal decomposition

At R=0.1, retained content and Kshield are numerically negligible (Kshield is
about 1e-100 MPa sqrt(m)). Consequently `H_full/H_zero-shield=1` throughout.
At cycle 12, `H_full/H_initial-radius` is 6.27e-6 at Kmax 12 and approximately
4e-13--1.5e-12 for Kmax 30--70.77. Direct source-linked blunting is therefore
the dominant state feedback.

At matched Kmax 30 and R=-0.95, cycle-12 Kshield is 0.621 MPa sqrt(m), and
zeroing it raises the frozen action by a factor 5.0. This is additional
suppression, not a resharpening mechanism.

## Signed reverse plasticity audit

The v7 transport law already has the requested parameter-free structure

```text
tau_glide = tau_applied(finite current radius) + tau_GND(signed densities).
```

The unsigned persistent-source backstress is not reinterpreted as kinematic.
At Kmax 30 and R=0.1, the minimum forward-projected glide stress remains
+0.451 GPa at cycle 12 and no reverse channel occurs. At identical Kmax and
R=-0.95, the minimum is -4.454 GPa and 49.2% of the saved-state phase/channel
samples have reverse drive. Thus the existing signed internal-stress
representation can produce reverse plasticity when the waveform supplies an
unloading reversal.

A diagnostic-only signed near-tip Orowan displacement was integrated as
`integral abs(m_alpha) b rho_mobile v_alpha exp(-x/l_blunt) dx dt`. At cycle 12:

| R | net source blunting count | radius (um) | signed displacement proxy (um) | reverse part (um) | physical returned source count | frozen H/cycle |
|---:|---:|---:|---:|---:|---:|---:|
| +0.10 | 11298.1 | 7.039 | 6.651 | 0 | 0 | 2.149e-10 |
| -0.95 | 11176.2 | 6.974 | 6.603 | 0.00492 | 5.62 | 4.160e-11 |

The signed displacement proxy corroborates the magnitude of the current
blunting state at high load. Its reverse part is only 0.074% of the net signed
displacement even at R=-0.95. Replacing topological return by this signed state
would therefore not cure positive-R passivation and would barely change the
matched negative-R radius. The audit does not justify adding a recovery
fraction or a phenomenological radius decay.

## Decision

Case 1 is rejected: no dimensional/accounting defect was found in the
source-slip-to-radius mapping.

Case 2 is not a justified blocker fix: a signed plastic-displacement state is
physically meaningful, but the measured reverse contribution is too small and
is exactly zero at R=0.1. Implementing an unvalidated projection would change
physics without solving the failure.

Case 3 is rejected in its ceiling form: the analytical cleavage ceiling is far
above all requested waiting-time rates. The interaction of physical blunting
with the low-exposure three-hit law does, however, make cleavage practically
inaccessible after the initial source burst.

The evidence selects Case 4. The current weak-T, 300 K, positive-R state is a
plastic-slip-dominated regime in which the specified cleavage first passage is
essentially inaccessible. The existing source-emission transition-state rate
cannot simply be relabeled as fracture: it describes dislocation nucleation,
does not represent bond separation, and would produce tens to hundreds of
thousands of activations in the first cycle. A surface-step crack event would
need a distinct, physically specified decohesion/tearing transition state and
its own validation. Deriving that event from the current parameters is not
unique.

No constitutive correction is implemented by this audit. Doing so would violate
the present immutable requirement that cleavage first passage remain the only
stochastic crack trigger. Consequently an ordinary-threshold first passage,
multiple crack events, a developed weak-T Paris curve, and the requested R and
frequency curves cannot be produced honestly from this parameterization under
the current mechanism contract.

## Change classification

- Physics: unchanged.
- Numerical representation: unchanged.
- Diagnostics: exact-state causal analyzer, analytical ceiling and dimensional
  audit, signed Orowan displacement proxy, CSV/JSON/PNG outputs.
- Launch tooling: unchanged.
