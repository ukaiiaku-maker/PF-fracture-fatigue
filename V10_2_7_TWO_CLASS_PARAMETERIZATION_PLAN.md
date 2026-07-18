# v10.2.7 exact reduced model and two-class parameterization plan

## Scope

This stage makes the reduced calibration path mechanically consistent with the
v10.2.6 state-resolved signed-dislocation formulation and prepares a new campaign
for two material classes:

1. **DBTT**: low-temperature brittle response and high-temperature plastic
   shielding/R-curve toughening.
2. **FCC-like weakT**: weak temperature dependence with deliberately weak but
   nonzero R-curve toughening.

The existing ceramic-like parameterization is frozen. It is evaluated only as a
regression control and is not included in the search variables.

## Exact shared physics

Every reduced candidate uses the production
`StateResolvedSignedBurgersTipEngine` and reconstructs all serialized 2-D
configuration fields:

- front and local cohesive-strength parameters, including `sigma_cap`;
- spatial active and wake MPZ grids;
- positive/negative mobile, retained, and slip populations;
- finite source capacity and geometry-controlled refresh;
- Taylor back stress, transport, recovery, escape, and moving-frame transfer;
- the authorized state-resolved signed shielding-kernel family;
- the same uncapped shielding law used by monotonic and fatigue paths.

No old scalar population, constant `(1,1)` shielding projection, shielding cap,
or fitted attenuation is permitted.

## Remaining reduced mechanical closure

The shielding operator is supplied by v10.2.6, but a reduced model also needs the
signed resolved shear that selects Burgers sign and emission magnitude. v10.2.7
therefore requires a second candidate-independent 2-D artifact:

```text
signed_drive_factor_s(q) = tau_signed_s(q) / sigma_local(q)
```

where the mechanical state is

```text
q = (r_eff/r0, sigma_local/sigma_cap, crack_extension).
```

The signed drive family must contain exactly the same state IDs and coordinates as
the shielding-kernel family, use reliable 2-D tensor probes, and remain within its
validated envelope. The reduced path sets

```text
drive_factor_s = abs(signed_drive_factor_s)
tau_signed_s   = signed_drive_factor_s * sigma_local
```

before calling the production engine. Extrapolation is a hard failure.

## Artifact gates before optimization

Both of the following JSON artifacts must explicitly contain
`production_parameterization_allowed=true`:

- the v10.2.6 state-resolved signed shielding-kernel family;
- the v10.2.7 state-resolved signed drive family.

Authorization is permitted only after:

1. interaction-integral contour and mesh convergence;
2. positive/negative multi-amplitude perturbation linearity;
3. source-normalization review;
4. complete state-envelope coverage;
5. cross-candidate checks showing that the mechanical families are sufficiently
   candidate independent;
6. one exact active-state monotonic replay and one cyclic replay.

A complete v10.2.6 engine-configuration JSON from an equilibrated 2-D trace is also
required. Candidate evaluation is blocked if any front, MPZ, kinetic,
anisotropic, campaign, or transport field is missing.

## Reduced R-curve calculation

The calibration observable is no longer only first passage. Each candidate is
loaded monotonically through repeated production-engine advances to a default
50 micrometre extension. The reduced calculation records

- initiation toughness;
- event-by-event R-curve;
- toughness at the target extension;
- mobile, retained, signed-line, source, blunting, and shielding histories;
- kernel and signed-drive state coordinates and interpolation weights.

The pilot uses four temperatures:

```text
300, 700, 900, 1200 K
```

and the modes

```text
full
plasticity_off
shielding_off
backstress_off
```

Ablations are evaluated at the 300 and 1200 K endpoints; the full model is
evaluated at all four temperatures.

## DBTT objective

The pilot strict gate requires:

- low-temperature initiation toughness in a physically useful range;
- final high-/low-temperature toughness ratio at least 1.5;
- low-temperature R-rise no more than 15%;
- high-temperature R-rise at least 20%;
- `plasticity_off` endpoint ratio no more than 1.25;
- at least half of the temperature rise removed by `shielding_off`;
- at least 30% of the high-temperature R-rise removed by `shielding_off`;
- positive R-rise with back stress removed, so back stress regulates rather than
  creates the transition;
- monotonic temperature ordering at at least 90% of adjacent temperature pairs;
- no state-family extrapolation or source-capacity violation.

The three historical systems are retained only as barrier-shape anchors:

- `A0002333`: legacy large DBTT rise;
- `A0003837`: legacy shielding sensitivity;
- `A0002277`: legacy non-cap-limited state.

Their old source counts are not reused. `source_sites_per_system` is sampled only
inside the mechanically derived common source-capacity interval supplied by the
signed kernel family.

## FCC-like weakT objective

The weakT class is intentionally not perfectly brittle or perfectly flat. The
strict pilot gate requires:

- maximum/minimum initiation toughness ratio no more than 1.20 across temperature;
- maximum/minimum final toughness ratio no more than 1.20;
- R-rise between 5% and 25% at every temperature;
- absolute R-rise at least 0.5 MPa sqrt(m);
- at least 30% of the mean R-rise removed by `plasticity_off`;
- at least 15% of the mean R-rise removed by `shielding_off`;
- no state-family extrapolation or source-capacity violation.

This produces an FCC-like response with weak temperature sensitivity and modest,
mechanistically plastic R-curve toughening rather than the previous nearly flat
no-toughening response.

## Ceramic-like control

The ceramic manifest is not sampled or modified. Before either search begins, it
must satisfy a frozen regression gate under the new shared physics:

- maximum/minimum final toughness ratio no more than 1.20;
- maximum absolute R-rise no more than 5%;
- complete calculations at all four temperatures.

Failure of this control stops the campaign because it would indicate that the new
mechanical closure changed a class that was meant to remain fixed.

## Search strategy

### Stage A: bounded Sobol pilot

Run 128 DBTT and 128 weakT candidates over a 50 micrometre reduced R-curve. Search
only inside the mechanics-derived source interval. Use the old DBTT systems and
current FCC-like weakT manifest as barrier-shape anchors.

Promote the best 12 candidates per class, but treat promotion only as selection for
further testing.

### Stage B: local cross-entropy refinement

For each class, fit a proposal distribution to the best 10-15% of Stage A and run
three to five generations of 256-512 candidates. Keep source capacity bounded by
the mechanical artifact and use truncated distributions for every material
parameter. Cleavage parameters receive narrower proposal widths than emission and
transport parameters so the search cannot satisfy the objective by a cleavage-only
shift.

### Stage C: rate robustness

Evaluate promoted candidates at three loading rates:

```text
0.1x, 1x, 10x reference Kdot
```

DBTT ordering and weakT near-flatness must persist. Candidates whose classification
exists only at one numerical loading rate are rejected.

### Stage D: full 2-D validation

For the top four candidates per class, run cap-free state-resolved signed 2-D
calculations:

- 300 and 1200 K, full and all three ablations;
- 100 micrometre R-curves at 300, 700, 900, and 1200 K;
- exact reduced/2-D state replay for one active endpoint per class;
- comparison of initiation toughness, final toughness, R-rise, shielding history,
  signed line content, and source consumption.

A reduced candidate is accepted only when the 2-D result preserves its class and
mechanism.

### Stage E: production curves

After endpoint validation, run 500 micrometre curves and the fatigue campaign. The
same state-resolved signed engine and artifacts must be used for monotonic fracture
and fatigue.

## Initial pilot command

The campaign runner requires explicit paths because it must not guess the current
FCC-like or ceramic manifests:

```bash
KERNEL_FAMILY=/path/to/authorized_v10_2_6_kernel_family.json \
DRIVE_FAMILY=/path/to/authorized_v10_2_7_drive_family.json \
ENGINE_TEMPLATE=/path/to/v10_2_6_2d_exact_engine_config.json \
WEAKT_ANCHOR=/path/to/current_FCC_like_weakT_manifest.csv \
CERAMIC_REFERENCE=/path/to/current_ceramic_like_manifest.csv \
OUTROOT=runs/v10_2_7_two_class_parameterization_pilot_v1 \
SAMPLES_DBTT=128 \
SAMPLES_WEAKT=128 \
WORKERS=2 \
TARGET_EXT_UM=50 \
bash scripts/run_v10_2_7_two_class_parameterization.sh
```

The runner refuses an existing output directory, unauthorized artifacts, a failed
ceramic control, source-capacity violations, and any state-envelope extrapolation.
