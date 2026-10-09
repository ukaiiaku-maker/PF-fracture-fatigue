# V5 one-void scientific closure V2 protocol

This protocol is frozen before any V2 sentinel or campaign result is generated.
It preserves every V1 result and all numerical tolerances in
`finalization_v3_schema.SCIENTIFIC_ACCEPTANCE_TOLERANCES`.

## Version separation

The following results are independent and must never overwrite one another:

- `NATURAL_BITWISE_SOURCE_REPLAY_V1`
- `NATURAL_FUTURE_PHYSICAL_REPLAY_V2`
- `STATIC_FAMILY_V1`
- `STATIC_FAMILY_V2`

V1 evidence remains immutable. A V2 pass cannot relabel a V1 failure.

## A. Child-tip constitutive handoff

The accepted analytical sharp-front model evaluates an intensity-derived field
as `K / sqrt(2*pi*r_eff)`. The current one-void continuation obtains an
area-weighted CST tensor at the child tip and sends its resolved opening stress
directly to the cleavage barrier. There is no accepted conversion from that
tensor to a stress intensity, no child-radius observation operator, and no
binding from `tip_process_state.r_tip_m` to `FrontEngine.f.r0` or
`FrontEngine.r_eff()`.

Consequently the prospective classification is:

```text
R_TIP_OWNERSHIP           = PASS
R_TIP_DISTINCT_FROM_VOID  = PASS
R_TIP_CAUSAL_LAW          = NOT_DEFINED
```

No `r_tip` or `R_void` multiplier may be introduced to change this result. A
future constitutive implementation requires a separately reviewed derivation
of the local FEM-to-sharp-front matching operator.

The signed analytical prediction for such a future implementation is frozen:
at fixed positive K and all else equal, decreasing `r_tip` must increase the
uncapped opening stress, must not increase the cleavage barrier, must not
decrease the cleavage rate, and must not delay crossing. Increasing `r_tip`
has the opposite weak inequalities. The exact stress ratios for the requested
peers are `sqrt(1/0.75)`, `1`, and `sqrt(1/1.25)`. V2 does not claim those
predictions have been implemented.

## B. Static-family V2

Before any new full matrix, retained V1 sources are reduced to a 33-row failure
atlas. Each row retains the complete geometry, mesh levels, quality-valid
levels, first failed predicate, reaction, compliance, energy, raw traction,
fixed-arc tensor, fixed-tip tensor, derivative, and geometry-match data.

A V2 family requires all of the following:

1. at least two quality-valid fine levels;
2. convergence toward one stable global reaction/compliance/energy limit;
3. final-level global accuracy under the frozen V1 limits;
4. final-level fixed-tip and, for cavity cases, fixed-arc tensor accuracy under
   the frozen tensor limit;
5. raw traction, equilibrium, topology, ownership, and geometry predicates;
6. energy/compliance derivative agreement and perturbation convergence for the
   derivative families.

All V1 measurements remain present even if V2 uses an additional prospective
operator. The unqualified recovery operator is not a production kinetic input.
A complete 33-family V2 matrix is permitted only after representative sentinel
families meet the frozen V2 contract.

## C. Controlled-history V3

The twelve-case registry is unchanged. The six previously passing histories
must remain passing. The six V2 failures receive new executions and distinct
V3 classifications:

- `accommodation_limited`: canonical spelling is
  `STABLE_SUBGRID_VOID_WITH_PLASTIC_ACCOMMODATION_MINIMUM_ALL_INTERVALS`.
- `diffusion_limited`: vacancy transport must be the strict minimum during
  every declared interval with a predeclared relative separation margin of
  `0.05` from the next-slowest channel.
- `delayed_downstream`: a qualified source must remain dormant with unchanged
  hazard, threshold, RNG, graph, and topology, then the same clock must cross
  after tensile reload.
- `downstream_zero_drive`: source qualification precedes the zero-opening and
  zero-effective-rate observation.
- `local_remesh_refinement`: the same physical case retains a coarse
  unqualified result and must produce a separately qualified refined result.
- `negative_offset`: the fixed-laboratory-crack positive/negative pair must
  report exact mirrored geometry and the signed tensor transformation. Any
  source-quality asymmetry is a failure unless a crystallographic distinction
  is derived before execution.

A source-unqualified terminal is a valid observed outcome but does not satisfy
the named downstream mechanism.

## D. Natural future-physical replay V2

V2 requires exact equality of case/seed identity, RNG state, threshold
identity, candidate and event identities/order, topology and mesh connectivity,
active-front ownership, phase, categorical outcome, and checkpoint lineage.

Continuous displacement, stress, reaction, energy, hazards, radius/inventory,
and rate diagnostics are compared numerically. Every limit is recorded per
quantity from its physical scale, normalized free residual, a recorded or
independently estimated stiffness condition number, and distance to the nearest
event threshold/selection bifurcation. The comparison fails closed if those
inputs are unavailable. A pass additionally requires one subsequent real
crossing from both states with identical selected event and accepted topology.

V2 is never described as bitwise equality.

## Evidence and terminal policy

The complete schema and validator must be native to the final implementation
before physical execution. No compatibility projection is allowed. Only small
canonical tables, manifests, hashes, decisions, and reconstruction tools enter
Git; large raw evidence belongs in durable workflow artifact storage.

The terminal classification is either
`V5_ONE_VOID_SCIENTIFIC_CLOSURE_QUALIFIED` or
`V5_ONE_VOID_FINAL_SCIENTIFIC_CLOSURE_COMPLETE_BUT_BLOCKED`. No RC branch or
fatigue coupling is allowed unless every mandatory one-void gate passes.
`r_tip != R_void` remains invariant.
