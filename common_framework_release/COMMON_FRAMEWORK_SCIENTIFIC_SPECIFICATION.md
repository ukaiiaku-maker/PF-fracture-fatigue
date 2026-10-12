# Common fracture framework: scientific specification

Status: prospective specification, not manuscript prose and not a claim that the
new production campaign has run.

## 1. Scientific question and causal sequence

The study asks how initiation and resistance during crack advance emerge when
thermally activated opening competes with crack-tip plastic accommodation while
the local microstructure evolves and is inherited by a moving or branching
front. The intended causal sequence is

activation landscapes and physical loading -> opening/plasticity competition ->
evolving tip and wake state -> initiation and resistance -> branch and cavity
pathway selection.

The four constructed parameter rows are fixed examples, not named-material
calibrations. Existing Ag observations motivate different timing of opening and
dislocation activity. No W experiment or unavailable mechanical history is
implied.

## 2. Common physical definitions

For a positive elementary mechanism rate,

    lambda_i = M_i nu_i exp[-Delta G_i(q_i,T,z)/(k_B T)].                 (1)

`M_i` is physical multiplicity, `nu_i` attempt frequency, `q_i` the
mechanism-specific conjugate drive, `T` temperature, and `z` evolving state.
Elementary, net, mean-population, completion and sampled-hazard rates are
distinct quantities. Signed net transport or return rates are never placed
inside a logarithm or treated as nonnegative event hazards.

For the set `E` of sampled channels only,

    S_next(t) = exp[- integral_(t0)^t sum_(i in E) Lambda_i(s) ds].       (2)

This is survival against the next sampled event, not necessarily survival
against fracture. Mean source-population, exchange and transport evolution
change the conditional hazards but are not additional sampled arrivals.
Sequential prerequisites are integrated as state evolution and completion;
their rates are not added as independent competing completions.

At a frozen state,

    lambda_e/lambda_c = (M_e nu_e)/(M_c nu_c)
      exp[-(Delta G_e-Delta G_c)/(k_B T)].                               (3)

Equation (3) is diagnostic. It does not replace integration of the coupled
trajectory, state retention or mechanically effective shielding.

At fixed local stress and state,

    partial_T ln lambda_i = partial_T ln(M_i nu_i)
      + [Delta G_i - T partial_T Delta G_i]/(k_B T^2),                   (4)

    partial_sigma ln lambda_i = partial_sigma ln(M_i nu_i)
      + V_i/(k_B T),   V_i = -partial_sigma Delta G_i.                   (5)

These derivatives include prefactors, multiplicity, renewal mappings and any
explicit temperature dependence of the stress scale. Activation enthalpy and
entropy remain separate.

For a monotone external controller `L` with speed `r>0`, a fixed initial state
and a fixed stochastic mark `Xi`,

    H_c(L*,T,r) = integral_(L0)^(L*) Lambda_c[L,T,z(L;T,r)]/r dL = Xi.    (6)

On a smooth event branch,

    dL*/dtheta = -(partial_theta H_c)/(partial_(L*) H_c).                (7)

At fixed state with no other rate dependence,

    dL*/dln(r) = Xi r/Lambda_c(L*) > 0.                                 (8)

Equations (6)-(8) do not cross event-order discontinuities. In the coupled
model, changing rate also changes state formation and retention, so the sign of
the full rate effect is not fixed by (8).

## 3. State and feedback ownership

- Source geography (`tip_surface`, `bulk_uniform`, `tip_plus_bulk`) is
  independent of source lifecycle (`persistent_reusable`, `finite_inventory`,
  `current_source`).
- Persistent sites supply a state-dependent reusable multiplicity. Finite sites
  consume and, where declared, renew inventory. Current-source state belongs to
  the evolving topology provider.
- Source Taylor backstress reduces the emission drive in the source equation.
  It is not subtracted directly from the cleavage/opening coordinate.
- Retained signed line content affects cleavage only through the native
  mechanical shielding kernel. Mobile and wake content, slip, escape, return
  and annihilation retain their signed state and physical eligibility.
- Blunting changes the physical tip/process geometry and therefore mechanics
  and multiplicity. It is not added as an independent scalar toughness.
- Partial hazard progress translates state in the moving coordinate but changes
  geometry only at a committed event. Wake/source/state mappings occur exactly
  once in an accepted transaction.

## 4. Crack length and resistance

`a0`, projected extension `Delta a`, path arclength and remaining solid ligament
are separate coordinates. Independent specimens at the same current length can
carry different inherited states. Primary resistance evidence therefore comes
from continuously evolved native histories under a declared loading protocol.

The observer stores its native quantity (`J`, `G`, or `K_J`), support, front,
state stage, physical time and geometry association. A pre-event `K_J` is not
assigned to a committed geometry unless a post-event equilibrium was actually
computed. Instantaneous geometry jumps are bracketed rather than interpolated.
Unload/reload compliance is one possible measurement protocol, not a universal
definition of resistance; conversely, connecting burst values does not create a
protocol-independent material curve.

## 5. Cohesive--Arrhenius--Kocks correspondence

For a reversible cohesive potential per area `phi(delta)` at fixed temperature
and internal state,

    t_coh(delta) = d phi/d delta,                                        (9)

    F(delta;t_n) = A_* [phi(delta) - t_n delta],                         (10)

    Delta G_c = A_* integral_(delta_m)^(delta_s)
                [t_coh(delta)-t_n] d delta,                              (11)

    V_c = A_* (delta_s-delta_m).                                         (12)

`A_*` is a physical activation area, not automatically a numerical cohesive
element area. Equations (9)-(12) are a prescribed-local-traction construction.
Under finite-body displacement control, the surrounding elastic energy and
compliance belong in the activation problem. An irreversible dissipative
softening path is not automatically a reversible free-energy potential.

For a triangular bilinear envelope with peak `t_c`, terminal separation
`delta_f`, and work `Gamma_0=t_c delta_f/2`, direct integration gives

    Delta G_c = A_* Gamma_0 (1-t_n/t_c)^2,                               (13)

    V_c = 2 A_* Gamma_0 (1-t_n/t_c)/t_c.                                 (14)

This is the `p=1, q=2` member of

    Delta G = Delta G_0 [1-(t_n/t_c)^p]^q.                               (15)

For a smooth nondegenerate traction maximum,
`t_coh=t_c-k(delta-delta_0)^2/2+...`, the local traction-controlled barrier
scales as `(t_c-t_n)^(3/2)`. This is a local one-coordinate asymptote, not a
universal spatial-nucleus exponent.

Kocks describes a barrier shape inside an Arrhenius rate; it is not an
alternative to Arrhenius kinetics. A conventional rate-independent cohesive
calculation makes a different evolution choice. A thermally activated use also
requires a rate, activation support and physical time/loading ensemble.

`arrhenius_fracture.cohesive_landscape` implements matched-peak-and-work
bilinear, exponential and specified polynomial potentials, their native
barriers, and activation volumes. It is a constitutive analysis interface, not
a substitute fracture trajectory. The current FEM/CZM penalty/degradation
interface is recorded as such and is not falsely called one of these reversible
potentials. The positive EXP-floor landscape remains a distinct selected
surface and is not fitted to a barrier-vanishing cohesive law.

## 6. Cavity candidate contract and unresolved physical choice

A local differential `J` and a complete-ligament finite operation are legitimate
but different observables. A finite operation is normalized by newly separated
solid area only; cavity free span is never charged as fractured solid. Kinetic
activation work and macroscopic separation work are not charged twice.

Two supported interpretations remain physically distinct:

1. `local_incremental_front`: resolve ligament approach/connection through
   local increments with fixed physical activation area and site multiplicity;
2. `collective_finite_ligament`: one coherent finite-ligament activation with an
   explicitly justified collective activation support and multiplicity one.

The existing equations do not uniquely select between these interpretations.
P7 production is gated on one author decision; no multiplier, distance rule or
candidate deletion is used to force agreement.

## 7. Prospective hypotheses and required controls

| Hypothesis | Primary observable | Control | Falsifying/limiting outcome |
|---|---|---|---|
| Opening softening competes with faster protective accommodation | first-event load/time, opening and emission/transport rates, retained signed state | matched opening-only arm at same physical rate and mark | response persists without state formation or reverses under refined coupled integration |
| A peak can arise from changing competition rather than a nonmonotone opening barrier | temperature-local slopes of native barriers/rates and coupled initiation | frozen-state derivative plus evolved path | peak tracks only a constitutive clamp or invalid raw scale |
| Rate shifts reflect exposure and state formation | transition location, `dL*/dln r`, state at event | 0.01/1/100 rates plus 0.1/10 local refinement | effect changes with numerical `dt` at fixed physical rate |
| Resistance depends on inherited state, not length alone | stage-valid `J/K_J`, reaction, extension, signed tip/wake state | independently prepared `a0`, evolved and reset/re-equilibrated diagnostics | histories collapse after numerical convergence or reset is not mechanically equilibrated |
| Branching is a physical pathway-selection event | spontaneous birth, daughter continuation, fields and clocks | no-bifurcation single-front control with same ancestry | only restored/preinserted fronts grow |
| Cavities alter pathway selection through mechanics and kinetics | same-operation drive, candidate barrier/rate, geometry and state | no-cavity, small/distant cavity and matched ancestry | difference disappears under comparable candidate mapping or is solely an observer switch |
| PF and FEM/CZM share a mechanically consistent reference | reaction, stored energy, stress and common-domain `J` at common loads | identical input plus production mesh/support sequences | nonconvergent unexplained scale factor remains |
| Cohesive shape changes barrier/activation-volume evolution at fixed peak and work | native `Delta G(t)` and `V(t)` | bilinear/exponential/polynomial matched peak/work | differences are caused by unmatched work or element area |

## 8. Domain and inference limits

The selected DBTT row has a raw emission stress scale that approaches zero at
1000 K and becomes negative above approximately 1001 K. The legacy 1 Pa clamp
is a numerical bound, not material physics. The prospective matrix retains all
nominal cells for accounting but marks 1000 K as requiring boundary
qualification and 1050--1300 K as blocked pending a physically justified law or
domain decision. No new DBTT or upper-branch claim may be produced by that
clamp.

All primary PF results for the eventual study are fresh trajectories from the
frozen framework. Old calculations remain immutable regression/reference
evidence and are never pooled based on curve appearance.
