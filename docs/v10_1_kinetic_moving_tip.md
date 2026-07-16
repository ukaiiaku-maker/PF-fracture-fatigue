# v10.1 kinetic moving-tip cell

## Purpose

v10.1 replaces the instantaneous process-zone translation associated with one
5 micrometre first-passage renewal.  The 5 micrometre length remains the outer
anisotropic-FEM field checkpoint and preserves the promoted calibration, but it
is no longer treated as a physical jump.

For cleavage action increment `dB`, the local moving-tip cell advances

```text
da = L_checkpoint * dB
v_c = L_checkpoint * lambda_c
```

and evolves source exposure, emission, Peierls transport, Taylor
retention/release, recovery, blunting, and shielding during the same physical
time interval.  The moving-frame MPZ is translated by each `da`; the checkpoint
commit does not translate it again.

The mean velocity is identical to the previous renewal interpretation: one
checkpoint length per mean first-passage time.  `kinetic_packet_length_m`
controls packet-rate and fluctuation diagnostics only.  It is not the numerical
geometry increment.

## Coupling algorithm

Each external FEM/J interval is integrated with bounded Strang-split substeps:

1. evolve half a plastic reaction/transport step at the current tip field;
2. recompute active shielding, blunting, tip stress, and cleavage rate;
3. integrate cleavage action and moving-frame distance;
4. continuously expose fresh source capacity and translate only crossed state
   into the wake;
5. evolve the second half plastic step at the updated state;
6. stop after at most one outer geometry checkpoint.

Substeps are limited independently by cleavage action and moving-frame distance.
The outer anisotropic FEM, directional J, crystal competition, multifront
lineage, remeshing, and coalescence implementation are unchanged.

## Current shielding scope

v10.1 enables instantaneous mobile as well as retained active-zone shielding.
The active contribution is signed internally, so anti-shielding is not silently
clipped.  The current kernel is still the reduced MPZ line-defect projection;
full tensorial `KI/KII` weight functions and FEM stress-tensor sampling remain a
subsequent anisotropic refinement.  The core moving-frame kinetic coupling is
validated before adding that refinement.

Wake shielding is disabled in the first causal gate.  Wake state is retained
for conservation and future fatigue/friction/bridging extensions but is not the
mechanism under test.

## First causal gate

```bash
OUTROOT=runs/v10_1_forward_zone_ablation_700K_10um_v1 \
bash scripts/run_v10_1_forward_zone_ablation_700K.sh
```

The runner executes otherwise identical cases:

- `full`: tip plasticity on, active shielding on;
- `active_shield_off`: tip plasticity on, active shielding off;
- `plasticity_off`: emission/transport off, active shielding off.

Wake shielding is off in all cases.  Every case writes:

- `v10_1_driver_modes.json`;
- `kinetic_tip_cell_audit_v101.json`;
- the inherited step/front/R-curve outputs.

## Required acceptance checks

1. `kinetic_micro_advance_total_m` approaches each 5 micrometre checkpoint
   continuously rather than appearing only at commit.
2. `mpz.advance_total_m` equals micro advance; it must not include an additional
   5 micrometre translation at commit.
3. source availability changes during partial checkpoint progress.
4. at most one outer geometry checkpoint occurs per FEM/J state.
5. `full` and `active_shield_off` share plastic activity but differ in effective
   crack drive when active shielding is significant.
6. `plasticity_off` has zero emitted/trapped/released populations.
7. ceramic, weakT, and DBTT retain their promoted initiation ordering.
8. no wake contribution is present in the causal comparison.

## Limitations of the first v10.1 implementation

- The local MPZ stress profile remains the reduced analytic tip profile rather
  than a system-resolved interpolation of the FEM tensor field.
- Directional cleavage competition remains in the outer solver; the local cell
  currently receives the selected scalar `K` state.
- The cyclic implementation uses phase-integrated hazards and an
  emission-weighted stress tau-leap, not atom-by-atom cycling.
- A stochastic packet diffusion term is reported through the packet variance but
  is not sampled in the deterministic production path.

These limitations are explicit.  They do not reintroduce a physical 5
micrometre jump or a separate cohesive-opening clock.
