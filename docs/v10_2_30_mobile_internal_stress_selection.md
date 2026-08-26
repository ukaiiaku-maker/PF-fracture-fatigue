# v10.2.30 reversible mobile-stress selection

## Decision

The initial unified solver uses the v10 tensor-derived signed slip-system
orientation and the existing finite-tip applied-stress magnitude for mobile
glide.  It does not add a local retained/GND stress.  This is candidate A from
the qualification contract, selected because it is the only formulation with
complete support in the qualified v10 state.

This is not a claim that a local dislocation self-stress is physically zero.
It is a fail-closed statement that v10 currently has no mechanically calibrated
operator mapping its signed retained/mobile bin populations to a local resolved
shear-stress field.

## Provenance and dimensional separation

The active v10.2.14 FEM atlas maps signed mobile/retained line content to a
mode-I shielding response in Pa sqrt(m).  Its metadata and implementation call
this a crack-tip stress-intensity operator.  It is therefore used only by
`K_shield`; it cannot be inserted as a bin-local stress in Pa without deriving
a different Green operator.

The anisotropic FEM probe maps the held two-dimensional stress tensor to signed
slip-system drives in Pa.  The sign of `_anisotropic_tau_signed_Pa` establishes
the positive Burgers orientation for each system.  Under an opening phase, a
population emitted with either Burgers sign must move away from the source; a
load reversal changes the applied drive and returns the same population toward
the source.  The reversible transport implements exactly that sign product.

The v7 `tau_gnd` closure is not imported.  Its normalization and state mapping
are not qualified for the v10 signed 2-D kernel state.  Likewise, `K_shield` is
not subtracted a second time from mobile transport.

## Candidate comparison

- A: tensor-oriented applied finite-tip drive. Units, sign, phase reversal, and
  relation to the existing v10 emission orientation are defined. Selected.
- B: A plus a local signed retained/GND stress. No v10/PF artifact currently
  supplies the required Pa-per-signed-line, system-by-bin operator. Rejected as
  underdetermined rather than assigned a fitted or v7-derived coefficient.
- Nonlocal alternative: the v10.2.14 signed kernel. It is retained for cleavage
  shielding only because its output units are Pa sqrt(m).

Frozen-state tests cover virgin, positive retained, sign-reversed retained,
mixed mobile/retained, and blunted source-slip states.  They require retained
sign reversal to reverse nonlocal shielding while leaving the selected local
mobile stress unchanged; the latter changes only when applied phase or the
explicit slip-system orientation changes.
