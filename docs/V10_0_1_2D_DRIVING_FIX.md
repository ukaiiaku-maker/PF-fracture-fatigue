# v10.0.1 first 2-D driving-force correction

## Observed v10.0 failure

The first 700 K three-class gate reported a small positive `KJ` near step 25,
then `KJ=0`, `sigma_tip=0`, `B=0`, and no crack extension for thousands of
continued displacement increments.

## Root cause

The published v10.0 repository omitted `arrhenius_fracture.crack_backend` and
`arrhenius_fracture.coalescence`. A clean CI environment therefore failed at
import. The user's existing conda environment also contained the older editable
`arrhenius-fem-czm` distribution under the same `arrhenius_fracture` namespace,
so the local run silently imported missing modules from the older checkout.

That mixed run additionally used the inherited full-field bulk plasticity
update. Its kinetic parameters and carrier state were not mapped to the promoted
ceramic, weakT, or DBTT material manifest. The result was not a valid v10
calculation.

## Dependency-closed audit

After vendoring a sharp-wake-only backend and geometry-only coalescence, a clean
anisotropic 12-step audit was run at 700 K with the surrounding FEM elastic and
the unified MPZ as the active plastic state.

Both directional-J conventions produced the same sequence:

```text
KJ [MPa sqrt(m)] = 2.876, 5.752, 8.629, 11.505, 14.381, 18.214
sigma_tip [GPa]  = 1.15,  2.29,  3.44,  4.59,  5.74,  7.27
```

The first physical crack event occurred at `KJ=14.381 MPa sqrt(m)`. The second
5 um renewal reached the requested 10 um extension on the next equilibrium
state.

Because `root_signed` and the one-front `abs_forward` diagnostic were identical,
the directional-J sign convention was not responsible for the v10.0 collapse.
The production path remains `root_signed` so anisotropic path selection and
branching retain signed configurational work.

## v10.0.1 policy

- Supported 2-D bulk mode: `tip_only`.
- Supported production J mode: `root_signed`.
- `abs_forward` remains a one-front diagnostic only.
- The inherited `full_field` mode is blocked until it uses the promoted material
  manifest and passes an independent constitutive and energy audit.
- Every 2-D case writes `v10_0_1_driver_modes.json`.
- Tests reject production modules imported from outside the current checkout.
