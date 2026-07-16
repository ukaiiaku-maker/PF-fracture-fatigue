# v10.1.3 local emission back stress

## Purpose

v10.1.2 bounded the recycling throughput of each aggregate crystallographic tip-emission channel. It did not directly reduce the emission driving stress as the forward dislocation cloud accumulated. v10.1.3 adds that missing physical feedback without introducing a fitted source-count, back-stress coefficient, or density scale.

## Distributed plasticity

The process zone remains a continuum Peierls--Taylor model. Mobile and retained populations, accumulated slip, trapping/release, recovery, escape, shielding, and moving-frame advection are unchanged. There is no discrete distributed-source inventory.

## Local density

For each crystallographic system, the explicitly evolved mobile-plus-retained population ahead of the crack tip is averaged with

```text
w(x) = exp(-x / Lback)
Lback = max(current blunted tip radius, configured blunting length, MPZ grid spacing)
```

The weighted count is divided by the existing MPZ strip area `dx * blunting_length`. The numerical forest-density floor is excluded, so the undeformed tip begins with zero emission back stress.

## Back stress and effective emission stress

The local Taylor shear back stress is

```text
tau_back,s = G b sqrt(rho_tip,s)
```

and the equivalent stress entering the promoted emission barrier is

```text
sigma_back,s = tau_back,s / m_taylor
sigma_emit_eff,s = max(sigma_tip - sigma_back,s, 0)
```

where `m_taylor` is the existing `taylor_stress_fraction`. No additional coefficient is fitted.

The aggregate tip-channel hazard is

```text
Lambda_s = source_sites_per_system * lambda_emit(sigma_emit_eff,s, T) * w_s
```

and the activity equation remains

```text
dq_s/dt = k_clear (1 - q_s) - Lambda_s q_s
```

with `k_clear = v_Peierls / r_tip`. The exact activity integration and v10.1.2 throughput bound remain active.

## Distinction from crack shielding

The emission back stress and crack shielding are not the same calculation:

- `sigma_back` suppresses additional dislocation nucleation at the tip.
- `K_active` is the elastic projection of the active dislocation population onto the crack driving force.

Both arise from the same evolving microstructure but enter different kinetic channels.

## Required diagnostics

Each continuum audit record includes:

- `tip_source_local_density_m2`
- `tip_source_backstress_shear_Pa`
- `tip_source_backstress_equivalent_Pa`
- `tip_source_effective_emission_stress_Pa`
- `tip_source_effective_emission_stress_min_Pa`

The 2-D gate rejects missing/nonfinite diagnostics, no-advance results, and initiation toughness above 100 MPa sqrt(m).

## First gate

Run only the weakT continuum case first. Acceptance requires bounded emission, finite growing back stress, nonzero cleavage action, and two 5 um geometry checkpoints before 250 outer steps.
