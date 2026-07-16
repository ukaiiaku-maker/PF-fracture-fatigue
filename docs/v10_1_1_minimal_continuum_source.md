# v10.1.1 minimal continuum source law

## Scope

This branch removes the runtime interpretation of `source_sites_per_system` as
a finite, depletable distributed-source inventory.

Distributed plasticity is represented only by the existing continuum fields:

- mobile population transported by the promoted Arrhenius Peierls law;
- retained population produced by the promoted Taylor encounter/storage law;
- Taylor release, retained recovery, mobile escape, and moving-frame advection;
- forest hardening calculated from the retained population.

No discrete distributed sources, spatial source particles, source barrier bins,
or new Kocks--Mecking fit constants are introduced.

## Tip-attached emission activity

Each crystallographic emission system has one dimensionless activity fraction
`q_s` in `[0,1]`.  This is not a source count.  It is the fraction of the
already calibrated tip-emission channel that is ready to emit again.

The firing propensity is

```text
lambda_fire,s = lambda_emit(sigma_tip,T) * h_s * w_s
```

where `h_s` is a Taylor-storage suppression factor and `w_s` is the existing
crystallographic system weight.

The source clears at

```text
k_clear = v_Peierls,tip / r_tip
```

using the current blunted tip radius.  The exact two-state evolution is

```text
dq_s/dt = k_clear*(1-q_s) - lambda_fire,s*q_s
```

so emission and clearing are sequential at atomic scales but produce a smooth
mean cycling rate over the continuum timestep.

Crack advance renews the local tip geometry without an additional fitted
length:

```text
q_s <- q_s + (1-q_s) * [1-exp(-da/r_tip)]
```

Thus a sharp tip renews its local emission geometry over a short distance,
while a blunted tip retains more history.

## Hardening/storage feedback

The near-tip retained population supplies an excess forest density above the
configured background.  Activity is reduced by

```text
h_s = 1 / [1 + sqrt(rho_excess,s/rho_c)]
```

where `rho_c` is the existing promoted Taylor correlation density.  This adds
no new hardening scale.  Storage, release, recovery, transport, and crack-tip
advection naturally create hysteresis.

## Parameter accounting

No new fitted source parameter is added.

`source_sites_per_system` remains only as the legacy emission-rate
multiplicity needed to preserve the promoted material calibration.  It is not
consumed and does not define an evolving source population.  A future
recalibration may absorb this multiplicity into the emission attempt
frequency.

## Validation gate

Run:

```bash
OUTROOT=runs/v10_1_1_source_model_gate_700K_10um_v1 \
bash scripts/run_v10_1_1_source_model_gate_700K.sh
```

The gate compares `continuum` and `finite_sites` for weakT and DBTT with the
same resolved mesh, moving-tip integration, active shielding, and no wake
shielding.

Required checks:

1. the continuum model does not collapse permanently to zero availability;
2. tip activity decreases during rapid emission and recovers through Peierls
   clearing and crack advance;
3. the retained/mobile fields remain the only distributed plastic state;
4. micro-advance equals committed geometry advance;
5. the DBTT response is not controlled by an arbitrary refresh length;
6. source activity, hardening factor, clearing rate, and effective
   multiplicity are present in the kinetic audit.
