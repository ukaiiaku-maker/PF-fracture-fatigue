# v10.2.3 shared-core uncapped shielding

## Purpose

Move the removal of the campaign-era hard shielding cap from the fatigue-only
wrapper into the shared `CampaignCalibratedTipEngine` used by both monotonic
temperature-dependent fracture calculations and cyclic fatigue calculations.

## Shared constitutive law

The shared engine now uses

```text
Kshield_effective = Kshield_raw
```

where `Kshield_raw` is the signed elastic superposition from the evolving active
dislocation population.  The historical manifest field
`max_K_shield_MPa_sqrt_m` is retained only as an audit reference.  It does not
enter tip stress, cleavage hazard, emission kinetics, source evolution, monotonic
fracture loading, or cyclic fatigue loading.

The unilateral opening law remains

```text
Ktip = max(Kapplied - Kshield, 0)
```

No fitted replacement saturation parameter is introduced.  Population growth is
limited by finite source capacity, Taylor backstress, transport and active-zone
escape, retained-population recovery, and moving-frame transfer into the wake.

## Architectural correction

v10.2.2 temporarily monkey-patched the campaign engine only inside the
fixed-DeltaK fatigue entry point.  v10.2.3 removes that runner-specific physics:

- `CampaignCalibratedTipEngine._active_shielding_signed` is permanently uncapped;
- the v10.2.2 context manager is now audit-only and does not patch the shielding law;
- the protected monotonic campaign entry point reports the cap as disabled;
- both monotonic and fatigue output identify the legacy manifest value as a
  diagnostic reference only.

## Required checks

```bash
python -m compileall -q arrhenius_fracture scripts
python -m pytest -q
```

The shared-core tests require a raw shielding value larger than the historical
manifest reference to pass through unchanged without entering the fatigue runner.
They also require the fatigue audit context to leave the shared constitutive
method untouched.

For a monotonic temperature-dependent fracture smoke run, inspect:

```text
v10_1_driver_modes.json
v10_1_1_source_model.json
kinetic_tip_cell_audit_v101.json
```

and require:

```text
manifest_K_shield_cap_enabled = false
legacy_manifest_K_shield_cap_reference_only = true
shared_monotonic_and_fatigue_core = true
campaign_calibration.shielding_cap_from_manifest = false
campaign_calibration.shielding_saturation = population_dynamics_only
```

The fatigue v10.2.2 audit remains valid and should continue to require raw and
effective shielding to agree within relative `1e-12`.
