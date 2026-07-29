# v10.2.30 room-temperature fatigue performance repair

This patch keeps the v10.2.30 fatigue physics unchanged and repairs the numerical
cycle-block search that made active cases appear stuck.

## Scope

The current fatigue campaign is restricted to 300 K and the four canonical paper
parameterizations:

- `v913_paper_peak01_0242980_persistent_sites`
- `v913_paper_dbtt01_0202500_persistent_sites`
- `v913_paper_weakT01_0129902_persistent_sites`
- `v913_paper_ceramic01_0077080_persistent_sites`

The monotonic parameter rows, persistent-site closure, phase-resolved cleavage
hazard, stochastic first-passage trigger, anisotropic direction competition, and
hazard-energy event-length gate are unchanged.

## Performance repair

The old nonlinear selector first trial-integrated the complete cycle horizon. A
`1e9`-cycle request could therefore enter a large adaptive coupled-hazard recursion
before the first output row was written. The repaired selector:

1. starts from the existing tangent block estimate;
2. expands the exact state-coupled trial block geometrically;
3. stops when an existing physical increment target is first exceeded;
4. refines the bracket in log-cycle space;
5. uses the same exact trial commit to choose the accepted block.

No cycle cap, fatigue law, Paris coefficient, toughness floor, or new constitutive
parameter is introduced. Search controls alter only the number of private trial
evaluations, not the physical acceptance targets.

Trial-engine cloning also copies NumPy generator state directly into an independent
bit generator. This preserves the random sequence while avoiding repeated
`SeedSequence` reconstruction during nested private trials.

## Canonical runner

Activate `arrhenius-sharp-front-v10`, set the four matching 300 K monotonic
reference roots, and run:

```bash
bash scripts/run_v10_2_30_300K_four_class_fatigue.sh
```

The wrapper rejects non-300 K runs and delegates to the existing four-class
qualification gate.
