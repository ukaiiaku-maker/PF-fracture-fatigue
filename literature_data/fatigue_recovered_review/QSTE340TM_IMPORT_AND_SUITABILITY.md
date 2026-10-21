# QSTE340TM import and calibration suitability

Date: 20 October 2026

## Outcome

The recovered package is imported intact and passes `python verify_import.py`.
The verifier checked 11 source files, 7,732 paired source-cell records, 23,207
cell/conversion assertions, all expected table counts, and the 52-file supplied
manifest. The raw RAR and XLSX files were not edited.

The 760 constant-amplitude points are ready for source-bound curve-shape
characterization. They are **not yet sufficient for an absolute alternative-G*
kinetic calibration or a stateful overload simulation**. No barrier fit and no
native simulation were run, because doing so would require assigning missing
temperature, frequency, waveform, CT dimensions/load history, and steel-specific
plasticity/local-stress inputs.

## Primary-method resolution

The primary paper and author dataset resolve the following:

- QSTE340TM automobile steel sheet, 2.5 mm thick;
- compact-tension specimens designed to ASTM E647;
- crack length and closure load measured by compliance with a crack-mouth clip
  gauge;
- E = 194 GPa, yield strength = 365 MPa, and UTS = 482 MPa;
- four load-ratio groups and constant-amplitude/single-overload observations.

The exact test temperature, loading frequency, waveform, dimensioned CT geometry,
force schedule, formal OLR definition, and overload timing/pulse sequence remain
unresolved. The full article and its dimensioned specimen figure were not present
locally. The publisher exposed metadata and indexed method snippets, but not the
paywalled full text through its unauthenticated text-mining endpoint. These fields
therefore remain null; no generic room-temperature value or ASTM-default geometry
has been substituted. The machine-readable record is
`audit/qste340tm_methods_resolution_2026-10-20.json`.

Primary sources:

- Lu, Yang and Chen, *Engineering Fracture Mechanics* 212 (2019) 81–94,
  <https://doi.org/10.1016/j.engfracmech.2019.03.029>
- Author-deposited data, <https://doi.org/10.17632/8pcx2mgfd4.1>

## Observation-only characterization

`scripts/analyze_qste340tm_import.py` performs no sorting, smoothing, barrier
fitting, or simulation. In source order, it reports whole-curve and contiguous
quartile log-log regressions only to expose the observed slope and curvature:

| R | points | whole-curve slope | R-squared | contiguous-quartile slope span |
|---:|---:|---:|---:|---:|
| 0.1 | 248 | 3.709 | 0.9924 | 3.115–4.621 |
| 0.3 | 219 | 3.378 | 0.9962 | 2.894–3.796 |
| 0.5 | 171 | 3.762 | 0.9980 | 3.243–4.082 |
| 0.7 | 122 | 2.920 | 0.9953 | 2.178–3.254 |

The differing slopes and within-curve spans are direct evidence that a fixed
`m = 4` target is inappropriate. These regressions are descriptive, not a Paris-law
replacement and not fitted alternative-G* parameters. Exact values, uncertainty,
rate ranges, and source-order bounds are in
`audit/qste340tm_constant_amplitude_characterization.csv`.

## Validation units

The fitting/validation unit must be a complete protocol block, not neighboring
rows. A defensible first design is leave-one-R-out rotation over the four complete
constant-amplitude curves. The eight `(R, OLR)` overload histories then remain
paired history/mechanism validation blocks, with no barrier refit per history.

The rate, a(N), and closure tables are correlated views of the same experiments.
They are not three independent likelihood contributions. Block identities and
counts are recorded in `audit/qste340tm_protocol_blocks.csv`. The primary Fig. 3,
4, and 5 views remain authoritative; later duplicate comparisons and the 14 extra
Fig. 15 points remain quarantined pending version reconciliation.

## Gate for the existing model

The existing analytical cycle-integrated hazard and state-informed implementation
remain the intended consumers. Execution is gated, not replaced:

1. Resolve or source-bind test temperature (or a documented nuisance interval),
   frequency, and waveform for the Arrhenius cycle integral.
2. Resolve the dimensioned CT geometry and force schedule for the common mechanics
   and local-drive mapping.
3. Resolve OLR definition and overload timing/pulse sequence for exact history
   replay.
4. Define a QSTE340TM-specific EXP_floor plasticity parameter source or explicit
   uncertainty model. A synthetic tungsten vector is not admissible evidence.
5. Define and independently justify the mechanical observation map for clip-gauge
   closure force. Closure force is not relabeled as K_open, blunting radius, or an
   additive plasticity fraction.

Only after those gates should one material vector be calibrated against selected
constant-amplitude blocks and evaluated, unchanged, against held-out R and overload
blocks. Temperature-class validation cannot be claimed from these nominally
isothermal observations.
