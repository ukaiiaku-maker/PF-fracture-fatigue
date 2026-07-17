# v10.1.8 forward interaction-zone memory

## Motivation

The v10.1.7 DBTT matrix showed that scalar source refresh and local back-stress
scales could change the instantaneous active population but could not generate a
developed high-temperature plastic R-curve.  The best high-temperature matched
late R-rise remained only about 0.05 MPa sqrt(m).

The missing crack-growth memory is placed in the interaction zone ahead of the
current crack tip, not in the crack wake.

## Spatial source state

Each slip system carries an available source-capacity field over the forward
moving coordinate `xi = x - a(t)`.  The promoted
`source_sites_per_system` remains the total virgin source content per system and
is distributed over a finite interaction length.  No temperature-dependent
source count is introduced.

Local emission uses the promoted Arrhenius emission surface evaluated at the
local forward stress minus the system Taylor back stress.  Emitted content is
inserted into the existing mobile field at the same spatial bin.

During crack growth the available source field translates toward `xi=0`.
Available capacity crossing the tip leaves the forward zone.  An equal material
slice enters at the far boundary carrying virgin source capacity.  The previous
uniform scalar refresh operation is discarded.

## Interaction length and retention

The reference interaction length is the promoted
`source_refresh_length_m`.  The dimensionless
`FORWARD_INTERACTION_LENGTH_SCALE` redistributes the same integrated virgin
content over a shorter or longer forward interval.

`FORWARD_RETENTION_SCALE` multiplies the Taylor encounter/trapping rate while
leaving emission, Peierls transport, Taylor release, recovery, and cleavage
barriers unchanged.  It is temperature independent.  Temperature and loading
rate affect the developed population only through the existing Arrhenius rates
and the residence time in the moving forward zone.

## Wake treatment

Active mobile, retained, and slip content crossing behind the tip is retained
for conservative bookkeeping.  The validation runner sets `--no-wake-shielding`.
The wake therefore does not contribute to toughness in this calibration.

## First matrix

The default four-candidate matrix uses:

- temperatures: 300 and 1100 K;
- interaction-length scales: 1 and 2;
- retention scales: 1 and 3;
- crack extension: 100 micrometres;
- one matched no-plasticity baseline per temperature.

A nine-candidate matrix can be requested with:

```bash
INTERACTION_SCALES="0.5 1 2" RETENTION_SCALES="0.3 1 3"
```

The analyzer reports matched plastic initiation and R-curve increments together
with late mobile/retained content, cumulative source consumption and inflow,
source and depletion centroids, back stress, and active shielding.
