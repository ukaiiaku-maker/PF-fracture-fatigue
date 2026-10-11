# Bounded current-source fracture pilots

## Outcome

All six full-spatial PF first-passage cases passed, and the current source
completed a developed DBTT 300 K path to
1004.926 µm. The V11 suffix and
exact pinned FEM/CZM calculation stopped at explicit provenance/physics gates;
neither produced an accepted new scientific state.

## Opening versus accommodation

| Condition | First K enabled | First K opening-only | ΔK | Ratio | Post-event K enabled | Post-event K opening-only |
|---|---:|---:|---:|---:|---:|---:|
| DBTT 300 K | 23.234 | 23.234 | 0.000 | 1.000 | 23.759 | 23.759 |
| DBTT 1200 K | 52.376 | 20.936 | 31.440 | 2.502 | 53.873 | 21.363 |
| Peak 900 K | 54.904 | 20.824 | 34.080 | 2.637 | 55.679 | 21.442 |


At 300 K the DBTT accommodation state is effectively inactive and the paired
fracture states coincide. At DBTT 1200 K and Peak 900 K, activated
Peierls–Taylor accommodation/backstress raises the first-event load by
31.440 and
34.080 MPa√m, respectively. This is a
paired mechanistic result for the registered rows and declared loading path,
not a fitted temperature law.

## Developed path

The current-head DBTT 300 K path accepted
203 geometry events and reached
1004.926 µm projected extension.
The 100, 500, and 1000 µm values in `PILOT_RESULTS.json` are discrete accepted
jump brackets; no interpolation was performed.

## Bounded stops

- **V11 migration:** both archive endpoints deserialize, but the current
  directional observer is unavailable after bounded refinement at the migrated
  post-event topology. Zero current-source intervals were accepted. A new
  matched pair must be prepared under current source from a pre-event topology
  with current kernel coverage; allow about 800 MiB before doing so.
- **FEM/CZM:** commit `6c129c2` repaired the shared handoff and its focused suite
  passed 17 tests. The pinned backend at `2934911` then rejected the live
  `d41b08…` kernel because it requires the missing immutable `a85b57…` input.
  Recover/regenerate and hash-verify that small kernel bundle (allow 10 MiB);
  do not weaken the guard.

## Storage and provenance

The batch occupies 63.6 MiB and leaves
13.71 GiB free on `/Volumes/Data`.
No local or external data were deleted or modified, and no rendered step-image
sequence was created. Raw paths and exact failure receipts are indexed in
`PILOT_RESULTS.json`. The final integrated PF suite passed 780 tests in 74.13 s;
the focused FEM-adapter suite passed 17 tests.
