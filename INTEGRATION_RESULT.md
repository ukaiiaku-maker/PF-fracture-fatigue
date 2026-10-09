# Integration result

The complete option-preserving fracture–fatigue family is integrated on `codex/shared-fracture-fatigue-options-v1`. The branch retains the original full spatial PF and separate full FEM/CZM realizations, exposes independent source geography and lifecycle choices, preserves signed/reversible cyclic state, branching, V5 voiding, long propagation, restart, stage-aware K/J extraction, fatigue analysis and plotting, and fails early for unsupported combinations.

## Source identity

- Ancestry base: `9e884fb0b0845da621d2612bdf1042e481b8df49`.
- Integrated mechanics/provider source commit: `baedb069c912db69032eea8b339cb01b4bf97278`.
- Donors: `9e2c43ebfb42d6694c27f01ea9092e79da7e3a06` (bulk, branch, V5 void), `94871be15702e7fb85116b92af62c1226c61be42` (reversible cyclic stack), and `0a0e522291882bd13e0d03c7e231fd8a6627aa4f` (V3.6 operator/feedback/ledger).
- Full FEM/CZM comparison: `ukaiiaku-maker/Arrhenius_FEM_CZM_MPZ` commit `293491157063484bb10df6adc479f847a6a08ba6`, branch `v10.0.5.18.4.0-theta0-pf-parity`. No FEM source change or companion branch was needed.
- Final pushed head and remote verification are recorded in `REMOTE_PUSH_RECEIPT.json`.

## Integrated capabilities

`SharedOptions` separates mechanics, loading, source location, source lifecycle, plasticity scope, crack topology, cavity processes, kinetics/integration and output retention. Seven named presets resolve to serialized provider receipts. Resolution verifies legacy providers in isolated interpreters, so import order cannot silently replace the chosen driver.

The callable family includes:

- tip/surface persistent or finite source paths and full-field bulk-uniform sources, including explicit tip-plus-bulk V5 use;
- monotonic, explicit cyclic, fixed-DeltaK and hold/dwell configuration identities;
- signed mobile/retained/wake state, physically eligible reversible mobile return, first-passage and high-cycle restart state;
- V3.6 operator-consistent projection, mechanism-specific feedback and swept-slip ledgers;
- V11–V13 audited branching/multifront ownership and mechanically separating V12 crack mechanics;
- full-order V5 vacancy/cavity scheduling with compressed current-state restart;
- chronological front/stage/segment observations, repeated-length preservation, discrete-jump K-at-length queries and schema-aware plots;
- developed fatigue event tables, censor/failure tables, provenance-preserving four-class aggregation and qualified Paris/local-slope analysis.

The detailed mapping from every protected capability to entry point and evidence is in `CAPABILITY_PRESERVATION_MATRIX.csv`.

## Qualification result

The selected release suite passed 267 PF tests plus 16 full FEM/CZM tests (283 total). The final committed-mechanics PF run completed 201 accepted events and reached 1002.903 µm extension. K_I/J queries at 100, 500 and 1000 µm all returned the correct accepted-event jump bracket without interpolation. Twelve protected 1000 µm histories were separately reanalyzed without alteration. See `SMOKE_REGRESSION_RESULTS.md` and its machine-readable companion.

One environment limitation remains explicit: the system base Python 3.13 pytest process terminated before collection with signal 11. The qualified Python 3.12.13 project environment is green. Legacy provider test families are run in isolated processes to prevent test-only import mutation; production provider resolution enforces the same isolation.

## Scope and retention

No production parameter sweep, parameter refit, manuscript rewrite, historical replacement declaration or deletion was performed. Raw smoke/long-run products and external kernel/FEM dependencies are not committed; compact receipts, source tables and plots are. No historical local or external scientific data were deleted or modified. Stateful Kitagawa was author-excluded and was not inspected, restored, ported, backed up, deleted or used as a blocker.
