# Output retention and storage

Output selection must not change mechanics, source ownership, random-number ownership, or event admission. Each run retains its option JSON, resolved provider receipt, exact command/environment, selected material/kernel hashes, chronological scalar/event history, terminal status, and a restart checkpoint appropriate to the selected profile.

| Profile | Always retain | Optional/thinned |
|---|---|---|
| `compact_smoke` | receipts, command, scalar/event history, terminal summary, one terminal restart | native fields and images omitted unless a failure requires them |
| `selected_fields` | compact contents plus named event/checkpoint fields | intermediate fields thinned by physical checkpoint, not arbitrary file count |
| `publication_restart` | full identity, chronological observers, event ledgers, restart generations needed to reproduce selected figures | redundant caches and regenerable images may be omitted |

The new long-PF regression uses solver flags `--save-snapshots 0 --no-plots --no-mpz-state-output`. These are output controls only; full spatial FEM/PF mechanics, spatial process-zone state, crack advance, state evolution, K/J evaluation, and terminal restart logic remain active. The physics-ablation flag `--no-pz-spatial-state` is explicitly prohibited for this regression.

Before a long run, record filesystem free space and estimate peak output from a comparable run. Pause only when the measured margin is unsafe. Do not impose a scientific crack-length, event-count, mesh, or test-size cap merely to reduce storage. Historical results are read-only and are not deleted by this integration.

The V5 artifact-heavy historical tests are not copied into the branch. Current-contract tests construct their own full state and round-trip a compressed checkpoint. Omitted archival checkpoints remain referenced as external evidence, not silently recreated.

## Measured qualification use

The final full-PF 1000 µm run began with approximately 4.4 GiB free on `/Volumes/Data` and completed with approximately 3.8 GiB free. Its retained raw directory occupied 54,800 KiB on disk (56,064,892 logical bytes). The large members were three 15 MB JSON audit ledgers plus 3.2 MB steps and 3.0 MB fronts tables; no snapshot sequence or MPZ field archive was written. The raw directory is ignored and was not staged. The committed evidence consists of a compact JSON summary, two small plots, protected-history query results, and rejection markers.

The canonical signed-kernel family remains an external dependency identified by path and SHA-256 in `integration_evidence/new_full_pf_1000um_summary.json`. The protected 1000 µm archive and the separate FEM repository were read only. No local or external scientific data were deleted.
