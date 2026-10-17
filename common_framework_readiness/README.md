# Common fracture framework — live readiness records

This directory is the engineering-readiness record for the current R01–R13
implementation. It is not a scientific-results release. Production execution
and manuscript modification remain disabled.

The current disabled campaign is `PROSPECTIVE_CAMPAIGN_V3.jsonl` (699 cases,
699 source-bound native commands). Rebuild and validate it with:

```bash
python scripts/build_prospective_campaign_v3.py --disabled --audit --dry-run
python scripts/audit_campaign_v3_acceptance.py \
  --output common_framework_readiness/CAMPAIGN_V3_ACCEPTANCE_AUDIT.json
python scripts/run_prospective_common_campaign.py --status
python scripts/run_prospective_common_campaign.py --status --block P9
```

`FRAMEWORK_COMPLETION_LEDGER.json` is the R01–R13 ledger;
`FRAC_GOALS_1.0_LIVE_PROGRESS.json` carries separate readiness, numerical,
scientific-coverage, conclusion, evidence and author-acceptance fields for all
18 charter goals. `fracture_final_current/methods/model_and_observer_contracts/
CAPABILITY_PRESERVATION_MATRIX.csv` is the capability matrix.

The original EXP-floor route and the two bilinear comparison routes have the
specified public IDs `exp_floor_activated_native`,
`bilinear_barrier_activated`, and `bilinear_czm_classical`. The latter two keep
their old internal names only as compatibility aliases. Their present material
support is constructed qualification input, not a production calibration. Both
public bilinear routes install `NativeBilinearFrontProvider` on the actual
current atomic-path-corridor specimen's fully wrapped
`CalibratedMixedModeMPZV911Engine`, retaining its requested mesh, loading,
source geography, signed persistent-source lineage, adaptive topology,
accepted event lifecycle and fresh-process checkpoint owner. The provider
replaces the inherited EXP-floor opening clock and advances the established
source owner once per accepted physical interval. Runtime verification rejects
a selected but disconnected provider. The older compact solved fixture remains
useful component evidence, but is not the campaign driver.

Current resource admission is workload-specific. R07 and R09 retain exact
accepted checkpoints and commands, but their substantial continuations are not
admitted while other multi-gigabyte scientific workers occupy the host and swap
is under pressure. This does not block lightweight independent implementation,
auditing, or release work.
`RESOURCE_ADMISSION_20261016.json` separates measured host state from workload
peak estimates; the existing disk preflight is explicitly disk-only and is not
represented as a RAM admission measurement.

The old v2 handoff remains reproducible through
`scripts/build_common_readiness_handoff.py`, but that script now writes only to
`common_framework_readiness/legacy_v2_regeneration` and cannot overwrite these
live v3 records.
