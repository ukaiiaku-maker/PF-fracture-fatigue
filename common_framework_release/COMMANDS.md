# Common-framework commands

These commands assume the repository root and the qualified Python environment.
They do not authorize the large prospective campaign.

## Portable code qualification

```bash
python scripts/run_common_framework_qualification.py portable
```

## Complete framework qualification

```bash
python scripts/run_common_framework_qualification.py full \
  --external-manifest common_framework_release/EXTERNAL_EVIDENCE_MANIFEST.json
```

The full command exits nonzero with `INCOMPLETE` when any mandatory external
artifact is absent or has the wrong content digest. It never turns that state
into an optional skip.

## Rebuild this compact handoff

```bash
python scripts/build_common_framework_release.py \
  --companion ../fem-czm-current-pf-companion
```

## Inspect the prospective plan (no solver execution)

```bash
python scripts/run_prospective_common_campaign.py --status
python scripts/run_prospective_common_campaign.py --block P1 --status
```

## Query an accepted continuous history

```bash
python scripts/query_k_at_crack_length.py HISTORY.csv \
  --target-um 1000 --coordinate projected_extension --stage equilibrated
```

Production launch remains disabled until the release gates in
`FRAMEWORK_STATUS.json` are resolved and a separate author-approved execution
stage names the frozen framework commit and output root.
