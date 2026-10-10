# Targeted completion review packet

This sub-3 MiB packet is the selected, Git-suitable evidence for numerical
source `b927dac18c58f2fb896fe3520fb4c23ba950ec7e`. It is not a replacement for
the full native checkpoints. `CLAIM_TO_EVIDENCE_INDEX.json` separates clean
committed-source execution, sealed development calculations, historical
bounded references, reanalysis, and tests.

The theta=40 receipts were generated from a clean detached worktree at the
exact source commit. The V5 calculations retain their sealed development
snapshots and are not relabeled as clean-source runs. The long-growth
trajectory was not rerun: the geometry export consumes the unchanged steps
and native event-coordinate files, and the packet includes both compressed
inputs and the resulting schema `/2` observations.

Verify packet bytes from this directory with:

```bash
shasum -a 256 -c SHA256SUMS
gzip -t growth/*.gz
```

Recreate the geometry export and queries with the committed scripts:

```bash
python scripts/export_developed_geometry_observations.py \
  --observations fracture_observations_corrected_348343f.jsonl \
  --geometry-events stochastic_avalanche_geometry_events.json \
  --out fracture_observations_geometry_aware.jsonl \
  --receipt developed_geometry_export.json
python scripts/query_k_at_crack_length.py \
  fracture_observations_geometry_aware.jsonl \
  --coordinate path_arclength --target-um 1002
```

Full artifact paths and raw hashes are recorded in the index. Only
`/Volumes/Data` was mounted during this pass, so newly generated full
checkpoints and histories still have one local copy. No external-backup claim
is made, and no scientific data were deleted.
