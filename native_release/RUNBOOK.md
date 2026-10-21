# Native fracture release runbook

The package is dry-run by default. Commands below assume the shell is in
`source/PF-fracture-fatigue`. The only required operator edits are the output
root and, on a different compatible host, the interpreter path.

```sh
PACKAGE_ROOT=$(cd ../.. && pwd)
PF_ROOT="$PACKAGE_ROOT/source/PF-fracture-fatigue"
PY=/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python
OUT=/replace/with/large/execution/volume/native-fracture-campaign
cd "$PF_ROOT"
```

## 1. Validate without advancing a solver

```sh
"$PY" native_release/native_campaign.py --python "$PY" --output-root "$OUT" validate
```

This verifies the source relationship, unchanged calculation closure, exact
FEM commit, preserved artifact hashes, safe/readable state archive, portable
kernel, all 663 case input paths, and interpreter. It does not execute a
fracture step. Separately confirm target free memory and disk against
`TARGET_HOST.example.json`; resource admission is case-local.

## 2. Inspect the staged command

The first eligible case is shown, but not launched:

```sh
"$PY" native_release/native_campaign.py --python "$PY" --output-root "$OUT" start
```

Select a phase or exact immutable case ID when desired:

```sh
"$PY" native_release/native_campaign.py --python "$PY" --output-root "$OUT" start --phase 1
"$PY" native_release/native_campaign.py --python "$PY" --output-root "$OUT" start --case-id CASE_ID
```

The dependency order is: phase 1 initiation; phase 2 state/rate/geometry,
orientation, constraint, and source/bulk comparisons; phase 3 developed
propagation; phase 4 selective independent FEM; phase 5 branch/void pairs;
phase 6 original-opening same-material controls. Independent eligible rows may
run while an unrelated conditional row remains blocked.

## 3. Explicitly start one admitted case

Only after reviewing target resources and the dry-run command:

```sh
"$PY" native_release/native_campaign.py --python "$PY" --output-root "$OUT" start \
  --case-id CASE_ID --execute --acknowledge START_STAGED_NATIVE_CAMPAIGN
```

The wrapper launches one background process group, records its PID, immutable
request and output directory, sets the packaged FEM roots, and delegates to the
existing source-attesting worker. It does not retune or fall back to another
physics path.

## 4. Status, safe stop/checkpoint, and resume

```sh
"$PY" native_release/native_campaign.py --python "$PY" --output-root "$OUT" status
"$PY" native_release/native_campaign.py --python "$PY" --output-root "$OUT" checkpoint-stop
"$PY" native_release/native_campaign.py --python "$PY" --output-root "$OUT" resume \
  --case-id CASE_ID --restart-from /exact/compatible/checkpoint \
  --execute --acknowledge START_STAGED_NATIVE_CAMPAIGN
```

`checkpoint-stop` signals only the PID group recorded by this package. Native
atomic checkpoint owners remain authoritative. Never relabel an interrupted
case as a physical censor; review its receipt and use only a compatible exact
restart. The wrapper never kills unrelated processes.

## 5. Incremental analysis inventory

```sh
"$PY" native_release/native_campaign.py --python "$PY" --output-root "$OUT" analyze
```

This writes an evidence index of summaries, attestations and checkpoints. The
existing case-specific analysis/export entry points remain in `scripts/` and
the accepted export behavior is reused. Classification is kept separate for
`INTERRUPTED/RESUMABLE`, numerical failure, domain exit and physical no-event
censor.

## Campaign boundaries

- `MAIN_SEQUENCE.jsonl` has 663 original activated-opening rows.
- `DEFERRED_OPTIONAL_CZM.jsonl` preserves 36 fitted-cohesive comparison rows;
  they are not in the default sequence.
- 585 main rows have versioned inputs and become executable only after target
  resource admission. Seventy-five retain qualification-only material
  selection status; three require matched populated checkpoints.
- Full scientific endpoints are preserved. Packaging timeouts are never used
  as physical endpoints.
- The accepted 17 October suites and native events are evidence to reuse, not
  commands in this runbook.

## Optional publication (not executed here)

After choosing a new remote branch/tag name, publish without force:

```sh
git push origin release/native-fracture-20261021
git push origin native-fracture-release-20261021T051215Z
```

The saved `pre-final-changes-20261017T231804Z` tag and GitHub release must not
be moved or replaced.
