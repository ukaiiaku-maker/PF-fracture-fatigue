# V13 merge-baseline review

`MERGE_CLEARED_RELATIVE_TO_ACCEPTED_SEVEN_FAILURE_BASELINE`

**BRANCHING_KINETICS_MODEL_UNCALIBRATED**

The merge gate is cleared **relative to the accepted seven-failure baseline**, not
as an all-green repository. No merge or push was performed. The seven inherited
failures remain failures; no physics or accepted scientific decision was changed.

## Exact disposition of the previous 31 failures

The original one-time publication suite remains untouched: **1,099 passed,
5 skipped, 31 failed**, with all original 131 V13-named tests passing.
Its review SHA-256 remains
`12bdb50e17896e877322fd6f4012b2ea2c3447f4db1b7190d6d9d8bc8d56a4c7`.

- **24 missing historical V5/V6 product fixtures:** absent generated JSON/report,
  archive or checkpoint-reference products, not V13 source regressions.
- **Seven established legacy failures:** the exact accepted node IDs and failure
  signatures, including the sandboxed `ps` environment limitation.

[Exact 31-test classification and current comparison](v13_merge_failure_comparison.json)
retains every historical missing-file message, every legacy failure message, and
the matching repository traceback frames. The seven node IDs are also present in
the earlier V6.1.1 accepted baseline record; its path and hash are recorded there.

| Accepted legacy test | Retained failure signature |
|---|---|
| `test_real_signed_builder_v10212.py::test_review_builder_emits_artifact_consumable_by_production_loader` | S00 audit schema rejected by the production loader; subprocess return code 1 |
| `test_v10214_capture_model_id.py::test_v10214_capture_model_id` | Expected v10.2.14 capture ID; actual v10.2.27 frozen-measurement-clone ID |
| `test_v10214_response_model_id.py::test_v10214_response_model_id` | Expected v10.2.14 response ID; actual v10.2.27 mesh-resolved response ID |
| `test_v10215_stage3_schema_and_status.py::test_stage3_status_script_parses_and_reports_missing_run` | Empty stdout because sandbox denies `ps`: `PermissionError: [Errno 1]` |
| `test_v10_2_27_zero_event_summary.py::test_zero_event_summary_is_recorded_explicitly` | `KeyError: zero_geometry_events_validated_against_zero_advances` |
| `test_v10_2_27_zero_event_summary.py::test_empty_geometry_rejects_nonzero_advances` | `Failed: DID NOT RAISE RuntimeError` |
| `test_v10_2_29_vhcf_nonlinear_selector.py::test_delegate_reenters_selector_under_global_cycle_cap` | `assert 123.0 is None` |

All paths above are beneath `tests/`. The `ps` failure is **environmental**,
not a scientific or V13 lifecycle failure. Message normalization removes only
volatile test-temporary paths, repr escapes and whitespace; raw messages remain
available. Repository traceback locations and exception types also match exactly.

## Historical fixtures: two verified passes, 22 explicit skips

The 24 affected tests now explicitly request the fixture-root loader. Source-only
tests in the same modules do not request it. No mutable simulation-tree fallback
exists, and all **207 original assertions across the nine edited modules** are
AST-identical to publication record `22cb426`.

One small in-repository package contains only three V5.2.1 JSON records used by
two tests. They are byte-identical to archival Git commit
`1816b4f73a7fb671aad29cb03948b10fc8a62b08`, which attests their origin in
`e4130052941a56dca80904eda85740f0a868610c`. The original e413005 object is not
available locally. The fixture manifest publishes SHA-256 and size for each
archived blob, and the loader checks both those hashes and the Git blobs on use.
These newly recorded SHA-256 values are not presented as an older publication.

The other **22 tests explicitly skip** with
`HISTORICAL_PRODUCT_FIXTURE_UNAVAILABLE`: the optional verified complete
historical pack is not installed. Some original assertions need compact archives,
source snapshots, figure files or checkpoint payloads in addition to JSON/CSV;
none was replaced with invented products or weakened assertions.

An explicitly configured missing, partial, corrupt, unpinned or path-escaping pack
**fails**, rather than skips. Nine new fixture-loader regressions passed.
[Fixture manifest](historical_product_fixture_manifest.json) and
[installation contract](../../tests/fixtures/HISTORICAL_PRODUCTS.md) record the
available members, exact test scope, provenance and optional-package contract.

## Validation: one focused run, then one full run

| Invocation | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Focused: historical products + seven legacy tests + all V13 tests | 144 | 22 | 5 |
| Complete repository suite | 1,110 | 27 | 7 |

Both runs passed **all original 131 V13 tests plus nine fixture-loader tests**
(140 V13-named tests). The full suite collected all original 1,135 node IDs plus
those nine new tests: 1,144 total. Its 27 skips comprise the 22 optional historical
fixture skips and the five unchanged pre-existing skips. No new failure exists.

The two zero-event legacy tests passed only in focused collection. Read-only source
inspection explains the distinction: `zero_event_summary_v10215.py` installs a
replacement summary function at import time; full collection imports that module,
while the focused tests retain the original function. **Both failures reproduce
in the full suite with their accepted signatures. They are not fixed.**

The initial focused review's strict seven-failure comparison remains recorded as
false. A separate [context review](focused_context_review.json) records this
collection-dependent distinction and the decision to proceed. Only validation
orchestration was adjusted to recognize that focused context; test/loader code
did not change after the focused run. Its original runner is preserved as
`focused_runner_source.py`. No focused or full test invocation was repeated.
The full-suite gate remained strict: all seven exact failures had to reproduce.

The prior publication suite was not rerun or overwritten. The new focused and
full invocations have separate exclusive claims, environments, XML inventories,
logs, source-file hashes and results. Qualified conda Python and single-thread
settings were retained. Temporary test files used Macintosh HD; compact records
remain on Data. No accepted data was removed for space.

- `compileall -q arrhenius_fracture scripts tests`: PASS.
- `git diff --check`: PASS.
- Numerical assertions unchanged: PASS.
- Exact seven full-suite failure messages and traceback frames: PASS.
- Complete test inventory and no new failures: PASS.

The four raw pytest log/XML files retain their exact emitted bytes, including
diagnostic trailing whitespace. Directory-local Git binary attributes preserve
those evidence files without text conversion; source, JSON and report files
remain subject to ordinary whitespace checks. Their original SHA-256 values
are unchanged.

## Immutable scientific package and provenance

The unchanged dataset remains **64 cases, 57 first branches, seven completed
nonbranching observations, zero early-gate censors**.

Frozen physics:
`35f4a03c622112cd5bc7cf9bc08c5b7fe2387e46`.

Accepted transition record:
`b2e0e2d85e048553ad2593d895a98974356213e2`.

Original final report/test producer:
`98f9b5d579e9d593c088bcfc1da9f7ccc8529093`.

This merge-review source is separately hash-bound in `full_claim.json` and
`full_result.json`; it is not mislabeled as the original physics/report producer.

Verification before and after both runs reproduced:

- 251 frozen source/registry files;
- 17,636 frozen transition artifacts and 3,620 earlier theta40 artifacts;
- 192 held-out terminal/checkpoint/field/ledger/launch seals;
- 12,913 report-input hashes;
- 37 publication files, including all accepted reports and figures;
- all 32 archive members (31 manifested members plus the manifest).

The unchanged review archive SHA-256 is
`35234f3e78afcbd6679b523aacf402ea025ad9b6ec48f982140729ddcb83e519`.

No simulation, continuation, restart, reseed, parameter adjustment, physical-model
edit, figure regeneration or accepted archive rewrite was performed. The original
publication record retains its historical blocked disposition; this separately
reviewed record supersedes only the merge-gate disposition.

[Final merge validation and provenance hashes](final_merge_validation.json)
bind the review records and test evidence. **No automatic merge.**
