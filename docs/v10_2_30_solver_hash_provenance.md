# Production Solver Hash Provenance (Gate S0)

## Finding

The mission-stated production solver SHA-256
(`c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b`) does not match
any individual production physics file at base commit `b7bd38b97da551aaef1b43d6d28c9ea44a06655c`,
nor a CRLF line-ending variant of any of them. It is, however, a genuine, long-established
literal constant (`SOLVER_SHA` / `production_solver_sha256`), first introduced at commit
`819dba3` (`scripts/complete_v10_2_30_two_scale_virtual_CT.py`, 2026-08-27) and referenced
unchanged across at least ten analysis/verification scripts spanning many subsequent commits,
always paired with `QUALIFIED_SOLVER_HEAD = "94871be15702e7fb85116b92af62c1226c61be42"` — the
same commit the mission cites as "production solver HEAD." `CODEX_PROGRESS.md:60` explicitly
states "the qualified production solver hash remains `c15a9571...`" across a run of
analysis-only commits, confirming this project's own convention treats the literal as an
unchanging fingerprint carried forward by assertion, not recomputed per script.

No script in the tracked repository history computes this literal from solver source (the only
`sha256(path)` helpers present are used to hash output/data CSVs and JSON manifests, never a
solver `.py` file). No documented multi-file manifest or concatenation recipe for "the solver"
as a hashed bundle was found (`grep` for `solver_manifest`, `SOLVER_FILES`, `combined_solver`,
`solver_bundle` across `*.py`/`*.md`/`*.json` returned nothing). The one distinct commit hash
found alongside it in one script (`SOURCE_HEAD = "d727cbde36f240214086ac2134985bcd023742fc"`,
`scripts/analyze_v10_2_30_analytical_overlay_all_1d.py`) exists in this repository's history but
predates the creation of `sharp_front_v10_2_30_fixed_deltaK.py` (the file does not exist at that
commit), so it cannot be the source of a single-file hash either.

## Independently verified facts

- Base commit `b7bd38b97da551aaef1b43d6d28c9ea44a06655c` is a real, clean commit and is exactly
  what the live temperature-fatigue campaign's `--expected-head` flag targets (confirmed via
  the running workers' process arguments).
- Seven core production physics files
  (`unified_front.py`, `kinetic_tip_cell.py`, `separated_source_tip.py`,
  `continuum_source_tip.py`, `campaign_calibrated_tip.py`,
  `persistent_site_cyclic_energy_gated_v10230.py`, `sharp_front_v10_2_30_fixed_deltaK.py`) are
  byte-identical across base commit `b7bd38b`, this branch's current HEAD (`c2cf2b8`), and the
  worktree's on-disk content — verified individually per file, not assumed.
- `94871be15702e7fb85116b92af62c1226c61be42` (the mission's cited "production solver HEAD") is
  an ancestor of `b7bd38b`, confirmed via `git merge-base --is-ancestor`.

## Classification

```
MISSION_SOLVER_HASH_UNREPRODUCED_BUT_BASE_COMMIT_VERIFIED
```

The origin of the mission's stated SHA-256 cannot be recovered from tracked repository history
(no generating script, no documented manifest recipe, no single-file or CRLF-variant match).
The base commit, the individual production-file hashes, and worktree content are all
independently verified and internally consistent. This is **not** classified as
`MISSION_SOLVER_HASH_STALE_BUT_BASE_COMMIT_VERIFIED`, since there is no evidence the hash
belonged to a superseded source version — only that its derivation is not reproducible from
what is currently tracked.

## Individual file hashes at base commit `b7bd38b97da551aaef1b43d6d28c9ea44a06655c`

| File | SHA-256 |
|---|---|
| `arrhenius_fracture/unified_front.py` | `377597fed7a137675aed0b8a724a01c78d4834673b4c411954a69d354920c868` |
| `arrhenius_fracture/kinetic_tip_cell.py` | `a4281c3310a1f3cc69796c58e1bd707648bf04086ac8f902a912fc637946b463` |
| `arrhenius_fracture/separated_source_tip.py` | `dca6d14ade4cec24872d5ac530e5020529dd55765a782d65d64d73fc2dee6f5c` |
| `arrhenius_fracture/continuum_source_tip.py` | `dea7fbda0f28e13e30aba8d2ce5e3d0ad37ac9dff96245b3b1529c70e1dd4ea0` |
| `arrhenius_fracture/campaign_calibrated_tip.py` | `4b7fbfa1b18cb0e9d921807fd5f12d457d343996b1db8f0cacc8d1fdb76eecc1` |
| `arrhenius_fracture/persistent_site_cyclic_energy_gated_v10230.py` | `af62e0dec3d3522d7d6cf455f1ee7f1d39f1544368b867e7aa935d0261f7d3b6` |
| `arrhenius_fracture/sharp_front_v10_2_30_fixed_deltaK.py` | `cee0b446a436165890605f08ca6ca73d60513890910ef7a6d2a31c0c431ed009` |

`git_tree_object_id` for `arrhenius_fracture/` at `b7bd38b` (a Git tree identity, not a
SHA-256 content hash, recorded separately per the interpretation rule):
`9cf18c341e6aa93cd5bbe2331f9afa5af035e43c`.

This record satisfies the mission's requirement to resolve the solver-hash discrepancy with a
definitive classification rather than leaving it as an unresolved footnote.
