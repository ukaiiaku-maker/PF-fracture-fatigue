# Clean historical Stage-II V3 neutrality comparison

The authoritative corrected Stage-II V3 implementation is
`326e3f5973ef623781ab8568798c495a3f68238c`, whose commit introduces
`scripts/qualify_v12_production_integration_v3.py`, schema
`v12.production-integration-qualified-evidence/3`, gate
`STAGE_II_BASE_V3_LOCAL`, and `tests/test_v12_stage2_base_v3.py`.
The revoked `codex/v12-qualified-production-base` scaffold is not the baseline.
Neither V5 production module exists at this historical commit.

`scripts/qualify_v5_disabled_neutrality_clean.py` requires a clean detached
worktree at that exact commit and a separate clean current implementation
worktree. It starts distinct Python processes importing the implementation
from each specified worktree. The harness itself does not modify either tree.

Four frozen cases execute V12 mechanics on both sides: monotonic advance,
fixed-mesh oblique advance, checkpoint/restore followed by the same advance,
and bounded unload/reload. The current side enters through the explicit
disabled-V5 dispatcher; the historical side uses the original Stage-II entry.
Both sides verify that no active void state exists. These software-forced
geometry histories test implementation neutrality; they are not physical
cleavage-kinetics qualification.

All accepted-state dataclass fields, including the complete mesh, P0 support,
model identity, ownership/certification metadata, fields, clocks, thresholds,
RNG, reactions, energies, and source/process states, enter the component
comparison. Arrays retain exact content hashes, shapes, and dtypes.
Native complete-state fingerprints are also recorded. Actual source commits
and the historical absence of the V5 field remain visible differences rather
than being removed to claim equality. A failed complete comparison is not
automatically a mechanics failure; its exact component differences establish
whether the discrepancy is provenance, representation, or physical state.

Run after freezing clean implementation worktrees:

```sh
python scripts/qualify_v5_disabled_neutrality_clean.py /tmp/closure-neutrality-a \
  --base-worktree /path/to/detached-stage-ii-v3 \
  --current-worktree /path/to/clean-current-implementation
```

The runner retains both checkpoint trees, four complete worker rows per side,
an exact component-difference ledger, and a SHA-256 inventory. A second run
must use the same frozen implementations and be compared recursively.
