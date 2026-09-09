# Part X final handoff

**Primary classification:** `SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE`
**Status:** `FINAL`

This is the operator-facing handoff for the completed signed-K crack-
rebonding compression-conditioned Part X campaign. For the full running
milestone-by-milestone log, see `CODEX_HANDOFF_PART_X.md` in this same
directory. For the complete terminal scientific decision, see
`part_x_final_decision.json` / `part_x_final_decision.md`.

## How to reproduce and verify

All commands below use the pinned interpreter:

```
/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python
```

1. **Regenerate every analysis artifact from tracked data** (never touches
   `runs/`, which is gitignored and not required to be present):

   ```
   scripts/build_part_x_px4_developed_analysis.py
   scripts/build_part_x_px4_stage1_evidence.py
   scripts/build_part_x_px4_stage1_slope_table.py
   scripts/build_part_x_px4_scientific_decision.py
   scripts/build_part_x_px5_evidence.py
   scripts/build_part_x_px5_static_shield_analysis.py
   scripts/build_part_x_px5_scientific_decision.py
   scripts/build_part_x_px6_synthesis.py
   scripts/plot_part_x_px6_figures.py
   scripts/build_part_x_local_slope_curvature_table.py
   scripts/build_part_x_admissibility_map.py
   scripts/build_part_x_producer_launch_provenance_closure.py
   scripts/build_part_x_final_decision.py
   scripts/render_part_x_final_decision_md.py
   scripts/build_part_x_artifact_manifest.py
   scripts/build_part_x_figure_manifest.py
   ```

   All of these are pure reductions of already-tracked
   `artifacts/crack_rebonding_part_x_v1/*.{json,csv}` files -- deterministic,
   reads-only-tracked-inputs, no physical simulation.

2. **Verify.** Run, in order:

   ```
   scripts/verify_part_x_px4_stage1.py
   scripts/verify_part_x_px5.py
   scripts/verify_part_x_terminal.py
   ```

   Each is independently portable: every one has been tested by moving the
   entire `runs/` directory aside, re-running, confirming the verifier still
   passes, and restoring `runs/` afterward. `verify_part_x_terminal.py` is
   the single comprehensive closure check -- it re-invokes the two stage
   verifiers, independently recomputes the campaign's key counts and
   classifications from the lowest-level tracked ledgers, checks producer-
   versus-launch source-identity closure, checks the artifact/figure
   manifests for internal consistency, and scans for forbidden claims.

3. **Regression tests.**

   ```
   pytest tests/ -k crack_rebonding -q
   ```

## What is and is not claimed

See `part_x_final_decision.json`'s `qualifiers`, `explicit_scope_limitations`,
and `not_claimed` fields for the authoritative list. In summary: this
campaign qualifies a signed-K compression-conditioned crack-rebonding
surrogate's fatigue-crack-growth-rate effect across five developed
protocols, two seeds, and (for two of those protocols) a static-versus-
dynamic mechanistic attribution. It does **not** claim a closure-corrected
DeltaK_eff, a resolved opposing-face contact model, topological crack
healing, a calibrated physical chemistry, or production-line merge
readiness.

## Where things live

- `artifacts/crack_rebonding_part_x_v1/` -- every Part X analysis artifact,
  table, figure, and decision document (tracked in git).
- `runs/crack_rebonding_part_x_v1/` -- raw physical result JSON files
  (gitignored, NOT required for verification or reproduction of any
  analysis -- every verifier is proven to pass with this directory hidden).

## Contact points for future extension

- Extending PX5's static-shield attribution to D1/D3/D6 would require new
  physical trajectories under the same `run_trajectory(...,
  static_shield_control=...)` mechanism already qualified for D2/D5 -- see
  `scripts/build_part_x_px5_static_shield_registry.py`.
- A closure-corrected DeltaK_eff or a resolved-contact model would require
  a genuinely new constitutive model, not an analysis-only extension of
  this campaign.
