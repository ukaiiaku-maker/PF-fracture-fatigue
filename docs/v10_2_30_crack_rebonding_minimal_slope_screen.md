# v10.2.30 Crack-Rebonding Minimal Local Slope Screen

## 1. Purpose and authorization chain

Bounded, staged follow-up to the completed two-seed causal pilot
(`codex/v10.2.30-crack-rebonding-causal-pilot-v2` @ `5bc56c9`, preserved
immutable, untouched by this worktree). Tests whether the reproducible
~0.042-decade cohesive waiting-time effect found at Kmax=18 MPa*sqrt(m) is
a flat rate offset or actually changes the local crack-growth slope, by
adding Kmax=15 and 21 MPa*sqrt(m) at the same two seeds (1720, 1001723).

**Authorization chain:**
1. The parent branch's regime-equivalence analysis (analysis-only, no new
   physics, commit `8737784`) found reversible and persistent kinetics
   presets equivalent for this purpose at both seeds
   (`|delta S_h| ~ 0.0001-0.0002` decade vs. a 0.005 threshold;
   action-weighted `K_rebond` identical to numerical precision) --
   authorizing a single-regime (reversible), 8-trajectory screen in place
   of a 16-trajectory two-regime one.
2. This worktree/branch was created from that parent's exact `5bc56c9`,
   per explicit review authorization, to run the new physics in isolation
   from the completed, preserved pilot.

**Explicitly not authorized by this screen** (regardless of outcome): the
full multi-K/R/frequency/dwell/passivation Part X campaign, or a merge
into the production line.

## 2. Frozen predictions (before any new trajectory)

`scripts/freeze_v2_slope_screen_predictions.py` reloads the reversible AND
persistent finite-cohesion configs verbatim from the parent's committed
`frozen_configuration.json` (hash-checked), then re-evaluates the
single-patch model's `A_on(K)`, `A_off(K)`, `p_B*(K)`, and predicted
`K_rebond(K)` at Kmax=15/18/21 MPa*sqrt(m) -- holding every kinetics/
cohesion parameter fixed (no barrier re-inversion, no activation-volume
search, no `Pi_K` rescaling). `Pi_K = K_rebond_max/Kmax` varies naturally
with load, as required:

```
reversible: Pi_K = 0.0600 (Kmax=15), 0.0500 (Kmax=18), 0.0429 (Kmax=21)
persistent: Pi_K = 0.0600 (Kmax=15), 0.0500 (Kmax=18), 0.0429 (Kmax=21)
```

## 3. Reused Kmax=18 point

The Kmax=18 reversible zero/finite-cohesion results are reused verbatim
from the parent's tracked `event_ledger.json` (seed 1720) and
`second_seed_event_ledger.json` (seed 1001723) -- not re-run. Config
hashes were confirmed identical to the parent's `frozen_configuration.json`
before use.

## 4. Exposure-gated staged execution

`scripts/run_v2_minimal_slope_screen.py` ran the zero-cohesion member of
each (Kmax, seed) pair first, requiring `>= 2` post-first-event intervals
with a complete negative-K excursion before launching that pair's
finite-cohesion twin:

| Load | Seed | Compression-containing intervals (of 6) | Exposure gate |
|---|---|---|---|
| Kmax=15 | 1720 | 2 | **PASS** |
| Kmax=15 | 1001723 | 2 | **PASS** |
| Kmax=21 | 1720 | 5 | **PASS** |
| Kmax=21 | 1001723 | **1** | **FAIL** |

All 4 zero-cohesion runs, plus the 3 finite-cohesion twins whose gate
passed (Kmax=15 both seeds, Kmax=21 seed 1720), completed uncensored with
7 accepted events each -- 7 trajectories total.

**Per the mission's explicit stopping rule, the screen stopped at
`K21MPa_seed1001723` without launching that pair's finite-cohesion
trajectory and without substituting a different load.** The full
8-trajectory, 3-point-per-seed slope fit cannot be completed.

This is itself informative: at Kmax=21 the native cleavage hazard is fast
enough, and the seed-1001723 threshold stream's particular event timing
places it, close enough to the exposure boundary that a single stochastic
realization drops below the 2-interval bar (seed 1720 at the same load
comfortably clears it with 5/6). This is a genuine sensitivity finding
about the higher end of this Kmax grid at the reference frequency, not a
software defect -- consistent with the same "reference protocol may not
sample compression" question the original V2-C preflight investigated at
Kmax=18.

## 5. Terminal status

**`STOPPED_EARLY_EXPOSURE_GATE_FAILED`** at `K21MPa_seed1001723`
(`artifacts/crack_rebonding_minimal_slope_screen_v1/slope_screen_decision
.json`). No `S_h(21)` value exists for seed 1001723, so no 3-point slope
fit, no secants, and none of the prospective classifications
(`REBONDING_RATE_OFFSET_LIKE` / `REBONDING_STEEPENS_LOCAL_RESPONSE` /
`REBONDING_FLATTENS_LOCAL_RESPONSE` / `REBONDING_SLOPE_EFFECT_WEAK_OR_
UNRESOLVED`) can be assigned. The screen returns without a slope
determination, exactly as its own frozen protocol requires when the
exposure gate fails.

`scripts/verify_v2_minimal_slope_screen.py`: **`overall_pass: true`** --
confirms the reused Kmax=18 point, the frozen predictions, the exposure
gate report, and the `STOPPED_EARLY` status are all mutually consistent
and match the tracked artifacts; confirms no unauthorized activity flag
(`multi_K_paris_slope_campaign_authorized`, `part_x_authorized`,
`production_line_merge_authorized`) was set. All 183 `crack_rebonding`
tests pass.

## 6. What remains open

Whether contact-gated rebonding steepens, flattens, or merely offsets the
local rate is **unresolved** at this Kmax grid under this exposure
protocol -- not because the mechanism is absent, but because Kmax=21 at
f=1000 Hz does not reliably sample two full compressive excursions across
both seeds. Any next step (e.g., a different Kmax upper bound, more seeds
at Kmax=21 specifically, or an explicitly-authorized frequency ablation
matching V2-C's original Option A/B framing) is new physics requiring its
own separate authorization -- not attempted here.
