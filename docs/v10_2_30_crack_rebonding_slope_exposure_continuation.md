# v10.2.30 Crack-Rebonding Slope-Exposure Continuation

## 1. Why this branch exists

The minimal slope screen (`codex/v10.2.30-crack-rebonding-minimal-slope-screen`
@ `c742722`, preserved unchanged, untouched by this worktree) correctly
stopped at its own prospectively frozen exposure gate: Kmax=21's
zero-cohesion trajectory under seed 1001723 found only 1 of 6
post-first-event intervals with a *complete* negative-K excursion (need
`>= 2`), so the finite-cohesion twin was never launched.

Per review: for the causal-pilot qualification question, requiring a
complete compressive excursion was the right admissibility test (it
proved a newly generated surface actually saw contact before any
contact-conditioned-bonding claim was made). For a *slope* question,
reduced compression exposure at high Kmax may itself be part of the
physical load dependence -- faster opening renewals leave less time for
bond formation, weakening the cohesive contribution, which is exactly the
mechanism that would make `S_h(K)` less negative (steepen the apparent
response) as Kmax increases. Excluding the point because it has little
exposure could exclude the very mechanism being measured. Three specific
reasons this matters, matched by what the data below actually shows:

1. Formation only needs `A_on = integral(k_CB dt) > 0`, not a complete
   excursion -- a partial negative-K interval can still form bonds.
2. A previously bonded patch's `K_rebond(t)` can still be sampled by the
   cleavage hazard in an interval with no new complete excursion.
3. Finite cohesion delays opening, which can itself carry the trajectory
   into more compression -- a coupled feedback the zero-cohesion-gated
   admission rule would never let run.

This branch (from the slope screen's own `c742722`) performs an
analysis-only partial calculation from already-tracked data, then runs
**exactly one** new trajectory -- the missing finite-cohesion twin at
Kmax=21, seed=1001723 -- under the exact frozen configuration, with no
exposure gate applied to its admission.

## 2. Partial analysis (no new physics)

`scripts/analyze_v2_slope_exposure_partial.py` computed, from tracked
artifacts alone (parent pilot's two ledgers + the slope screen's own
ledger), the five already-available `S_h` points:

```
K15MPa_seed1720:    S_h=-0.125007 decade
K15MPa_seed1001723: S_h=-0.122650 decade
K18MPa_seed1720:    S_h=-0.042846 decade
K18MPa_seed1001723: S_h=-0.042051 decade
K21MPa_seed1720:    S_h=-0.026521 decade
```

`accepted_lengths_identical` confirmed `True` for every pair before using
the actual-rate-ratio definition (which reduces exactly to the
waiting-time simplification here). The pending pair's zero-cohesion side
(Kmax=21, seed 1001723) already showed 1 complete + 5 *partial* negative-K
intervals (i.e. some compressive contact in all 6, just not a full lobe in
5 of them) -- direct empirical support for point 1 above, and the reason
the completion run was expected to be informative rather than moot.

## 3. The one new trajectory

`scripts/run_v2_slope_exposure_completion.py`: Kmax=21 MPa*sqrt(m),
seed=1001723, RB2 reversible finite cohesion, config reloaded verbatim
(hash-checked against the parent's `frozen_configuration.json` --
identical bond/rupture barriers, attempt frequencies, activation volumes,
`restored_work_of_separation_J_m2`, `rebond_K_geometry_factor`, R, f,
`mpz_n_bins`, `n_phase`, event target, numerical controls; nothing
retuned). **No exposure gate applied to admission** -- labeled
`EXPOSURE_UNCONDITIONED_COMPLETION`, explicitly not a retroactive pass of
the screen's own rule.

Completed uncensored, 7 accepted events, wall time 148s. Compression
exposure (informational only): **0 complete, 6 partial** of 6 intervals --
*less* complete exposure than its own zero-cohesion twin (which had 1
complete + 5 partial), consistent with point 3 above: the finite-cohesion
shielding shifted event timing relative to the zero-cohesion twin, and
here that shift happened to reduce (not increase) the complete-excursion
count while every interval still carried partial compressive contact.

**Incidental fix**: while wiring this run, found that
`crack_rebonding_causal_pilot_v2_v10230.py::run_trajectory`'s returned
`"seed"` field was hardcoded to the module constant `SEED` (1720)
regardless of which seed was actually configured on the engine before the
call -- a pre-existing, harmless (no gate or comparison anywhere ever read
this field) mislabeling present since the original causal pilot. Fixed by
adding an explicit `hazard_rng_seed` parameter, recorded verbatim in the
result; the actual RNG behavior was never affected (it is set by
`Engine.configure_hazard(seed=...)` before `run_trajectory` is called, not
by this field).

## 4. Full three-point result (both seeds now complete)

`scripts/analyze_v2_slope_exposure_continuation.py`:

| Kmax (MPa*sqrt(m)) | S_h, seed 1720 | S_h, seed 1001723 |
|---|---|---|
| 15 | -0.125007 | -0.122650 |
| 18 | -0.042846 | -0.042051 |
| 21 | -0.026521 | **-0.026123** |

`|S_h|` falls **monotonically** with Kmax for both seeds, and the two
seeds agree closely at every point (largest relative difference ~2%,
despite Kmax=21's wildly different complete-excursion counts: 2 vs 0).

```
delta_m (least-squares, 3-point): seed 1720 = 0.6849   seed 1001723 = 0.6713
secant(15->18):                   seed 1720 = 1.0376   seed 1001723 = 1.0179
secant(18->21):                   seed 1720 = 0.2439   seed 1001723 = 0.2379
```

Both secants are positive and both seeds' `delta_m` clear the 0.25
steepening gate by nearly 3x, and the two seeds' `delta_m` values differ
by only 0.0136 (well under the "seeds differ a lot" bar) despite their
starkly different Kmax=21 compression-exposure profile -- direct evidence
that this result is **not** an artifact of exposure-counting variance
between seeds; the underlying steepening signal is robust across two very
different realizations of how much complete-vs-partial compression each
one actually saw.

## 5. Classification

**`REBONDING_STEEPENS_LOCAL_RESPONSE`**
(`artifacts/crack_rebonding_slope_exposure_continuation/
slope_exposure_continuation_decision.json`).

Per the revised gate set: not `RATE_OFFSET_LIKE` (endpoint span ~0.098
decade, far above the 0.01 threshold); seeds do not disagree in sign or
differ by more than 0.25 in `delta_m`, and the high-K difference is not
traced to exposure disparity (checked explicitly and found false) --
`delta_m >= 0.25` for both seeds, cleanly clearing
`REBONDING_STEEPENS_LOCAL_RESPONSE`.

Physical reading: contact-gated cohesive shielding suppresses the local
crack-growth rate most strongly at low Kmax (~29% rate reduction at
Kmax=15, `10^0.125 = 1.334`) and gets much weaker at high Kmax (~6% rate
reduction at Kmax=21, `10^0.0265 = 1.063`) -- consistent with the
mechanism this review proposed: faster opening renewals at high Kmax
reduce the fraction of the cleavage-hazard-weighted history available for
contact-conditioned bond formation, even though (per Section 3) some
compressive contact is present in essentially every interval at every
load studied.

## 6. Verification

`scripts/verify_v2_slope_exposure_continuation.py`: depends ONLY on
tracked artifacts (this branch's own, the slope screen's, and the parent
pilot's -- never any gitignored `runs/...` file). Independently rebuilds
the bare A_NATIVE engine, reproduces `frozen_configuration_sha256` from
scratch, confirms the completion trajectory's config hash and
classification label, re-derives all 6 `S_h` values and both seeds'
`delta_m` directly from the tracked ledgers, and confirms no
unauthorized-activity flag was set. **`overall_pass: true`**. All 183
`crack_rebonding` tests pass.

## 7. What this does and does not authorize

This one-trajectory completion and its analysis do **not** constitute
Part X, a developed Paris-law campaign, or a production-line merge. The
result is a *minimal local slope screen* at three points (15/18/21
MPa*sqrt(m)), two seeds, one regime (reversible, shown equivalent to
persistent for this purpose in the parent branch) -- sufficient to
demonstrate a reproducible sign and approximate magnitude for a local
slope correction, not to characterize a full Paris curve, a different `R`,
a different frequency, or passivation/repassivation behavior. No further
seeds, no additional Kmax points, no frequency ablation, and no scope
expansion were run or are proposed here.
