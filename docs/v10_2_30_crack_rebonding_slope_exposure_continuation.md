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

`scripts/analyze_v2_slope_exposure_continuation.py` (v2, hardened per
review -- see Section 5 for what changed and why):

| Kmax (MPa*sqrt(m)) | S_h, seed 1720 | S_h, seed 1001723 |
|---|---|---|
| 15 | -0.125007 | -0.122650 |
| 18 | -0.042846 | -0.042051 |
| 21 | -0.026521 | -0.026123 |

`|S_h|` falls **monotonically** with Kmax for both seeds, and the two
seeds agree closely at every point (largest relative difference ~2%,
despite Kmax=21's wildly different *complete-excursion* counts: 2 vs 0 --
see Section 5.3 on why that count is the wrong metric to judge this by).

**Corrected physical units** (a reporting error in the prior draft is
fixed here: `10^0.125 = 1.334` is the *waiting-time multiplier*, not a
"29% rate reduction"):

| Kmax | rate ratio `10^S_h` | rate reduction `1-10^S_h` | waiting-time increase `10^-S_h - 1` |
|---|---|---|---|
| 15, seed 1720 | 0.7499 | **25.01%** | **33.35%** |
| 15, seed 1001723 | 0.7540 | **24.60%** | **32.63%** |
| 18, seed 1720 | 0.9061 | **9.39%** | **10.37%** |
| 18, seed 1001723 | 0.9077 | **9.23%** | **10.17%** |
| 21, seed 1720 | 0.9408 | **5.92%** | **6.30%** |
| 21, seed 1001723 | 0.9416 | **5.84%** | **6.20%** |

`da/dN` in m/cycle (`= g_m_per_s / f`, `f=1000 Hz`) is recorded alongside
the m/s rate in every `per_pair[...]["windows"][...]` record for both
completeness and direct comparability with conventional Paris-law units;
since every trajectory here uses the same frequency, the m/s ratio and the
m/cycle ratio are identical to `S_h`, so no reported conclusion changes --
this is a units/reporting addition, not a re-derivation.

```
delta_m (least-squares, 3-point, all-event window):
    seed 1720 = 0.6849   seed 1001723 = 0.6713
secant(15->18): seed 1720 = 1.0376   seed 1001723 = 1.0179
secant(18->21): seed 1720 = 0.2439   seed 1001723 = 0.2379
```

**Curvature is substantial**: the 15->18 secant (~1.03) is roughly 4x the
18->21 secant (~0.24) for both seeds. The single 3-point least-squares
`delta_m` is a valid finite-window average, but the more precise physical
statement is that rebonding produces **strong low-K suppression that
weakens rapidly with Kmax**, not a uniform Paris-exponent shift -- an
apparent steepening/onset-shift over the sampled window rather than a
constant slope correction. The 18->21 secant (~0.24) already sits just
under the 0.25 steepening gate on its own, so the LOW-K interval is what
mainly drives the overall-window classification.

### 4.1 Persistence check: does this survive into a "late" window?

Seven events from an initially empty bonded wake can include wake
establishment and early-occupancy transients, not yet a developed
moving-frame response. `analyze_v2_slope_exposure_continuation.py` now
also computes `S_h` over a `post_first` window (events 1-6) and a `late`
window (events 3-6, prospectively defined) for every (Kmax, seed) pair:

```
late-window delta_m: seed 1720 = 0.6895   seed 1001723 = 0.6860
  (all-window was:   seed 1720 = 0.6849   seed 1001723 = 0.6713)
```

The late-window classification is **identical**
(`REBONDING_STEEPENS_LOCAL_RESPONSE`) and `delta_m` changes by less than
0.02 for either seed. The result is not an artifact of the wake's initial
transient -- it persists, essentially unchanged, when the first event
(closest to the empty-wake initial condition) is dropped and only events
3-6 are used.

## 5. Classifier repair and classification

### 5.1 The bug found on review

`classify_slope_effect_v2`'s original `exposure_driven` check was
computed *only inside* the branch already guarded by
`seeds_disagree_in_sign or seeds_differ_a_lot` -- so it could never
independently change the classification (dead code), and it compared only
the *binary complete-excursion count* (2 vs 0 at Kmax=21), not a
continuous exposure/action quantity. Both are fixed in
`crack_rebonding_slope_exposure_continuation_v10230.py`'s v2:

- `seeds_disagree_in_sign`, `seeds_differ_a_lot`, and
  `exposure_disparity_exceeds_threshold` are now three genuinely
  independent conditions -- any one alone can trigger
  `REBONDING_PHASE_EXPOSURE_SENSITIVE`, none nested inside another.
- Exposure disparity is now measured via `total_negative_K_contact_time_s`
  (integrated compressive-contact **duration**, continuous), not the
  complete-lobe count.

### 5.2 What the continuous metric actually shows

```
total_negative_K_contact_time_s at Kmax=21 (finite-cohesion trajectory):
    seed 1720:    0.0628 s
    seed 1001723: 0.0760 s
    ratio: 1.21x  (threshold for "disparity": 2.0x -- NOT exceeded)
```

Despite the complete-excursion counts differing starkly (2 vs 0), the
*continuous* contact-time exposure differs by only ~21% between seeds --
nowhere near a 2x disparity. This is the metric review identified as more
physically meaningful (partial excursions still contribute real contact
duration and real formation opportunity), and by this corrected metric the
two seeds' exposure histories are actually quite similar, not starkly
different.

### 5.3 Result: `REBONDING_STEEPENS_LOCAL_RESPONSE` (confirmed, non-tautologically)

Per the fixed, independent gate set: not `RATE_OFFSET_LIKE` (endpoint span
~0.098 decade, far above the 0.01 threshold); `seeds_disagree_in_sign` is
`False`; `seeds_differ_a_lot` is `False` (diff 0.0136 vs 0.25 gate);
`exposure_disparity_exceeds_threshold` is `False` (1.21x vs 2.0x
threshold) -- none of the three `PHASE_EXPOSURE_SENSITIVE` triggers fire,
so the result falls through to `delta_m >= 0.25` for both seeds ->
`REBONDING_STEEPENS_LOCAL_RESPONSE`. Unlike the prior draft, this is no
longer "not exposure-driven by construction of a tautological check" --
it is "checked against a continuous exposure metric and found not
exposure-driven."

### 5.4 Load-dependence decomposition diagnostic (no new physics)

Does the steepening come mainly from the *fixed absolute* cohesive scale
becoming a smaller *fraction* of a larger `Kmax` (`Pi_K =
K_rebond_max/Kmax` falling with load, acting on an unchanged absolute
shielding), or from a genuinely load-dependent bonded-occupancy/contact-
exposure mechanism? Comparing the already-tracked, already-measured
action-weighted `K_rebond` the finite-cohesion trajectory actually reaches
at each `Kmax` (no counterfactual re-simulation):

```
seed 1720:    varies 15.2% across the Kmax=15/18/21 grid
seed 1001723: varies  0.0% across the Kmax=15/18/21 grid
```

Both are well inside a 30% "roughly constant" band. This is consistent
with the steepening being explained **mainly by the fixed absolute
shielding scale becoming a smaller fraction of a larger `Kmax`**, not by a
strongly load-dependent bonded-occupancy mechanism -- though this
comparison is a diagnostic on already-measured quantities, not a
controlled counterfactual (which would require a new, fixed-`K_rebond`
simulation and is out of scope here).

### 5.5 Scientific qualifiers

The single machine label is retained, decorated with explicit qualifiers
recorded in `slope_exposure_continuation_decision.json["scientific_
qualifiers"]`:

```
REBONDING_STEEPENS_LOCAL_RESPONSE
  + CONTACT_GATED_REBONDING_FINITE_WINDOW_STEEPENING_DEMONSTRATED
  + RELATIVE_SLOPE_CORRECTION_POSITIVE_OVER_KMAX_15_TO_21
  + STRONG_CURVATURE_OR_ONSET_SHIFT_PRESENT
  + DEVELOPED_PARIS_SLOPE_NOT_YET_QUALIFIED
  + PHASE_EXPOSURE_AND_ABSOLUTE_SHIELDING_CONTRIBUTIONS_NOT_YET_DECOMPOSED
  + PHYSICAL_CHEMISTRY_REALISM_NOT_CLAIMED
```

In prose: **contact-gated rebonding steepens the finite-window local
crack-growth response primarily by suppressing the low-K rate; the
correction weakens substantially above the intermediate load point.** The
new trajectory and the fixed classifier corroborate this at both a
different seed and, within each seed, a late/developed-leaning event
window -- but Section 5.4's decomposition is a diagnostic, not a
quantitative separation, and the result applies to the **reversible**
preset specifically (shown equivalent to persistent *at Kmax=18 only*
in the parent branch; not independently re-tested across 15-21 here).

## 6. Seed-metadata fix (documented, not a data rewrite)

`run_trajectory`'s returned `"seed"` field was hardcoded to the module
constant `SEED` (1720) regardless of which seed was actually configured
on the engine before the call -- confirmed harmless (no gate or
comparison anywhere in this campaign ever read this nested field for a
decision; e.g. the second seed's own `second_seed_event_ledger.json` has
the correct seed 1001723 at its top level, with the mislabeled `1720`
only in each trajectory's own nested `"seed"` field). Fixed via an
explicit `hazard_rng_seed` parameter recorded verbatim going forward. No
prior physical result was rewritten; this section documents the
mislabeling for anyone reading the older nested fields directly.

## 7. Verification

`scripts/verify_v2_slope_exposure_continuation.py` (v2, hardened):
depends ONLY on tracked artifacts (this branch's own, the slope screen's,
and the parent pilot's -- never any gitignored `runs/...` file).
Independently rebuilds the bare A_NATIVE engine, reproduces
`frozen_configuration_sha256` from scratch, re-derives all 6 `S_h` values
and both seeds' `delta_m` (all-window) directly from the tracked ledgers,
and **reruns `classify_slope_effect_v2` and requires exact equality**
(not membership in an allowed set) with the saved classification and its
full diagnostics dict. Additionally hard-checks, for **all six**
zero/finite pairs: accepted-event-length identity, event-index-by-index
`hazard_threshold_action` sequence identity (the common-random-numbers
design this whole campaign relies on -- confirmed to match exactly, 7 of
7 events, at every one of the six pairs), and certified
(`all_bulk_action_qualified`) status on every admitted event. **60/60
checks pass, `overall_pass: true`.** All 183 `crack_rebonding` tests pass.

## 8. What this does and does not authorize

This one-trajectory completion and its analysis do **not** constitute
Part X, a developed Paris-law campaign, or a production-line merge. The
result is a *minimal, finite-window local slope screen* at three points
(15/18/21 MPa*sqrt(m)), two seeds, one regime (reversible, shown
equivalent to persistent *at Kmax=18* in the parent branch) -- sufficient
to demonstrate a reproducible sign, approximate magnitude, and
late-window persistence for a local slope correction, not to characterize
a full Paris curve, a different `R`, a different frequency, or
passivation/repassivation behavior. No further seeds, no additional Kmax
points, no frequency ablation, and no scope expansion were run or are
proposed here.

A **developed-response confirmation** (fresh reversible zero/finite pairs
at Kmax=15/18/21, seed 1720 first, run to the ~100 micrometre / 18-event
developed-growth and stationarity standard used by the qualified A_NATIVE
campaigns, with seed 1001723 replicated only if the late-window slope
correction remains positive and exceeds 0.25) is a scientifically
reasonable next step suggested on review -- but it is new physics, is not
authorized by this document, and was not run.
