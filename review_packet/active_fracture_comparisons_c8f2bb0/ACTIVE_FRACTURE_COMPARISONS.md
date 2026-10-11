# Active fracture comparisons

Date: 2026-10-11  
Scope: bounded original-model calculations authorized after physical completion packet v4  
Main calculation branch: `codex/shared-fracture-fatigue-options-v1`  
FEM/CZM companion branch: `codex/current-pf-companion`

## Scientific result

### Hot-path controller and event clock

The retained input is combined checkpoint generation
`85f8ef062b9a298f1f2a92fb` at accepted step 1701, 202 committed geometry
events, and 998.191926254 µm projected extension. Its three payload hashes were
verified before copying it into isolated restart roots. No source checkpoint
pointer or payload was changed.

The nominal increments are 0.2 µm and 8.4 s, so the prescribed opening rate is
23.8095238 nm/s. The adaptive controller scales displacement and physical time
by the same accepted fraction and therefore preserves that loading rate. It
restores the full driver, FEM, material, crack path, source ledgers, pending
hazard action, threshold, and RNG state on retry and restart.

Changing only the minimum adaptive fraction from `1e-8` to `1e-9` and `1e-10`
did not change the calculation. The accepted fraction remained
`1.83196570306e-8`, above all three floors. Native step, front, crack-path,
geometry-event, wake-advance, and summary files are byte-identical across the
three runs. Event 203 occurred at 1699.838679147 s, 40.4723495035 µm opening,
110.872783699 MPa√m native K_J, and advanced 6.733837222 µm to
1004.925763476 µm projected extension. The floor did not cause the rapid event.

The active resolution control was the dimensionless target for accepted
hazard-action increments. The checkpoint retained action 1.23108095246 against
its own mark 1.46557256245. Refining the action target while fixing the floor at
`1e-10` gave:

| action target | new accepted steps to event | time after checkpoint (ns) | event opening (µm) | event K_J (MPa√m) | event N_em |
|---:|---:|---:|---:|---:|---:|
| 0.150000 | 2 | 307.750 | 40.4723495035 | 110.872783699 | 40623.555 |
| 0.075000 | 4 | 307.750 | 40.4723495035 | 110.872783699 | 40466.589 |
| 0.037500 | 7 | 269.279 | 40.4723495026 | 110.872783697 | 40517.423 |
| 0.018750 | 13 | 250.043 | 40.4723495021 | 110.872783695 | 40907.399 |
| 0.009375 | 25 | 240.425 | 40.4723495019 | 110.872783695 | 40571.334 |

The two finest timing changes halve (19.236 then 9.618 ns); the finest
accepted step is 9.618 ns. Event geometry and event-length/mark identity are
byte-identical at every target. The event K_J range is `3.96e-11` relative;
N_em spans 1.08% over all targets and 0.96% over the three finest targets.
Thus the event load, mechanics, and geometry are numerically stable at the
tested scope. Exact nanosecond timing and emitted-state values retain their
reported resolution. The rapid sequence is a conditional quasistatic/kinetic
model prediction, not a measurement or validation of dynamic crack velocity.

### Full FEM/CZM comparison and exact-condition PF anchor

The companion correction preserves an explicitly supplied zero-degree crystal
orientation instead of replacing falsey zero by the 45-degree default. The
focused companion suite passed (21 tests), and the corrected launcher and
matrix are committed at `8a86519` (orientation correction `cbb6e51`).

The original full FEM/adaptive-CZM run executed 500 accepted native increments
at Peak-0242980, 1000 K, theta=0 degrees, and seed 8666. It reached the
predeclared finite endpoint of 100 µm opening and 4200 s with no cohesive or
fracture advance. Final native values are K_J = 3.716348 MPa√m,
J = 31.044982 J/m², tip stress = 1.394 GPa, N_em = 1.40021e-10, and crack
length = 0.500 mm. Full-field local Peierls-Taylor state evolved for 1000 bulk
updates; spatial density transport is intentionally absent. Four native state
snapshots and the terminal bulk array are retained. The directional traction
observer is reliable but its phase error is -2.15878 degrees, 0.15878 degrees
outside the strict +/-2 degree diagnostic tolerance; this caveat does not
change the accepted no-event state.

The separately executed current PF anchor uses the same selected row,
temperature, theta, seed, nominal dU/dt, and mesh-count/tip-resolution controls.
Its sharp-wake model accepted the first 5 µm event at step 161, 32.2 µm opening,
1352.4 s, and native K_J = 54.687864 MPa√m, then a second event to exactly
10 µm extension. The PF and FEM results are not constitutive parity: their
geometries, sharp-wake versus adaptive-CZM crack representations, accommodation
resolution, and native force/J normalization differ. Reaction is retained as
N in FEM and is not relabeled as force per thickness. No transfer coefficient
was fitted, and native K/J values were not renamed across solvers.

### Coupled cavity continuation

At the immutable 5.9 s ramp endpoint, the enabled candidate-specific operation
uses the qualified complete-ligament topology energy without substituting it
for invalid local isolated-tip J. The solid ligament is 7.358229034 µm; level-3
and level-4 values are 44.3878234 and 40.4943938 J/m². Their 8.771% difference
passes the existing mechanical criterion but is not a kinetic-rate error bound.

The exact native frozen-state query gives:

| arm | effective rate (s^-1) | retained-mark crossing time (s) |
|---|---:|---:|
| suppressed | 7.72104e-3 | 45.3559892 |
| cavity enabled | 5.39093e-58 | 2.23393e56 |

These are frozen-state/frozen-rate quantities only. The enabled finite-ligament
observation enters the existing directional kinetic row as a distinct complete-
topology energy source; it neither replaces local J nor adds a second energy
gate.

Two narrow dispatch/numerical defects were corrected and tested. A suppressed
state with no cavity now preserves its ordinary native directional row instead
of dereferencing nonexistent cavity geometry. Paired rollback/bisection now
treats vacancy Picard nonconvergence like the existing fail-closed concentration
bound while keeping both arms on one accepted time partition.

The predeclared continuation is an explicit 10 µs fixed-opening dwell after the
historical 1 µm/s ramp completed its declared 6.300 µm endpoint. It is not
misreported as continued ramping. The corrected pair passed directional
dispatch, then rejected the full interval on Picard convergence. After bounded
common bisection it reached a distinct enabled-arm capability stop:
`resolved cavity loop source recovery failed qualification`. No common segment
was accepted and no physical time, loading, RNG, transport, or topology state
was published. Consequently the evolving exposure remains unresolved; the
tiny enabled frozen rate cannot be promoted to a general no-void-effect claim.

### Independent-seed PF check

Seed 91733 was executed at both active anchors with accommodation enabled and
with the original full-spatial opening-only control. Each run retained the
same physical loading, material/orientation, numerical controls, correlated
threshold/event-length law, and 10 µm target guard used for seed 1733.

| anchor | arm | K first, seed 1733 (MPa√m) | K first, seed 91733 (MPa√m) | seed change | events / extension, seed 91733 |
|---|---|---:|---:|---:|---:|
| DBTT 1200 K | accommodation | 52.37574 | 52.92102 | +1.04% | 3 / 14.4631 µm |
| DBTT 1200 K | opening only | 20.93593 | 21.05050 | +0.55% | 3 / 14.4631 µm |
| Peak 900 K | accommodation | 54.90392 | 55.70689 | +1.46% | 3 / 14.4631 µm |
| Peak 900 K | opening only | 20.82369 | 21.01520 | +0.92% | 3 / 14.4631 µm |

The accommodation-minus-opening first-event gaps are 31.4398 and 31.8705
MPa√m for DBTT seeds 1733 and 91733, and 34.0802 and 34.6917 MPa√m for Peak.
Within seed 91733, all four geometry-event files are byte-identical because
they correctly share the same marks and threshold-scaled event-length stream.
Across seeds, the event count and lengths differ. Two seeds are a pilot
variability check only; no confidence interval, tail estimate, or completed
replicate uncertainty is claimed.

## Identity, preservation, and status

- Current main producer commit: `c8f2bb0` plus this report-only descendant.
- FEM/CZM producer commit: `8a86519`; corrected orientation source commit:
  `cbb6e51`.
- Main remote is `origin` at `ukaiiaku-maker/PF-fracture-fatigue.git`; the
  working branch has no configured upstream. No push was attempted.
- The companion branch is local and has no reported remote update. No default
  branch was changed.
- Calculation source is clean. Raw evidence directories remain intentionally
  untracked; unrelated retained evidence was not added or removed.
- `/Volumes/Data` and the backup root have different device IDs (16777237 and
  16777234). At preservation time they reported 16 GiB and 90 GiB available.
- The corrected FEM archive is independently copied and verified as SHA-256
  `121fe0f7cd284948bb802cd59de065b6b95e5770d7bddd368aeae8b7298cacee`.

## Next finite campaign

1. Diagnose and qualify the resolved cavity-loop source recovery on the exact
   retained enabled mesh; then rerun the unchanged 10 µs paired dwell. Do not
   alter physical rates, cavity ownership, or the 10 ns floor to obtain passage.
2. Only if a quantitative nanosecond event time is needed, add one action-target
   level at 0.0046875 from the same hot checkpoint; the present event-load and
   geometry conclusion needs no new 1 mm run.
3. Add further PF seeds only after choosing an ensemble claim and sample-size
   design. The two-seed pilot is not a statistical stopping rule.
4. Treat the FEM/CZM finite no-event endpoint and PF event result as independent
   native-model evidence unless an author-approved common specimen/normalization
   study is defined. Do not infer or fit a conversion from this pair.

V3.6 active-history qualification and combined branching-plus-cavity cases
remain separate workstreams and were neither erased nor made gates for these
monotonic results.
