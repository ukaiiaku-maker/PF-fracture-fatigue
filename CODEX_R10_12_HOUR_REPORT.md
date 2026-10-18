# R10 autonomous joint CZM/EXP_floor campaign

## Decision

The strict Paris target has not been claimed. The campaign found physically
recurrent smooth-mixed cohesive growth over rates from the HCF range to fast
growth, but every completed adjacent-slope sequence so far is much steeper and
less regular than the required 3.4--4.6 band. Censored, unsupported, burst,
cascade, and resource-stopped cases remain separate throughout the exported
tables.

The strongest HCF material is `R10F_V2_22871__MS`. At 300 K, R=0.1 and
1000 Hz, factor 1.01 gives 61 physical one-micrometre advances and a developed
20--60 micrometre rate of 5.499977510625358e-12 m/cycle. The developed window
runs from cycle 4,161,735.0197 to 11,616,310.9558. Its first event is at cycle
542,018.9866; positive inter-event waits span 4,540.0851 to 803,733.0207
cycles. Exact-versus-accelerated initial action and state overlap is bit exact
for this trajectory. A no-refit six-point amplitude ladder is complete.  The
clean same-seed, same-resolution frozen-state control also crossed the window
at 5.5000253968e-12 m/cycle, only 8.71e-6 higher in relative terms than the
evolving rate, and therefore closes the long-exposure fatigue attribution as
opening dominated at this protocol.

## Constitutive and numerical scope

- New opening arms use the existing smooth-exponential or bilinear athermal/
  mixed CZM owners. No new opening family, empirical cycle damage, fitted
  endpoint law, per-cell delay, or numerical advance cap was introduced.
- Emission, Peierls transport, Taylor release, activation entropy, persistent
  sources, return motion, backstress, blunting, wake/history, contact, and
  geometry remain owned by the current EXP_floor full-state backend.
- Every amplitude is an independently initialized fixed-applied-K history
  unless explicitly labeled as a restart continuation of the same protocol.
- Developed rates use the established first-event-at-or-beyond 20 and 60
  micrometre window. No rate is imputed to a censor or unsupported case.
- The HCF restart identity now records and checks phase resolution in addition
  to material, load, temperature, R, frequency, and plastic-feedback mode.

## Scientific findings

### Best-material table

The table uses complete consumed-row hashes, not descriptive IDs, as material
identity. Full-precision cohesive and EXP_floor vectors are exported in
`runs/r10_12h_autonomous/analysis/finalist_parameterizations.csv` and the two
`materialized_round_*.csv` registries; the portable manifest hashes those
files byte-for-byte.

| Complete material | Complete-row SHA-256 | Current evidence | Joint status |
|---|---|---|---|
| `R10F_V2_22871__MS` | `4c3ecdf86c50d3ec969d8c114fa8e3215598450b0a406aeff3cd1e5f69354d75` | six developed HCF amplitudes from 5.500e-12 to 5.654e-9 m/cycle; matched frozen fatigue rate within 8.71 ppm; plastic-state-created Peak-T fracture | strongest HCF frontier; five adjacent slopes 146.67--199.71 fail |
| `R10F_V2_17780__MS` | `a332c34e4d5af32f312523893c24db3296516eb211c8239df20dfd5a0b820cc5` | recurrent multi-amplitude fatigue; seven-temperature Weak-T span 1.209; R/f holdouts | qualified joint near-frontier, failed slope |
| `R10F_V2_28475__MS` | `0aaa671d1b7c714c23b8ad07f23c84fa2f8d3246c5dd3cbe0a919190e316b1da` | 1.154e-9--2.225e-8 m/cycle, seven amplitudes and seed bounds | HCF-rate near-frontier, failed slope; high-T fracture unsupported |
| `R10F_V2_11566__MS` | `883c76bd8081c89c8efae6de6da935826e91de95307bb4df23b85666a12ecbfe` | one-cell full-state versus five-cell frozen burst and 1e12-cycle arrest at factor 1.06; phase-sensitive factor-1.10 recurrence | consequential-plasticity frontier; recurrent rate not phase-qualified |
| `R10F_12H_FC_07172_VP_27940__MS` | `d987e245733fb89f38a85e7678aafdc33f99f6c635840eada5f48b42e3caa245` | replicated finite bursts and arrest versus frozen cascades; strong closing response | mechanism boundary, not a fatigue curve |
| `R10F_V2_00116__MS` | `6e28416309e81ae3593dd818514b481b64506ea26eb964fef25f1feb87dbb38f` | three developed rates from 8.090e-11 to 3.783e-10 m/cycle | HCF-rate near-frontier; adjacent slopes 95.06/215.48 fail |
| `R10F_V2_01796__MS` | `cfa9d6949a406ae5d05e5a9171c9d56d85d3a6dd6404f94f4345ed55b503bdb0` | intermediate recurrent rates and bounded thermal response | opening-controlled comparison; failed slope/full thermal class |

No row is described as globally optimal or experimentally calibrated.

### Slow recurrent HCF and fracture response

`R10F_V2_22871__MS` is the slow-growth frontier. Its full-state monotonic
fracture roots at 300--1200 K are 31.1, 50.5, 48.9, 44.4, 40.1, 36.1, and
32.3 MPa sqrt(m). The matched frozen-plastic roots are 31.1, 29.2, 27.7,
26.1, 24.5, 23.0, and 21.4 MPa sqrt(m). Thus the Peak-T response is created by
evolved EXP_floor feedback rather than intrinsic cohesive hardening. Early
same-seed, same-resolution full/frozen fatigue event times differ by 164 cycles
(0.0303 percent) at the first event.  The absolute offset remains only
164--319 cycles and the relative difference decreases to 0.00329 percent by
event 55.  The completed frozen developed window gives 5.5000253968e-12
m/cycle versus 5.4999775106e-12 m/cycle for the evolving path (an 8.71-ppm
relative difference).  The opening-controlled fatigue attribution therefore
remains valid through the developed window even though the same material's
thermal-fracture class is created by plastic-state evolution.

The first completed new amplitude point, factor 1.05, traverses the same
first-event-at-or-beyond 20/60 micrometre window from cycle 3,282.05879 to
10,534.05336, giving 5.6536170e-9 m/cycle.  Its positive waits span
2.011--934.014 cycles.  The
factor-1.01-to-1.05 endpoint secant is about 178, already excluding a hidden
moderate four-slope response across that broad interval.  Factor 1.04 has also
completed the developed window, from cycle 20,542.01894 to 67,063.03504, at
8.8132211e-10 m/cycle.  The directly adjacent factor-1.04/1.05 local slope is
194.23, independently rejecting the target band.  Factors 1.02 and 1.03 have
also matured at 2.7381964e-11 and 1.9214544e-10 m/cycle; the inserted factor
1.015 point gives 1.1348203e-11 m/cycle.  The resulting six-point factors
1.01/1.015/1.02/1.03/1.04/1.05 have five adjacent slopes 146.67, 179.25,
199.71, 157.65, and 194.23.  This is a connected three-decade HCF response and
a decisive failed moderate-slope test, not an interpolation through censors.
Nonoverlapping 60--100 micrometre rates at factors 1.04 and 1.05 are
1.0075324e-9 and 7.0710278e-9 m/cycle, 14.3 and 25.1 percent above their
respective developed windows; recurrence persists but remains mildly
path-conditioned.

At 450 K, unchanged-material frequency controls at 10, 100, and 1000 Hz have
no physical advance through 1e6 cycles at fixed Kmax values of 29 and
49 MPa sqrt(m).  The intermediate 40 MPa sqrt(m) histories enter the
unsupported post-escape mixed transit and are retained as unresolved rather
than zero-rate observations.  Consequently these controls bound a
temperature-dependent accessible interval but do not support a 450-K
frequency-rate curve.

The developed factor-1.05 protocol panel at 300 K uses the same complete
material and fixed Kmax=32.5455357059 MPa sqrt(m), without refitting:

| Protocol | Developed rate (m/cycle) | Native interpretation |
|---|---:|---|
| R=0.1, 10 Hz | 5.6893127e-7 | 120-cell developed trajectory |
| R=0.1, 100 Hz | 5.6549816e-8 | 61-event developed prefix |
| R=0.1, 1000 Hz | 5.6536170e-9 | 120-cell developed trajectory |
| R=0.3, 1000 Hz | 7.9149896e-9 | 62-event developed prefix |
| R=0.5, 1000 Hz | 1.1057230e-8 | 61-event developed prefix |

The three frequency rates become 5.6893e-6, 5.6550e-6, and 5.6536e-6 m/s
after multiplying by frequency (0.63-percent full spread).  This nearly exact
elapsed-time collapse is the expected signature of the retained mixed
first-passage owner, not an empirical per-cycle damage law.  Increasing R from
0.1 to 0.3 and 0.5 raises the per-cycle rate by 40.0 and 95.6 percent at fixed
Kmax.  Same-Kmax 600-K and 900-K comparisons remain unresolved opening-physics
cases (a 120-cell post-escape cascade and a zero-accepted-cycle unsupported
state, respectively), so no thermal rate is manufactured from them.

The final uncertainty block remains deliberately below the developed-rate
threshold: the factor-1.04 alternate seed has 36 advances in 37,610.95 cycles,
the factor-1.05 same-seed 32-phase refinement has 25 advances in 7,335.99
cycles, and the factor-1.05 alternate seed has 42 advances in 6,988.06 cycles.
All three are `RESOURCE_STOP` prefixes with exact restart checkpoints.  They
demonstrate continued recurrence but contribute no developed rate or slope.

### Weak-T recurrent control and protocol holdouts

`R10F_V2_17780__MS` has a qualified seven-temperature fracture span of 1.209
(33.5--40.5 MPa sqrt(m)). It has developed 1000-Hz rates at multiple amplitudes,
but replicate-aware local slopes are approximately 123--148 rather than four.
At fixed Kmax=35.38547195021402 MPa sqrt(m), developed rates change from
2.128e-5 at 10 Hz to 2.905e-6 at 100 Hz, about 2.32e-7 at 1000 Hz, and
2.412e-8 m/cycle at 10 kHz. The supplementary 10-kHz point is reported as such;
the required comparison is 10/100/1000 Hz. R=0.3 and R=0.5 histories are also
retained at unchanged material and Kmax.
At the three independently repeated reference amplitudes, the observed
maximum/minimum rate ratios are 1.274, 1.346, and 1.389.  These are finite
stream ranges, not confidence intervals; every stream and censor is retained
in `replicate_rate_bounds.csv`.

The unchanged-material `17780` protocol panel is:

| Protocol change at fixed Kmax=35.38547 MPa sqrt(m) | Native outcome |
|---|---|
| 300 K, R=0.1, 10/100/1000 Hz | developed rates 2.128e-5 / 2.905e-6 / median 2.316e-7 m/cycle |
| 300 K, R=0.3 / 0.5, 1000 Hz | developed prefixes 3.944e-7 / 4.408e-7 m/cycle; later resource stop retained |
| 600 / 900 K, R=0.1, 1000 Hz | 120-cell target-truncated cascades, not Paris rates |
| 1200 K, R=0.1, 1000 Hz | unsupported mixed first-passage approximation; unresolved |

The temperature, load-ratio, and frequency rows use the same complete material
without refitting.  The separately retained 10 kHz result is supplemental.

### Near-frontier recurrent and transition controls

- `R10F_V2_28475__MS` gives seven supported developed rates from 1.154e-9 to
  2.225e-8 m/cycle. Independent low/middle/high streams preserve substantial
  stochastic scatter. Its adjacent slopes remain far outside the target, and
  its evolving/frozen event times are bit-identical through 120 events at the
  matched factor-1.01 protocol, identifying opening-dominated fatigue there.
  Its three-stream low/middle/high observed rate ratios are 1.271, 1.273, and
  1.110, again reported as observed ranges rather than inferred confidence
  bounds.
- `R10F_V2_17444__MS` gives a six-point recurrent fast boundary. Its adjacent
  slopes range from about -1271 to +255 and therefore reject a moderate Paris
  interval despite genuine recurrent remapping.
- `R10F_V2_01796__MS` fills an intermediate recurrent control. Factor 1.01
  reaches a developed 9.315e-7 m/cycle; factor 1.03 reaches 8.285e-6 m/cycle;
  factor 1.06 is a target-truncated cascade. Its evolving fracture root stays
  21.3 MPa sqrt(m) through 600 K, rises to 23.1 by 900--1050 K, and becomes an
  unsupported mixed crossover at 1200 K, so it is not assigned a full seven-
  temperature class.
- `R10F_V2_00116__MS` validates the high-barrier acquisition direction with
  developed rates of 8.0902798e-11, 1.2985738e-10, and 3.7834404e-10 m/cycle
  at factors 1.00001, 1.005, and 1.01.  The directly adjacent local slopes are
  95.06 and 215.48, so rare opening was recovered but not the desired broad
  moderate response.

### Consequential plastic-state arrest

The coherent active-plastic `R10F_12H_FC_07172_VP_27940__MS` material
reproducibly changes from finite bursts to arrest and then to a same-excursion
cascade over a narrow amplitude bracket. At factor 1.04 it gives 2--6 events
across streams before 1e6-cycle arrest; at factor 1.05 it gives 33--60 events
before the same exposure censor. Matched frozen-state calculations cascade,
so the arrest is caused by evolving plastic feedback. A same-material reduced
2-D transfer restores the developed state bit exactly and measures 5.447 GPa
of closing traction; even the no-feedback interface remains ramp-censored in
that reduced geometry, so this is a bounded feedback comparison rather than a
spatial propagation claim.

The new high-barrier acquisition found `R10F_V2_11566__MS`. At factor 1.06 it
opens once at cycle 0.03644, evolves from zero mobile population to 2564.51,
raises tip radius from 1.0 to 1.41579 micrometres, develops 1.399 GPa source
backstress, changes from bonded-metastable to bonded-stable, and then remains
arrested through the physical 1e12-cycle censor.  The exact final HCF checkpoint
was consumed at its one-cell advanced geometry by the reduced 2-D interface:
state restore was bit exact, the retained operator supplied 1.3068 GPa of
closing traction, and the factor-100-delta0 terminal opening fell from
9.564e-11 m with feedback disabled to 6.256e-11 m with feedback active.  Both
paths remain ramp-limit censored, so this is a 34.6-percent same-state opening
reduction, not a spatial propagation threshold.  The same-seed, same-phase
frozen-initial-plastic factor-1.06 control makes five immediate advances before
arresting through the same 1e12-cycle exposure.  Evolving EXP_floor state thus
suppresses four otherwise accessible opening transactions and establishes the
one-cell arrest as a consequential feedback effect; neither history is a
developed Paris interval.  The matched monotonic fracture control resolves
300--750 K: evolving/frozen roots are 21.9/19.2, 20.8/17.0, 21.0/14.7, and
16.4/12.4 MPa sqrt(m).  Evolving plasticity therefore raises the resolved roots
by 14--43 percent, but 900--1200 K encounters the unsupported mixed crossover
and the material is not assigned a complete Weak-T, Peak-T, or DBTT class.
At factor 1.10 the same evolving material reactivates after a 1,424-cycle late
wait and then reaches the established 20--60 micrometre developed window.  The
first-event-at-or-beyond window runs from cycle 2,440.05896 to 4,675.97360 and
gives 1.8337015e-8 m/cycle; positive waits span 1.071e-4--1,424 cycles.  This sharp
factor-1.06 arrest/factor-1.10 recurrent-growth transition is retained as a
state-sensitive mechanism bracket, not promoted to a moderate-slope claim.  A
nonoverlapping 60--100 micrometre window gives 2.4478776e-8 m/cycle over
1,634.069 cycles, 33.5 percent above the 20--60 micrometre rate and therefore
evidence of recurrent but still path-conditioned acceleration rather than a
stationary rate.

This factor-1.10 rate is not promoted as numerically qualified.  A same-seed
32-phase refinement matches the first two event times bit exactly, but at the
100-cycle checkpoint its stress, barrier, mobile population, backstress, and
tip radius differ from the 16-phase path.  It then reaches the full 1e12-cycle
censor with only those two advances and changes to `BONDED_STABLE`; the
16-phase path's third event was at cycle 136.973 and ultimately reached 120
advances.  This is a decision-changing phase-resolution sensitivity, not
stochastic scatter.
The 16-phase developed and late-window rates remain useful diagnostic
measurements but are excluded from the qualified recurrent portfolio unless a
subsequent refinement resolves the discrepancy.

## Search coverage and acquisition history

The campaign met or exceeded each prospective fidelity allocation:

| Fidelity | Requested allocation | Executed distinct coverage |
|---|---:|---:|
| cheap current-law physical tuples | toward 32,768 | 32,768 allocated; 23,691 domain-valid; 94,764 opening arms |
| coupled state-informed physical vectors | approximately 512 | 554 vectors; 9,350 cyclic-state records |
| current full-state 1-D physical vectors | 64, rising toward 96 | 105 vectors; 112 execution identities; 213 results |
| diverse deepened materials | up to six | seven principal rows plus retained boundary controls |

Of the 105 native physical-vector source identities, 103 are not among the
four supplied baseline archive sources.  The native work spans nine numbered
execution rounds plus targeted discrepancy, transition, mechanism, and HCF
follow-ups.  At the stable numerical freeze the production archive contained
71 HCF-tested physical vectors, 182 finalized HCF results, 12 distinct
protocols, 56 effective developed-rate records, 3.002478927e12 accepted
cycles, and 3.002802954e9 simulated seconds.  Those trajectories commit 9,395
physical opening events.  Shape, mode, temperature, R, frequency, seed, and
numerical refinement remain condition or execution counts rather than inflated
physical-vector counts.

Acquisition moved from the retained 1024-arm information registry through new
native discrepancy blocks, coherent active-plastic crossings, corrected
athermal/bilinear transition brackets, exact reconstruction of previously
unpromoted Sobol vectors, and two new activation-area/barrier cohorts. The
second cohort discovered supported rare-opening actions of 1.53e-5/cycle for
`00116` and a state-active no-path-to-opening crossover for `11566`; native
execution followed immediately rather than stopping at descriptor selection.
The 1024 exact native-information rows alone span 319 distinct cohesive and 414
distinct EXP_floor fingerprints.  Their sampled cohesive ranges are
Gamma(300 K)=0.5110--7.9073 J/m2, cohesive entropy
-0.0020099--0.0057147 J/(m2 K), activation area
1.042e-20--1.024e-18 m2, delta0=4.772e-12--2.929e-10 m, peak traction
8.409--78.526 GPa, and zero-traction barrier 0.3537--5.9340 eV.  The paired
plastic ranges include emission H0=0.9704--5.1216 eV, Peierls
H0=1.9176--10.3334 eV, Taylor H0=0.2014--6.8726 eV, and the original nonzero
Peierls/Taylor activation-entropy coordinates.  These are search ranges, not
independent fitted endpoint laws; every native case consumes one coherent row.

The discrepancies changed the next selections rather than being treated as a
post hoc correction.  For `22871`, the current cyclic reduction predicted the
first event at cycle 539,217.19 and the native trajectory located it at
542,018.99 (0.520 percent later), validating promotion into a deep HCF ladder.
For `11566`, the initial reduction reported zero completed-cycle thermal action
and a 7.50e6-cycle plastic-state timescale, whereas the native first excursion
mechanically committed at cycle 0.03644 and then the evolved state removed the
escape path through a 1e12-cycle exposure.  That qualitative miss caused the
next acquisition to bracket the state-created arrest at factor 1.06 with a
higher-load factor-1.10 reactivation trajectory and a matched frozen-state
control; it was not converted into a fitted predictor multiplier.

## Numerical integrity and evidence boundaries

The accelerated high-cycle path was exercised on eventful trajectories, not
qualified only on zero-event overlap.  It advances current EXP_floor state,
localizes a current-CZM first passage, commits/remaps a material cell, consumes
the residual waveform, and continues event-to-event.  Exact-versus-accelerated
initial overlap is bit exact for the strongest HCF case.  Atomic checkpoints
carry the active state, accepted geometry, branch/hazard state and histories;
the final restart inventory reports only checkpoints that exist on disk.

One provenance defect was found during deepening, bounded, and repaired.  A
legacy 22871 frozen checkpoint did not encode phase resolution and was
mistakenly continued from 16 to 64 phases.  Its immutable 55-event phase-16
result remains a valid prefix, but the two-event mixed-resolution suffix is
excluded.  A subsequent one-event replay with the wrong seed is also excluded.
The clean replacement uses 16 phases and the production seed 202610184101 from
cycle zero.  New restarts bind candidate, complete-row hash, T/R/frequency,
feedback mode, phase count, stochastic seed, patch length and observation-path
length.  Neither excluded suffix contributes to coverage, slopes, or mechanism
classification.

A late supervisor process audit also detected several transient duplicate
invocations with the *same* execution identity while their original long-wall
owners were still active.  The newer invocations were interrupted immediately.
Final evidence uses the completed original owner or, for the already completed
10-Hz case, a unique replay from the last atomic checkpoint to the declared
120-cell target.  Progress, result, and checkpoint identities were reconciled
again before packaging; no duplicate prefix is counted as an independent run.
That replay exposed one terminal-transaction defect: an event committed in the
exact fractional-cycle remainder could reach the path target after the last
block-level checkpoint.  The runner now seals every accepted non-numeric-stop
terminal state before writing its result.  The 10-Hz path was replayed from
event 119 and now has identical 197.9513686586 cycles and 120 events in result,
progress, and restart state.

The principal limitations are physical or evidentiary rather than hidden rate
substitutions:

- No completed adjacent local secant currently falls inside 3.4--4.6, so no
  strict Paris pass is claimed.
- A resource stop is not a runout.  It retains an exact restart and contributes
  only its accepted prefix; it receives no zero rate.
- A static no-path state, a post-escape transit requiring unsupported physics,
  a physical finite-horizon runout, an arrested burst, and a path-truncated
  cascade remain separate dispositions.
- The reduced 2-D comparisons consume the retained closing-traction operator;
  they bound finite-state feedback but are not presented as a full spatial
  crack-propagation qualification.
- These are model calculations, not experimental calibration or uncertainty
  bounds on a material measurement.

## Execution record and reproduction

The fixed campaign clock is 2026-10-18 06:08:49--18:08:49 UTC.  It began at
`6018f4c40df68139fd0a54bda6ab10c48510eca0` on
`codex/r10-athermal-mixed-czm-comparison`; the final execution-source revision,
including the terminal-checkpoint seal, is `2a897297`.  Across 135 HCF result files carrying
process accounting, the accepted trajectories report 82,182.99 CPU-seconds
and 117,506.01 aggregate worker-wall-seconds.  The host exposed 14 logical
CPUs.  The supervisor limited numerical work to ten one-thread workers,
reserving four logical CPUs for analysis, validation, and the host.
The independently supplied baseline archive hash is
`b304aa7159e261aba755c5473a8b47468e794291b8e9cda0bf267405f5aa3c1a`.

Repository-wide validation passed 912 tests in 3,475.33 seconds during the
campaign.  After the terminal-checkpoint repair, 25 focused HCF, reduction,
analysis, packaging, and archive-verification tests passed in 2.32 seconds, and
the repaired 120-event trajectory was reproduced natively from event 119.  The
final aggregate contains 27 actionable restart checkpoints;
the portable `analysis/restart_inventory.csv` records their exact material,
protocol, accepted exposure, event count, and checkpoint path.

The production interpreter is
`/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`.
Representative exact commands, run from this worktree, are:

```sh
PY=/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python
$PY scripts/run_r10_czm_hcf.py --registry runs/r10_12h_autonomous/acquisition_round_0_1024/native_information_registry.csv --candidate-id R10F_V2_22871__MS --factor 1.01 --horizon 1000000000000 --out runs/r10_12h_autonomous/hcf_mechanism_controls_22871/frozen_phase16_same_seed --seed 202610184101 --phases 16 --max-wall-s 14400 --frozen-initial-plastic-state
$PY scripts/analyze_r10_12h_campaign.py --campaign runs/r10_12h_autonomous --out runs/r10_12h_autonomous/analysis --finalist-source-ids R10F_V2_22871,R10F_V2_17780,R10F_V2_28475,R10F_V2_17444,R10F_V2_01796,R10F_V2_07172,R10F_V2_11566,R10F_V2_00116
$PY scripts/analyze_r10_12h_scientific_findings.py --campaign runs/r10_12h_autonomous --out runs/r10_12h_autonomous/analysis
MPLCONFIGDIR=/private/tmp/r10-mpl $PY scripts/plot_r10_12h_campaign.py --campaign runs/r10_12h_autonomous --analysis runs/r10_12h_autonomous/analysis --out runs/r10_12h_autonomous/figures
$PY scripts/package_r10_12h_campaign.py --campaign runs/r10_12h_autonomous --report CODEX_R10_12_HOUR_REPORT.md --out R10_12H_AUTONOMOUS_EVIDENCE.zip
$PY scripts/verify_r10_12h_campaign_archive.py R10_12H_AUTONOMOUS_EVIDENCE.zip
```

Reissuing an HCF command with the identical output directory consumes its
atomic `checkpoint/high_cycle_live_checkpoint.json`; restart identity checks
material hash, protocol, phase count, stochastic seed, feedback mode, patch,
and observation-path length before accepting it.  The portable
`analysis/restart_inventory.csv` lists every still-actionable state and its
accepted exposure rather than merely naming hypothetical checkpoint paths.
