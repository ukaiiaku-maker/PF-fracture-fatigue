# Claude code-audit handoff: PF-fracture-fatigue v10.4.2

## 1. Audit request

Please perform an independent code and model audit of branch
`v10.4.2-plastic-flow-terminal` in repository
`ukaiiaku-maker/PF-fracture-fatigue`.

There are **two separate workstreams**. They must not be collapsed into one
problem:

1. **Generated scheduler, inherited-case reuse, and restart correctness.**
   This is an active software defect. Source-level tests pass, but the final
   generated shell scheduler still rejects cases that were independently
   approved for reuse.
2. **Physical and numerical validity of the plastic-flow terminal.**
   The earlier DBTT/1000 K result that appeared to show plasticity suppressing
   fracture was primarily caused by a directional-J sign defect. That defect is
   fixed. The remaining question is whether a genuine sustained-plastic-flow
   terminal can be identified without changing the validated hazard-based
   sharp-fracture physics.

Please report these workstreams separately even if the implementation changes
are delivered in one pull request.

---

## 2. Repository access

### Repository and branch

- Repository: <https://github.com/ukaiiaku-maker/PF-fracture-fatigue>
- Audit branch: <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/tree/v10.4.2-plastic-flow-terminal>
- Commit history for the branch: <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/commits/v10.4.2-plastic-flow-terminal>
- Production-code baseline represented by the recorded run evidence:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/commit/c90df55cbd762459dd0ccda82fb21e27ef17febe>

The branch contains documentation and audit-export tooling added after
`c90df55...`. Use the current branch head for the audit; use `c90df55...` when
comparing the production-code state against the recorded local run evidence.

### Clone into a new checkout

```bash
git clone \
  --branch v10.4.2-plastic-flow-terminal \
  --single-branch \
  https://github.com/ukaiiaku-maker/PF-fracture-fatigue.git \
  PF-fracture-fatigue-v10.4.2-audit

cd PF-fracture-fatigue-v10.4.2-audit
git branch --show-current
git rev-parse HEAD
git status --short
```

If the repository is private, authenticate using the GitHub account or token
that has access to `ukaiiaku-maker/PF-fracture-fatigue`.

### Existing local worktree

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_4_2_plastic_flow_terminal
```

### Environment

```text
Conda environment: arrhenius-sharp-front-v10
Package:           arrhenius-sharp-front-mpz 10.4.2
```

Set up and validate with:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate arrhenius-sharp-front-v10
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python -m pip install -e . --no-deps
bash scripts/validate_v10_4_bulk_peierls_taylor.sh
```

Recorded result at code baseline `c90df55...`:

```text
74 passed
31 passed
```

These passing tests did **not** catch the final generated-scheduler failure.
The audit must therefore execute or faithfully emulate the final generated
shell scheduler, not only inspect builder source or string-position tests.

---

## 3. One-document code map

Read the following files directly on the audit branch.

### 3.1 Production entry and underlying solver

- Audited v10.4.2 model entry:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/sharp_front_v10_4_2_plastic_flow_audited.py>
- Underlying transformed sharp-front solver:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/sharp_front.py>
- Domain J-integral implementation:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/j_integral.py>
- Full-field bulk Peierls/Taylor audited entry inherited from v10.4.1:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/sharp_front_v10_4_bulk_peierls_taylor_audited.py>
- Detailed-balance bulk-plasticity overlay:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/bulk_plastic_detailed_balance_v1041.py>

### 3.2 Corrected directional-J and hazard-energy path

- Positive raw signed directional-J overlay:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/directional_j_positive_v1042.py>
- Hazard-energy mechanics observer:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/hazard_energy_observer_v10230.py>
- Observed hazard-energy engine:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/hazard_energy_observed_engine_v10230.py>
- Event energy gate:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/hazard_energy_gate_v10230.py>

Required production convention:

```text
J_effective = max(J_signed, 0)
```

Negative directional J is non-driving. `abs(J)` is not the production measure.
The first-nonzero directional-J sign latch must remain disabled in both the
solver and the observer.

### 3.3 Plastic-flow terminal and accepted work

- Terminal criteria and contour diagnostics:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/plastic_flow_terminal_v1042.py>
- Accepted constitutive plastic-work and peak-load contour accounting:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/plastic_flow_accepted_work_v1042.py>
- v10.4.2 case classifier:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/classify_v10_4_2_case.py>
- Fracture/plastic-temperature plotter:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/plot_v10_4_2_fracture_plastic_temperature.py>

The terminal status is:

```text
plastic_flow_no_sharp_fracture
```

It is not a ductile-fracture model. The code does not model void growth,
necking, localization-driven rupture, or ductile cohesive tearing.

### 3.4 Inherited-case reuse

- v10.4.1-to-v10.4.2 verifier and materializer:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/reuse_v1041_v1042.py>
- Earlier v10.4.0-to-v10.4.1 verifier used by some inherited cases:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/reuse_v1040_v1041.py>

The v10.4.2 verifier checks source completion, detailed-balance provenance,
required-file SHA-256 values, materialized-file hashes, and the complete
root-front directional-J relation through first passage.

### 3.5 Generated launcher chain

The executable scheduler is assembled through nested transforms. Read these in
this order:

1. Public wrapper:
   <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/run_v10_4_paper_four_class_orientation_rate.sh>
2. Reuse-aware builder:
   <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/build_v10_4_2_reuse_aware_launcher.py>
3. Positive-J builder:
   <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/build_v10_4_2_positive_J_launcher.py>
4. Plastic-terminal builder:
   <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/build_v10_4_2_plastic_terminal_launcher.py>
5. v10.2.30 rate/orientation builder:
   <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/build_v10_2_30_rate_enabled_orientation_launcher.py>
6. v10.2.28 base wrapper:
   <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh>
7. Underlying long-R-curve scheduler source:
   <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves.sh>

The critical audit object is the **final generated scheduler text and its actual
shell control flow**. A builder unit test that only checks token order is not an
adequate acceptance test.

### 3.6 Tests

- Master validation script:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/validate_v10_4_bulk_peierls_taylor.sh>
- Directional-J tests:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_2_directional_j_positive.py>
- Hazard-energy tests:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_2_30_hazard_energy_gate.py>
- Plastic-terminal tests:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_2_plastic_flow_terminal.py>
- Launcher-adapter tests:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_2_launcher_adapter.py>
- Reuse-aware builder test:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_2_reuse_aware_launcher.py>
- Bulk plasticity tests:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_bulk_peierls_taylor.py>
- Provenance tests:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_provenance.py>
- Detailed-balance tests:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_1_detailed_balance.py>
- Campaign contract tests:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_1_campaign_contract.py>
- Selective-reuse tests:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_1_selective_reuse.py>

### 3.7 Existing supporting audit documents and exporter

- Existing audit narrative:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/docs/v10_4_2/AUDIT_HANDOFF.md>
- GitHub access/index page:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/docs/v10_4_2/GITHUB_ACCESS.md>
- Compact local evidence record:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/docs/v10_4_2/LOCAL_EVIDENCE_20260802.md>
- Deterministic local evidence exporter:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/export_v10_4_2_audit_bundle.py>

---

## 4. Physical invariants that any repair must preserve

1. Cleavage is a thermally activated first-passage process.
2. There is no absolute athermal `Gc` criterion.
3. Forward configurational work is exactly:

   ```text
   J_effective = max(J_signed, 0)
   ```

4. Negative directional J is non-driving.
5. `abs(J)` is not used in production.
6. Full-field bulk plasticity remains active.
7. Bulk net slip uses detailed balance and is exactly zero at zero stress.
8. Tip and bulk source populations remain distinct.
9. Direct tip-to-bulk density transfer remains disabled.
10. Plastic work does not enter fracture J, the cleavage hazard, or the event
    energy gate.
11. Contour shielding remains diagnostic only.
12. The non-fracture terminal is called `plastic_flow_no_sharp_fracture`, not
    ductile fracture.
13. Completed source campaign data and quarantined partial runs must not be
    deleted or silently altered.
14. Incomplete v10.4.1 cases must not be reused.

---

## 5. Workstream A: scheduler, reuse, and restart defect

### 5.1 Intended production campaign

Parameterizations:

- peak
- DBTT
- weak temperature dependence
- ceramic-like

Temperatures:

```text
300 600 800 900 950 1000 1050 1100 1150 1200 1250 1300 K
```

Target:

```text
1000 um projected ligament extension
```

Total planned cases:

```text
48
```

A prior v10.4.1 campaign contains 17 completed fracture cases. All 17 passed an
independent positive-directional-J compatibility audit. Intended accounting:

```text
17 verified inherited cases
31 live v10.4.2 calculations
```

### 5.2 Source and destination run roots

Source v10.4.1 campaign:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_1_theta0_rate1x_bulk_PT_four_class_1000um_selective_reuse_base3621_v1
```

Compatibility report:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_1_theta0_rate1x_bulk_PT_four_class_1000um_selective_reuse_base3621_v1/v10_4_2_positive_directional_J_compatibility_report.json
```

Materialized v10.4.2 production root:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_2_theta0_rate1x_bulk_PT_positiveJ_plastic_terminal_four_class_1000um_reuse17_base3621_v1
```

Recorded compatibility result:

```text
completed_cases_checked:       17
compatible_for_v10_4_2_reuse: 17
must_be_rerun:                  0
```

Materialization result:

```text
materialized_cases: 17
verified_cases:     17
```

The inherited contents, including `COMPLETE`, are symlinks. `find -type f`
therefore gives incorrect counts for those markers. Verification and monitoring
must be symlink-aware.

### 5.3 Compatible inherited cases

| class | temperature (K) | seed | first-passage step | first-passage J (J/m²) |
|---|---:|---:|---:|---:|
| DBTT | 300 | 1003621 | 90 | 1221.189468132166 |
| DBTT | 600 | 1004630 | 148 | 1187.1413210881537 |
| DBTT | 800 | 1005639 | 400 | 1147.764614088523 |
| DBTT | 900 | 1006648 | 452 | 1121.230213222208 |
| DBTT | 950 | 1007657 | 461 | 1069.8948636914229 |
| peak | 300 | 3621 | 83 | 1257.3445216074792 |
| peak | 600 | 4630 | 79 | 1142.3155515312178 |
| peak | 800 | 5639 | 231 | 5740.056891208428 |
| peak | 900 | 6648 | 264 | 6759.449836717835 |
| peak | 950 | 7657 | 270 | 6708.43130061901 |
| peak | 1000 | 8666 | 177 | 6530.805687956452 |
| peak | 1050 | 9675 | 179 | 6423.722908003451 |
| peak | 1100 | 10684 | 177 | 6232.954572253016 |
| peak | 1150 | 11693 | 174 | 5972.951721950498 |
| peak | 1200 | 12702 | 167 | 5420.873232151028 |
| peak | 1250 | 13711 | 176 | 5710.78899755089 |
| peak | 1300 | 14720 | 172 | 5341.842038362321 |

Every checked case had zero reported error in the required relation through
first passage.

### 5.4 Actual executable failure

The final generated scheduler rejected each inherited case with:

```text
ERROR: terminal-looking case failed contract verification
FAILED: <case> (exit=3)
```

It then launched calculations that should have been skipped.

The one-case inherited smoke produced the contradictory result:

```text
FAILED: v913_paper_peak01_0242980_persistent_sites:T300K:seed3621 (exit=3)
Campaign acceptance: planned=1 complete=1 failed_or_incomplete=0
Campaign complete: failures=1
```

This proves that separate shell and Python aggregation paths disagree about the
same materialized case.

A reuse-aware builder was added with the intended short circuit:

1. detect `v10_4_2_reuse_audit.json`;
2. call `verify_materialized_case(root)`;
3. verify the referenced source case;
4. exit successfully before native v10.4.2 command checks.

Source-level tests passed, but the real generated-scheduler smoke still failed.
Do not assume the intended short circuit exists in the verifier actually called
by `run_case()`.

### 5.5 Required diagnosis

Inspect the final generated scheduler and determine:

1. Which `verified_complete()` or equivalent function `run_case()` actually
   calls.
2. Whether the reuse short circuit is inserted in that exact function.
3. Whether multiple verifier blocks exist and the transform patched the wrong
   occurrence.
4. Whether the Python verifier exits zero while the containing shell function
   continues or returns nonzero.
5. Whether a second native-only verifier rejects the case later.
6. Whether the case contract is overwritten before reuse detection.
7. Whether stale `RUN_FAILED`, `INCOMPLETE`, or contract files are involved.
8. How symlinked `COMPLETE` markers are handled by every shell and Python path.
9. Why the campaign can report both `complete=1` and `failures=1`.
10. Whether inherited cases should use an explicit successful scheduler status,
    such as `SKIP_REUSED_VERIFIED`, rather than the native completion path.

### 5.6 Required executable acceptance

For peak, 300 K, seed 3621, require all of the following:

```text
v10_4_2_reuse_audit.json verified
source hashes verified
positive-J history verified
no native v10.4.2 command checks applied
no new solver process launched
no RUN_FAILED created
no FAILED line emitted
planned=1
complete=1
failed_or_incomplete=0
failures=0
```

Then add a full-matrix dry run or preflight that explicitly reports:

```text
17 verified inherited cases
31 live cases
```

The full production campaign must not be launched until both acceptance checks
pass.

### 5.7 Restart state and quarantine

The erroneous process tree was stopped. Recorded PIDs terminated:

```text
38550 38563 38582 40551 40556 40915 40920 41145
```

Eleven partially started cases were moved to:

```text
/Volumes/Data/Data/Nanopillar_calculation/quarantine/v10_4_2_pre_reuse_scheduler_fix_20260802_074842
```

Quarantined cases:

- DBTT: 1000, 1050, 1100, 1150, 1200, 1250, 1300 K
- weakT: 300, 600, 800, 900 K

The 17 materialized reuse cases remained in the production root. The latest
restart attempt then stopped because the PID file already existed.

Before any new launch, independently verify:

- whether the PID in the PID file is live;
- the complete process tree for the production root;
- all remaining non-reuse case directories;
- all stale failure and incomplete markers;
- all 17 reuse audits and their symlink targets;
- that no quarantined case was moved back implicitly.

---

## 6. Workstream B: plastic accommodation versus sharp fracture

### 6.1 Original apparent suppression was not a valid terminal

The original DBTT/1000 K long run appeared to have no crack drive. Its final
2000-step window was instead nearly elastic:

```text
plastic fraction:             0.0003478
elastic fraction:             0.999624
final force / peak force:     1.0
normalized tangent stiffness: 1.039
```

The terminal detector correctly rejected this state. The failed criteria were
plastic-accommodation dominance, flat elastic storage, and collapse of
load-carrying capacity.

### 6.2 Directional-J defect

The inherited solver latched the sign of the first nonzero raw directional J. A
small negative startup value fixed the reference sign at `-1`. Later positive
raw J was multiplied by `-1` and clipped to zero. A second independent latch
existed in the hazard-energy observer.

Both paths were changed to:

```text
J_effective = max(J_signed, 0)
```

without using an absolute value.

### 6.3 Corrected DBTT/1000 K smoke

Run root:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_2_DBTT_1000K_positiveJ_20um_smoke_seed1008666_v2
```

Case root:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_2_DBTT_1000K_positiveJ_20um_smoke_seed1008666_v2/v913_paper_dbtt01_0202500_persistent_sites/T1000K_th0_seed2008666
```

Observed values:

```text
root-front rows:                         570
raw J range:                            -0.0715728678 to 8346.4647749 J/m²
max J_effective relation error:          0.0 J/m²
unique J_sign_ref:                       [1.0]
first positive raw J:                    step 107
first accepted crack event:              step 536
first-event KJ:                          57.8738079982 MPa sqrt(m)
target reached:                          step 570
projected extension:                     24.7195429200 um
classification:                          complete_target_extension
mode:                                    brittle
nominal checkpoint advances:             5
accepted geometry events:                2
```

Integrated event-energy results:

```text
event 1 available:    0.07652887434541052 J/m
event 1 dissipated:   0.00011015683040985367 J/m
event 1 margin:       0.07641871751500066 J/m

event 2 available:    0.11789423901256565 J/m
event 2 dissipated:   0.0001286970054494601 J/m
event 2 margin:       0.1177655420071162 J/m
```

No failed integrated balance was found. This establishes that the original
DBTT/1000 K apparent suppression was primarily a numerical directional-J
artifact, not a demonstrated plastic-flow terminal.

### 6.4 Open terminal-physics audit

Please audit the following questions.

#### A. Accepted plastic work

The overlay requests `update_plasticity(..., return_info=True)`, obtains
`dWp_accepted_gp`, and sums positive accepted contributions over every stagger
iteration. Determine whether this is physically accepted constitutive work or
whether nonlinear stagger iterations can revisit and double-count increments.

#### B. Energy balance

Determine whether the terminal energy residual is merely reported after
acceptance or actually gates acceptance. Consider a scale-aware criterion based
on:

```text
Delta W_external
  ~= Delta U_elastic
   + Delta W_bulk_plastic
   + Delta W_tip_emission
   + other explicitly tracked terms
```

Recommend an absolute-plus-relative tolerance tied to the work scale and solver
accuracy.

#### C. Reaction force and tangent stiffness

Confirm that terminal force and tangent stiffness are evaluated from the fully
converged accepted mechanics state, not a stale or pre-relaxation iterate.

#### D. Plastic dominance after mechanical collapse

Incremental external, elastic, and plastic work may all approach zero after
collapse. Determine whether a state with previously demonstrated sustained
positive plastic activity should remain eligible even if the final incremental
plastic fraction becomes numerically ill-conditioned.

#### E. Tip J and stress tolerances

Confirm that residual-J and residual-tip-stress tolerances are tied to numerical
resolution and physically meaningful scales rather than arbitrary constants.

#### F. Cleavage horizon

Confirm that the terminal uses the same active cleavage hazard, action variable,
and energy gate as production first-passage fracture.

#### G. Failed-candidate diagnostics

Even when terminal classification fails, require a candidate audit that records:

- every metric and threshold;
- pass/fail state for each criterion;
- failed-criterion list;
- external/elastic/plastic/emission energy ledger;
- accepted plastic-work source;
- stagger-iteration count;
- force and stiffness history;
- positive-J and tip-stress history;
- remaining cleavage-time estimate and loading horizon.

#### H. Semantics

The result must be described as `plastic_flow_no_sharp_fracture`, never as
simulated ductile fracture.

#### I. Controlled reachability tests

Add at least these tests:

1. nearly elastic, load-bearing, no-crack state — must fail;
2. plastically active but still load-bearing state — normally fail;
3. sustained plastic accommodation with collapsed stiffness and inaccessible
   cleavage first passage — should pass;
4. positive crack-driving J with accessible hazard — must fail.

#### J. Reassessment after positive-J repair

Reassess any case previously suspected of plastic termination under the corrected
positive raw directional-J convention.

---

## 7. Local generated evidence and how Claude accesses it

The complete run trees are generated data, large, partially symlinked, and
contain mutable restart state. They should not be committed wholesale to normal
Git history. The branch instead contains:

- this complete handoff;
- a compact evidence record;
- direct source and test links;
- a deterministic exporter that packages the audit-relevant local files with
  original paths, sizes, and SHA-256 hashes.

Exporter:

<https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/export_v10_4_2_audit_bundle.py>

On the machine containing `/Volumes/Data`, run:

```bash
cd /Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_4_2_plastic_flow_terminal

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate arrhenius-sharp-front-v10
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python scripts/export_v10_4_2_audit_bundle.py \
  --output /Volumes/Data/Data/Nanopillar_calculation/v10_4_2_audit_bundle_20260802.zip
```

Provide the resulting ZIP to Claude together with this document and the audit
branch. The exporter prints the ZIP path and SHA-256 digest.

Expected bundle content includes:

- corrected smoke `fronts_1000K.csv` and `steps_1000K.csv`;
- corrected smoke status, summary, command, and bounded run log;
- hazard-energy gate, geometry-event, and energy-ledger JSON audits;
- the 17-case compatibility report;
- all 17 materialized `v10_4_2_reuse_audit.json` records;
- the materialized-reuse manifest;
- campaign lock and kernel-resolution records;
- one-case reuse smoke and restart logs;
- PID evidence and process inventory;
- inventories and bounded log excerpts from the 11 quarantined partial cases;
- Git branch, head, and worktree status;
- a manifest of bundled files with source paths, sizes, and SHA-256 values.

Claude cannot obtain `/Volumes/Data` through GitHub. The ZIP must be uploaded or
mounted separately. Do not ask Claude to infer raw histories from this summary
when the audit bundle is available.

---

## 8. Recommended audit sequence

1. Clone the branch and record the current head.
2. Run the full validation suite.
3. Generate and save the final scheduler text for the one-case reuse fixture.
4. Trace `run_case()` through the exact completion verifier it calls.
5. Reproduce the peak/300 K reuse failure without launching a solver.
6. Fix the control flow and add an executable final-scheduler regression test.
7. Require `failures=0` as well as `failed_or_incomplete=0`.
8. Add a full 17-reuse/31-live preflight.
9. Inspect the audit ZIP and independently verify the corrected positive-J smoke
   and the 17-case compatibility report.
10. Audit plastic-work accounting and terminal metrics independently of the
    scheduler repair.
11. Add controlled terminal pass/fail fixtures.
12. Provide a pinned commit, test outputs, and exact safe restart commands.

---

## 9. Required deliverables

### Report A: scheduler/reuse/restart

Provide:

- exact failure location in the final generated scheduler;
- explanation of why builder tests passed while executable control flow failed;
- corrected reuse short circuit;
- executable final-scheduler regression test;
- one-case result with:

  ```text
  planned=1
  complete=1
  failed_or_incomplete=0
  failures=0
  ```

- full preflight reporting exactly 17 verified inherited and 31 live cases;
- symlink-safe monitoring and marker counting;
- process/PID/stale-directory cleanup procedure;
- pinned commit SHA and exact update, validation, and restart commands.

### Report B: plastic-terminal physics

Provide:

- accepted plastic-work accounting assessment;
- energy-balance assessment and recommended gate/tolerance;
- converged force/stiffness-state assessment;
- criterion-by-criterion terminal review;
- controlled reachability tests;
- assessment of whether any corrected positive-J case genuinely terminates in
  plastic flow instead of fracturing;
- any proposed changes, explicitly identifying whether they modify physics or
  diagnostics only.

---

## 10. Explicit prohibitions

Do not:

- add an absolute athermal `Gc`;
- use `abs(J)` as the production fracture measure;
- reintroduce a first-nonzero sign latch;
- feed plastic work into fracture J, cleavage hazard, or event energy gate;
- feed contour shielding into the production fracture hazard;
- classify plastic flow as simulated ductile fracture;
- delete source campaign data or quarantined partial runs;
- reuse incomplete cases;
- launch the full campaign before the executable reuse smoke and full preflight
  both pass.

The primary objective is not merely to make the smoke test pass. It is to restore
auditable campaign control and to establish whether the plastic-flow terminal is
physically meaningful while preserving the hazard-based fracture formulation.
