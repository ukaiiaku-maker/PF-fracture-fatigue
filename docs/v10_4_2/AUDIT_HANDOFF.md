# v10.4.2 external audit handoff

## Start here

Repository and direct file links:

- [GitHub access and file index](GITHUB_ACCESS.md)
- [Local run evidence snapshot](LOCAL_EVIDENCE_20260802.md)
- [Audit-bundle exporter](../../scripts/export_v10_4_2_audit_bundle.py)

Repository branch:

```text
ukaiiaku-maker/PF-fracture-fatigue
v10.4.2-plastic-flow-terminal
```

Code baseline before the documentation/export-only commits:

```text
c90df55cbd762459dd0ccda82fb21e27ef17febe
```

The branch includes later documentation and exporter commits.  Begin from the
current branch head, and use `c90df55...` when comparing the production-code
state against the recorded local evidence.

## Audit objective

There are two distinct workstreams.  They must not be collapsed into one bug.

1. **Campaign reuse/restart correctness remains unresolved.**
2. **The physical and numerical validity of the plastic-flow terminal remains an
   open model audit.**

The earlier DBTT/1000 K result that appeared to show plasticity suppressing
fracture was primarily caused by a directional-J sign defect.  That defect has
been corrected.  The remaining question is whether a *genuine* sustained
plastic-accommodation terminal can be identified without altering the validated
hazard-based sharp-fracture physics.

# 1. Repository and environment

Local worktree:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_4_2_plastic_flow_terminal
```

Conda environment:

```text
arrhenius-sharp-front-v10
```

Package:

```text
arrhenius-sharp-front-mpz 10.4.2
```

Production output repository:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1
```

Recorded source-level validation at code baseline `c90df55...`:

```text
74 passed
31 passed
```

These tests did not catch the final generated-scheduler failure described below.
The audit therefore must execute the actual generated shell scheduler.

# 2. Intended physical invariants

The active production entry is:

```text
arrhenius_fracture.sharp_front_v10_4_2_plastic_flow_audited
```

The following invariants must be preserved:

- Cleavage is a thermally activated first-passage process.
- There is no absolute athermal `Gc` criterion.
- Positive forward configurational work is:

  ```text
  J_effective = max(J_signed, 0)
  ```

- Negative directional J is non-driving.
- The first-nonzero directional-J sign latch is disabled.
- `abs(J)` is not the production fracture measure.
- Full-field bulk plasticity is active.
- Bulk net slip uses detailed balance and is exactly zero at zero stress.
- Tip and bulk source populations are distinct.
- Direct tip-to-bulk density transfer is disabled.
- Plastic work does not enter fracture J, the cleavage hazard, or the event
  energy gate.
- Contour shielding is diagnostic only.
- A successful non-fracture terminal is called:

  ```text
  plastic_flow_no_sharp_fracture
  ```

- The model does not simulate void growth, necking, ductile tearing, or ductile
  fracture.  The plastic terminal must never be described as simulated ductile
  rupture.

# 3. Workstream A — generated scheduler, reuse, and restart

## 3.1 Intended campaign

Four parameterizations:

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

A prior v10.4.1 campaign contains 17 completed fracture cases.  Every completed
case independently passed the corrected positive-J history audit.  Intended
production accounting is therefore:

```text
17 verified inherited cases
31 live v10.4.2 calculations
```

## 3.2 Source campaign and compatibility report

Source campaign:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_1_theta0_rate1x_bulk_PT_four_class_1000um_selective_reuse_base3621_v1
```

Compatibility report:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_1_theta0_rate1x_bulk_PT_four_class_1000um_selective_reuse_base3621_v1/v10_4_2_positive_directional_J_compatibility_report.json
```

Recorded result:

```text
completed_cases_checked:       17
compatible_for_v10_4_2_reuse: 17
must_be_rerun:                  0
```

The full case list and first-passage values are recorded in
[LOCAL_EVIDENCE_20260802.md](LOCAL_EVIDENCE_20260802.md).

## 3.3 Materialized destination

Production root:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_2_theta0_rate1x_bulk_PT_positiveJ_plastic_terminal_four_class_1000um_reuse17_base3621_v1
```

Materialization reported:

```text
materialized_cases: 17
verified_cases:     17
```

The inherited contents, including `COMPLETE`, are symlinks.  A count using
`find -type f` is therefore wrong.  Reuse verification must be symlink-aware.

## 3.4 Actual failure

The generated scheduler rejected every inherited case:

```text
ERROR: terminal-looking case failed contract verification
FAILED: <case> (exit=3)
```

It then began running cases that should have been skipped.

The one-case inherited smoke also reported the contradictory combination:

```text
FAILED: v913_paper_peak01_0242980_persistent_sites:T300K:seed3621 (exit=3)
Campaign acceptance: planned=1 complete=1 failed_or_incomplete=0
Campaign complete: failures=1
```

This proves that separate shell and Python aggregation paths disagree about the
same case.

A reuse-aware builder was added, but the executable smoke still failed.  The
likely possibilities include:

- the short circuit was inserted into an intermediate verifier that is not the
  verifier called by `run_case()`;
- there are multiple generated verifier blocks and the wrong occurrence was
  patched;
- the Python verifier exits zero but the containing shell function continues or
  returns nonzero;
- a second native-only verification block rejects the case later;
- a stale `RUN_FAILED` or rewritten case contract is present;
- symlink handling differs among shell, Python, and campaign aggregation;
- the scheduler records a failure before the aggregator later recognizes the
  inherited case as complete.

## 3.5 Required scheduler audit

Inspect the final generated scheduler produced by this chain:

```text
run_v10_4_paper_four_class_orientation_rate.sh
  -> build_v10_4_2_reuse_aware_launcher.py
  -> build_v10_4_2_positive_J_launcher.py
  -> build_v10_4_2_plastic_terminal_launcher.py
  -> build_v10_2_30_rate_enabled_orientation_launcher.py
  -> run_v10_2_28_paper_four_class_theta30_1000um.sh
  -> run_v10_2_27_paper_four_class_30deg_long_rcurves.sh
```

Determine:

1. Which `verified_complete()` or equivalent function `run_case()` actually
   calls.
2. Whether the v10.4.2 reuse short circuit is in that exact function.
3. Whether the case contract is written or overwritten before reuse detection.
4. Whether `RUN_FAILED` is created before or after completion verification.
5. Whether there are duplicate native audit checks after the supposed short
   circuit.
6. Why the shell failure count and campaign-complete count disagree.
7. How symlinked markers are handled in every path.
8. Whether inherited cases should emit an explicit successful status such as
   `SKIP_REUSED_VERIFIED` rather than passing through native completion logic.

## 3.6 Required executable acceptance test

For peak, 300 K, seed 3621:

```text
- verify v10_4_2_reuse_audit.json
- verify source-case hashes
- verify positive-J history
- do not apply native v10.4.2 command checks
- do not launch the solver
- do not create RUN_FAILED
- emit no FAILED line
- planned=1
- complete=1
- failed_or_incomplete=0
- failures=0
```

Then perform a full-matrix dry-run/preflight that explicitly reports:

```text
17 verified inherited cases
31 live cases
```

Do not launch production until both checks pass.

## 3.7 Restart state

The first erroneous process tree was stopped.  Eleven partially started cases
were moved to:

```text
/Volumes/Data/Data/Nanopillar_calculation/quarantine/v10_4_2_pre_reuse_scheduler_fix_20260802_074842
```

The latest attempted restart then stopped because the production PID file still
existed.  Before any new launch, independently verify:

- actual launcher and child processes;
- PID-file contents and whether the PID is live;
- absence of model processes referring to the production root;
- all non-reuse case directories remaining in the production root;
- all `RUN_FAILED`, `INCOMPLETE`, and stale contract files;
- the 17 reuse audits and their symlink targets.

# 4. Workstream B — genuine plastic accommodation versus sharp fracture

## 4.1 Original apparent suppression was not plastic flow

The original DBTT/1000 K long run appeared to have no crack drive.  Its final
window was instead nearly elastic:

```text
plastic fraction:              0.0003478
elastic fraction:              0.999624
final force / peak force:      1.0
normalized tangent stiffness:  1.039
```

The plastic-flow classifier correctly rejected that state.  It failed the
plastic-dominance, flat-elastic-storage, and load-collapse gates.

## 4.2 Directional-J sign defect

The inherited solver latched the sign of the first nonzero raw directional J.
A small negative startup value fixed the sign reference at `-1`.  Later large
positive raw J was multiplied by `-1` and clipped to zero.

The same independent sign latch also existed in the hazard-energy observer.
Both paths were repaired to use:

```text
J_effective = max(J_signed, 0)
```

without `abs(J)`.

## 4.3 Corrected smoke result

Corrected DBTT/1000 K smoke root:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_2_DBTT_1000K_positiveJ_20um_smoke_seed1008666_v2
```

The result was brittle fracture:

```text
first accepted event: step 536
KJ:                   57.87380799822452 MPa sqrt(m)
target reached:       step 570
projected extension:  24.719542920012962 um
```

Across all 570 root-front rows:

```text
max error in J_effective=max(J_signed,0): 0.0
unique J_sign_ref: [1.0]
```

Both accepted geometry events passed integrated energy balance.  Exact values
and local files are recorded in [LOCAL_EVIDENCE_20260802.md](LOCAL_EVIDENCE_20260802.md).

This establishes that the original DBTT/1000 K apparent suppression was a
numerical directional-J artifact, not a valid plastic terminal.

## 4.4 Remaining open terminal-physics questions

Audit the current terminal implementation and accepted-work accounting:

1. **Accepted plastic work** — Does summing `dWp_accepted_gp` over every stagger
   iteration represent physically accepted constitutive work, or can the
   nonlinear iteration revisit and double-count increments?
2. **Energy balance** — Is the external/elastic/plastic/emission residual only
   reported after acceptance, or is it an actual terminal gate?  Define a
   scale-aware tolerance if it should gate acceptance.
3. **Reaction force and tangent stiffness** — Are these measured from the fully
   converged accepted state rather than a stale or pre-relaxation state?
4. **Plastic dominance after collapse** — Should a mechanically collapsed state
   remain eligible if prior sustained positive plastic work has been established
   but all final incremental work terms approach zero?
5. **Tip J and stress tolerances** — Are tolerances tied to numerical and physical
   scales rather than arbitrary constants?
6. **Cleavage horizon** — Does the terminal use the same active hazard and energy
   gate as the production first-passage calculation?
7. **Candidate diagnostics** — Are all metrics and failed criteria written even
   when terminal classification does not pass?
8. **Semantics** — Is the outcome consistently described as plastic flow without
   sharp fracture, never ductile fracture?
9. **Reachability tests** — Add controlled pass/fail cases for elastic no-crack,
   plastic-but-load-bearing, collapsed plastic accommodation, and positive-J
   accessible-cleavage states.
10. **Effect of corrected J** — Reassess all previously suspected plastic-terminal
    cases under the corrected positive raw J convention.

# 5. Constraints on any repair

Do not:

- add an absolute athermal `Gc`;
- replace signed J with `abs(J)`;
- feed plastic work into fracture J, cleavage hazard, or event energy gate;
- feed contour shielding into the fracture hazard;
- call the terminal simulated ductile fracture;
- delete or alter source campaign outputs;
- reuse incomplete source cases;
- delete quarantined partial runs;
- launch the full campaign before executable reuse acceptance is clean.

Preserve:

```text
J_effective = max(J_signed, 0)
```

and preserve the cryptographic and positive-J history checks for inherited
cases.

# 6. Requested deliverables

Provide two separately labeled reports, even if implemented in one pull request.

## A. Scheduler/restart report

- exact generated-scheduler failure location;
- corrected control flow;
- executable final-scheduler regression test;
- one-case reuse result with `failures=0`;
- full 17-reuse/31-live preflight;
- process/PID/stale-directory cleanup procedure;
- pinned commit SHA and exact update/validation/restart commands.

## B. Plastic-terminal physics report

- accepted plastic-work accounting audit;
- energy-balance audit and recommended gate/tolerance;
- force/stiffness state audit;
- terminal-criterion recommendations;
- controlled reachability tests;
- assessment of whether any corrected positive-J case genuinely terminates in
  plastic flow rather than fracturing.

# 7. Providing the local data

The complete simulation trees are generated data and are not appropriate for
normal Git history.  Git contains a compact evidence record and a deterministic
exporter.

On the machine containing `/Volumes/Data`:

```bash
cd /Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_4_2_plastic_flow_terminal

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate arrhenius-sharp-front-v10

python scripts/export_v10_4_2_audit_bundle.py \
  --output /Volumes/Data/Data/Nanopillar_calculation/v10_4_2_audit_bundle_20260802.zip
```

Provide the resulting ZIP and its printed SHA-256 digest to the auditor together
with the GitHub branch.  The ZIP contains the corrected smoke histories and
energy audits, the 17-case compatibility report, all materialized reuse audits,
restart logs/PID evidence, directory inventories, and quarantined partial-case
log excerpts.
