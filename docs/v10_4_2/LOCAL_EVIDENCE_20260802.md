# v10.4.2 local evidence snapshot — 2026-08-02

This file records the small, audit-critical facts extracted from the local run
trees and terminal transcripts.  The complete simulation outputs remain under
`/Volumes/Data` because they are large generated data, not source-controlled
inputs.  Use `scripts/export_v10_4_2_audit_bundle.py` to package the exact local
files with hashes for an external auditor.

## Code baseline represented by this evidence

- Repository: `ukaiiaku-maker/PF-fracture-fatigue`
- Branch: `v10.4.2-plastic-flow-terminal`
- Code baseline before this documentation update:
  `c90df55cbd762459dd0ccda82fb21e27ef17febe`
- Package: `arrhenius-sharp-front-mpz 10.4.2`
- Conda environment: `arrhenius-sharp-front-v10`
- Worktree:
  `/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_4_2_plastic_flow_terminal`

Validation at that baseline completed with:

```text
74 passed
31 passed
```

Passing source-level tests did **not** imply that the final generated scheduler
accepted materialized reuse cases.  The executable one-case reuse smoke still
failed, so generated-shell control flow remains an open defect.

## Corrected positive directional-J smoke

Run root:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_2_DBTT_1000K_positiveJ_20um_smoke_seed1008666_v2
```

Case root:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_2_DBTT_1000K_positiveJ_20um_smoke_seed1008666_v2/v913_paper_dbtt01_0202500_persistent_sites/T1000K_th0_seed2008666
```

Observed audit values:

- root-front rows: 570
- raw signed J range: `-0.07157286780799363` to
  `8346.464774873848 J/m^2`
- maximum error in `J_effective=max(J_signed,0)`: `0.0 J/m^2`
- unique `J_sign_ref`: `[1.0]`
- first positive raw J: step 107,
  `J=0.004810635466324909 J/m^2`,
  `KJ=0.04626173534009141 MPa sqrt(m)`
- first accepted crack event: step 536,
  `J_signed=J_effective=7528.74742507725 J/m^2`,
  `KJ=57.87380799822452 MPa sqrt(m)`
- target completion: step 570,
  projected extension `24.719542920012962 um`
- terminal status: `complete_target_extension`
- fracture mode: `brittle`
- nominal checkpoint advances: 5
- accepted geometry events: 2

Integrated event-energy records:

1. Available `0.07652887434541052 J/m`, dissipated
   `0.00011015683040985367 J/m`, margin
   `0.07641871751500066 J/m`.
2. Available `0.11789423901256565 J/m`, dissipated
   `0.0001286970054494601 J/m`, margin
   `0.1177655420071162 J/m`.

Thirty-seven repeated/embedded audit records were found, no failed integrated
balance was found, and both geometry-event records had
`integrated_energy_balance_pass=true`.

## Positive-J compatibility of inherited v10.4.1 cases

Source campaign:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_1_theta0_rate1x_bulk_PT_four_class_1000um_selective_reuse_base3621_v1
```

Compatibility report:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_1_theta0_rate1x_bulk_PT_four_class_1000um_selective_reuse_base3621_v1/v10_4_2_positive_directional_J_compatibility_report.json
```

Result:

```text
completed_cases_checked:       17
compatible_for_v10_4_2_reuse: 17
must_be_rerun:                  0
```

Every checked case reported zero error in the required relation through first
passage.  The compatible cases were:

| class | temperature (K) | seed | first-passage step | first-passage J (J/m^2) |
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

## Materialized production root

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_2_theta0_rate1x_bulk_PT_positiveJ_plastic_terminal_four_class_1000um_reuse17_base3621_v1
```

Materialization reported:

```text
materialized_cases: 17
verified_cases:     17
```

The inherited case contents and `COMPLETE` markers are symlinks.  Therefore,
commands such as `find ROOT -type f -name COMPLETE` incorrectly report zero.
Use `find ROOT -name COMPLETE`, `find -L`, or explicit `Path.is_file()` checks.

## Unresolved generated-scheduler/restart defect

The generated scheduler rejected each materialized case with:

```text
ERROR: terminal-looking case failed contract verification
FAILED: <case> (exit=3)
```

It then launched cases that should have been skipped.  The one-case acceptance
smoke produced an internally inconsistent summary:

```text
ERROR: terminal-looking case failed contract verification
FAILED: v913_paper_peak01_0242980_persistent_sites:T300K:seed3621 (exit=3)
Campaign acceptance: planned=1 complete=1 failed_or_incomplete=0
Campaign complete: failures=1
```

The full restart similarly rejected all 17 inherited cases and began launching
DBTT 1000 K and 1050 K.  The latest attempted restart then stopped at:

```text
ERROR: PID file already exists
```

Consequently, branch `v10.4.2-plastic-flow-terminal` must not be treated as
production-ready solely because its pytest groups pass.  The final generated
shell scheduler must be executed in a reuse fixture and must report both
`failed_or_incomplete=0` and `failures=0`.

## Stopped and quarantined partial calculations

The erroneous campaign process tree was stopped.  The local stop command found
and terminated PIDs:

```text
38550 38563 38582 40551 40556 40915 40920 41145
```

Eleven live case directories produced before the stop were moved to:

```text
/Volumes/Data/Data/Nanopillar_calculation/quarantine/v10_4_2_pre_reuse_scheduler_fix_20260802_074842
```

They comprise:

- DBTT: 1000, 1050, 1100, 1150, 1200, 1250, and 1300 K
- weakT: 300, 600, 800, and 900 K

The 17 materialized reuse cases remained in the production root.

## Original apparent fracture suppression

The earlier DBTT/1000 K long run that appeared to suppress fracture was not a
valid plastic-flow terminal.  Its final 2000-step window was approximately:

- plastic fraction: `0.0003478`
- elastic fraction: `0.999624`
- final force / peak force: `1.0`
- normalized tangent stiffness: `1.039`

It was nearly elastic and retained full load-carrying capacity.  The terminal
criteria correctly rejected it.  The missing fracture drive was traced to the
first-nonzero directional-J sign latch.  A small negative startup J set the
reference sign to `-1`, causing later large positive raw J to be clipped to
zero.  Both the main solver and hazard-energy observer were changed to use
`max(J_signed,0)` without an absolute value.

## Files to export from the local machine

The audit bundle should include, at minimum:

- corrected smoke case:
  - `fronts_1000K.csv`
  - `steps_1000K.csv`
  - `stage3_case_status.json`
  - `summary.json`
  - `stochastic_avalanche_geometry_events.json`
  - `v10_2_30_hazard_energy_gate_audit.json`
  - `v10_2_27_energy_ledger_output_audit.json`
  - `command.sh`
  - `run.log`
- source campaign:
  - `v10_4_2_positive_directional_J_compatibility_report.json`
- materialized production campaign:
  - `v10_4_2_materialized_reuse_manifest.json`
  - campaign lock and kernel-resolution JSON files
  - all `v10_4_2_reuse_audit.json` files
  - one-case reuse smoke log
  - restart logs and PID file, if present
- quarantine:
  - directory inventory
  - command/status/log excerpts for each partial case

Run `scripts/export_v10_4_2_audit_bundle.py` to create a bounded ZIP containing
these files, hashes, and log excerpts.
