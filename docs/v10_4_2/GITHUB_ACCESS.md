# v10.4.2 GitHub access and audit file index

This page is the entry point for an external code auditor.  It identifies the
branch, gives clone/update commands, links every file directly relevant to the
positive directional-J repair, plastic-flow terminal, accepted plastic-work
accounting, inherited-case reuse, and generated campaign scheduler, and explains
how to obtain the local simulation evidence.

## Repository, branch, and code baseline

- Repository: <https://github.com/ukaiiaku-maker/PF-fracture-fatigue>
- Audit branch:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/tree/v10.4.2-plastic-flow-terminal>
- Code baseline before the audit documentation/export additions:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/commit/c90df55cbd762459dd0ccda82fb21e27ef17febe>
- Latest branch commits:
  <https://github.com/ukaiiaku-maker/PF-fracture-fatigue/commits/v10.4.2-plastic-flow-terminal>

The branch link should be used for the current audit because documentation and
export tooling were added after code baseline `c90df55...` without altering the
production physics represented by that baseline.

## Clone from GitHub

```bash
git clone \
  --branch v10.4.2-plastic-flow-terminal \
  --single-branch \
  https://github.com/ukaiiaku-maker/PF-fracture-fatigue.git \
  PF-fracture-fatigue-v10.4.2-audit

cd PF-fracture-fatigue-v10.4.2-audit
git branch --show-current
git rev-parse HEAD
```

## Update an existing checkout or worktree

```bash
BRANCH=v10.4.2-plastic-flow-terminal

git fetch --force origin \
  "refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"

git switch "$BRANCH"
git merge --ff-only "refs/remotes/origin/${BRANCH}"

git rev-parse HEAD
git status --short
```

For the existing local worktree used in the campaign:

```text
/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_4_2_plastic_flow_terminal
```

## Environment and validation

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate arrhenius-sharp-front-v10

unset PYTHONPATH
export PYTHONNOUSERSITE=1

python -m pip install -e . --no-deps
bash scripts/validate_v10_4_bulk_peierls_taylor.sh
```

The recorded result at code baseline `c90df55...` was:

```text
74 passed
31 passed
```

The generated scheduler still failed an executable reuse smoke despite those
passing tests.  The generated-shell control flow therefore remains an explicit
audit target.

# Direct file index

## Audit documents and evidence tooling

- Main handoff:
  [docs/v10_4_2/AUDIT_HANDOFF.md](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/docs/v10_4_2/AUDIT_HANDOFF.md)
- Local evidence summary:
  [docs/v10_4_2/LOCAL_EVIDENCE_20260802.md](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/docs/v10_4_2/LOCAL_EVIDENCE_20260802.md)
- This access/index page:
  [docs/v10_4_2/GITHUB_ACCESS.md](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/docs/v10_4_2/GITHUB_ACCESS.md)
- Local evidence ZIP exporter:
  [scripts/export_v10_4_2_audit_bundle.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/export_v10_4_2_audit_bundle.py)

## Production entry and directional-J physics

- v10.4.2 audited model entry:
  [arrhenius_fracture/sharp_front_v10_4_2_plastic_flow_audited.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/sharp_front_v10_4_2_plastic_flow_audited.py)
- Positive raw signed directional-J overlay:
  [arrhenius_fracture/directional_j_positive_v1042.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/directional_j_positive_v1042.py)
- Hazard-energy mechanics observer:
  [arrhenius_fracture/hazard_energy_observer_v10230.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/hazard_energy_observer_v10230.py)
- Observed hazard-energy tip engine:
  [arrhenius_fracture/hazard_energy_observed_engine_v10230.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/hazard_energy_observed_engine_v10230.py)
- Hazard-energy gate implementation:
  [arrhenius_fracture/hazard_energy_gate_v10230.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/hazard_energy_gate_v10230.py)
- Underlying domain J-integral implementation:
  [arrhenius_fracture/j_integral.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/j_integral.py)
- Underlying sharp-front solver transformed by the overlays:
  [arrhenius_fracture/sharp_front.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/sharp_front.py)

The required production convention is:

```text
J_effective = max(J_signed, 0)
```

Negative directional J remains non-driving.  `abs(J)` is not the production
measure, and the first-nonzero sign latch must remain disabled.

## Plastic-flow terminal and accepted plastic work

- Plastic-flow terminal criteria and contour diagnostics:
  [arrhenius_fracture/plastic_flow_terminal_v1042.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/plastic_flow_terminal_v1042.py)
- Accepted constitutive plastic-work and peak-load contour accounting:
  [arrhenius_fracture/plastic_flow_accepted_work_v1042.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/plastic_flow_accepted_work_v1042.py)
- Full-field bulk Peierls/Taylor audited entry inherited from v10.4.1:
  [arrhenius_fracture/sharp_front_v10_4_bulk_peierls_taylor_audited.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/sharp_front_v10_4_bulk_peierls_taylor_audited.py)
- v10.4.1 detailed-balance overlay:
  [arrhenius_fracture/bulk_plastic_detailed_balance_v1041.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/bulk_plastic_detailed_balance_v1041.py)
- v10.4.2 case classifier:
  [scripts/classify_v10_4_2_case.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/classify_v10_4_2_case.py)
- Fracture/plastic-temperature plotter:
  [scripts/plot_v10_4_2_fracture_plastic_temperature.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/plot_v10_4_2_fracture_plastic_temperature.py)

The terminal state is `plastic_flow_no_sharp_fracture`.  It is not a ductile
fracture model.  Plastic dissipation and contour shielding remain diagnostic and
must not be fed into the cleavage hazard or fracture J.

## Inherited-case reuse

- v10.4.1-to-v10.4.2 source and materialized-case verifier:
  [arrhenius_fracture/reuse_v1041_v1042.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/reuse_v1041_v1042.py)
- Earlier v10.4.0-to-v10.4.1 verifier used by some inherited cases:
  [arrhenius_fracture/reuse_v1040_v1041.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/arrhenius_fracture/reuse_v1040_v1041.py)

The v10.4.2 verifier checks target completion, required-file hashes,
detailed-balance provenance, and the complete root-front relation through first
passage.  All 17 completed source cases passed the independent compatibility
audit, but the final generated scheduler still rejected them.

## Generated launcher chain

Read these in order because the executable scheduler is assembled through nested
source transforms:

1. Public wrapper:
   [scripts/run_v10_4_paper_four_class_orientation_rate.sh](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/run_v10_4_paper_four_class_orientation_rate.sh)
2. Reuse-aware builder:
   [scripts/build_v10_4_2_reuse_aware_launcher.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/build_v10_4_2_reuse_aware_launcher.py)
3. Positive-J launcher builder:
   [scripts/build_v10_4_2_positive_J_launcher.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/build_v10_4_2_positive_J_launcher.py)
4. Plastic-terminal launcher builder:
   [scripts/build_v10_4_2_plastic_terminal_launcher.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/build_v10_4_2_plastic_terminal_launcher.py)
5. v10.2.30 rate/orientation builder:
   [scripts/build_v10_2_30_rate_enabled_orientation_launcher.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/build_v10_2_30_rate_enabled_orientation_launcher.py)
6. v10.2.28 base launcher:
   [scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh)
7. Underlying long-R-curve scheduler source:
   [scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves.sh](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves.sh)

The external audit must inspect and execute the **final generated scheduler**.
String-position tests on an intermediate builder are insufficient.

## Validation and regression tests

- Master v10.4 validation command:
  [scripts/validate_v10_4_bulk_peierls_taylor.sh](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/scripts/validate_v10_4_bulk_peierls_taylor.sh)
- Directional-J regression tests:
  [tests/test_v10_4_2_directional_j_positive.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_2_directional_j_positive.py)
- Hazard-energy gate and observer tests:
  [tests/test_v10_2_30_hazard_energy_gate.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_2_30_hazard_energy_gate.py)
- Plastic-flow terminal tests:
  [tests/test_v10_4_2_plastic_flow_terminal.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_2_plastic_flow_terminal.py)
- Initial launcher-adapter tests:
  [tests/test_v10_4_2_launcher_adapter.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_2_launcher_adapter.py)
- Reuse-aware builder test:
  [tests/test_v10_4_2_reuse_aware_launcher.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_2_reuse_aware_launcher.py)
- Bulk plasticity tests:
  [tests/test_v10_4_bulk_peierls_taylor.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_bulk_peierls_taylor.py)
- Bulk provenance tests:
  [tests/test_v10_4_provenance.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_provenance.py)
- Detailed-balance tests:
  [tests/test_v10_4_1_detailed_balance.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_1_detailed_balance.py)
- Campaign contract tests:
  [tests/test_v10_4_1_campaign_contract.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_1_campaign_contract.py)
- Selective-reuse tests:
  [tests/test_v10_4_1_selective_reuse.py](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/tests/test_v10_4_1_selective_reuse.py)

## Package metadata

- Python package/build metadata:
  [pyproject.toml](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/blob/v10.4.2-plastic-flow-terminal/pyproject.toml)

# Local generated data and how to provide it to an auditor

The complete run directories are not committed because they are generated,
large, partially symlinked, and include mutable restart state.  Git now contains:

1. the source and tests;
2. a compact evidence summary with exact measured values and local paths; and
3. a deterministic exporter that packages the relevant local files with hashes.

On the machine containing `/Volumes/Data`, run:

```bash
cd /Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_4_2_plastic_flow_terminal

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate arrhenius-sharp-front-v10

python scripts/export_v10_4_2_audit_bundle.py \
  --output /Volumes/Data/Data/Nanopillar_calculation/v10_4_2_audit_bundle_20260802.zip
```

The script prints the ZIP path and SHA-256 digest.  Provide that ZIP to Claude in
addition to the GitHub branch.  It includes the corrected smoke CSV/JSON data,
energy-gate and geometry-event audits, the 17-case compatibility report, all
materialized reuse audits, campaign/restart logs and PID files, directory
inventories, quarantined partial-case evidence, Git metadata, and a file manifest
with source paths, sizes, and SHA-256 values.

Use `--include-full-logs` only if complete logs are required; otherwise bounded
head/tail excerpts are embedded while full-file hashes are retained.
