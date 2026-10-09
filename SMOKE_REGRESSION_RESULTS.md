# Smoke and regression results

Outcome: the option-preserving PF integration passed 267 selected PF tests, the pinned full FEM/CZM pair passed 16 tests, and the committed-mechanics full-PF run reached 1002.903 µm extension. The total selected regression count is 283 passed and 0 failed in the recorded release environment.

## Test layers

| Layer | Result | Scope |
|---|---:|---|
| PF options, observers, plotting, V3.6 operator, reversible transport, void restart, interaction integral | 48 passed | typed resolution and provider receipts; repeated-length/stage-aware queries; real projection; signed return; exact element mask |
| PF bulk, V11/V12 branch mechanics, provider transition and restart | 126 passed | full-field bulk evolution; audited branching; multifront topology; mechanically separating wake; primal crack mechanics |
| PF persistent cyclic/event/fixed-DeltaK stack | 71 passed | negative-R propagation, first passage, high-cycle state, atomic event admission, fixed-DeltaK and analysis provenance |
| PF finite/persistent/kinetic source paths | 22 passed | continuum-source, anisotropic-emission and kinetic-tip contracts in a clean interpreter |
| Pinned original full FEM/CZM | 16 passed | theta=0 parity, full-field metadata/parity, cohesive trial and conservative bulk remesh transfer |

PF commands used `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python` (Python 3.12.13) with pytest plugin autoload disabled. The exact file groupings and counts are serialized in `SMOKE_REGRESSION_RESULTS.json`. `compileall`, shell syntax checks and `git diff --check` also passed.

The system base Python 3.13 pytest executable terminated with signal 11 before collection. It is not the qualified project environment. Provider-related groups are intentionally separate interpreter invocations because older compatibility adapters mutate module bindings at import time; runtime option resolution itself verifies providers in isolated subprocesses.

## Developed propagation and K-at-length

The final run used mechanics source commit `baedb069c912db69032eea8b339cb01b4bf97278`, full `sharp_front_2d` spatial mechanics, active spatial process-zone state, 5 µm nominal checkpoints, the DBTT persistent-site material at 300 K and theta=0, and seed 2001733. Only output retention was reduced (`save_snapshots=0`, no requested plots, no MPZ state dumps). The run exited 0 after 201 accepted geometry events at 1002.903 µm projected extension.

| Requested extension | Bracket (µm) | Upper K_I (MPa sqrt(m)) | Upper J (J/m²) | Result |
|---:|---:|---:|---:|---|
| 100 | 99.855–104.855 | 32.0555 | 2309.743 | discrete jump; no interpolation |
| 500 | 498.845–503.697 | 43.5671 | 4266.550 | discrete jump; no interpolation |
| 1000 | 997.903–1002.903 | 39.5423 | 3514.651 | discrete jump; no interpolation |

Compact numeric evidence is in `integration_evidence/new_full_pf_1000um_summary.json`; the source-table plot is `integration_evidence/new_full_pf_1000um_K_history.png`. Raw 54 MB run output is intentionally ignored rather than committed.

The protected archive was also re-read without modification: 12 cases, 207 accepted events per case, each reaching 1000.011 µm. All 36 queries at 100, 500 and 1000 µm returned discrete-jump brackets with no interpolation. See `integration_evidence/protected_1000um_k_at_length.json`.

## Exclusions and limits

Two exploratory runs are not evidence: one disabled spatial process-zone state, and one preceded the immutable final source identity. Their `REJECTED_RUN.md` markers are retained. Compact tests qualify the named combinations and fixtures, not every future material, mesh or option cross-product. No production sweep or parameter refit was run. Stateful Kitagawa was not inspected or changed.
