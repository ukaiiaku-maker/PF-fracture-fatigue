# Restore and storage boundary

Updated 2026-10-12T23:32:48.939068+00:00. Production execution is not authorized.

- PF checkout: `/Volumes/Data/working-papers/Fracture/shared-fracture-fatigue-options-v1` at coupled calculation/test head `8982d4222970d24a1b476a601996cc791788402e`; compact qualification record commit `9c122ea65fa7058caa79a7792c90b06abf6802e6` passed 901 tests in a real clean clone with mandatory manifest evidence restored.
- Independent FEM/CZM checkout: `/Volumes/Data/working-papers/Fracture/fem-czm-current-pf-companion` at `90cb6671523bb676cb241eb850b7f5fd41cf5461`.
- Python environments: `arrhenius-sharp-front-v10-codex` and `arrhenius-fem-czm`.
- External evidence remains referenced by the release manifest and is not duplicated here. Restore only the dependency subset required by a qualification.
- Capacity is admitted from free bytes against **peak transient + reserve**, one substantial worker at a time. See `QUALIFICATION_ONLY_MANIFEST.json` for the measured snapshot and per-job estimates.
- The earlier 28.35 GiB figure is a sum of retained-output estimates, not a peak-space proof. Current free space admits the preliminary peak estimates, but Q-PF-FEM-3LEVEL and Q-BRANCH-RECURSIVE remain unexecuted because their named adapters are unresolved—not because storage was falsely declared inadequate.
- Retain configuration/source identities, scalar histories, residual and convergence tables, selected accepted states and restart inputs. Do not retain movies, dense field sequences, per-iteration vectors or duplicate external evidence by default.
