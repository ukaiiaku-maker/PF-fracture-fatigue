# V11 Unchanged Baseline Tests

## Environment

- Starting source HEAD: `9e884fb0b0845da621d2612bdf1042e481b8df49`
- Provenance-only HEAD while tests ran: `44f1b8d`
- Environment: `arrhenius-sharp-front-v10`
- Python: `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10/bin/python` (3.12.13)
- Package: `/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v11_branching/arrhenius_fracture/__init__.py`

No package source had been changed when these commands ran.

## Repository-wide baseline

Command:

```bash
conda run -n arrhenius-sharp-front-v10 python -m pytest -q
```

Result with normal process-table access: **529 passed, 6 failed, 0 skipped**
in 15.85 seconds.

An initial sandboxed invocation reported 528 passed and 7 failed in 24.72
seconds. Its additional failure was
`test_stage3_status_script_parses_and_reports_missing_run`, caused solely by
the sandbox denying the status script access to `ps`. The normal-access rerun
passed that test.

The six reproducible pre-existing failures are:

1. `tests/test_real_signed_builder_v10212.py::test_review_builder_emits_artifact_consumable_by_production_loader`
   rejects the fixture's response schema before producing the atlas.
2. `tests/test_v10214_capture_model_id.py::test_v10214_capture_model_id`
   expects the old v10.2.14 identifier while production exports v10.2.27.
3. `tests/test_v10214_response_model_id.py::test_v10214_response_model_id`
   expects the old v10.2.14 identifier while production exports v10.2.27.
4. `tests/test_v10_2_27_zero_event_summary.py::test_zero_event_summary_is_recorded_explicitly`
   expects a summary key that is absent.
5. `tests/test_v10_2_27_zero_event_summary.py::test_empty_geometry_rejects_nonzero_advances`
   expects an inconsistency exception that is not raised.
6. `tests/test_v10_2_29_vhcf_nonlinear_selector.py::test_delegate_reenters_selector_under_global_cycle_cap`
   expects `force_cycles` to be cleared, but the delegate forwards `123.0`.

These failures were reproduced before any v11 source change. No version or
model-identifier assertion was changed to make the baseline green.

## Focused authoritative contracts

Command:

```bash
conda run -n arrhenius-sharp-front-v10 python -m pytest -q \
  tests/test_stochastic_hazard_pilot_v10172.py \
  tests/test_anisotropic_emission_v10174.py \
  tests/test_v10221_persistent_site_source.py \
  tests/test_v10221_no_source_inventory.py \
  tests/test_physical_shielding_v1022.py \
  tests/test_signed_burgers_shared_v1025.py \
  tests/test_kinetic_tip_cell_v101.py \
  tests/test_stochastic_avalanche_pilot_v10173.py \
  tests/test_stochastic_avalanche_deterministic_equivalence_fix.py \
  tests/test_v10_2_30_hazard_energy_gate.py \
  tests/test_v10_2_30_energy_gate_contract.py \
  tests/test_v10_2_30_transactional_engine.py \
  tests/test_v10_2_30_event_growth_v5.py \
  tests/test_v10_2_30_high_cycle_engine.py \
  tests/test_v10_2_30_high_cycle_launcher.py
```

Result: **79 passed, 0 failed, 0 skipped** in 7.99 seconds.

This focused set covers stochastic cleavage first passage and emission,
persistent sites, signed shielding, moving-tip state, stochastic event length,
energy-gated geometry commit/rollback, event-to-event fatigue growth,
checkpoint/restart continuation, production launcher provenance, and generated
launcher repository-root binding.

## Baseline classification

- Newly observed v11 failures: none.
- Reproducible pre-existing failures: six.
- Environment-only failures: one in the sandboxed run; absent with required
  process-table access.
- Skips: none reported by either authoritative invocation.
