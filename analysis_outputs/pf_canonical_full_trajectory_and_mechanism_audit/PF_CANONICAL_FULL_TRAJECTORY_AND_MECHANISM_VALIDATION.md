# PF Canonical Full-Trajectory and Mechanism Validation

## Result

- Focused full-trajectory/mechanism audit: **15 passed**.
- Combined canonical PF campaign and new audit: **68 passed**.
- Full repository suite: **600 passed, 7 legacy failures**.
- New failures relative to the qualified baseline: **0**.
- `compileall`: **passed**.
- `git diff --check`: **passed**.
- CSV/Parquet schema and round-trip checks: **passed**.
- Raw steps SHA-256 checks: **288/288 passed**.
- Recorded artifact-hash checks: **passed**.
- Deterministic regeneration: **byte-identical** trajectory and figure-manifest
  fingerprints across two clean regenerations.
- Visual QA: **91/91 PNG figures inspected** through five contact sheets; passed.

## Unchanged legacy failures

The same seven pre-existing failures remain in:

1. `test_real_signed_builder_v10212.py`
2. `test_v10214_capture_model_id.py`
3. `test_v10214_response_model_id.py`
4. `test_v10215_stage3_schema_and_status.py` (sandboxed `ps` permission)
5. two checks in `test_v10_2_27_zero_event_summary.py`
6. `test_v10_2_29_vhcf_nonlinear_selector.py`

No production code was changed to address or mask these unrelated failures.

## Safety

No stochastic PF trajectory or FEM/CZM run was launched. No material row,
barrier, equation, lifecycle policy, wake rule, canonical trajectory, or raw
result was changed. The only mechanics execution was the authorized
deterministic zero-history PF sharp-wake orientation/source diagnostic.
