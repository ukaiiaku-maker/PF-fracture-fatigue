# Voiding V3 evidence supersession

PR #60 is retained as historical scaffold evidence only. Its valid scoped
results are the source-hash audit, default-off standalone shell, explicit
cavity geometry, one cavity-only FEM smoke solve, isolated localized-clock
tests, ray-circle intersection and length-ledger tests, and standalone JSON
round trip.

The former broad gates are revoked:

- `V12_VOIDING_DISABLED_NEUTRALITY`: `OPEN`
- `V12_EXPLICIT_CRACK_VOID_STATIC_MECHANICS_QUALIFIED`: `OPEN`
- `V12_CRACK_VOID_TRANSACTION_QUALIFIED`: `OPEN`
- `V12_VOID_LIFECYCLE_QUALIFIED`: `OPEN`
- `V12_ONE_VOID_END_TO_END_DEMONSTRATED`: `OPEN`
- `V12_VOID_PROMOTION_AND_GROWTH_QUALIFIED`:
  `FAIL_INCORRECT_3D_INVENTORY_AND_NO_GEOMETRIC_PROMOTION`
- `V12_BOUNDED_NATURAL_STOCHASTIC_CASE`:
  `NOT_RUN_THRESHOLD_ONLY_DIAGNOSTIC`

Future qualification must be based on runner-generated production case rows
with state fingerprints, executed operations, solver observables, and direct
comparisons. The original artifacts remain unchanged and must not be read as
production qualification.
