# V5 semantic-hardening checkpoint

Status: `AUDIT_REPAIR_IN_PROGRESS`; broad finalization remains prohibited.

This checkpoint follows, and does not rewrite, exact green checkpoint
`2b39339d7738cc3d251146b69f10ce1a9e0e17c4`.

The active-front ledger now distinguishes connected cavity span from traversed
cavity span. Connection stops at the near boundary. Downstream first passage
books the centered projected diameter and the new sharp segment separately.

The solver-backed causal comparison uses one frozen incident element and records
its node, element, operator identity, and unit recovery weight in every row.

`cavity_free_surface_certificate` certifies only a closed traction-free cavity
cycle. The separate `crack_void_connection_certificate` additionally checks the
intended graph endpoint, graph/cavity incidence, absence of a surviving bridge,
crack and wake exclusion from the cavity, and a single combined incidence
identity.

Single-arm V5 transactions filter proposal selection to one-arm proposals.
Unselected simultaneous events remain pending at the original completion time
for reconsideration on rebuilt geometry.

Interior oblique ray/cavity-edge intersections are inserted as explicit
boundary nodes inside the isolated trial before graph realization and support
rebuilding. The semantic suite includes centered, spatially offset, and oblique
transactions plus rollback after intersection alignment.

Initial seed area is part of finite defect inventory and is debited atomically.
New causality, inventory, and combined-topology evidence predicates resolve raw
source rows and recompute their scientific decisions.

Repository-wide CI is not green. The machine-readable comparison at
`artifacts/voiding_v5_semantic_hardening/general_ci_inheritance.json` records the
same seven failure identities at the exact base and retained V5 checkpoint:
`V5_INTRODUCED_NEW_GENERAL_CI_FAILURES=NO_OBSERVED` and
`GENERAL_REPOSITORY_CI=FAIL_INHERITED_BASELINE`.
