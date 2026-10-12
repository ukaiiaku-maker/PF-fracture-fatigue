# Common fracture framework release contract

This release is a code and protocol product. The existing manuscript package is
frozen as an intermediate record and is not regenerated here.

## Independent spatial realizations

- PF identity: this repository, full spatial sharp-front/finite-element
  mechanics family.
- FEM/CZM identity: the separately pinned
  `fem-czm-current-pf-companion` repository. Its mesh, crack representation and
  cohesive state remain native. It is not source-merged into PF.
- Shared objects: material/activation definitions, source/support terminology,
  loading/time conventions, observer contracts, constitutive benchmarks and
  common-reference protocols.

## Release gates

A capability is `production_ready` only when its requested option reaches a
native consumer in a fresh worker, has active-state restart evidence where the
state matters, and passes its declared numerical/physical qualification. A
component import, topology fixture or retained file is not enough.

Current gates are machine-readable in `common_framework_release/FRAMEWORK_STATUS.json`.
In particular:

- portable option resolution, persistent/finite/current lifecycles, tip-only,
  pure bulk, tip-plus-bulk, monotonic/cyclic/hold, reversible signed state,
  long-growth analysis and complete restart have executable native paths;
- recursive physical branch birth plus developed daughter continuation remains
  a production qualification gate, not a claim established by a restored
  two-front fixture;
- cavity topology/restart and current physical evolution are preserved, while
  the primary P7 comparison is gated on the candidate-rate interpretation;
- identical-mesh PF/FEM elasticity is exact, but production mesh/support
  sequences must converge before cross-representation science is released;
- the prospective matrix emits native arguments only for resolved adapters;
  gated P3/P4/P6/P7/P8 configurations have an empty argument vector and an
  explicit `UNRESOLVED_ADAPTER_GATE` status rather than inheriting a nearby PF
  command line;
- the cohesive landscape module is a verified constitutive analysis interface;
  it does not silently change the current FEM/CZM degradation law.

## Test layers

1. Portable code suite: committed source and compact/generated fixtures only.
2. Native integration suite: storage-light full-order execution, active state,
   restart, event and analysis paths.
3. External-evidence qualification: requires every mandatory artifact in the
   content-addressed manifest; absence is `INCOMPLETE`, never an optional green
   skip.
4. Publication/export checks: separate and deferred for this mission.

Commands are recorded in `common_framework_release/COMMANDS.md`. Production
campaign execution is disabled in its generated manifest until all mandatory
framework gates pass and the cavity interpretation is selected.
