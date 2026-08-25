# oneD V2 predictive result imported for fatigue integration

The completed reduced-model program selected four shared screening rows and established fit-for-purpose onset, topology, and sensitivity fidelity for parameter exploration. The fatigue integration must import the material rows while preserving all existing cyclic state, signed reversible transport, physical-return, hazard, event-length, energy-gate, geometry-transaction, and high-cycle acceleration semantics.

See `ONE_D_V2_TO_FATIGUE_INTEGRATION_HANDOFF.md` for the complete integration contract and `data/oneD_v2_four_class_screening_registry.csv` for the exact parameter vectors.

This handoff is based on `codex/v9.14-minimal-reversible-fatigue` and must not be merged by replacing that branch with monotonic oneD lifecycle code.
