# Fracture-to-fatigue state contract

Schema: `voiding-v1.0`; file: `analysis_outputs/voiding_v1/voiding_v1_state_schema.json`.

The backend-neutral payload contains capability configuration, candidate sites, cavity objects, first-passage clocks and RNG states, local defect inventory, explicit-boundary node IDs, geometry/component lineage, event history, transaction identity, and separate length ledgers. Enum values are stable strings and SI quantities carry unit suffixes.

The contract permits embryo healing, persistent stable/resolved cavities, signed relaxation in a future defect provider, and transactional crack connection under cyclic loading. It introduces no cyclic kinetic law and modifies no fatigue physics.
