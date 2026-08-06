# V11 Crack-Network and Physical Competition Design

## Compatibility-first architecture

The v11 representation is a rooted directed crack graph. `CrackNetworkState`
owns deterministically ordered branch records and global accounting. A branch
record has a stable ID, optional parent ID, generation, initiation event, root,
tip, path/segments, current and historical orientations, status (`active`,
`arrested`, `merged`, `terminated`), physical length, projected extension and
local state references.

Local state ultimately includes cleavage candidate actions/thresholds,
event-length latents, event index/history, moving MPZ/persistent-site state,
signed shielding/source history, fatigue state, energy-gate state, and a
deterministic RNG identity. Global state includes every segment/tip, sorted
active IDs, total physical length, primary projected extension, fracture and
plastic dissipation, branch-creation history, shielding interactions, energy
reservations and a monotonically increasing geometry generation.

The initial implementation is intentionally narrower: an immutable/pure-data
one-tip network with stable IDs, topology validation, length accounting and
canonical serialization. It wraps a copied path only. The existing production
engine remains authoritative and is neither called nor reseeded by the
container. Thus disabling branching creates no new draws, clocks, numerical
operations, output changes or checkpoint bytes.

## Identity and ordering

- Root branch ID is `b00000000`; future IDs are allocated from the committed
  network generation/event counter, never trial order or RNG.
- Candidate IDs combine branch ID with a canonical crystallographic direction
  key; floating angles are not identities.
- Branches serialize by ID, segments by `(branch ID, local segment index)`, and
  active tips by branch ID.
- Ties use `(crossing physical time, candidate kind rank, orientation key,
  branch ID)` solely as a deterministic numerical tie breaker. Kind rank cannot
  grant priority when crossing times differ.

## Candidate generation and competing hazards

At each active tip the mechanics provider supplies admissible crystallographic
directions in the global frame. The current direction is the continuation
candidate; other physically allowed cleavage directions are daughter candidates.
Plastic emission remains a competing process. No random angle or branch
probability exists.

For candidate `j`, store a persistent threshold `Xi_j`, accumulated action
`B_j`, and rate `lambda_j(t)` derived from the same Arrhenius cleavage surface,
orientation-dependent local drive, signed shielding and local state used by the
validated model. Emission uses its existing Arrhenius clock/state evolution.
Over a trial interval all rates are integrated from the same pretrial network
and loading history. Solve every crossing time, choose the global minimum, and
advance physical time once. Losing candidates retain their updated sub-crossing
actions; they are not reset or redrawn.

Continuation and daughter initiation use separate persistent clocks because
they are distinct physical paths. Candidate thresholds are deterministically
streamed by network seed/branch/candidate identity. A candidate's event-length
latent is derived at its threshold draw using the existing threshold-scaled,
mean-preserving mapping; losing candidates retain it. Emission and cleavage
competition is evaluated in the same interval. `no event` means no threshold
crosses before the interval ends.

Crossings equal within a documented time tolerance are a numerical tie, not a
simultaneous energy entitlement. They enter the canonical tie order and only
the first accepted transaction commits; mechanics, hazards and available energy
are recomputed before another event. This prevents two events from spending one
elastic state. A daughter event creates one new daughter tip. The parent remains
active only if its continuation candidate remains physically admissible;
otherwise it becomes arrested. No automatic symmetric second daughter is
created.

## Energy contract

The existing hazard-energy gate supplies the event resistance and fixed-opening
release calculation. A network transaction evaluates the proposed oriented
event length and charges its complete sharp-front fracture work, plastic work
and orientation factor. It records available local/network elastic energy and
all existing signed shielding contributions.

Energy is reserved globally before geometry creation. Reservations have unique
transaction IDs and reduce the common available-energy ledger. A proposal may
commit only when its full dissipation is covered; unused reservation is released.
Two tips can never each see the same unreserved energy. No athermal toughness,
branch threshold, fitted coefficient or alternate gate is introduced. A
daughter and parent continuation, if ever modeled as one compound physical
event, must pay the sum of both surfaces in one atomic reservation; the initial
design treats them as competing single events.

## Geometry transaction

An event snapshots canonical network state, affected local engine/MPZ state,
candidate clocks and RNG states, energy ledger and geometry generation. It then
proposes the existing stochastic event length in its physical candidate
direction and checks zero length, angular resolution, self/existing-segment
intersection, minimum separation, domain boundary, process-zone admissibility
and energy.

Only after all checks pass are the segment, topology, state translation, energy
charge, histories and generation committed atomically. A veto restores the
snapshot byte-for-byte. It consumes no redraw and preserves the candidate's
existing threshold/latent according to the governing v10 geometry-veto contract.
Crossing/merging requires an explicit physically supported merge event; otherwise
intersection vetoes. Arrest is caused by absent admissible drive/energy or a
documented boundary/interaction state, never arbitrary pruning.

## Shielding and process zones

A `NetworkDrivingForceProvider` will return the signed local driving state for
every active tip from the complete geometry. Its contract separates self,
parent, sibling, wake and plastic contributions and states their coordinate
frames. The current prescribed-geometry kernel may serve the one-tip case only.
It cannot be declared additive, rotation-correct or valid for nearby branches
without dedicated verification.

Each persistent site has one owner. When process zones overlap, an interaction
map assigns sites deterministically from physical position and branch geometry,
or a future shared-site law evolves them once and maps their signed effect to
each tip. Copying populations is forbidden. Parent-to-daughter transfer must
conserve emitted, mobile, retained and wake ledgers.

## Restart design

Network checkpoint schema `v11.crack-network.checkpoint/1` contains source
provenance, network seed, canonical branches/segments/active IDs, topology,
geometry generation, global energy/dissipation, per-tip MPZ/source/shielding and
fatigue states, every candidate action/threshold/event-length latent/history,
and RNG bit-generator state or stable stream identity. Arrays use a manifest
with owner ID, field name, shape and offsets.

Loading fails closed on unknown schema, missing/duplicate IDs, broken parents,
cycles, noncanonical ordering, path/tip/segment disagreement, invalid lengths,
action beyond threshold without a pending event, absent RNG state, inconsistent
energy totals, or incompatible source/parameter/geometry provenance. A network
checkpoint is never silently downgraded to the v10 singleton schema.

For one tip, the compatibility adapter continues to use the existing v10.2.30
checkpoint unchanged. Migration occurs only after a test proves uninterrupted
and restarted network trajectories identical for fixed seeds.

## Required qualification sequence

1. Pure one-tip container invariants and canonical round trip.
2. Disabled-mode v10 output/hash equality with no production call-path change.
3. Analytical constant-rate candidate competition and losing-action retention.
4. Transactional geometry/energy rollback with RNG-state hashes.
5. Network shielding provider verification.
6. Multi-tip checkpoint corruption and restart equivalence.
7. Minimal two-branch smoke, then fatigue integration.

Broad multi-tip production and high-cycle acceleration remain explicitly outside
the initial milestone.
