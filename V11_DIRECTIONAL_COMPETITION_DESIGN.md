# V11 Analytical Directional Competition Contract

## Milestone boundary

- Actual branching implemented: no
- Production driver modified: no
- FEM topology trials implemented: no
- Hazard equations modified: no
- Athermal fracture criterion added: no
- Empirical branch probability added: no
- Completed action preservation: implemented
- Energy reservation: transactional only

This milestone supplies a pure state and decision model. It consumes canonical
crystallographic candidates, externally supplied signed directional `J`, and
candidate rates previewed through the existing cleavage engine. It neither
creates geometry nor enters the accepted single-front call path.

## Physical identity and inventory

A candidate identity is derived from its plane family, crystallographic
variant, normalized global direction and an explicit crystal-orientation
convention. Coordinates are quantized only after normalization and signed zero
is canonicalized. List position, helper enumeration order, front/daughter
numbers and ordinary floating-point formatting are excluded from identity.

Candidate direction and normal must be finite unit vectors, orthogonal within
tolerance, with positive finite `gamma_rel`. Inventories reject duplicate IDs
and serialize in canonical ID order. The adapter to `crystal.py` accepts the
existing tungsten cleavage-trace records and applies only exact validity,
positive-forward and positive-`gamma_rel` filters; no legacy branch heuristics
or replacements are used.

## Directional drive and rates

The drive adapter uses `J_positive = max(J_signed, 0)` and
`K_directional = sqrt(Eprime * J_positive)`. It never takes `abs(J)`, latches a
sign, or converts negative `J` into drive. Candidate-specific positive drive is
mapped through the existing engine's `sigma_tip` and `lambda_cleave` methods.
The existing relative plane resistance is applied as a rate-driving scale
`K/sqrt(gamma_rel)`, consistent with energy release divided by relative surface
work. The engine is deep-snapshotted before and after preview; mutation is a
hard error. No cleavage barrier or cooperative-hit equation is copied.

## Exact constant-rate transaction

Accepted action is dimensionless integrated physical hazard. For constant rate,
`B1 = B0 + lambda*dt`. Every integer boundary strictly above the initial action
and at or below `B1` becomes a pending completed event at
`t0 + (k-B0)/lambda`. The full action and fractional residual are preserved;
multiple crossings are allowed. Preview returns an immutable transition and
does not mutate state. Commit verifies the proposal began from the exact current
state, then replaces accepted action and appends events. Event consumption is a
separate explicit transaction.

The interval interface accepts exact integrated increments and crossing-time
records so a future FEM/ramped-rate adapter can supply quadrature without
changing state semantics.

## Correlation, proposals and ties

Two physically distinct first-pending events are eligible for a joint action
when their completion-time difference is at most the existing cleavage
correlation interval `tau_c`, including equality. Proposal construction creates
one-arm actions for each candidate and at most two-arm correlated actions. Each
proposal names exact immutable event IDs/ordinals and consumes nothing.

Temporal separation greater than the numerical tolerance gives the earlier
event priority. Truly degenerate alternatives are ordered by a SHA-256 key over
the global hazard seed, network competition-event index, and canonical physical
member IDs. This key resolves equivalent alternatives only; it changes no rate,
admissibility or permission to branch.

## Reservation ownership

Reservation is not an energy calculation. It gives one proposal exclusive
ownership of its pending event IDs while an external whole-topology energy trial
is performed. An event cannot belong to two active reservations. Acceptance
consumes exactly the reserved events; rejection releases ownership and consumes
nothing. Immutable snapshot/replace transitions provide rollback semantics.

## Serialization and restart mapping

Directional state uses `v11.directional-competition/1`, canonical JSON fields
and candidate/event/reservation ordering. Loading rejects nonfinite values,
unknown candidates, duplicate events, inconsistent ordinals, missing reservation
members and multiply owned events. Identical state yields identical bytes.

The existing `v11.crack-network/1` one-tip representation is unchanged and
remains readable byte-for-byte. Future `v11.crack-network/2` migration will add
an optional directional-competition payload beneath each tip while mapping an
absent payload to “analytical competition not initialized.” It will not replace
the v10.2.30 checkpoint until uninterrupted/restarted topology trials qualify.

## Future coupling

The FEM adapter will supply positive signed directional `J` per canonical
candidate. Pending actions will reserve exact stochastic rewards, propose
reversible topology, evaluate whole-topology energy release against
hazard-derived dissipation, then either accept and consume or reject and release.
This analytical layer deliberately makes none of those topology or energy
decisions.
