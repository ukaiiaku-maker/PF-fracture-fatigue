# Conservative positive vacancy-transport closeout

Production transport is now `MIXED_RT0_P0_CONSERVATIVE_POSITIVE_FLUX_CORRECTION`.
The centered RT0/P0 solve remains the high-order reference.  When, and only
when, its cell state is negative, the registered shared-face flux vector is
minimally projected onto positive backward-Euler cell balances.  Concentration
is reconstructed from the balance equation; it is not clipped or renormalized.
The unchanged `0.01` dilute upper limit remains fail-closed.

Simulation source is `854d32d`; the spatial-convergence test was added at
analysis/test source `d57087c`.  Full evidence is under
`integration_evidence/positive_transport_854d32d/` and is intentionally not in
Git history.

The exact retained `0.50002 -> 0.60002 s` failure now passes in both paired
arms.  The pair reaches `1.00002 s` without a cavity or fracture event; that is
a valid censored no-event result, not a voiding prediction.  Interrupted and
uninterrupted accepted states are bitwise identical.

The original Neumann refinement table is correction-inactive. A supplemental
full-order graded/sheared refinement now exercises the correction while
comparing the concentration field, reservoir flux, site capture, and birth
action. The errors fall toward the 4,096-cell full-order reference, while the
integrated correction collapses and is inactive on that finest mesh. The ADMM
solve now has an explicit KKT acceptance gate, and nonlinear acceptance requires
the final capture active set to match the assembled reaction coefficients. See
`docs/v5_vacancy_transport_equation_to_code.md` and `TRANSPORT_RESULTS.json`.

The earlier 25-to-55 micrometre cavity run remains a test-only extreme-reservoir
fixture. A targeted archive lookup subsequently recovered the independently
evolved, matched full-order V11 permitted/suppressed pair. Its bounded physical
result and exact provenance are recorded in `PHYSICAL_CAVITY_RECOVERY.md`.
Every one of its 781 retained rolling states used the centered RT0/P0 path; the
historical positivity limiter was inactive. The trajectory is retained under
its actual launch source and is not rebadged as a new-flux calculation. The
branching-plus-cavity preset likewise resolves all independent owners but has
no qualified shared executor (`restart_family=unmapped`); this genuinely new
composition remains independently pending while qualified single-front cavity
and multifront modes remain preserved.

The requested active V3.6 restart addition could not be honestly promoted from
the established loads: the highest registered load, 15.462474735549586
MPa sqrt(m), produced only `2.561535e-40` maximum projected plastic distortion
after two phases.  The existing exact six-phase low-activity restart evidence
is preserved.

The external volume `/Volumes/My Passport` was present but mounted read-only.
No supplemental duplicate was attempted; all local evidence was retained.
