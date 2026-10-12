# R6 current-state matched execution

R6 continues from packaged R5 head `a96c7d9e92fd804d95b9456989e9bc53639a308b`
on branch `codex/r6-current-state-matched-execution`.

The 42 nm V3 donor remains the named
`SUBGRID_ZERO_FAR_FIELD_CORRECTION_42NM` control. Its evolving radius is not
cited as a fracture-feedback result. The separate 55 um off-axis fixture uses
the previously certified finite full-spatial response, a common current-engine
owner, and an identical external load/time schedule in active, passive, and
void-disabled arms. It remains explicitly a pre-existing fixed-cavity fixture;
it is not natural nucleation or a qualified moving-cavity response.

R6 changes the current source refresh from a diagnostic partition sweep to a
transactional full-versus-two-half controller. Private trials include the
complete current engine, stochastic owner, and geometry descriptors. Acceptance
controls population, gross/returned/permanent slip, effective radius, source
area, source/return rates, small absolute action, event identity, and event
time. Unused time is consumed by the fixed-horizon owner. The accepted path is
the two-half path; rejected state is never published.

The reduced/spatial replay is localized causally. It agrees through the first
event mark and event length. Its first structural divergence is the first
post-event geometry commit: the spatial path installs a nonzero tangent and
re-equilibrates/remaps, while the reduced replay retains one-dimensional path
length. The large population residual begins in the next source update. This is
a controlled reduced-geometry limitation, not a stress-only error or exact
spatial correspondence.

Portable inputs are in `dependencies/`. They are byte-identical copies of the
reviewed observation and declared-input JSON files. `preflight/` contains the
exact R5 source archive. Generated execution JSON, restart payloads, commands,
focused tests, and the final full-suite log are added after execution.

The accepted-increment dataflow is emitted by
`arrhenius_fracture.matched_execution_r6.accepted_increment_dataflow`. Bulk FEM
plastic strain remains the vacancy/direct-exposure producer. Tip MPZ emitted
line content and signed return remain a distinct source/transport owner and are
not relabeled as bulk equivalent plastic strain. Cavity-emitted loop swept area
is booked once through its native growth/deposited-forest ledger.

## Results

- Matched finite fixture: passed. Active/passive/no-void arms used the same
  schedule; passive and no-void terminal states match; all arms crossed the
  pulse event, continued to common later time, and restarted exactly.
- Adaptive 8.4 s hold: 26 accepted subsegments (not a fixed 128-slice rule),
  terminal population 1918.31 versus 1913.81 for the 128-slice reference
  (0.235% difference).
- Causal replay: first divergence is the `load_20um` geometry commit. The
  common event advance is 2.2973400956 um; the spatial tangent becomes
  6.97449 degrees while the reduced replay remains one-dimensional. The next
  `load_30um` population error is -55.659%.
- Dynamic zero-correction control: passed, including native event, later
  interval, same-time cavity/fracture root ordering, and exact complete owned
  restart. The 42 nm radius and no-void crack drives are bitwise equal.
- Fatigue: phase, outer-source-refresh, and native-action refinements were
  executed separately at a common 3-cycle endpoint; elapsed time is monotone.
  Accuracy remains explicitly unqualified because no prospective first-event
  tolerance was met/declared for the phase sequence.
- Focused suite: 24 passed in 747.29 s.
- Final suite: 818 passed in 777.07 s.

The supported finite fixture still does **not** qualify a moving-cavity
mechanics response. That limitation is emitted as
`moving_cavity_response_qualified=false`; R6 does not manufacture or infer a
nonzero nanocavity correction.
