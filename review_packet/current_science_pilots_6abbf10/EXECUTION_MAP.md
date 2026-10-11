# Current-source pilot execution map

This map separates historical retained evidence from calculations executable on
the current integrated source. A named provider or old report is not presented
as current-source execution evidence.

| Scientific item | Simulation source | Observer/report source | Decision |
|---|---|---|---|
| Opening–accommodation first passage | Current commit `6abbf10`; `sharp_front_v10_2_27_audited.main` over full spatial mechanics | Native accepted `steps`/`crack_path` histories; compact results assembled in this packet | Execute six bounded paired cases |
| Developed propagation near 1 mm | Current commit `6abbf10`; same executor and DBTT-300 K accommodation arm | Native accepted history and discrete event brackets | Rerun current head after short-case pass; prior `baedb06` trajectory is sizing/reference evidence only |
| V11 cavity comparison | Historical finite result at `a2a90cd`; immutable archive | Historical K(a), cavity and field products | Retain as historical evidence; do not relabel |
| V11 corrected-transport suffix | Historical accepted 5.900 s endpoints migrated once into current commit `6abbf10`; `shared_v5_cavity_adapter.main` | Current paired audit and native new checkpoints | Execute one 10 microsecond matched interval with explicit boundary |
| Corrected vacancy transport qualification | Current qualification commits through `610f033` | `TRANSPORT_RESULTS.json` and equation-to-code audit | Preserve; do not rerun as a paper trajectory |
| Original full FEM/CZM | Separate clean checkout at exact commit `2934911`; theta=0 parity entry | Native FEM/CZM audit and histories | Execute one selective 10 micrometre consistency case |
| Theta-40 two-front | Current retained exact restart evidence | Targeted-completion receipt | Preserve; outside this pilot batch |
| V3.6 and branching+cavity composition | Current independent development paths | Their own receipts | Not blockers and not part of this batch |

The full numerical protocol, function identities, parameters, endpoints, and
consumer mapping are machine readable in `PILOT_PROTOCOL.json`.

## Post-execution disposition

- The six PF first-passage cases and current-head 1 mm path passed.
- The V11 migration inputs loaded, but zero current-source suffix intervals were
  accepted because the required directional observation was unavailable after
  bounded refinement.
- The FEM adapter handoff was repaired at local commit `6c129c2`; the pinned
  backend then correctly rejected the mutated live kernel because the required
  immutable `a85b57…` family is not locally available.

See `PILOT_RESULTS.json`, `CURRENT_SCIENCE_PILOT_REPORT.md`, and
`NEXT_BATCH_PLAN.md` for quantitative results and exact preparation gates.
