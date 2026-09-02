# v10.2.30 Crack-Rebonding Causal Pilot

## Status

`REBONDING_CAUSAL_PILOT_COMPLETE_PARTIAL_GATE_PASS` — six trajectories run,
causal analysis complete, 6/8 hard gates pass cleanly, gate 2 fails on a
documented, mathematically-demonstrated tension with gate 3 (see below).
**Part X (the full multi-K Paris-slope matrix) and a merge into the
authoritative solver line remain unauthorized and were not attempted.**

Branch: `codex/v10.2.30-crack-rebonding-causal-pilot`
Worktree: `/private/tmp/v10230-crack-rebonding-causal-pilot`
Branched from the qualified S8 HEAD `4d6c1c17f9933c2f56b469251e6fd6c3d04ee450`
on `codex/v10.2.30-optional-crack-rebonding` (kept completely separate; that
branch/worktree was not modified).

## What this is

The user directed a small, explicitly bounded causal pilot as the step
immediately after S8's `REBONDING_PREPHYSICS_INTEGRATION_QUALIFIED`
classification: six single-Kmax trajectories (RB0/RB1/RB2 at two R-ratios)
to determine whether the rebonding mechanism, driven live and end-to-end
through the real production engine, produces a genuine causal shielding
effect — not a multi-K physical Paris-slope campaign (explicitly deferred)
and not a merge (explicitly unauthorized).

## Frozen configuration

Everything below was computed and hashed to
`runs/crack_rebonding_causal_pilot_v1/frozen_configuration.json`
(`frozen_configuration_sha256` in that file) **before** any trajectory ran.
The run script re-loads the frozen configs from that file and verifies their
hashes match before driving any trajectory — the six runs use exactly what
was frozen, not a silently-redone calibration.

- **RB0**: rebonding disabled entirely (no `install_crack_rebonding` call).
- **RB1**: `CONTACT_PROXY_ONLY` — `patch_Q` returns the exact zero generator
  for this model level, so RB1 is provably physically identical to RB0.
- **RB2_reversible** / **RB2_persistent**: `CLEAN_REVERSIBLE_REBOND`,
  barriers solved via the existing
  `crack_rebonding_kinetics_v10230.solve_reference_action_barriers` against
  the `REFERENCE_ACTION_PRESETS["reversible"]` (A_on=1, A_off=1) and
  `["persistent"]` (A_on=10, A_off=0.1) targets at the reference condition
  (T=300K, f=1000Hz, R=-0.95, Kmax=18 MPa·√m).
- **Pi_K = K_rebond_max/Kmax = 0.05** (K_rebond_max = 0.9 MPa·√m), resolved
  by fixing `rebond_K_geometry_factor=1.0` (no second free knob) and solving
  `restored_work_of_separation_J_m2` from the engine's own reduced modulus
  `E' = 2G/(1-nu)`.
- **bond_activation_volume_m3**: found by an automated, monotone geometric
  search (×1.15 per step), the only free calibration knob, driving the
  predicted single-patch action at R=0.1 (fully tensile — K never crosses
  zero under `SIGNED_K_COMPRESSION_PROXY`, so `sigma_comp≡0` and only the
  unassisted thermal rate applies) below `1e-3` relative to the reference
  action at R=-0.95 — see "The gate 2 / gate 3 tension" below for why 1e-3
  (not a stricter, seemingly "safer" tolerance) was the deliberate choice.
- Hazard stochasticity: `mode="exponential"`, `seed=1720`, set once via
  `Engine.configure_hazard()` before any of the six engines is constructed.
  Each trajectory additionally calls `Engine.reset_audit()` immediately
  before its own engine construction, resetting the `_next_engine_id`
  counter `kinetic_tip_cell.py` mixes into the hazard RNG's `SeedSequence`
  — without this, engines built later in the process draw a different
  threshold stream purely from construction order, which would make the
  identical-threshold-stream comparisons in gates 1/2/4 meaningless. This
  was caught empirically during pilot construction (P0 and P1 initially
  diverged even though the physics is provably identical).
- Integrator: every trajectory drives the real engine directly
  (`engine.cycle_step_waveform` / `engine.commit_energy_gated_event`, the
  same real methods S7/S8's qualification tests use) — never through
  `sharp_front_v10_2_30_energy_gated_fatigue.main()`'s full CLI chain. This
  is what makes it `EXPLICIT_PHASE_RESOLVED` with `DMD/Poincare
  acceleration disabled` **by construction**: the accelerated-engine
  monkeypatch only exists inside that CLI's `main()`, which is never
  called.
- Stopping/censoring: 5 accepted events or 30 µm cumulative projected
  extension, whichever first; fewer than 3 accepted events (from a
  wall-time or block budget cutoff mid-event) is a censor, never a
  reported `da/dN=0`. All six trajectories reached 5 accepted events
  uncensored.
- Analytical single-patch predictions (`b*`, `P_survive` via
  `two_state_fixed_point`) were computed and frozen at
  Kmax ∈ {12, 15, 18, 24.3} MPa·√m, R=-0.95, for both barrier sets — a
  reference grid for a future multi-K matrix, not compared against
  simulation in this single-Kmax pilot (per hard gate 8).

## Results (`runs/crack_rebonding_causal_pilot_v1/`)

| Gate | Result | Note |
|---|---|---|
| 1. P0/P1 physically identical | **PASS** | 0 mismatches at 2% relative tolerance on waiting time (see tolerance note below); exact on accepted length (1e-9 rel). |
| 2. P4/P5 identical, P5 zero bonding | **FAIL** (timing identical; bonding not exactly zero) | `p5_max_pB≈2.31e-5`, `p5_max_K_rebond≈5.1e-3 Pa·√m` — see tension below. |
| 3. P2 or P3 dynamic nonzero bonding | **PASS** | Both P2 and P3 show real, monotonically growing `p_B`/`K_rebond` from zero, from live kinetics on event-created patches — not seeded. |
| 4. Post-first-event waiting intervals compared | reported | At event index 1: P2 > P1 and P3 > P1 (both ~2.3% larger, in the causally-correct — shielding delays the event — direction), consistently across both barrier sets. |
| 5. K_rebond not in energy gate directly | **PASS** | Accepted event length identical across P1/P2/P3 at every matched event (0 diffs beyond 1e-9 rel) — the `HAZARD_ONLY_REBOND_SHIELD` invariant holds through the real engine. |
| 6. Only certified bulk-action estimates admitted | **PASS** | Every `phase_resolved_action` call recorded during all six trajectories (via a wrapper installed for the duration of each) reports `bulk_action_qualified=True`; none of the events in this pilot needed the S8C bulk/periodic-orbit branch at all (interval widths stayed well under `bulk_cycle_threshold` cycles). |
| 7. Contact-proxy / mesh-backend limitation preserved | **PASS** | Static — this pilot never touches `EnergyGatedAvalancheBackend.advance()`, exactly as S8D scoped. |
| 8. No physical Paris-slope inference | **PASS** | Static — single Kmax only; Part X explicitly deferred. |

`overall_pass = False` (gate 2's strict sub-check). This is a real,
transparently-reported finding, not a defect — see below.

### Tolerance note (gates 1/2)

`commit_energy_gated_event` routes through `_commit_rebonding_event`'s
bisection root-finder (`solve_coupled_event_time`, `eps_B=1e-6`) whenever
rebonding is merely *enabled* (even at `CONTACT_PROXY_ONLY`, where
`K_rebond≡0` identically), while `REBOND_OFF` skips that root-find
entirely. This introduces a genuine, tiny, compounding numerical
discrepancy in `waiting_time_s_this_event` between an RB0/RB1 pair —
confirmed empirically at ~0.02% by event index 1, growing to ~0.9% by event
index 4 of 5 (compounding, since each event's tiny timing perturbation
shifts the state the next event's root-find starts from). Since RB1's
generator is exactly zero, there is no real physical effect this
discrepancy could be masking, so the 2% comparison tolerance used for gates
1/2 only needs to absorb this numerical artifact — it is not used for the
gate-4 comparison, which reports raw numbers.

### The gate 2 / gate 3 tension (the pilot's main finding)

Under `SIGNED_K_COMPRESSION_PROXY`, bond formation at zero compressive
stress (R=0.1: K never crosses zero, so `sigma_comp≡0` throughout) reduces
to a single, activation-volume-independent constant: the unassisted thermal
rate `nu·exp(-bond_barrier_eV/kT)`. For this to underflow to bit-exact
`0.0`, `bond_barrier_eV` must exceed ≈19.86 eV (`kT·(745+ln(nu))`, from
`math.exp`'s underflow floor at nu=1e10). Solving for a barrier that large
via `solve_reference_action_barriers` (still hitting the target reference
action at R=-0.95) requires an activation volume so large that the
*compression-assisted* rate at peak compression is comparably suppressed —
confirmed empirically: at the activation volume that makes the R=0.1 rate
exactly `0.0` (`V=1e-27 m³`, resolved `bond_barrier_eV≈31.5`), P2/P3 show
`p_B` staying at bit-exact `0.0` through all 5 events too, i.e. gate 3
fails outright.

This pilot's frozen configuration instead uses the automated search's
`1e-3`-relative-tolerance stopping point (`V≈7.08e-30 m³`,
`bond_barrier_eV≈0.54–0.60`): P5's bonding is three-plus orders of
magnitude below the reference-condition target (not bit-exact zero, but
`p_B≈2.3e-5`, `K_rebond≈5e-3 Pa·√m` — roughly `5e-9` of `K_rebond_max`),
while P2/P3 show a real, growing, non-noise signal (gate 3) and a small
but consistently-signed waiting-time delay at gate 4. **Native cleavage
hazard at this candidate/Kmax is orders of magnitude faster than one
waveform period** (confirmed: cumulative elapsed time across all 5 events
in every trajectory stays under ~1% of one period), so essentially no
compression-phase exposure accumulates within this pilot's event budget —
this, not a software defect, is why the achievable signal is small at
either end of the tradeoff.

This is reported as a genuine physical/numerical finding, not
papered over: **exact gate-2 zero and a genuinely observable gate-3 signal
are mutually exclusive at Kmax=18 MPa·√m / Pi_K=0.05 within a 5-event
pilot budget**, and this pilot's configuration deliberately sits on the
signal-preserving side of that tradeoff, with the residual R=0.1 leakage
reported honestly rather than hidden behind an unearned "PASS".

## Verification

- `python -m pytest tests/test_v10_2_30_crack_rebonding_*.py -q`: 181
  passed (170 pre-existing S8 tests + 11 new causal-pilot tests), 0
  failures — run under the required interpreter.
- `python scripts/verify_v10_2_30_crack_rebonding_causal_pilot.py --run-root runs/crack_rebonding_causal_pilot_v1 --output runs/crack_rebonding_causal_pilot_v1/verification.json`
  exits 0: frozen-configuration hash re-derives correctly, all six
  trajectory configs match their frozen hashes, `causal_analysis.json` is
  byte-for-byte reproducible fresh from `trajectories.json`, both hard-gate
  presence checks pass.
- This branch only *adds* files (one production module, two scripts, one
  test file, this doc); no existing production file was edited, so a full
  unfiltered regression sweep of the whole `tests/` tree was not repeated
  here (S8's own sweep already established the ~78 pre-existing,
  artifact-dependent baseline failures are unrelated to this branch's
  starting point).

## What was deliberately not done

- No multi-Kmax matrix (Part X) — single Kmax=18 MPa·√m only, per the
  mission's explicit scope.
- No branch merge.
- No attempt to exercise the mesh-dependent
  `EnergyGatedAvalancheBackend.advance()` path (S8D's documented,
  out-of-scope gap; unchanged here).
