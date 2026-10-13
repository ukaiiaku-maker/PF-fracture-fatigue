# R8 current-backend/full-spatial integration

## Source identity

- Branch: `codex/r8-current-backend-spatial-integration`
- Calculation commit: `97aaf1cc2a0963f118deab398a17c6cbf3111acd`
- Calculation tree: `e7823788ef1ce8ac8722799b6ff87120e6ea98a6`
- Parent R7 packaging commit: `b9184e7d144fa77d6f41fce8c227b2e36173e782`
- Runner: `scripts/execute_r8_current_backend_spatial.py`
- Python: `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`

The evidence was executed from the exact calculation tree above before this
README and the evidence files were packaged. No barrier, source density,
return fraction, event reward, admission cost, material parameter, or rate
multiplier was changed.

## Implemented boundary

The accepted state's per-front payload now owns either
`LEGACY_UNIFIED_MPZ_V5` or `CURRENT_PERSISTENT_REVERSIBLE_R8`. The legacy
default remains available. The current payload reconstructs the native engine
through its registered factory and restores complete active arrays, absolute
ledgers, stochastic state, geometry, signed drive, and material-frame data.
Root and downstream-child construction inherit the selected strategy.

The current spatial adapter resolves signed Schmid channels from the accepted
FEM tensor and accepted crack tangent. Its native mark and moving-MPZ update
own cleavage time, continuous micro-translation, reward, and successor mark.
Directional mechanics supplies plane selection and the existing topology
admission transaction only. The legacy `1e300` threshold interlock is never
used for the current strategy.

The R7 rate-observable fallback was also corrected: gross source and return
rates are accepted cumulative-ledger increments divided by their actual
consumed interval, not absent result keys interpreted as zero.

## Executed outcomes

All three arms used the same external schedule: 51.1004675 MPa sqrt(m) at
900 K for 1.0 microsecond, followed by 0.25 microsecond at the same load and
temperature. Every arm executed one native `DIRECTIONAL_CLEAVAGE` event,
committed a 5 micrometre increment, re-equilibrated the changed geometry, and
completed the later exposure. Final current-engine time is 1.25 microseconds.

- `no_void`: cavity-free initialization.
- `passive`: the same pre-existing 55 micrometre cavity initialization as the
  active arm; only cavity-to-tip mechanical feedback is disabled by supplying
  the no-void reference tensor. Bulk, cavity, transport, clocks, current tip,
  topology, and admission remain active.
- `active`: the current cavity-bearing FEM tensor is supplied to the signed
  tip channels at every accepted observation.

The active and passive arms retain the 55 micrometre prescribed cavity as a
functional pre-existing fixture; this is not claimed as natural nucleation.
The output records bulk strain/density measures, tip mobile/retained/source/
return state, cavity and vacancy ledgers, directional and native marks,
resolved/path/projected lengths, consumed signed inputs, and topology results.

The interrupted active checkpoint reconstructs all owning state from its JSON
manifest and compressed state file. Its 0.25 microsecond continuation matches
the uninterrupted continuation exactly by accepted-state fingerprint.

## Geometry correspondence

The turning comparison begins from one complete refined pre-event state and
uses the same mechanics, native random owner, duration, mark, reward, and
current engine in the direct and full-spatial paths. The accepted tangent is
6.974492530416284 degrees (error `-1.1369e-13` degrees), the path increment is
5.000000000000006 micrometres, and projected extension is
4.963001541062606 micrometres. The material-fixed wake origin remains at the
pre-event tip while the accepted tip position and material frame rotate.

Direct-current versus spatial native state after the event has maximum active
relative error `6.7763e-20`, maximum ledger relative error `7.5894e-19`, and
exact stochastic-state equality under a declared `1e-12` comparison tolerance.
The later autonomous reduced approximation under the same external load has
population relative error `8.37145e-4` and radius relative error
`3.95323e-8`; these are measured approximation errors, not imported spatial
populations or future event times.

## Verification

- Focused affected suite: `22 passed in 336.32s`.
- New R8 suite alone: `4 passed in 64.23s`.
- Full repository suite: `824 passed in 1814.39s (0:30:14)`.

Primary result: `r8_current_backend_spatial.json` (`passed: true`).
