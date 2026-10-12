# F1/F4 durable repair record

Date: 12 October 2026. Production execution remains unauthorized.

## Authority and lineage

The C16 paper runner deliberately chose the historical finite-inventory
v10.1.7.4 path for elapsed-time reasons. That choice is retained as historical
alternative physics and is ineligible as an automatic baseline substitute.
The present common-framework monotonic tip baseline instead resolves to the
audited v10.2.27 persistent-site path.

Relevant recovered commits and roles:

- `676b6160`: historical v10.1.7.4 anisotropic finite-source implementation;
  preserved, not selected automatically.
- `39d56ccb`: explicit transport-mode entry and the validated scalar transport
  contract inherited by the current persistent path.
- `6243bc3a`: shared signed Burgers population, Peierls transport,
  mobile/retained exchange, Taylor completion and moving-frame wake state.
- `3380e472`: persistent reusable source with implicit simultaneous
  backstress-limited injection and state-dependent physical multiplicity.
- `8ff99cf3` and `013877dc`: complementarity repair and its audited-entry
  installation.
- `e1b10769`: grid-independent physical front-width correction.
- `d2e7c29f`: audited v10.2.27 executable binding.
- `5a82b6a4`: isolated, restart-aware shared execution.
- `995402f2e30988c032cc966fb46350d82ff12957`: independently cited F4
  material-surface reference; the current `material_manifest.py` behavior at
  the counterexample is unchanged before the diagnostic repair.

Current intended call chain:

`shared_execution_worker` -> `sharp_front_v10_2_27_audited` ->
`sharp_front_v10_2_27` -> `sharp_front_v10_2_22` ->
`sharp_front_v10_2_17` with `AuditedPersistentSiteStateResolvedTipEngine`.
The live MPZ object must bind `_persistent_emit`, `_signed_evolve_scalar` (or
the explicitly requested reversible overlay), `_persistent_advance`, native
mobile/retained exchange and the native emission-rate surface. The finite-site
v10.1 entry remains separately selectable only through
`source_lifecycle=finite_inventory`.

## F1 repair

`persistent_runtime_contract.py` now inspects the callables bound to the actual
evolving object, their source files and hashes. The audited engine checks the
contract after construction and before/after every accepted step. The shared
worker requires the post-run binding receipt for a persistent tip request.
Provider failure, missing required input, runtime-binding corruption and a
simulated scheduler timeout fail closed; none dispatches v10.1 or another
physics family. Output profile does not enter the physics contract.

Active DBTT 950 K native qualification and fresh-process restart results are
recorded in `F1_ACTIVE_NATIVE_QUALIFICATION.json`. The public execution path ran
48 accepted steps, committed a complete outer-plus-kinetic checkpoint, then a
second interpreter restored it and advanced through step 64. The first 48 CSV
rows are text-identical across the boundary. Injection, retained state,
backstress, signed shielding, slip, radius feedback, escape, accepted mechanics
and cleavage-clock evolution are appreciable; the selected transport and
exchange/release paths were evaluated and their actual zero/tiny increments are
reported rather than inflated. No fracture event was forced and no finite-source
substitute was started. If interrupted, use the exact command in that receipt
and retain the same option, kernel, seed, marks, state and physical time.

## F4 repair

The historical `PARAMETER_PARENT_DOSSIER.json` is preserved unchanged as v1.
`ExpFloorBarrier.diagnostic` evaluates derivatives of the same piecewise native
surface used by `values_eV` and `rate`. It reports the zero-stress amplitude
slope, complete fixed-physical-stress/frozen-state derivative, separate
fixed-normalized-stress derivative, activation entropy, stress derivatives,
curvature, activation volume, active floor/gate branches and kinks.

The immutable DBTT emission counterexample at 950 K and
266302770.76474476 Pa returns 5.337216317262275 eV and a complete fixed-stress
thermal derivative near -0.0031065770305 eV/K, distinct from
`dG0/dT=+0.0061668124636635 eV/K`. The corrected native-backed output is
`PARAMETER_PARENT_DOSSIER_V2_NATIVE.json`.

## Impact and unfinished work

- No production campaign or broad material search is authorized.
- Historical C16 v10.1.7.4 finite-inventory results remain preserved and are
  ineligible substitutes for the persistent-source baseline.
- Present persistent-source campaign outputs whose runtime receipts identify
  the intended bindings remain valid; the repair did not alter their equations.
- The v1 parameter dossier remains a historical erroneous diagnostic. Selection
  coefficients and numerical arrays were not changed; current consumers use the
  versioned native-backed v2 dossier.
- No current specimen result was found to depend on the abbreviated dossier
  derivative, so no production specimen array is queued for rerun. Future
  production work must use the enforced persistent path and is still unauthorized
  on this computer.
- Parameter coefficients and temperature laws remain unchanged. The known
  DBTT upper-temperature domain problem remains a material-input decision.
