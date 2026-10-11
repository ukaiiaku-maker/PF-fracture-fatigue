# R1 execution record

Interpreter:
`/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`

Environment for tests and Python runs:
`PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Fresh cross-fidelity event/reload gate used the existing wrapper with candidate
`DBTT_A0002277`, 1200 K, theta 45 degrees, 80 MPa sqrt(m) cap, 10 um target,
5 um checkpoint/event quantum, and `KINETIC_MAX_INTERNAL_STEPS=100000`. The
wrapper writes the exact 2-D invocation, replay invocation and comparison log
under `runs/r1_v1023_crossfidelity_reload/`; its committed source is
`scripts/run_v10_2_3_shared_state_equivalence_gate.sh`.

Public cyclic V5 command:

```text
python scripts/run_shared_fracture_options.py --preset pf_cavity_v5_cyclic_single_front --out runs/r1_v5_cyclic_public_qualified --source-policy qualified --native-args-json ["--load-factors","1.0,0.2,1.0","--steps","3","--duration-s","1e-5"]
```

The attestation records clean calculation source
`b8adedc82503a2ff6698d35bf3d03af671ea9f24`, a fresh interpreter, all loaded
module hashes, exact argv, options fingerprint, seed and successful post-source
verification. V5 restart schema is `shared-v5-physical-pair-checkpoint/2`;
reduced replay summary schema is `v10.2.3_shared_reduced_state_equivalence`.

Known failed scientific/numerical attempts are retained under `runs/`: active
reduced attempts with insufficient internal-step budgets, V5 negative applied
opening (now rejected at the boundary), missing generated-registry launches,
and the production negative-R run. The final negative-R run used
`DeltaK=12.0677888 MPa sqrt(m)`, `R=-0.95`, 300 K, seed 2001726, a 10 um target,
1,000,000-cycle cap and 600 s watchdog. It timed out at 273,958 cycles with no
accepted advance, gross source activity 200.43442068988253, gross return
activity zero, and an unresolved process-zone mesh (`hbar_tip=1.385e-6 m` versus
`L_pz=1e-6 m`). None was overwritten or reclassified as a successful physical
result.
