# R2 execution record

Interpreter:
`/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`

Working tree: `PF-fracture-fatigue_worktrees/r1-reduced-spatial-physics-restoration`

Commands:

```text
PYTHONPATH=. python scripts/run_r2_reduced_runtime_contract.py --out runs/r2_reduced_runtime_contract_fresh2
PYTHONPATH=. python scripts/attest_r2_runtime_install_isolation.py --output runs/r2_runtime_install_isolation/receipt.json
PYTHONPATH=. python scripts/run_r2_autonomous_reduced_fatigue.py --output runs/r2_autonomous_reduced_fatigue
PYTHONPATH=. python scripts/run_r2_reduced_void_reference.py --output runs/r2_reduced_void_reference/result.json
PYTHONPATH=. python scripts/write_r2_candidate_c_receipt.py --output runs/r2_candidate_c_dual/receipt.json
MPLCONFIGDIR=/private/tmp/mpl-r2 PYTHONPATH=. python scripts/run_r2_active_v5_cycle.py --output runs/r2_active_v5_cycle_compact
```

The full runtime receipts, raw a(N), cycle histories, native checkpoints, and
V5 state output are retained under ignored `runs/`. The compact cross-check is
`runtime_summary.json`. R1 evidence is not copied or rewritten here.

Focused R2 plus donor-regression result: `33 passed`.

Full repository regression: `791 passed in 122.13s`.

The native V5 deterministic connection was also attempted with its unmodified
energy admission. `ligament_transaction` returned `RuntimeError: ligament event
rejected`; no bypass or forced topology mutation was applied.

No remote push, material fit, rate scaling, source reduction, site movement, or
historical-data deletion was performed.
