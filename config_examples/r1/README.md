# R1 reusable examples

Use the qualified interpreter and `PYTHONPATH=.` from this checkout. Install the
generated four-class runtime input before native fixed-range fatigue work:

```sh
python scripts/install_v10_2_27_four_class_registry.py
```

Resolve or run the restored identities:

```sh
python scripts/run_v10_2_3_shared_reduced.py --help
python scripts/run_shared_fracture_options.py --preset pf_monotonic_tip_persistent --plan-only
python scripts/run_shared_fracture_options.py --preset pf_cavity_v5_single_front --plan-only
python scripts/run_shared_fracture_options.py --preset pf_cavity_v5_disabled_control --plan-only
```

Public positive-R cyclic V5 execution:

```sh
python scripts/run_shared_fracture_options.py \
  --preset pf_cavity_v5_cyclic_single_front \
  --out runs/r1_v5_cyclic_example \
  --source-policy qualified \
  --native-args-json '["--load-factors","1.0,0.2,1.0","--steps","3","--duration-s","1e-5"]'
```

Continue from its pair checkpoint with the identical history identity:

```sh
python scripts/run_shared_fracture_options.py \
  --preset pf_cavity_v5_cyclic_single_front \
  --out runs/r1_v5_cyclic_example_restart \
  --source-policy qualified \
  --restart-from runs/r1_v5_cyclic_example/checkpoint/accepted_v5_pair.json \
  --native-args-json '["--load-factors","1.0,0.2,1.0","--steps","1","--duration-s","1e-5"]'
```

Run the explicitly non-material finite-activity lifecycle fixture:

```sh
python scripts/run_r1_v5_seeded_finite_activity.py --out runs/r1_v5_seeded_example
```

The exact matched full-PF/reduced replay command and its numerical budget are
recorded in `integration_evidence/r1_restoration/EXECUTION_RECORD.md`.
