# v10.0.2.1 wake-ablation correction

The v10.0.2 wake-on/off archive did not execute an ablation. `run_args.json`
reported `wake_shielding: true` for both output trees, and the steps, fronts,
summary and crack-path files were byte-identical for ceramic, weakT and DBTT.

`--wake-shielding` uses `argparse.BooleanOptionalAction` with default true.
Therefore the off case must pass `--no-wake-shielding`; omitting the positive
flag leaves shielding enabled.

The archive also exposed a diagnostic-schema mismatch. The unified MPZ state
returns `mpz_total_K_shield_Pa_sqrt_m` and `mpz_wake_retained_count`, while the
inherited long-run CSV writer requested the older keys
`mpz_K_shield_Pa_sqrt_m` and `mpz_wake_retained_total`. v10.0.2.1 provides an
entry-point compatibility mapping and records the resolved wake mode in both
`run_args.json` and `v10_0_1_driver_modes.json`.

The previous wake-ablation results must not be interpreted physically. The
resolved-mesh 50 um wake-on baseline remains valid for class-dependent
propagation, but its wake diagnostic columns are incomplete. Rerun only the
matched wake on/off ablation under v10.0.2.1.
