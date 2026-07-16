# v10.0.2.1 wake-ablation routing and diagnostics correction

The v10.0.2 wake-on/off archive demonstrated that the intended ablation was not executed:

- `run_args.json` reported `wake_shielding: true` in both output trees;
- the only run-argument difference was the output path;
- steps, fronts, summaries and crack paths were byte-identical for all three material classes.

The root cause was that `--wake-shielding` uses `argparse.BooleanOptionalAction` with default `true`. Omitting the positive flag does not disable it; the off case must pass `--no-wake-shielding` explicitly.

A second problem affected diagnostics. `UnifiedMPZState.diagnostics()` returns:

- `mpz_active_K_shield_Pa_sqrt_m`;
- `mpz_wake_K_shield_Pa_sqrt_m`;
- `mpz_total_K_shield_Pa_sqrt_m`;
- `mpz_wake_mobile_count`;
- `mpz_wake_retained_count`.

The 2-D CSV writer was still looking for obsolete keys `mpz_K_shield_Pa_sqrt_m` and `mpz_wake_retained_total`, causing zeros even when the state existed.

v10.0.2.1:

1. passes exactly one of `--wake-shielding` or `--no-wake-shielding`;
2. records the resolved wake mode in `v10_0_1_driver_modes.json` and `run_args.json`;
3. maps the CSV fields to the current active/wake diagnostic keys;
4. adds explicit active, wake and total shielding columns plus wake mobile/retained counts;
5. tests the on/off runner tokens and diagnostic-key mapping.

The previous wake-ablation outputs must not be used to infer that the wake is mechanically irrelevant. The resolved-mesh 50 um wake-on baseline remains useful for class-dependent propagation, but its wake diagnostic columns are incomplete and should be superseded for causal wake analysis.
