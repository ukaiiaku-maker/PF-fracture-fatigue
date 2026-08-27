# Canonical PF V2 execution-output policy

This addendum changes only serialization frequency. It does not alter the
immutable V2 condition matrix, material rows, mechanics/source families,
loading rates, random seeds, stochastic lifecycle, or scientific-plan
fingerprint.

Production continuation uses `--save-snapshots 0 --observer-mode off`.
Intermediate mechanical-field images and event-boundary spatial profiles are
not retained. Each run retains its complete scalar step history, front and
crack-path histories, stochastic avalanche geometry transactions, sharp-wake
advance log, summary, launch/result provenance, and final moving-process-zone
field under `mpz_state_snapshots_{temperature_tag}.json` → `final_fronts`.
Selected full image sequences may be reconstructed later as explicitly
labelled replays using the frozen source, material row, loading contract, and
seed.

The family coordinate-envelope tolerance and interpolation-error metadata are
not interchangeable. The former is `1e-10` and controls fail-closed envelope
membership. Empirical leave-one-out spatial cross-validation error is recorded
separately as unavailable (`null`) because the active spatial curves have only
two measured endpoints; it is not assigned the envelope-tolerance value.

The project owner explicitly waived the original 30 GiB free-space launch
gate on 2026-08-27 and will provision additional capacity independently.
