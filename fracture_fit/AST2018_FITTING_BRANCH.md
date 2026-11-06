# Ast 2018 per-temperature joint fitting snapshot

Published snapshot prepared 5 November 2026 from the 4 November handoff. This branch preserves the **current fitting implementation**, including opt-in full-section forest-floor plasticity and signed tip pile-up. It is a spatial **1-D fitting model with scalar cantilever mechanics**, not a validated evolving 2-D production fracture fit. Existing fit quality and physical limitations are described in `../CODEX_AST2018_PER_TEMPERATURE_JOINT_FIT_HANDOFF_20261104.md`.

The publication includes the 31 Python modules required by the fitting/plotting entry points, their five focused test files, the exact native-source pin, and the immutable material/elastic/data inputs required for initialization. `fixtures/ast2018_per_temperature_handoff_20261104/source_snapshot.json` records source SHA-256 values. The scientific source files are copied byte-for-byte from the local working version; the publication adds input preparation, documentation and manifests.

## Preserved model decisions

- Fit each temperature independently: 21, 100, 250 and 500 °C. Do not impose barrier temperature slopes during these fits.
- Preserve the existing Arrhenius opening/emission/glide barrier forms. Family A is the retained EXP-floor branch; family B remains historical code.
- A successful fit must retain active tip emission and report tip activations, shielding and radius change. Low objective with emission switched off is insufficient.
- `full_section` uses forest-floor carriers with `bulk_nucleation=False`; tip-only remains an option.
- The `signed_pileup` option distinguishes same-sign pile-up content from mixed-sign forest interactions. Taylor phi remains uncapped.
- The current elastic field supports the declared 0.98 µm fitting window. Longer measured extensions, bound-limited fits and the hinge-yield/crack-drive problem remain unresolved.

## Setup from a fresh checkout

Use Python 3.12 with NumPy, SciPy, Matplotlib and pytest installed. Native source must be a separate Git checkout at exactly:

`https://github.com/ukaiiaku-maker/Arrhenius_FEM_CZM_MPZ.git`

Commit: `ce2bf9d4e868468b6979f0fc8398de95ce983367`.

The preserved fitting launcher currently reads `/private/tmp/ast2018_native_source`. On the original Mac this is a symlink to a durable checkout. Supply that path (or an equivalent symlink) to the pinned checkout; do not overwrite an existing source directory. The native loader checks its commit and import paths. `AST2018_NATIVE_SOURCE` selects the native-dependent test fixture path; the legacy fitting `SOURCE` constant still uses the path above.

From `fracture_fit/`:

```sh
export PYTHONPATH=..:.
export MPLCONFIGDIR=/private/tmp/ast-mpl
export OMP_NUM_THREADS=1
export AST2018_NATIVE_SOURCE=/private/tmp/ast2018_native_source
python scripts/prepare_ast2018_fit_inputs.py
python -m pytest -q tests/test_ast2018_per_temperature_joint_fit.py tests/test_ast2018_signed_pileup.py tests/test_ast2018_section_plasticity.py tests/test_ast2018_source_bound_roundoff.py tests/test_ast2018_implicit_source.py
```

The input-preparation script verifies fixture hashes, restores the experimental data and exact seed inputs into ignored `runs/` paths, and refuses to overwrite differing existing files. Historical seed rows and the top-three v4 starts per temperature are included only as immutable initialization inputs; they are not published as complete campaign ledgers or new fit results. The original bulk-flow starting values are included wherever available. Generated trajectories and large historical campaigns remain excluded from Git.

Two prior test-fixture errors have been corrected: the elastic check now uses the exact specimen identity at E=404 GPa rather than a rounded K=8.4, and the active plasticity test uses a genuinely active emission fixture. Original numerical tolerances were retained and an activation-count check was added. The full five-file test set passed 36 tests before this publication; see the publication manifest for fresh-checkout validation.

## Fitting and plotting entry points

These commands start real optimization, unlike input preparation and source inspection:

```sh
python -m experimental_fits.ast2018.per_temperature_joint_fit 250 4 A --signed-pileup
python -m experimental_fits.ast2018.per_temperature_joint_fit 500 4 A --emission-active
```

They preserve the handoff's existing `OUTPUT=runs/ast2018_per_temperature_joint_v5` and `PREVIOUS=runs/ast2018_per_temperature_joint_v4`. Prepare a new output campaign before extending the existing local results, as specified in the handoff. No optimization was launched merely to publish this branch.

```sh
python -m experimental_fits.ast2018.plot_per_temperature_joint_fit 250
```

The plotter requires real evaluation logs in the selected output directory. The published immutable fixtures are not substitutes for those logs. `section_plasticity_replay.py` is included as an imported historical seed helper; its older executable replay mode enables bulk nucleation and is not the current approved fitting launcher.

This branch snapshots the fitting work without incorporating concurrent uncommitted parent production-code changes or unrelated local campaigns. The original dirty working copy was preserved during publication using a separate temporary Git checkout.
