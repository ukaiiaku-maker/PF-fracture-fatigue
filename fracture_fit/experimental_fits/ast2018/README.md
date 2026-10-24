# Ast 2018 tungsten joint fit

This package fits the supplied Ast 2018 data with the pinned v9.13 stateful
0-D reduction and confirms the retained material vector with its paired
spatial 1-D state model. It does not call a 2-D solver. The native source is
loaded from an explicit checkout and its commit and module hashes are saved
in each campaign configuration.

From `fracture_fit`, using an environment with NumPy, SciPy, Matplotlib, and
the native source dependencies installed:

```sh
python -m experimental_fits.ast2018.run_all \
  --native-source /path/to/Arrhenius_FEM_CZM_MPZ \
  --archive Ast2018_tungsten_joint_fit_data.zip \
  --run-dir runs/ast2018_joint_fit_v1 --workers 8
```

The launcher checkpoints the space-filling search, bounded direct
refinement, numerical convergence, paired confirmation, conditional holdouts,
sensitivity/profiles, stochastic ensembles,
barrier surfaces, figures, and `FIT_REPORT.md`. It skips completed stages on
resume. It verifies all 59 supplied files against the archive's `SHA256SUMS.txt`
before starting or resuming. Keep the native checkout at the pinned commit recorded in
`ANALYTICAL_1D_MODEL_REFERENCE.txt` and the package source fixed while a search is running;
the campaign configuration records the adapter hash. Search proposals and
completed scores are append-only JSONL under `runs/`.

The complete launcher limits each initial search vector to 900 seconds. A
timeout is recorded as a numerical rejection with condition progress and a
runtime audit plot. It does not change the material law or a physical rate.

For a single stage, use `python -m experimental_fits.ast2018.campaign --help`,
`python -m experimental_fits.ast2018.refine --help`, or
`python -m experimental_fits.ast2018.confirm --help`. The complete launcher is the supported
reproduction command. Its `--workers` limit is eight and its default initial
search is 128 candidates for the primary source scenario, followed by an
independent 32-candidate alternative-source search warmed from the primary
solution.

The fit uses a deterministic unit-action first-passage trajectory as its
working statistic. Common-seed exponential-threshold ensembles are saved
separately. The declared cantilever dimensions, notch radius, 20 nm physical
event advance, and initial compliance convention are explicit assumptions;
read the final report and sensitivity tables before interpreting fitted
barriers. Numerical load-step size is separate from event advance.

The FIT_REPORT gate requires both 31-condition paired confirmations, complete
ensemble and sensitivity files, full J-curve support audits, barrier tables,
and PNG/SVG figure pairs. A missing artifact causes report generation to
fail rather than presenting an unfinished campaign as a fit.
