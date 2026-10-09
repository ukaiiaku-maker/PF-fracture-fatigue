# Shared fracture–fatigue options v1 implementation plan

Base: `9e884fb0b0845da621d2612bdf1042e481b8df49` (`v10.2.30-hazard-energy-gated-fatigue-events`). This base owns the persistent-site, signed-state, cyclic first-passage, stochastic event-length, energy-admission and restart lifecycle. It is ancestry, not the final default for every physical option.

1. Add one typed, serializable option contract that separates mechanics, loading, source location, source lifecycle, plasticity/state, crack path, cavity process, kinetics/integration strategy and output retention. Preserve distinct physical alternatives and fail early for unsupported combinations.
2. Add runtime-resolution receipts and compatibility adapters around the existing launchers; do not replace established launcher meanings or install competing monkey patches.
3. Recover donor functions by exact diff: continuous/bulk-source and simultaneous exchange–transport corrections; reversible signed return; V3.6 projection/root/anchor fixes; V12/V13 provider/multifront lifecycle; V5 cavity scheduler. Port only callable dependencies and preserve donor identities.
4. Add a versioned chronological observation schema and stage-aware K-at-crack-length API/CLI. Retain repeated lengths, fronts, loading segments, contour validity and discrete-jump brackets.
5. Preserve old history/field loaders, propagation/resistance, fatigue, branch/cavity and plotting entry points through explicit adapters and regression tests.
6. Keep PF and FEM/CZM as separate repositories/backends. Add a shared comparison manifest and test the clean full-FEM commit/configuration pair; create a companion FEM branch only if a required FEM source change is unavoidable.
7. Run layered unit, full-model option, restart, interaction, analysis/plotting and output-parity tests. Re-analyze protected long-growth data, then run a compact original full-PF long path near 1000 µm if measured peak space remains safe.
8. Commit in coherent feature groups, re-run from committed source, push only this new branch, and verify the remote head.

Constraints: no parameter refit, broad production sweep, historical-data deletion, donor reset/clean, global editable install, force push, default-branch merge, or Stateful Kitagawa work.
