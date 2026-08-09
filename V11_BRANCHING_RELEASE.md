# v11 Hazard-Branching Production Release

- Release branch: `v11.0.0-hazard-branching-production`
- Release tag: `v11.0.0-hazard-branching-production`
- Qualified solver SHA: `df1c759feafa8e65f5623dfefbd2f8bab44fd59d`
- Release date: 2026-08-09

Qualified campaign: four material parameterizations at 300 K and 1000 K,
theta = 30 deg, seed = 3621, and 1000 um target extension.

Major production physics: explicit sharp crack network; directional Arrhenius
first-passage clocks; thermodynamic topology gate; exact-topology live FEM;
compliance-aware multi-tip kinetics; physical 50 um process-zone/handoff scale;
stochastic exponential thresholds; explicit process-state ownership; full-network
mechanical crack shielding; and no geometric branch pruning.

Parameter registry:
`arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv`

Qualified options:

- `v913_paper_peak01_0242980_persistent_sites`
- `v913_paper_dbtt01_0202500_persistent_sites`
- `v913_paper_weakT01_0129902_persistent_sites`
- `v913_paper_ceramic01_0077080_persistent_sites`

Campaign root:
`runs/v11_four_class_300K_1000K_theta30_seed3621_1000um`

This provenance-only release commit does not change the qualified solver source.
