# V11 Direct-Kernel Preflight

The v10.2.28 self-generating production preflight was run from the clean v11
workspace before the authoritative two-dimensional reference. Generated files
remain ignored runtime artifacts under `runs/`.

## Command

```bash
OUTROOT="$ROOT/runs/v11_single_front_kernel_preflight" \
KERNEL_CACHE_ROOT="$ROOT/runs/v11_direct_kernel_cache" \
PREFLIGHT_ONLY=1 MAX_JOBS=1 \
bash scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh
```

Result: exit 0, `PREFLIGHT_COMPLETE`.

## Resolved family

- Configuration fingerprint: `4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873`
- Provider: `v10.2.28_direct_prescribed_geometry_fem_v1`
- Family SHA-256: `06b887f47a9d19c467921c60564a2b5568142fad446f44f2c628fc67caf4b8b1`
- Physics fingerprint: `77340db33df73106746728f72d306b3c2a2c0c6af080ff5307795ab56b9f0b5f`
- Resolution: `recalculated` (not reused)
- State count: 7
- Measured extension envelope: 0–1185 µm
- Direct-provider validation: passed
- Production physics modified: false

The resolution file, campaign lock, family, build/validation manifests,
prescribed snapshots, load-invariance reports, normalization and coverage audit
all reside below the v11 workspace. No stored artifact references
`scripts/Archive.zip`, the original v10 campaign workspace, or an old campaign
output. The final family bytes match both the resolution and campaign-lock hash.

The direct build declares:

- `prior_kernel_family_required = false`
- `material_parameter_option_required = false`
- `hazard_seed_required = false`
- `stochastic_trajectory_required = false`
- `production_physics_modified = false`

Private validation creates temporary snapshots and some audit payloads retain
those temporary construction paths. They are not resolved runtime dependencies;
the promoted family and validation manifest point to the persistent v11 copies.

## Bounded two-dimensional reference

The fresh family drove one weak-T case at 700 K, crystal θ=30°, seed 3621,
`max_fronts=1`, and a 10 µm projected-extension target through
`arrhenius_fracture.sharp_front_v10_2_28_audited`. It completed in 86 steps with
three stochastic geometry commits, first passage at 14.9460525662 MPa√m, and
10.3312160824 µm projected / 10.5667663512 µm physical path extension.

The authoritative machine-readable result is
`v11_branching_disabled_baseline.json`. The earlier one-dimensional stochastic
engine run is supplemental only and is not the v11 2-D regression target.
