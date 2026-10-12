# PF rate, orientation, and source sensitivities

**Decision:** `COMPLETE_VALIDATED`

All 40 predeclared sensitivity-matrix rows have verified original spatial-PF executions and physical first events.

## Loading rate

Across the five anchors, two arms, and 0.01×–100× range, the largest absolute first-event change from the matched 1× case is 52.12%.

## Crystal orientation

The paired 45° minus 0° first-event changes span -61.91% to +0.00% across the four selected conditions and two arms.

## Source geography

DBTT, 1200 K, accommodation enabled, seed 1733:

| Source geography | First-event KJ (MPa√m) | Change from tip |
|---|---:|---:|
| tip_surface | 52.376 | +0.00% |
| bulk_uniform | 20.934 | -60.03% |
| tip_plus_bulk | 52.414 | +0.07% |

These are question-driven local sensitivities at predeclared anchors, not a Cartesian parameter atlas. The 45-degree operator is the original FEM-measured, two-pass self-consistent signed-kernel family.

Machine-readable report: `review_packet/final_fracture_campaign_a5a5883/PF_SENSITIVITIES.json`

Analysis command: `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python scripts/analyze_final_pf_sensitivities.py`
