# Reproduction commands

Use `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`.

```bash
python scripts/design_r10_thermal_accessibility_iteration.py --joint-registry runs/r10_joint_search/joint_level_iteration/joint_level_candidates.csv --out-root runs/r10_joint_search/thermal_accessibility_iteration
python scripts/run_r10_current_fracture_panel.py --help
python scripts/run_r10_retained_bridge.py --help
python scripts/analyze_r10_thermal_accessibility_iteration.py --root runs/r10_joint_search/thermal_accessibility_iteration --continuation-state runs/r10_joint_search/continuation_state.json
python -m pytest -q
```
