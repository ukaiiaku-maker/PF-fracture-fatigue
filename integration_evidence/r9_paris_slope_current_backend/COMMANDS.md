# R9 replay commands

Run from the repository root.  The registry locations and their SHA-256 values
are also frozen in `production_preflight.json`.

```sh
OPT=/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/v3-fatigue-informed-joint-barrier-search/analysis_outputs/v3_1_coupled_slope_continuation/v3_1_optimized_candidate_registry.json
PARENT=/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/v3-fatigue-informed-joint-barrier-search/analysis_outputs/v3_fatigue_informed_joint_barrier_search/v3_complete_candidate_registry.json

python scripts/run_r9_paris_baseline.py --optimized-registry "$OPT" --parent-registry "$PARENT" --candidate-id V3_ceramic_V2_P25_007__V3_1_TRUST1 --out-root runs/r9_anchor_seed1720 --loads 4.781762498950186,8.598738351065925,15.462474735549586,27.809,50 --seed 1720 --target-events 20 --target-extension-um 100

python scripts/analyze_r9_paris_baseline.py --case runs/r9_anchor_seed1720/Kmax_4p7817625/result.json --case runs/r9_anchor_seed1720/Kmax_8p59873835/result.json --case runs/r9_anchor_seed1720/Kmax_15p4624747/result.json --case runs/r9_anchor_seed1720/Kmax_27p809/result.json --case runs/r9_anchor_seed1720/Kmax_50/result.json --out-root runs/r9_paris_five_point_analysis

python scripts/analyze_r9_seed_confirmation.py --case runs/r9_anchor_seed1720/Kmax_4p7817625/result.json --case runs/r9_anchor_seed1720/Kmax_50/result.json --case runs/r9_confirm_low_s2718/Kmax_4p7817625/result.json --case runs/r9_confirm_high_s2718/Kmax_50/result.json --case runs/r9_confirm_low_s3141/Kmax_4p7817625/result.json --case runs/r9_confirm_high_s3141/Kmax_50/result.json --out-root runs/r9_seed_confirmation

python -m pytest -q tests/test_paris_slope_r9.py tests/test_reduced_persistent_reference_r2.py tests/test_resolved_stochastic_fatigue_r4.py
python -m pytest -q tests --junitxml=integration_evidence/r9_paris_slope_current_backend/full-suite.xml
python scripts/package_r9_paris_evidence.py
(cd integration_evidence/r9_paris_slope_current_backend && shasum -a 256 -c PACKAGE_SHA256.txt)
```

The primary archive stores the original separately launched case directories,
so its paths differ from the compact single-command replay root above; the
physical inputs and random ownership are identical.
