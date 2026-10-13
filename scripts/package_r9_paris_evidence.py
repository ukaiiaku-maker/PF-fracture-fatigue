#!/usr/bin/env python3
"""Create a compact, reviewable R9 evidence package from completed runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"integration_evidence/r9_paris_slope_current_backend"


def copy(source: Path, relative: str):
    target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    primary={
        "low":ROOT/"runs/r9_anchor_low_seed1720/Kmax_4p7817625/result.json",
        "middle":ROOT/"runs/r9_anchor_middle_seed1720/Kmax_8p59873835/result.json",
        "upper":ROOT/"runs/r9_anchor_upper_seed1720/Kmax_15p4624747/result.json",
        "high1":ROOT/"runs/r9_anchor_high1_seed1720/Kmax_27p809/result.json",
        "high2":ROOT/"runs/r9_anchor_high2_seed1720/Kmax_50/result.json",
    }
    controls={
        "v3_low":ROOT/"runs/r9_control_low_seed1720/Kmax_4p7817625/result.json",
        "v3_mid":ROOT/"runs/r9_control_mid_seed1720/Kmax_15p4624747/result.json",
        "v3_high":ROOT/"runs/r9_control_high_seed1720/Kmax_50/result.json",
        "dbtt_low":ROOT/"runs/r9_dbtt_low_seed1720/Kmax_4p6283897/result.json",
        "dbtt_mid":ROOT/"runs/r9_dbtt_mid_seed1720/Kmax_13p0301395/result.json",
        "dbtt_high":ROOT/"runs/r9_dbtt_high_seed1720/Kmax_36p6832844/result.json",
    }
    confirmations={
        "low_seed2718":ROOT/"runs/r9_confirm_low_s2718/Kmax_4p7817625/result.json",
        "high_seed2718":ROOT/"runs/r9_confirm_high_s2718/Kmax_50/result.json",
        "low_seed3141":ROOT/"runs/r9_confirm_low_s3141/Kmax_4p7817625/result.json",
        "high_seed3141":ROOT/"runs/r9_confirm_high_s3141/Kmax_50/result.json",
    }
    for name,path in {**primary,**controls,**confirmations}.items():copy(path,f"trajectories/{name}.json")
    for folder,label in (("r9_paris_five_point_analysis","primary"),("r9_control_three_point_analysis","v3_control"),("r9_dbtt_three_point_analysis","dbtt_control"),("r9_seed_confirmation","seed_confirmation")):
        for path in (ROOT/"runs"/folder).iterdir():
            if path.is_file():copy(path,f"analysis/{label}/{path.name}")
    preflight=ROOT/"runs/r9_anchor_low_seed1720/production_preflight.json";copy(preflight,"production_preflight.json")
    summary=json.loads((DEST/"analysis/primary/slope_width_summary.json").read_text())
    seeds=json.loads((DEST/"analysis/seed_confirmation/seed_confirmation_summary.json").read_text())
    readme=f"""# R9 restored-current Paris-slope result

The exact archived `V3_ceramic_V2_P25_007__V3_1_TRUST1` row was bound to the
R8 persistent/reversible current backend at 300 K, R=0.1 and 1000 Hz.  Every
amplitude began from the same virgin physical state and its own PCG64
seed/substream.  The five seed-1720 trajectories each reached 20 events and
117.7965 micrometres.

## Result

All four adjacent developed secants are inside 3.4--4.6.  The largest connected
interval is DeltaK {summary['largest_connected_m4_interval']['DeltaK_low']:.6g}
to {summary['largest_connected_m4_interval']['DeltaK_high']:.6g} MPa sqrt(m),
with {summary['largest_connected_m4_interval']['rate_width_decades']:.6f}
decades in da/dN.  This exceeds the two-decade target.

The three-seed 20--60 micrometre endpoint check gives slope
{seeds['endpoint_slope_mean']:.6f} +/- {seeds['endpoint_slope_sample_sd']:.6f}
(sample SD) and mean rate width {seeds['rate_width_decades_mean']:.6f} decades.

The earlier V3 ceramic and optimized DBTT three-point controls span 3.92834 and
3.79749 rate decades, respectively.  No additional parameter perturbation was
promoted because the primary archived row already satisfies the objective and
is the widest tested row.

## Scope

The package contains exact event/state histories, common-window tables,
opening-only comparisons, local slopes, seed uncertainty, plots, and the full
production preflight.  The R8 spatial/void fixtures remain unchanged.  A
same-material spatial/void follow-up was not claimed: R8's spatial factory does
not yet accept the archived embedded material binding, and substituting its
native 900 K material would violate the comparison contract.
"""
    (DEST/"README.md").write_text(readme)
    commands="""# R9 replay commands

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
"""
    (DEST/"COMMANDS.md").write_text(commands)
    entries=[]
    for path in sorted(p for p in DEST.rglob("*") if p.is_file()):
        if path.name=="PACKAGE_SHA256.txt":continue
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(DEST)}")
    (DEST/"PACKAGE_SHA256.txt").write_text("\n".join(entries)+"\n")
    return 0
if __name__=="__main__":raise SystemExit(main())
