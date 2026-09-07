"""PX5 (mission section 8, review-scoped): authorize the 20 prescribed
static-shield control jobs -- D2 and D5 ONLY (per the review's explicit
scope restriction: "do NOT expand to D1/D3/D6 without new prospective
reason"), two static controls each (ceiling, analytical periodic-orbit-
matched), across the five-point seed=1720 Kmax grid already qualified for
PX4 (12/15/18/21/24.3 MPa*sqrt(m)).

Uses campaign_stage="px5_static_shield" in canonical_job_key -- distinct
from PX4's "developed" tag -- specifically because the ceiling control's
K_rebond_max_target_Pa_sqrt_m (900000.0) is numerically IDENTICAL to PX4's
own finite-cohesion dynamic-rebonding K_target at the same Kmax/protocol;
without a distinguishing campaign_stage, the ceiling-static job would
alias to the already-completed PX4 dynamic job (the exact PX4.1 aliasing
bug this campaign_stage mechanism was built to prevent, mission section
3's "campaign_stage distinguishes screen vs developed budget" now doing
double duty distinguishing developed-dynamic vs developed-static-shield
CONTROL_MODE at otherwise-identical Kmax).

cohesion column carries "ceiling_static" / "orbit_matched_static" (never
"finite"/"zero") -- part_x_run_one_job.py's STATIC_SHIELD_COHESIONS set
routes these to rebonding_cfg=None + static_shield_control instead of the
Markov-kinetics path.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

from build_part_x_kinetic_regime_registry import canonical_job_key  # noqa: E402

SEED = 1720
R_REF = -0.5
FREQUENCY_HZ = 1000.0
HOLD_S = 0.0
CHEMISTRY_FACTOR_PLACEHOLDER = 1.0  # not used by the static-shield physics path; recorded for schema uniformity only
T_K = 300.0
INTEGRATOR_MODE = "explicit"
CAMPAIGN_STAGE = "px5_static_shield"
PROTOCOL_ROW = {"D2": "COMPETING_REVERSIBLE", "D5": "PASSIVATION_LIMITED"}
FIELDNAMES = [
    "protocol", "row_name", "config_hash", "material_row_hash", "Kmax_Pa_sqrt_m", "R", "frequency_Hz",
    "minimum_load_hold_s", "chemistry_factor", "K_rebond_max_target_Pa_sqrt_m", "seed", "integrator_mode",
    "physical_producer_sha", "canonical_job_key", "cohesion", "note", "alias_of_protocol", "alias_of_row_name", "status",
]


def _current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _material_row_hash() -> str:
    return json.loads((ARTIFACTS_DIR / "source_provenance.json").read_text())["material_row"]["complete_active_material_row_sha256"]


def main() -> None:
    registry = json.loads((ARTIFACTS_DIR / "kinetic_regime_registry.json").read_text())["rows"]
    predictions = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_predictions.csv")))
    pred_by = {(r["protocol"], round(float(r["Kmax_Pa_sqrt_m"]))): r for r in predictions}

    material_hash = _material_row_hash()
    producer_sha = _current_head()

    rows_out = []
    for protocol, row_name in PROTOCOL_ROW.items():
        config_hash = registry[row_name]["config_hash"]
        for control_mode, K_field in (
            ("ceiling_static", "ceiling_K_b_static_Pa_sqrt_m"),
            ("orbit_matched_static", "periodic_orbit_matched_K_b_static_Pa_sqrt_m"),
        ):
            for Kmax in [12.0e6, 15.0e6, 18.0e6, 21.0e6, 24.3e6]:
                pred = pred_by[(protocol, round(Kmax))]
                K_b_static = float(pred[K_field])
                key = canonical_job_key(
                    material_row_hash=material_hash, rebonding_config_hash=config_hash, Kmax=Kmax, R=R_REF,
                    nominal_frequency_Hz=FREQUENCY_HZ, minimum_load_hold_s=HOLD_S, T_K=T_K,
                    chemistry_factor=CHEMISTRY_FACTOR_PLACEHOLDER, K_rebond_max_Pa_sqrt_m=K_b_static, seed=SEED,
                    integrator_mode=INTEGRATOR_MODE, physical_producer_sha=producer_sha, campaign_stage=CAMPAIGN_STAGE,
                )
                rows_out.append({
                    "protocol": protocol, "row_name": row_name, "config_hash": config_hash,
                    "material_row_hash": material_hash, "Kmax_Pa_sqrt_m": Kmax, "R": R_REF,
                    "frequency_Hz": FREQUENCY_HZ, "minimum_load_hold_s": HOLD_S,
                    "chemistry_factor": CHEMISTRY_FACTOR_PLACEHOLDER, "K_rebond_max_target_Pa_sqrt_m": K_b_static,
                    "seed": SEED, "integrator_mode": INTEGRATOR_MODE, "physical_producer_sha": producer_sha,
                    "canonical_job_key": key, "cohesion": control_mode, "note": "", "alias_of_protocol": "",
                    "alias_of_row_name": "", "status": "AUTHORIZED_PX5",
                })

    if len(rows_out) != 20:
        raise RuntimeError(f"expected exactly 20 PX5 static-shield rows, built {len(rows_out)}")
    if len({r["canonical_job_key"] for r in rows_out}) != 20:
        raise RuntimeError("canonical_job_key collision among the 20 PX5 rows")

    out_path = ARTIFACTS_DIR / "px5_static_shield_job_registry.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {out_path}: {len(rows_out)} AUTHORIZED_PX5 rows (producer_sha={producer_sha})")


if __name__ == "__main__":
    main()
