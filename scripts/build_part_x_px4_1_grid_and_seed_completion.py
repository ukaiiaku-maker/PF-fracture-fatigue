"""PX4.1 sections 4 and 6: complete the five-point seed-1720 developed grid
(D1/D2/D5 were missing their Kmax=18 MPa sqrt(m) point -- it had silently
ALIASED to the already-completed 12-event/60um PX3 screen result at that
same Kmax, since the original canonical_job_key formula hashed only
physical CONDITION, never campaign BUDGET) and add the required
second-seed (1001723) confirmation rows at Kmax=15/18/21 for D1/D2/D3/D5/D6.

D1_confirm/D2_confirm ALREADY EXIST in developed_job_registry.csv (built
by build_part_x_kinetic_regime_registry.py's own PX2-era _developed_job_
confirmation calls, at exactly the right seed/Kmax grid, status
BLOCKED_PENDING_STAGE1_DECISION -- correctly gated on Stage-1's own
completion, which is now done). Those 12 rows are REUSED here: refreshed
to the current producer commit and campaign_stage="developed" (for
uniform tagging with the newly-appended rows below, though seed alone
already disambiguates them from any screen job), then authorized.

D3_confirm/D5_confirm/D6_confirm did not exist yet (PX2 predates their
frequency/passivation/persistent selections) -- 18 new rows appended.

The 6 Kmax=18 grid-completion rows (seed=1720) are new appends using
campaign_stage="developed" specifically because THIS is the one condition
that would otherwise collide with the already-completed PX3 screen result
at the same Kmax/seed (canonical_job_key's PX4.1 fix).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

from build_part_x_kinetic_regime_registry import canonical_job_key, _config_hash  # noqa: E402
from part_x_run_one_job import _base_rebonding_cfg  # noqa: E402

REF_T_K = 300.0
INTEGRATOR_MODE = "explicit"
GRID_COMPLETION_KMAX = 18.0e6
SECOND_SEED = 1001723
SECOND_SEED_KMAX_GRID = [15.0e6, 18.0e6, 21.0e6]

# (row_name, R, frequency_Hz, minimum_load_hold_s, chemistry_factor, K_b_target)
PROTOCOL_CONDITIONS = {
    "D1": ("COMPETING_REVERSIBLE", -0.95, 1000.0, 0.0, 1.0, 900000.0),
    "D2": ("COMPETING_REVERSIBLE", -0.50, 1000.0, 0.0, 1.0, 900000.0),
    "D3": ("COMPETING_REVERSIBLE", -0.50, 316.227766016837952, 0.0, 1.0, 900000.0),
    "D5": ("PASSIVATION_LIMITED", -0.50, 1000.0, 0.0, 1.0, 900000.0),
    "D6_conditional_persistent": ("COMPETING_PERSISTENT", -0.50, 316.227766016837952, 0.0, 1.0, 900000.0),
}
# Protocol label used for second-seed rows of each base protocol.
CONFIRM_LABEL = {
    "D1": "D1_confirm", "D2": "D2_confirm", "D3": "D3_confirm",
    "D5": "D5_confirm", "D6_conditional_persistent": "D6_confirm",
}


def _row_config(row_name: str):
    payload = json.loads((ARTIFACTS_DIR / "kinetic_regime_registry.json").read_text())["rows"][row_name]["config"]
    return _base_rebonding_cfg({"config": payload})


def _material_row_hash() -> str:
    payload = json.loads((ARTIFACTS_DIR / "source_provenance.json").read_text())
    return payload["material_row"]["complete_active_material_row_sha256"]


def _current_head() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _new_job(*, protocol: str, row_name: str, R: float, frequency_Hz: float, hold_s: float, chem: float,
             K_b_target: float, Kmax: float, cohesion: str, seed: int, material_hash: str, producer_sha: str,
             config_hash: str, note: str) -> dict:
    K_target = K_b_target if cohesion == "finite" else 0.0
    key = canonical_job_key(
        material_row_hash=material_hash, rebonding_config_hash=config_hash, Kmax=Kmax, R=R,
        nominal_frequency_Hz=frequency_Hz, minimum_load_hold_s=hold_s, T_K=REF_T_K, chemistry_factor=chem,
        K_rebond_max_Pa_sqrt_m=K_target, seed=seed, integrator_mode=INTEGRATOR_MODE,
        physical_producer_sha=producer_sha, campaign_stage="developed",
    )
    return {
        "protocol": protocol, "row_name": row_name, "config_hash": config_hash,
        "material_row_hash": material_hash, "Kmax_Pa_sqrt_m": Kmax, "R": R, "frequency_Hz": frequency_Hz,
        "minimum_load_hold_s": hold_s, "chemistry_factor": chem, "K_rebond_max_target_Pa_sqrt_m": K_target,
        "seed": seed, "integrator_mode": INTEGRATOR_MODE, "physical_producer_sha": producer_sha,
        "canonical_job_key": key, "cohesion": cohesion, "note": note,
        "alias_of_protocol": "", "alias_of_row_name": "", "status": "AUTHORIZED_PX4",
    }


def main() -> None:
    registry_path = ARTIFACTS_DIR / "developed_job_registry.csv"
    with registry_path.open() as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    existing_keys = {r["canonical_job_key"] for r in rows}
    material_hash = _material_row_hash()
    producer_sha = _current_head()
    config_hash_cache = {name: _config_hash(_row_config(name)) for name in
                          {v[0] for v in PROTOCOL_CONDITIONS.values()}}

    # --- Reuse existing D1_confirm/D2_confirm rows in place. ---
    n_reused = 0
    for row in rows:
        if row["protocol"] not in ("D1_confirm", "D2_confirm"):
            continue
        if row["status"] != "BLOCKED_PENDING_STAGE1_DECISION":
            raise RuntimeError(f"expected {row['protocol']} to be BLOCKED_PENDING_STAGE1_DECISION, found {row['status']}")
        base_protocol = row["protocol"].replace("_confirm", "")
        row_name, R, freq, hold_s, chem, K_b_target = PROTOCOL_CONDITIONS[base_protocol]
        new_key = canonical_job_key(
            material_row_hash=row["material_row_hash"], rebonding_config_hash=row["config_hash"],
            Kmax=float(row["Kmax_Pa_sqrt_m"]), R=float(row["R"]), nominal_frequency_Hz=float(row["frequency_Hz"]),
            minimum_load_hold_s=float(row["minimum_load_hold_s"]), T_K=REF_T_K, chemistry_factor=float(row["chemistry_factor"]),
            K_rebond_max_Pa_sqrt_m=float(row["K_rebond_max_target_Pa_sqrt_m"]), seed=int(row["seed"]),
            integrator_mode=row["integrator_mode"], physical_producer_sha=producer_sha, campaign_stage="developed",
        )
        if new_key in existing_keys:
            raise RuntimeError(f"canonical_job_key collision refreshing {row['protocol']}: {new_key[:12]}")
        existing_keys.discard(row["canonical_job_key"])
        existing_keys.add(new_key)
        row["physical_producer_sha"] = producer_sha
        row["canonical_job_key"] = new_key
        row["status"] = "AUTHORIZED_PX4"
        row["note"] = row["note"] + " -- PX4.1: authorized, refreshed to current producer, campaign_stage=developed"
        n_reused += 1
    if n_reused != 12:
        raise RuntimeError(f"expected exactly 12 D1_confirm/D2_confirm rows to reuse, found {n_reused}")

    new_rows = []

    # --- Section 4: five-point grid completion (D1/D2/D5 Kmax=18, seed=1720). ---
    for protocol in ("D1", "D2", "D5"):
        row_name, R, freq, hold_s, chem, K_b_target = PROTOCOL_CONDITIONS[protocol]
        for cohesion in ("finite", "zero"):
            job = _new_job(
                protocol=protocol, row_name=row_name, R=R, frequency_Hz=freq, hold_s=hold_s, chem=chem,
                K_b_target=K_b_target, Kmax=GRID_COMPLETION_KMAX, cohesion=cohesion, seed=1720,
                material_hash=material_hash, producer_sha=producer_sha, config_hash=config_hash_cache[row_name],
                note="PX4.1 five-point grid completion (campaign_stage=developed avoids aliasing to the PX3 screen result at this same Kmax/seed)",
            )
            if job["canonical_job_key"] in existing_keys:
                raise RuntimeError(f"canonical_job_key collision for grid-completion {protocol}/{cohesion}: {job['canonical_job_key'][:12]}")
            existing_keys.add(job["canonical_job_key"])
            new_rows.append(job)

    # --- Section 6: second-seed confirmation for D3/D5/D6 (D1/D2 reused above). ---
    for protocol in ("D3", "D5", "D6_conditional_persistent"):
        row_name, R, freq, hold_s, chem, K_b_target = PROTOCOL_CONDITIONS[protocol]
        for Kmax in SECOND_SEED_KMAX_GRID:
            for cohesion in ("finite", "zero"):
                job = _new_job(
                    protocol=CONFIRM_LABEL[protocol], row_name=row_name, R=R, frequency_Hz=freq, hold_s=hold_s,
                    chem=chem, K_b_target=K_b_target, Kmax=Kmax, cohesion=cohesion, seed=SECOND_SEED,
                    material_hash=material_hash, producer_sha=producer_sha, config_hash=config_hash_cache[row_name],
                    note="PX4.1 second-seed confirmation (campaign_stage=developed)",
                )
                if job["canonical_job_key"] in existing_keys:
                    raise RuntimeError(f"canonical_job_key collision for {CONFIRM_LABEL[protocol]}/Kmax={Kmax}/{cohesion}: {job['canonical_job_key'][:12]}")
                existing_keys.add(job["canonical_job_key"])
                new_rows.append(job)

    rows.extend(new_rows)
    with registry_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reused (authorized in place) {n_reused} D1_confirm/D2_confirm rows")
    print(f"Appended {len(new_rows)} new rows "
          f"(expected 6 grid-completion + 18 D3/D5/D6-confirm = 24)")
    if len(new_rows) != 24:
        raise RuntimeError(f"expected exactly 24 appended rows, got {len(new_rows)}")
    print(f"Total PX4.1 jobs authorized this run: {n_reused + len(new_rows)} (expected 36)")


if __name__ == "__main__":
    main()
