#!/usr/bin/env python3
"""Reduce one generic FEM native run to a compact, hash-bound receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


EVIDENCE_NAMES = {
    "adapter": "generic_current_pf_companion_contract_v1.json",
    "transfer": "four_class_parameter_transfer_v10_0_5_18_3_9.json",
    "summary": "anisotropic_calibrated_tip_first_passage_summary.json",
    "production": "persistent_site_production_manifest_v10_0_5_18_3_9.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(run_root: Path) -> dict:
    root = run_root.expanduser().resolve()
    paths = {key: root / name for key, name in EVIDENCE_NAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing generic FEM evidence: " + ", ".join(missing))

    adapter = json.loads(paths["adapter"].read_text())
    transfer = json.loads(paths["transfer"].read_text())
    summary = json.loads(paths["summary"].read_text())
    if isinstance(summary, dict):
        row = summary
    elif isinstance(summary, list) and len(summary) == 1:
        row = summary[0]
    else:
        raise ValueError("generic FEM qualification requires exactly one summary row")
    production = json.loads(paths["production"].read_text())

    checks = {
        "native_entry_completed": adapter["run_completed_without_exception"] is True,
        "registry_material_class_peak": adapter["material"]["material_class"] == "peak",
        "transfer_material_class_peak": transfer["material_class"] == "peak",
        "summary_material_class_peak": row["target_class"] == "peak",
        "temperature_900K": float(row["T_K"]) == 900.0,
        "orientation_17p5deg": float(row["crystal_theta_deg"]) == 17.5,
        "plane_strain_declared": (
            adapter["supported_native_request"]["constraint"] == "plane_strain"
        ),
        "tip_state_updated": int(row["bulk_state_update_calls"]) > 0,
        "native_J_finite_positive": float(row["J_first_J_per_m2"]) > 0.0,
        "adaptive_czm_backend": "adaptive_czm" in str(
            production.get("crack_backend", production)
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"generic FEM native qualification failed: {checks}")

    return {
        "schema": "generic-fem-native-run-qualification/1",
        "status": "PASS_NATIVE_CRACKED_BODY_ONE_STEP",
        "evidence_directory": root.name,
        "checks": checks,
        "observed": {
            "J_first_J_per_m2": row["J_first_J_per_m2"],
            "KJ_reference_first_MPa_sqrt_m": row["KJ_reference_first_MPa_sqrt_m"],
            "N_em_final": row["N_em_final"],
            "mesh_nx": adapter["supported_native_request"]["nx"],
            "mesh_ny": adapter["supported_native_request"]["ny"],
            "material_option": adapter["material"]["option_key"],
            "registry_sha256": adapter["material"]["registry_sha256"],
        },
        "sha256": {path.name: _sha(path) for path in paths.values()},
        "acceptance_boundary": (
            "This proves registry-driven native plane-strain/tip-only execution "
            "outside the Peak/1000 K preset. It is not three-level convergence, "
            "opening-only, plane-stress/generalized constraint, bilinear specimen "
            "integration, or active-state restart evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-evidence-dir", type=Path)
    args = parser.parse_args()
    evidence_root = args.run_root
    if args.compact_evidence_dir is not None:
        evidence_root = args.compact_evidence_dir
        evidence_root.mkdir(parents=True, exist_ok=True)
        for name in EVIDENCE_NAMES.values():
            shutil.copy2(args.run_root / name, evidence_root / name)
    payload = qualify(evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
