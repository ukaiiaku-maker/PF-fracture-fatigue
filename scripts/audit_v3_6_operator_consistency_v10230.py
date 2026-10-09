#!/usr/bin/env python3
"""Reproduce the independent V3.5 spatial audit before V3.6 changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.operator_consistent_projection_v10230 import (
    MaterialSlipPatch, OperatorConsistentProjection, area_weighted,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v35", type=Path, required=True)
    p.add_argument("--portable-zip", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    a = p.parse_args(); a.out_root.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(a.v35 / "matched_first_event_pilot.csv")
    old_coeff = pd.read_csv(a.v35 / "conditional_operator_coefficients.csv")
    old = np.load(a.v35 / "conditional_operator_fields.npz")
    projection = OperatorConsistentProjection()

    recon = []
    selectors = (("MIDDLE", "MECHANICAL"), ("HIGH", "LEGACY"), ("HIGH", "MECHANICAL"))
    for condition, mode in selectors:
        row = events[(events.condition == condition) & (events["mode"] == mode)].iloc[0]
        patch = MaterialSlipPatch(0, 0.0, (float(row.gamma_system0), float(row.gamma_system1)))
        result = projection.project([patch], crack_extension_m=0.0, K_MPa_sqrt_m=float(row.Kmax_MPa_sqrt_m))
        recon.append({
            "condition": condition, "mode": mode,
            "old_reduced_bin_max_engineering_eigenstrain": float(row.maximum_engineering_eigenstrain),
            "reconstructed_fixed_patch_max_engineering_eigenstrain": float(np.max(abs(result["engineering_eigenstrain"]))),
        })
    pd.DataFrame(recon).to_csv(a.out_root / "v3_5_spatial_deformation_reconstruction.csv", index=False)

    coeff = []
    for system in range(2):
        delta = old[f"delta_sigma_per_gamma_s{system}"]
        row = {"system": system}
        for name, mask in projection.observation_masks.items():
            row[f"v35_equal_element_delta_sigma_yy_per_gamma_{name}_Pa"] = float(np.mean(delta[1, mask]))
            row[f"v36_area_weighted_delta_sigma_yy_per_gamma_{name}_Pa"] = area_weighted(delta[1], projection.mesh.area_e, mask)
            row[f"v35_saved_coefficient_{name}_Pa"] = float(old_coeff.loc[old_coeff.system == system, f"delta_sigma_yy_per_gamma_{name}_Pa"].iloc[0])
        coeff.append(row)
    pd.DataFrame(coeff).to_csv(a.out_root / "old_new_observation_operator_audit.csv", index=False)

    refinement = []
    for nxyh in ((80,160,.125e-6),(100,200,.100e-6),(120,240,.080e-6),(160,320,.060e-6)):
        q = OperatorConsistentProjection(nx=nxyh[0], ny=nxyh[1], tip_h_fine_m=nxyh[2], orientation_signs=(1,-1))
        virgin = q.virgin_correspondence()
        for system in range(2):
            result = q.project([MaterialSlipPatch(0,0.0,tuple(1.0 if j==system else 0.0 for j in range(2)))],crack_extension_m=0.0,K_MPa_sqrt_m=q.reference_K,compute_kinematics=False)
            refinement.append({"nx":nxyh[0],"ny":nxyh[1],"tip_h_fine_m":nxyh[2],"elements":q.mesh.ne,"system":system,
              "support_width_m":q.support_width_m,"patch_length_m":q.patch_length_m,
              "virgin_opening_Pa":virgin["opening_Pa_at_reference_K"],
              "delta_opening_per_gamma_Pa":result["opening_Pa"]["activation_8_16"]-virgin["opening_Pa_at_reference_K"],
              "virgin_conjugate_shear_Pa":virgin["resolved_shear_Pa_at_reference_K"][system],
              "represented_area_m2":q.unit_response(system,0.0)["meta"]["represented_area_m2"]})
    pd.DataFrame(refinement).to_csv(a.out_root / "fixed_support_mesh_refinement.csv", index=False)

    members = []
    with zipfile.ZipFile(a.portable_zip) as z:
        for info in sorted(z.infolist(), key=lambda x: x.filename):
            if info.is_dir(): continue
            payload = z.read(info.filename)
            disk = a.v35 / info.filename
            members.append({
                "member": info.filename, "size": len(payload),
                "zip_member_sha256": hashlib.sha256(payload).hexdigest(),
                "matching_export_exists": disk.exists(),
                "matching_export_sha256": sha256(disk) if disk.exists() else None,
                "byte_identical": bool(disk.exists() and disk.read_bytes() == payload),
            })
    pd.DataFrame(members).to_csv(a.out_root / "reviewed_portable_source_bytes.csv", index=False)
    result = {
        "schema": "v10.2.30_v3_6_independent_audit_reproduction_v1",
        "portable_zip_sha256": sha256(a.portable_zip),
        "reviewer_SOURCE_AUDIT_available": False,
        "reviewer_audit_results_available": False,
        "absence_reason": "pasted sandbox links were not materialized in either supplied attachment",
        "reproduced_values": recon,
        "all_portable_members_match_exported_bytes": all(x["byte_identical"] for x in members),
        "missing_manifest_entry_not_missing_in_workspace": (a.v35 / "shielding_same_operator_refinement.csv").exists(),
    }
    (a.out_root / "audit_reproduction.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    projection.save_reference_fields(a.out_root / "operator_reference_fields.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
