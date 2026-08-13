#!/usr/bin/env python3
"""Materialize the exact v9.14 A--D rows for the v10.2.31 spatial transfer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

IDS = {
    "A_0462": "v914_endurance_knee_0462",
    "B_0658": "v914_endurance_knee_0658",
    "C_0554": "v914_endurance_knee_0554",
    "D_0133": "v914_endurance_knee_0133",
}
SHARED = {
    "n_slip_channels": "2", "rho_forest_floor_m2": "5000000000000",
    "peierls_stress_fraction": "0.5773502691896258",
    "taylor_stress_fraction": "0.5773502691896258",
    "mobile_shield_fraction": "0", "source_recovery_rate_s": "0",
    "L_pz_um_recommended": "50", "n_bins_recommended": "80",
    "retained_recovery_rate_s": "0", "source_refresh_length_um": "0",
    "recovery_nu0_s": "0", "recovery_H0_eV": "0",
    "recovery_activation_entropy_kB": "0", "reference_source_area_um2": "25",
    "reference_front_width_um": "10", "source_zone_length_um": "2",
    "legacy_source_sites_active": "0", "legacy_source_refresh_active": "0",
    "explicit_recovery_active": "0",
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    with a.source.open(newline="") as f: source=list(csv.DictReader(f))
    by_id={row["candidate_id"]:row for row in source}
    missing=set(IDS.values())-set(by_id)
    if missing: raise SystemExit(f"missing authoritative candidates: {sorted(missing)}")
    rows=[]
    for label,candidate in IDS.items():
        original=by_id[candidate]; row=dict(original); row.update(SHARED)
        row.update({"option_key":candidate,"candidate_id":candidate,"material_class":label[0],
                    "role":"v10.2.31 exact endurance-knee spatial transfer",
                    "mechanism_summary":{"A":"direct barrier","B":"plastic-state controlled","C":"timescale crossover","D":"mixed"}[label[0]],
                    "validation_status":"pending sparse 2-D validation"})
        rows.append(row)
    fields=list(rows[0]); a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    audit={"schema":"v10.2.31_endurance_knee_registry_transfer_v1","source":str(a.source.resolve()),
           "source_sha256":sha(a.source),"output":str(a.out.resolve()),"candidate_ids":IDS,
           "shared_spatial_constants":SHARED,"parameter_refit":False}
    a.out.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    return 0
if __name__ == "__main__": raise SystemExit(main())
