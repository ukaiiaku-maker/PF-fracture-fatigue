#!/usr/bin/env python3
"""Fail-closed completion gate for the v10.2.32 hybrid response."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


TABLES = [
    "abcd_1D_accelerated_explicit_rates.csv", "abcd_2D_accelerated_explicit_rates.csv",
    "abcd_hybrid_rates.csv", "abcd_mode_switch_diagnostics.csv",
    "abcd_explicit_event_intervals.csv", "dbtt_peak_hybrid_rates.csv",
]
FIGURES = [
    "abcd_four_path_da_dN_vs_deltaK", "abcd_hybrid_1D_2D_da_dN_vs_deltaK",
    "abcd_cycles_to_100um_vs_deltaK_hybrid", "abcd_event_intervals_vs_deltaK",
    "abcd_mode_switch_regime_map", "dbtt_peak_four_path_da_dN_vs_deltaK",
]


def require(condition: bool, message: str) -> None:
    if not condition: raise AssertionError(message)


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--analysis",type=Path,default=Path("runs/v10_2_32_endurance_knee_ABCD_hybrid_HCF_LCF_v1/analysis")); args=ap.parse_args()
    out=args.analysis; require(out.is_dir(),f"missing analysis directory: {out}")
    frames={}
    for name in TABLES:
        path=out/name; require(path.is_file() and path.stat().st_size>20,f"missing/nonempty table: {name}")
        frames[name]=pd.read_csv(path); require(len(frames[name])>0,f"empty table: {name}")
    for stem in FIGURES:
        for suffix in ("png","pdf","svg"):
            path=out/f"{stem}.{suffix}"; require(path.is_file() and path.stat().st_size>1000,f"missing figure: {path.name}")
        path=out/f"{stem}_plot_data.csv"; require(path.is_file() and path.stat().st_size>20,f"missing plot data: {path.name}")
    require((out/"HCF_LCF_HYBRID_VALIDATION_REPORT.md").stat().st_size>1000,"scientific report incomplete")
    one=frames[TABLES[0]]; two=frames[TABLES[1]]; canon=frames[TABLES[5]]
    for cls in "ABCD":
        require(len(one[(one["class"]==cls)&(one["integration_mode"]=="explicit")])>=3,f"{cls}: fewer than 3 explicit 1-D points")
        require(len(two[(two["class"]==cls)&(two["integration_mode"]=="explicit")])>=3,f"{cls}: fewer than 3 explicit 2-D points")
    for cls in ("DBTT","Peak"):
        require(len(canon[(canon["class"]==cls)&(canon["dimensionality"]=="1D")&(canon["integration_mode"]=="explicit")])>=2,f"{cls}: sparse explicit 1-D missing")
        require(len(canon[(canon["class"]==cls)&(canon["dimensionality"]=="2D")&(canon["integration_mode"]=="explicit")])>=2,f"{cls}: sparse explicit 2-D missing")
    all_rows=pd.concat([one,two,canon],ignore_index=True)
    bad=all_rows[(all_rows["plot_kind"].isin(["censor","partial"])) & all_rows["da_dN_m_per_cycle"].notna()]
    require(bad.empty,"censor/partial rows carry artificial rates")
    resolved=all_rows[all_rows["plot_kind"]=="resolved"]
    require((resolved["da_dN_m_per_cycle"]>0).all(),"resolved rate is not positive")
    provenance=pd.read_csv(out/"hybrid_provenance_inventory.csv")
    explicit=provenance[(provenance["dimensionality"]=="2D")&(provenance["integration_mode"]=="explicit")]
    require(len(explicit)>=16,"fewer than 16 explicit 2-D provenance rows")
    for _,r in explicit.iterrows():
        contract=Path(r["contract_path"]); require(contract.is_file(),f"missing contract: {contract}")
        data=json.loads(contract.read_text())
        for path_key,hash_key in (("registry","registry_sha256"),("family_json","family_sha256")):
            source=Path(data[path_key]); require(source.is_file(),f"missing contracted input: {source}")
            actual=hashlib.sha256(source.read_bytes()).hexdigest(); require(actual==data[hash_key],f"hash mismatch: {source}")
        require(data["repository_clean"] is True,"2-D launch was not from clean worktree")
        require(data["cycle_integration_mode"]=="explicit","2-D contract mode mismatch")
        require(data["seed"]==1720 and data["temperature_K"]==300 and data["R"]==.1 and data["frequency_Hz"]==1000,"physics/loading contract mismatch")
    print(f"PASS: {len(one)} 1-D rows, {len(two)} A-D 2-D rows, {len(explicit)} explicit 2-D contracts, six figures in three formats")
    return 0


if __name__=="__main__": raise SystemExit(main())
