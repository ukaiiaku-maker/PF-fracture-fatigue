#!/usr/bin/env python3
"""Matched-stress constitutive audit for the v10 unified sharp-front MPZ state."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.sharp_front import FrontConfig, default_cleavage_barrier, default_emission_barrier
from arrhenius_fracture.unified_front import UnifiedMPZFrontEngine
from arrhenius_fracture.unified_mpz import MPZConfig

K_INIT = {"ceramic": 11.820868, "weakT": 16.949365, "DBTT": 29.197177}


def build(class_name: str, length_um: float, bins: int):
    manifest = MaterialManifest.from_csv(default_manifest_path(class_name))
    mat = ElasticProperties()
    f = FrontConfig(); f.r0 = 1.0e-6; f.sigma_cap = 0.0; f.da = 5.0e-6; f.L_pz = length_um * 1.0e-6
    return UnifiedMPZFrontEngine(
        f, default_cleavage_barrier(), default_emission_barrier(mat.b),
        mat.G, mat.nu, mat.b, manifest,
        MPZConfig(length_m=length_um * 1.0e-6, n_bins=bins, wake_length_m=length_um * 1.0e-6),
    )


def row_for(class_name: str, T: float, hold_s: float, length_um: float, bins: int):
    engine = build(class_name, length_um, bins)
    K = K_INIT[class_name] * 1.0e6
    sigma = engine.sigma_tip(K)
    lc, lc_raw, Gc = engine.lambda_cleave(sigma, T)
    le, _, Ge = engine.lambda_emit(sigma, T)
    total_sites = float(engine.mpz.site_capacity.sum())
    before = engine.mpz.diagnostics(engine.G, engine.nu, engine.b, engine.f.r0)
    evolve = engine.mpz.evolve(hold_s, T, sigma, engine.b)
    after = engine.mpz.diagnostics(engine.G, engine.nu, engine.b, engine.f.r0)
    return {
        "class": class_name,
        "T_K": T,
        "K_init_MPa_sqrt_m": K_INIT[class_name],
        "sigma_tip_Pa": sigma,
        "G_cleave_eV": Gc / 1.602176634e-19,
        "G_emit_eV": Ge / 1.602176634e-19,
        "lambda_c_raw_s-1": lc_raw,
        "lambda_c_multihit_s-1": lc,
        "self_consistent_opening_time_s": 1.0 / lc if lc > 0.0 else float("inf"),
        "lambda_e_per_site_s-1": le,
        "total_source_sites": total_sites,
        "time_to_one_emission_full_inventory_s": 1.0 / (le * total_sites) if le * total_sites > 0.0 else float("inf"),
        "Pi_open_over_emit": (1.0 / lc) / (1.0 / (le * total_sites)) if lc > 0.0 and le * total_sites > 0.0 else 0.0,
        "hold_s": hold_s,
        "dN_emit_hold": evolve["dN_emit"],
        "mobile_hold": after["mpz_mobile_count"],
        "retained_hold": after["mpz_retained_count"],
        "active_Kshield_hold_Pa_sqrt_m": after["mpz_active_K_shield_Pa_sqrt_m"],
        "available_fraction_before": before["mpz_available_site_fraction"],
        "available_fraction_after": after["mpz_available_site_fraction"],
        "state_model": after["mpz_state_model"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=["ceramic", "weakT", "DBTT"])
    p.add_argument("--T-K", type=float, default=700.0, dest="T_K")
    p.add_argument("--hold-s", type=float, default=100.0)
    p.add_argument("--mpz-length-um", type=float, default=100.0)
    p.add_argument("--mpz-n-bins", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("runs/v10_matched_stress_audit"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [row_for(c, args.T_K, args.hold_s, args.mpz_length_um, args.mpz_n_bins) for c in args.classes]
    with (args.out / "matched_stress_v10.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.out / "matched_stress_v10.json").write_text(json.dumps(rows, indent=2, default=str))
    for row in rows:
        print(" ".join(f"{k}={v}" for k, v in row.items()))
    return rows


if __name__ == "__main__":
    main()
