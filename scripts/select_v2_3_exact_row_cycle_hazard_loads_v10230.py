#!/usr/bin/env python3
"""Select V2.3 exact-row fatigue loads from fresh-state cycle hazard only."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from arrhenius_fracture.canonical_v2_registry_v10230 import load_row, surface_adapters


EXTERNAL_ROOT = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search")
EXTERNAL_COMMIT = "a74c4c38aad706209c7c9a8157971c847d2bf9bd"
FATIGUE_SHA = "088b5b73d6b804aa22aaa815fc1ce74a75dfbc9949a969f8e07fe0248ea0c0a0"
OUT = ROOT / "analysis_outputs/v2_3_exact_row_cycle_hazard_fatigue"
CANDIDATE_ID = "P25_TJBSV2_S_002987"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_external_fatigue():
    assert subprocess.check_output(
        ["git", "-C", str(EXTERNAL_ROOT), "rev-parse", "HEAD"], text=True
    ).strip() == EXTERNAL_COMMIT
    assert not subprocess.check_output(
        ["git", "-C", str(EXTERNAL_ROOT), "status", "--porcelain=v1"], text=True
    )
    assert sha(EXTERNAL_ROOT / "arrhenius_fracture/fatigue_v914.py") == FATIGUE_SHA
    name = "immutable_v914"
    if name not in sys.modules:
        package = EXTERNAL_ROOT / "arrhenius_fracture"
        spec = importlib.util.spec_from_file_location(
            name, package / "__init__.py", submodule_search_locations=[str(package)]
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load immutable v9.14 package")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.fatigue_v914")


def exact_candidate_and_physics(fatigue):
    row = load_row(CANDIDATE_ID)
    parent = json.loads(row["parent_complete_registry_row_json"])
    opening, emission = surface_adapters(row)
    types = importlib.import_module("immutable_v914.emergent_gnd_types_v913")
    PT = types.PTMechanism
    candidate = types.CandidateParameters(
        candidate_id=CANDIDATE_ID,
        cleavage=opening,
        emission=emission,
        peierls=PT(
            H0_eV=float(parent["peierls_H0_eV"]),
            activation_entropy_kB=float(parent["peierls_activation_entropy_kB"]),
            exp_a=float(parent["peierls_exp_a"]), exp_n=float(parent["peierls_exp_n"]),
            nu0_s=float(parent["peierls_nu0_s"]),
            stress_fraction=float(parent["peierls_stress_fraction"]),
        ),
        taylor=PT(
            H0_eV=float(parent["taylor_H0_eV"]),
            activation_entropy_kB=float(parent["taylor_activation_entropy_kB"]),
            exp_a=float(parent["taylor_exp_a"]), exp_n=float(parent["taylor_exp_n"]),
            nu0_s=float(parent["taylor_nu0_s"]),
            stress_fraction=float(parent["taylor_stress_fraction"]),
        ),
        rho_source0_m2=float(parent["rho_source0_m2"]),
        source_refresh_length_m=float(parent["source_refresh_length_um"]) * 1e-6,
        taylor_corr_rho_c_m2=float(parent["taylor_corr_rho_c_m2"]),
        taylor_corr_scale=float(parent["taylor_corr_scale"]),
        recovery_nu0_s=float(parent["recovery_nu0_s"]),
        recovery_H0_eV=float(parent["recovery_H0_eV"]),
        recovery_activation_entropy_kB=float(parent["recovery_activation_entropy_kB"]),
        c_blunt=float(parent["c_blunt"]),
    )
    physics = types.CommonPhysics(
        mpz_length_m=float(parent["L_pz_um_recommended"]) * 1e-6,
        n_bins=int(parent["n_bins_recommended"]),
        source_zone_length_m=float(parent["source_zone_length_um"]) * 1e-6,
        cleavage_nu0_s=float(row["opening_attempt_frequency_s"]),
        cleavage_hits=float(row["physics__cleavage_hits"]),
        cleavage_correlation_time_s=float(row["physics__cleavage_correlation_time_s"]),
        emission_nu0_s=float(row["emission_attempt_frequency_s"]),
        blunting_length_m=float(parent["physics__blunting_length_m"]),
        blunting_slip_fraction=float(parent["physics__blunting_slip_fraction"]),
        persistent_backstress_scale=float(parent["physics__persistent_backstress_scale"]),
        encounter_efficiency=float(parent["physics__encounter_efficiency"]),
        taylor_phi_max=float(parent["physics__taylor_phi_max"]),
        mobile_transport_velocity_scale=float(parent["physics__mobile_transport_velocity_scale"]),
    )
    return candidate, physics, row


def main() -> int:
    fatigue = load_external_fatigue()
    candidate, physics, row = exact_candidate_and_physics(fatigue)
    grid = np.arange(1.0, 30.0 + 0.125, 0.25)
    records = []
    for Kmax in grid:
        state = fatigue.EmergentGNDState(candidate, physics)
        loading = fatigue.FatigueLoading(
            deltaK_MPa_sqrt_m=0.9 * float(Kmax), R=0.1,
            frequency_Hz=1000.0, temperature_K=300.0, phase_steps=64,
        )
        action = float(fatigue.constant_state_action_per_cycle(state, loading))
        projected = math.log(2.0) / action if action > 0.0 else math.inf
        records.append({
            "candidate_id": CANDIDATE_ID, "Kmax_MPa_sqrt_m": float(Kmax),
            "DeltaK_MPa_sqrt_m": 0.9 * float(Kmax),
            "fresh_state_cycle_hazard_action": action,
            "projected_median_first_passage_cycles": projected,
            "barrier_or_renewal_retuned": False,
        })
    frame = pd.DataFrame(records)
    targets = [10000.0, 1000.0, 100.0]
    available = set(frame.index)
    selected = []
    for target in targets:
        finite = [i for i in available if math.isfinite(frame.loc[i, "projected_median_first_passage_cycles"])]
        index = min(finite, key=lambda i: (
            abs(math.log10(frame.loc[i, "projected_median_first_passage_cycles"]) - math.log10(target)),
            frame.loc[i, "Kmax_MPa_sqrt_m"],
        ))
        available.remove(index)
        record = frame.loc[index].to_dict()
        record["target_median_first_passage_cycles"] = target
        selected.append(record)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "cycle_hazard_load_selection_grid.csv", index=False)
    payload = {
        "schema": "v10.2.30_v2_3_selected_exact_row_fatigue_loads_v1",
        "candidate_id": CANDIDATE_ID,
        "complete_bound_row_sha256": row["complete_bound_row_sha256"],
        "selection_rule": "nearest_log10_fresh_state_cycle_hazard_projection",
        "selected_loads_in_ascending_Kmax": sorted(selected, key=lambda x: x["Kmax_MPa_sqrt_m"]),
        "monotonic_ramp_cap_applied": False,
        "barrier_or_renewal_retuned": False,
        "external_driver_commit": EXTERNAL_COMMIT,
        "external_driver_sha256": FATIGUE_SHA,
        "candidate_contract_sha256": hashlib.sha256(json.dumps(
            asdict(candidate), sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "physics_contract_sha256": hashlib.sha256(json.dumps(
            asdict(physics), sort_keys=True, separators=(",", ":"), allow_nan=True
        ).encode()).hexdigest(),
    }
    (OUT / "selected_exact_row_fatigue_loads.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
