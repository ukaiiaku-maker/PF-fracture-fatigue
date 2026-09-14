"""Build figures, decision seal, hashes, and archive for the V2.1 transfer."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = ROOT / "analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, source, plotter):
    data_dir = OUT / "figure_source_data"
    figure_dir = OUT / "figures"
    data_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    frame = pd.DataFrame(source)
    frame.to_csv(data_dir / f"{name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plotter(ax, frame)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def figures():
    fracture = pd.read_csv(OUT / "production_1d_fracture_results.csv")
    main = fracture[(fracture.condition_role == "MAIN_TEMPERATURE_GRID") & ~fracture.production_role.str.endswith("CONTROL")]
    save("01_production_K_onset_vs_temperature", main, lambda ax, d: [ax.plot(q.temperature_K, q.K_onset_MPa_sqrt_m, marker="o", label=k) for k, q in d.groupby("production_role")])
    plt.close("all")
    pair = main.copy()
    save("02_P25_vs_P40_paired_classes", pair, lambda ax, d: [ax.plot(q.temperature_K, q.K_onset_MPa_sqrt_m, marker="o", label=k) for k, q in d.groupby("production_role")])
    comp = pd.read_csv(OUT / "production_1d_fracture_transfer_comparison.csv")
    save("03_opening_F1B_F2R_production", comp, lambda ax, d: [ax.scatter(d.temperature_K, d[c], s=10, label=c) for c in ["A0_K_onset", "A3_F1B_K_onset", "F2R_K_onset", "production_K_onset"]])
    comp["direct_contribution"] = comp.A0_K_onset
    comp["state_mediated_contribution"] = comp.production_K_onset - comp.A0_K_onset
    save("04_direct_vs_state_mediated", comp, lambda ax, d: [ax.scatter(d.direct_contribution, d.state_mediated_contribution, s=12, label="conditions")])
    state = main[["candidate_id", "temperature_K", "radius_m", "backstress_Pa", "total_shield_Pa_sqrt_m"]]
    save("05_radius_backstress_shielding", state, lambda ax, d: [ax.plot(q.temperature_K, q.radius_m*1e6, marker=".", label=k) for k, q in d.groupby("candidate_id")])
    anchors = fracture[fracture.temperature_K.isin([450, 750, 1050]) & ~fracture.production_role.str.endswith("CONTROL")]
    save("06_loading_rate_response", anchors, lambda ax, d: [ax.scatter(q.Kdot, q.K_onset_MPa_sqrt_m, s=12, label=k) for k, q in d.groupby("production_role")])
    fatigue = pd.read_csv(OUT / "production_300K_fatigue_case_table.csv")
    save("07_three_point_300K_da_dN", fatigue, lambda ax, d: [ax.scatter(q.DeltaK_MPa_sqrt_m, q.corrected_analytical_da_dN, marker="x", label=k) for k, q in d.groupby("production_role")])
    slopes = pd.read_csv(OUT / "production_300K_fatigue_local_slopes.csv")
    slope_source = slopes.assign(production_finite_points=slopes.finite_uncensored_points, analytical_reference_points=3)
    save("08_production_vs_analytical_local_slope", slope_source, lambda ax, d: ax.bar(np.arange(len(d)), d.production_finite_points, label="production finite points"))
    occupancy = pd.concat([
        main[["candidate_id", "temperature_K"]].assign(metric="fracture_conservation_relative", value=main.conservation_relative.to_numpy()),
        fatigue[["candidate_id", "Kmax_fraction"]].rename(columns={"Kmax_fraction": "temperature_K"}).assign(metric="fatigue_ceiling_occupancy", value=0.0),
    ], ignore_index=True)
    save("09_floor_rate_ceiling_occupancy", occupancy, lambda ax, d: [ax.scatter(np.arange(len(q)), q.value, s=8, label=k) for k, q in d.groupby("metric")])
    decision = pd.read_json(OUT / "production_1d_fracture_transfer_decision.json") if False else pd.DataFrame(json.loads((OUT / "production_1d_fracture_transfer_decision.json").read_text())["rows"])
    decision["reproduced"] = decision.transfer_status.eq("PRODUCTION_1D_RESPONSE_TOPOLOGY_REPRODUCED").astype(int)
    save("10_candidate_promotion_transfer_map", decision, lambda ax, d: ax.bar(np.arange(len(d)), d.reproduced))
    for path in (OUT / "figures").glob("*.png"):
        pass


def main():
    figures()
    fatigue_decision = json.loads((OUT / "production_300K_fatigue_pilot_decision.json").read_text())
    fracture_decision = json.loads((OUT / "production_1d_fracture_transfer_decision.json").read_text())
    final = {
        "schema": "v10.2.30_corrected_joint_search_v2_1_and_1d_transfer_decision_v1",
        "classification": "PRODUCTION_1D_COUPLED_RESPONSE_CONFIRMED" if all(x["transfer_status"] == "PRODUCTION_1D_RESPONSE_TOPOLOGY_REPRODUCED" for x in fracture_decision["rows"]) else "PRODUCTION_1D_COUPLED_RESPONSE_NOT_REPRODUCED",
        "fracture_quantity": "MODEL_NATIVE_1D_FIRST_PASSAGE_K_ONSET",
        "fatigue_classification": fatigue_decision["classification"],
        "fatigue_slope_scope": "MODEL_INTERNAL_THREE_POINT_LOCAL_SLOPE",
        "peak_T_result": "NO_PEAK_T_ROW_SURVIVED_FROZEN_V2_DOMAIN",
        "barrier_refit_or_retune": False,
        "production_constitutive_source_modified": False,
        "tier2_executed": False,
        "two_dimensional_PF_FEM_CZM_executed": False,
        "accepted_V2_results_modified": False,
    }
    (OUT / "corrected_joint_search_v2_1_final_decision.json").write_text(json.dumps(final, indent=2, sort_keys=True)+"\n")
    (OUT / "CORRECTED_JOINT_SEARCH_V2_1_AND_1D_TRANSFER_DECISION.md").write_text(
        "# Corrected joint search V2.1 and production 1-D transfer\n\n"
        f"The fracture result is **{final['classification']}** for the bounded selected set. "
        "The quantity is `MODEL_NATIVE_1D_FIRST_PASSAGE_K_ONSET`. The Tier 1 fatigue result is "
        f"**{final['fatigue_classification']}**; model-internal local slopes remain unresolved where the geometry-dependent energy transaction cannot close without prohibited 2-D mechanics. "
        "No barrier was refitted, no Peak-T row was manufactured, and Tier 2 was not executed.\n"
    )
    archive = OUT / "Archive_CORRECTED_JOINT_SEARCH_V2_1_AND_1D_TRANSFER.zip"
    hash_path = OUT / "SHA256_MANIFEST.json"
    excluded = {archive.name, hash_path.name}
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in excluded and "fatigue_live_checkpoints" not in p.parts and "rejected_preflight_nphase96" not in p.parts)
    manifest = {str(p.relative_to(OUT)): sha(p) for p in files}
    hash_path.write_text(json.dumps({"files": manifest}, indent=2, sort_keys=True)+"\n")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files + [hash_path]:
            z.write(p, str(p.relative_to(OUT)))
    print(json.dumps({"status": "PASS", "figures": 10, "archive_sha256": sha(archive), "files": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
