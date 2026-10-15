import csv
import json

import pytest

from scripts.qualify_pf_fem_common_mechanics import build_receipt


def _pf_level(tmp_path, name, h, nodes, force, kj, *, emission=0.0):
    path = tmp_path / name
    path.mkdir()
    with (path / "steps_0900K.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "Uapp_m", "Ftop_N", "KJ_Pa_sqrtm", "lambda_e", "N_em",
            "W_emit_J_per_m", "pz_store_total", "pz_mobile_total",
            "pz_escape_total", "pz_emit_total",
        ])
        writer.writeheader()
        writer.writerow({"Uapp_m": 2e-7, "Ftop_N": force, "KJ_Pa_sqrtm": kj,
                         "lambda_e": emission, "N_em": 0, "W_emit_J_per_m": 0,
                         "pz_store_total": 0, "pz_mobile_total": 0,
                         "pz_escape_total": 0, "pz_emit_total": 0})
    (path / "summary.json").write_text(json.dumps([{"hbar_tip_m": h, "n_nodes": nodes}]))
    (path / "run_args.json").write_text(json.dumps({"tip_h_fine": h}))
    return path


def _fem_receipt(tmp_path):
    levels = []
    for ratio, spacing, nodes, force, energy, j, g in (
        (.245, 12.25e-6, 100, 39060, .003906, 1.932, 1.938),
        (.18, 9e-6, 200, 39040, .003904, 1.945, 1.952),
        (.12, 6e-6, 400, 38955, .003896, 1.937, 1.950),
    ):
        levels.append({"support_h_over_L_pz": ratio, "support_spacing_m": spacing,
                       "node_count": nodes, "triangle_count": 2*nodes,
                       "top_reaction_N_per_unit_thickness": force,
                       "stored_energy_J_per_m": energy,
                       "native_full_energy_J_median_J_per_m2": j,
                       "matched_outer_virtual_G_J_per_m2": g})
    path = tmp_path / "fem.json"
    path.write_text(json.dumps({"config": {"total_grip_opening_m": 2e-7},
                                "levels": levels,
                                "observer_correspondence": {"passed": True}}))
    return path


def test_build_receipt_keeps_common_and_native_observers_separate(tmp_path):
    dirs = tuple(_pf_level(tmp_path, f"pf{i}", h, 100+i, force, kj)
                 for i, (h, force, kj) in enumerate((
                     (2e-6, 43560, 358500), (1.3e-6, 43570, 358350),
                     (1e-6, 43580, 358330))))
    receipt = build_receipt(dirs, _fem_receipt(tmp_path))
    assert receipt["PF"]["support_converged_below_2_percent"]
    assert receipt["FEM"]["support_converged_below_2_percent"]
    assert receipt["fine_level_common_observer_comparison"]["PF_over_FEM_reaction"] > 1.1
    assert "PF native domain-J/K_J" in receipt["interpretation"]["noninterchangeable_native_observers"]


def test_build_receipt_rejects_nonzero_opening_only_state(tmp_path):
    dirs = tuple(_pf_level(tmp_path, f"pf{i}", 1e-6, 100 + i, 43580, 358330,
                           emission=1.0 if i == 1 else 0.0) for i in range(3))
    with pytest.raises(ValueError, match="not identically zero"):
        build_receipt(dirs, _fem_receipt(tmp_path))
