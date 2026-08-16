from pathlib import Path

from scripts.verify_v913_joint_fracture_fatigue_causality import verify_figure_triplets


def test_figure_triplet_gate_requires_png_pdf_and_sidecar(tmp_path: Path):
    errors: list[str] = []
    verify_figure_triplets(tmp_path, ("figure",), errors)
    assert len(errors) == 3

    for suffix in (".png", ".pdf", "_plot_data.csv"):
        (tmp_path / f"figure{suffix}").write_text("evidence")
    errors = []
    verify_figure_triplets(tmp_path, ("figure",), errors)
    assert errors == []
