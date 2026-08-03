from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_v10_4_4_bulk_plasticity_orientation_launcher.py"
    spec = importlib.util.spec_from_file_location("v1044_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_preserves_campaign_source_and_inserts_full_field_dual_terminal():
    root = Path(__file__).parents[1]
    source = (
        root / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
    ).read_text()
    builder = _load_builder()
    generated = builder.transform(source)

    assert builder.MODEL_ENTRY in generated
    assert "v10.4.4_full_field_bulk_plasticity_orientation_rate_lock_v1" in generated
    assert "--bulk-plasticity-mode full_field" in generated
    assert "--n-stagger 80" in generated
    assert "--plastic-flow-terminal" in generated
    assert "PLASTICITY_DOMINATED" in generated
    assert "plasticity_dominated_after_partial_fracture" not in generated
    assert "summarize_v10_4_4_bulk_plasticity_campaign.py" in generated
    assert "--nx 36 --ny 72" in generated
    assert "--tip-h-fine 1e-6 --tip-ratio 1.20" in generated
    assert "--target-crack-extension-um \"$TARGET_EXT_UM\"" in generated
    assert "v913_paper_peak01_0242980_persistent_sites" in generated
    assert "v913_paper_dbtt01_0202500_persistent_sites" in generated
    assert "v913_paper_weakT01_0129902_persistent_sites" in generated
    assert "v913_paper_ceramic01_0077080_persistent_sites" in generated
