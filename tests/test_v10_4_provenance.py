from __future__ import annotations

import importlib.util
from pathlib import Path


def test_v104_launcher_marks_production_physics_modified():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_v10_4_bulk_rate_orientation_launcher.py"
    spec = importlib.util.spec_from_file_location("v104_builder_provenance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = (
        root / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
    ).read_text()
    generated = module.transform(source)

    assert '"production_physics_modified": True' in generated
    assert (
        '"production_physics_change": '
        '"full_field_bulk_peierls_taylor_coupling"'
    ) in generated
    assert '"production_physics_modified": False' not in generated
