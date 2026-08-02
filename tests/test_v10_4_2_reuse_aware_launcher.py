from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = ROOT / "scripts" / "build_v10_4_2_reuse_aware_launcher.py"
    spec = importlib.util.spec_from_file_location("v1042_reuse_aware_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reused_cases_exit_before_native_v1042_command_checks():
    builder = _load_builder()
    source = (
        ROOT / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
    ).read_text()
    transformed = builder.transform(source)

    reuse_marker = 'v1042_reuse_path = root / "v10_4_2_reuse_audit.json"'
    verify_marker = "verify_materialized_case(root)"
    source_verify_marker = "verify_source_case(root)"
    exit_marker = "raise SystemExit(0)"
    native_marker = "bulk_audit = json.loads("

    assert transformed.count(reuse_marker) >= 2
    early = transformed.index(reuse_marker)
    verify = transformed.index(verify_marker, early)
    source_verify = transformed.index(source_verify_marker, verify)
    success_exit = transformed.index(exit_marker, source_verify)
    native = transformed.index(native_marker, success_exit)

    assert early < verify < source_verify < success_exit < native
    assert "--plastic-flow-terminal" in transformed
    assert "J_eff=max(J_signed,0)" not in transformed  # launcher source, not banner
