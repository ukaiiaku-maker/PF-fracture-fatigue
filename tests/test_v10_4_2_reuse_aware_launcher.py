from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = ROOT / "scripts" / "build_v10_4_2_reuse_aware_launcher.py"
    spec = importlib.util.spec_from_file_location("v1043_scheduler_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generate_final_scheduler(tmp_path: Path) -> str:
    builder = _load_builder()
    source = (
        ROOT / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
    ).read_text()
    outer = builder.transform(source)
    outer_path = tmp_path / "outer.sh"
    outer_path.write_text(outer)
    subprocess.run(["bash", "-n", str(outer_path)], check=True)

    match = re.search(
        r'OUTROOT="\$OUTROOT" "\$PYTHON_BIN" - <<[\'\"]PY[\'\"]\n(.*?)\nPY\n',
        outer,
        flags=re.DOTALL,
    )
    assert match is not None, "nested final-scheduler generator not found"
    generator = tmp_path / "generator.py"
    generator.write_text(match.group(1))

    final_scheduler = tmp_path / "final_scheduler.sh"
    final_plotter = tmp_path / "plotter.py"
    outroot = tmp_path / "out"
    outroot.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "SOURCE_SCHEDULER": str(
                ROOT / "scripts" / "run_v10_2_27_paper_four_class_30deg_long_rcurves.sh"
            ),
            "SOURCE_PLOTTER": str(
                ROOT / "scripts" / "plot_v10_2_27_paper_four_class_rcurves.py"
            ),
            "GENERATED_SCHEDULER": str(final_scheduler),
            "GENERATED_PLOTTER": str(final_plotter),
            "OUTROOT": str(outroot),
        }
    )
    subprocess.run([sys.executable, str(generator)], cwd=ROOT, env=env, check=True)
    subprocess.run(["bash", "-n", str(final_scheduler)], check=True)
    return final_scheduler.read_text()


def _extract_verified_complete(final_scheduler: str) -> str:
    match = re.search(
        r'(verified_complete\(\) \{.*?\n\})\n\nrun_case\(\)',
        final_scheduler,
        flags=re.DOTALL,
    )
    assert match is not None, "verified_complete shell function not found"
    return match.group(1)


def _make_required_fixture(function_text: str, case_root: Path) -> None:
    before_guard = function_text.split(
        'v1042_reuse_path = root / "v10_4_2_reuse_audit.json"', 1
    )[0]
    names = set(re.findall(r'root / "([^"]+)"', before_guard))
    # The terminal-marker contract requires exactly one terminal marker, and
    # failure/incomplete markers must be absent.
    names.difference_update({"PLASTIC_FLOW", "RUN_FAILED", "INCOMPLETE"})
    names.add("COMPLETE")
    names.add("v10_4_2_reuse_audit.json")
    for name in names:
        path = case_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "fixture\n")


def _write_stub_reuse_module(stub_root: Path) -> None:
    package = stub_root / "arrhenius_fracture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "reuse_v1041_v1042.py").write_text(
        """from __future__ import annotations
import os
from pathlib import Path


def _log(message):
    path = Path(os.environ["STUB_REUSE_LOG"])
    with path.open("a") as stream:
        stream.write(message + "\\n")


def verify_materialized_case(root):
    _log(f"materialized:{Path(root)}")
    return {"source_case": os.environ["STUB_SOURCE_CASE"]}


def verify_source_case(root):
    _log(f"source:{Path(root)}")
    return {"verified": True}
"""
    )


def _run_verifier(
    tmp_path: Path,
    function_text: str,
    *,
    allow_legacy_reuse: bool,
) -> subprocess.CompletedProcess[str]:
    case_root = tmp_path / ("case_allow" if allow_legacy_reuse else "case_reject")
    case_root.mkdir()
    _make_required_fixture(function_text, case_root)

    stub_root = tmp_path / "stubs"
    if not stub_root.exists():
        _write_stub_reuse_module(stub_root)
    source_case = tmp_path / "source_case"
    source_case.mkdir(exist_ok=True)
    log_path = tmp_path / ("allow.log" if allow_legacy_reuse else "reject.log")

    script = tmp_path / ("run_allow.sh" if allow_legacy_reuse else "run_reject.sh")
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        f"PYTHON_BIN={sys.executable!s}\n"
        "TARGET_EXT_UM=1000\n"
        "THETA=0\n"
        "SAVE_SNAPSHOTS=20\n"
        "SNAPSHOT_COLS=5\n"
        + function_text
        + "\nverified_complete \"$CASE_ROOT\" option candidate 300 3621\n"
    )
    env = os.environ.copy()
    env.update(
        {
            "CASE_ROOT": str(case_root),
            "PYTHONPATH": str(stub_root),
            "STUB_SOURCE_CASE": str(source_case),
            "STUB_REUSE_LOG": str(log_path),
            "ALLOW_V1041_REUSE_AFTER_STAGGER_FIX": (
                "1" if allow_legacy_reuse else "0"
            ),
        }
    )
    return subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_final_generated_scheduler_orders_and_hardens_reuse(tmp_path: Path):
    final = _generate_final_scheduler(tmp_path)
    function = _extract_verified_complete(final)
    guard = 'v1042_reuse_path = root / "v10_4_2_reuse_audit.json"'
    assert final.count("verified_complete()") == 1
    assert function.index(guard) < function.index("for key, value in expected.items():")
    assert "SKIP_REUSED_VERIFIED" in function
    assert "FAILED_REUSE_VERIFICATION" in function
    assert "RERUN_REQUIRED_STAGGER_TIME_CORRECTION" in function
    assert 'verify_source_case(Path(reuse_audit["source_case"]))' in function
    assert 'find "$OUTROOT" -name COMPLETE -print -quit' in final
    assert "acceptance_failures=$(" in final
    assert "failed_or_incomplete_cases" in final


def test_final_verifier_rejects_legacy_reuse_by_default(tmp_path: Path):
    final = _generate_final_scheduler(tmp_path)
    function = _extract_verified_complete(final)
    result = _run_verifier(
        tmp_path, function, allow_legacy_reuse=False
    )
    assert result.returncode != 0
    assert "RERUN_REQUIRED_STAGGER_TIME_CORRECTION" in result.stdout
    assert "SKIP_REUSED_VERIFIED" not in result.stdout


def test_final_verifier_audit_opt_in_uses_recorded_source(tmp_path: Path):
    final = _generate_final_scheduler(tmp_path)
    function = _extract_verified_complete(final)
    result = _run_verifier(
        tmp_path, function, allow_legacy_reuse=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIP_REUSED_VERIFIED" in result.stdout
    log = (tmp_path / "allow.log").read_text().splitlines()
    assert log[0].startswith("materialized:")
    assert log[1] == f"source:{tmp_path / 'source_case'}"


def test_public_wrapper_declares_fresh_v1043_semantics():
    wrapper = (
        ROOT / "scripts" / "run_v10_4_paper_four_class_orientation_rate.sh"
    ).read_text()
    assert "build_v10_4_2_reuse_aware_launcher.py" in wrapper
    assert "v10.4.3 stagger-consistent bulk plasticity" in wrapper
    assert "Legacy v10.4.1 reuse: disabled for corrected production" in wrapper
