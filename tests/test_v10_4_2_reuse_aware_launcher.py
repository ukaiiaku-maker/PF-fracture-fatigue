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
    spec = importlib.util.spec_from_file_location("v1042_reuse_aware_builder", path)
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

    candidates = []
    pattern = re.compile(
        r'^OUTROOT="\$OUTROOT" "\$PYTHON_BIN" - <<[\'\"]?PY[\'\"]?\n'
        r'(?P<body>.*?)^PY$',
        flags=re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(outer):
        body = match.group("body")
        if "SOURCE_SCHEDULER" in body and "GENERATED_SCHEDULER" in body:
            candidates.append(body)
    assert len(candidates) == 1

    generator = tmp_path / "generate_final_scheduler.py"
    generator.write_text(candidates[0])
    outroot = tmp_path / "out"
    outroot.mkdir()
    generated = tmp_path / "final_scheduler.sh"
    generated_plotter = outroot / "plotter.py"
    env = os.environ.copy()
    env.update(
        {
            "SOURCE_SCHEDULER": str(
                ROOT / "scripts" / "run_v10_2_27_paper_four_class_30deg_long_rcurves.sh"
            ),
            "SOURCE_PLOTTER": str(
                ROOT / "scripts" / "plot_v10_2_27_paper_four_class_rcurves.py"
            ),
            "GENERATED_SCHEDULER": str(generated),
            "GENERATED_PLOTTER": str(generated_plotter),
            "OUTROOT": str(outroot),
        }
    )
    subprocess.run([sys.executable, str(generator)], cwd=ROOT, env=env, check=True)
    subprocess.run(["bash", "-n", str(generated)], check=True)
    return generated.read_text()


def _extract_shell_function(shell: str, name: str, following: str) -> str:
    start = shell.index(f"{name}() {{")
    end = shell.index(f"\n}}\n\n{following}", start) + 3
    return shell[start:end]


def _make_materialized_fixture(case_root: Path, scheduler: str) -> None:
    verifier_start = scheduler.index("verified_complete() {")
    early_guard = scheduler.index(
        'v1042_reuse_path = root / "v10_4_2_reuse_audit.json"',
        verifier_start,
    )
    prefix = scheduler[verifier_start:early_guard]
    required_names = set(
        re.findall(r"root / ['\"]([^'\"]+)['\"]", prefix)
    )
    required_names.update(
        {
            "stage3_case_status.json",
            "v10_2_27_case_contract.json",
            "v10_2_27_paper_four_class_parameter_transfer.json",
            "command.sh",
            "v10_4_2_reuse_audit.json",
        }
    )
    required_names.discard("RUN_FAILED")
    required_names.discard("PLASTIC_FLOW")

    case_root.mkdir(parents=True)
    source_root = case_root.parent / "source_case"
    source_root.mkdir()
    source_complete = source_root / "COMPLETE"
    source_complete.write_text("complete\n")
    (case_root / "COMPLETE").symlink_to(source_complete)
    for name in sorted(required_names - {"COMPLETE"}):
        path = case_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("{}\n")

    required_paths = [case_root / name for name in required_names]
    assert all(path.is_file() for path in required_paths), [
        str(path) for path in required_paths if not path.is_file()
    ]


def _write_fixture_module(stub_root: Path) -> None:
    stub_root.mkdir(parents=True)
    (stub_root / "v1042_reuse_fixture.py").write_text(
        """from __future__ import annotations
import os
from pathlib import Path


def verify_materialized_case(root):
    if os.environ.get('REUSE_STUB_FAIL') == '1':
        raise RuntimeError('synthetic corrupt reuse audit')
    print(f'VERIFY_MATERIALIZED_STUB {root}')
    return {'source_case': str(Path(root).parent / 'source_case')}


def verify_source_case(root):
    print(f'VERIFY_SOURCE_STUB {root}')
    return {'verified': True}
"""
    )


def _run_generated_reuse_case(
    tmp_path: Path,
    scheduler: str,
    *,
    corrupt: bool,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    option = "v913_paper_peak01_0242980_persistent_sites"
    outroot = tmp_path / ("campaign_corrupt" if corrupt else "campaign_valid")
    case_root = outroot / option / "T300K_th0_seed3621"
    _make_materialized_fixture(case_root, scheduler)

    stub_root = tmp_path / ("stub_corrupt" if corrupt else "stub_valid")
    _write_fixture_module(stub_root)
    sentinel = tmp_path / ("solver_corrupt_started" if corrupt else "solver_valid_started")
    python_wrapper = tmp_path / ("python_corrupt.sh" if corrupt else "python_valid.sh")
    python_wrapper.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-u" && "${2:-}" == "-m" ]]; then
  : > "$SOLVER_SENTINEL"
  exit 99
fi
exec "$REAL_PYTHON" "$@"
"""
    )
    python_wrapper.chmod(0o755)

    verified_complete = _extract_shell_function(
        scheduler,
        "verified_complete",
        "run_case() {",
    )
    production_import = (
        "from arrhenius_fracture.reuse_v1041_v1042 import ("
    )
    assert verified_complete.count(production_import) == 1
    verified_complete = verified_complete.replace(
        production_import,
        "from v1042_reuse_fixture import (",
        1,
    )
    run_case = _extract_shell_function(scheduler, "run_case", "pids=()")
    runner = tmp_path / ("run_corrupt.sh" if corrupt else "run_valid.sh")
    runner.write_text(
        f"""#!/usr/bin/env bash
set -u
set -o pipefail
ROOT={str(ROOT)!r}
OUTROOT={str(outroot)!r}
PYTHON_BIN={str(python_wrapper)!r}
SKIP_FINISHED=1
RESTART_INCOMPLETE=1
TARGET_EXT_UM=1000
THETA=0
SAVE_SNAPSHOTS=20
SNAPSHOT_COLS=5
FAMILY_JSON=dummy-family.json
REGISTRY=dummy-registry.csv
STEPS=10
PERSISTENT_SOURCE_MIN_WIDTH_UM=0

candidate_for_option() {{
  printf '%s\n' v913_zeroD_sobol_0242980
}}

write_case_contract() {{
  echo 'ERROR: write_case_contract must not be reached' >&2
  return 97
}}

{verified_complete}

{run_case}

set +e
run_case {option!r} 300 3621
rc=$?
set -e
echo "RUN_CASE_RC=$rc"
exit "$rc"
"""
    )
    runner.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "REAL_PYTHON": sys.executable,
            "SOLVER_SENTINEL": str(sentinel),
            "PYTHONPATH": os.pathsep.join(
                [str(stub_root), str(ROOT), env.get("PYTHONPATH", "")]
            ),
            "REUSE_STUB_FAIL": "1" if corrupt else "0",
        }
    )
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, sentinel


def test_final_generated_scheduler_orders_reuse_before_native_checks(tmp_path: Path):
    scheduler = _generate_final_scheduler(tmp_path)
    verifier_start = scheduler.index("verified_complete() {")
    skip = 'print(f"SKIP_REUSED_VERIFIED {root}")'
    native = "expected = {"
    assert scheduler.count(skip) == 1
    assert scheduler.index(skip, verifier_start) < scheduler.index(native, verifier_start)
    assert "verify_source_case(Path(reuse_audit[\"source_case\"]))" in scheduler
    assert "from arrhenius_fracture.reuse_v1041_v1042 import (" in scheduler
    assert "find \"$OUTROOT\" \\( -type f -o -type l \\) -name COMPLETE" in scheduler
    assert "acceptance_rc=$?" in scheduler
    assert "FAILED_REUSE_VERIFICATION" in scheduler


def test_valid_reused_case_skips_without_solver_launch(tmp_path: Path):
    scheduler = _generate_final_scheduler(tmp_path)
    result, sentinel = _run_generated_reuse_case(tmp_path, scheduler, corrupt=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERIFY_MATERIALIZED_STUB" in result.stdout
    assert "VERIFY_SOURCE_STUB" in result.stdout
    assert "SKIP_REUSED_VERIFIED" in result.stdout
    assert "SKIP verified complete" in result.stdout
    assert "RUN_CASE_RC=0" in result.stdout
    assert not sentinel.exists()
    assert not any((tmp_path / "campaign_valid").rglob("RUN_FAILED"))


def test_corrupt_reuse_fails_closed_without_solver_launch(tmp_path: Path):
    scheduler = _generate_final_scheduler(tmp_path)
    result, sentinel = _run_generated_reuse_case(tmp_path, scheduler, corrupt=True)
    assert result.returncode == 3
    assert "FAILED_REUSE_VERIFICATION" in result.stderr
    assert "RUN_CASE_RC=3" in result.stdout
    assert not sentinel.exists()


def test_public_wrapper_uses_reuse_aware_builder():
    wrapper = (
        ROOT / "scripts" / "run_v10_4_paper_four_class_orientation_rate.sh"
    ).read_text()
    assert "build_v10_4_2_reuse_aware_launcher.py" in wrapper
    assert "verify v10.4.2 reuse audit before native command checks" in wrapper
