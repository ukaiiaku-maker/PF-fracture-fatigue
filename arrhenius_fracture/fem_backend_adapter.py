"""Read-only verifier for the separately versioned full FEM/CZM backend."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .shared_runtime import FEM_BRANCH, FEM_COMMIT, FEM_REPOSITORY

DEFAULT_FEM_ROOT = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "Arrhenius_FEM_CZM_MPZ_v10_0_5_17_paper7_self_contained"
)
ENTRY_RELATIVE = Path(
    "arrhenius_fracture/"
    "mode_i_first_passage_v10_0_5_18_4_0_theta0_pf_parity.py"
)
ENTRY_MODULE = "arrhenius_fracture.mode_i_first_passage_v10_0_5_18_4_0_theta0_pf_parity"
GENERIC_FEM_ROOT = Path(
    "/Volumes/Data/working-papers/Fracture/fem-czm-current-pf-companion"
)
GENERIC_ENTRY_RELATIVE = Path("arrhenius_fracture/generic_current_pf_companion.py")
GENERIC_ENTRY_MODULE = "arrhenius_fracture.generic_current_pf_companion"
GENERIC_FEM_COMMIT = "6247534831263c1dbf5dcc06ebcf4211dca7c730"
GENERIC_FEM_BRANCH = "codex/current-pf-companion"
DEFAULT_FEM_PYTHON = Path(
    "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-fem-czm/bin/python"
)

def _run(root: Path, *args: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(args, cwd=root, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=environment)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "external command failed")
    return result.stdout.strip()

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

@dataclass(frozen=True)
class FEMBackendReceipt:
    schema: str
    repository: str
    root: str
    expected_commit: str
    observed_commit: str | None
    expected_branch: str
    observed_branch: str | None
    clean: bool | None
    entry_file: str
    entry_sha256: str | None
    constructor: str
    interpreter: str
    available: bool
    execution_verified: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

def resolve_fem_backend_metadata(root: str | Path | None = None) -> FEMBackendReceipt:
    """Resolve the configured dependency without Git or Python execution."""
    root = Path(root or os.environ.get("FRACTURE_FEM_ROOT", DEFAULT_FEM_ROOT)).resolve()
    entry = root / ENTRY_RELATIVE
    interpreter = Path(os.environ.get("FRACTURE_FEM_PYTHON", DEFAULT_FEM_PYTHON)).resolve()
    installed = root.is_dir()
    return FEMBackendReceipt(
        "fem-backend-receipt/2", FEM_REPOSITORY, str(root), FEM_COMMIT, None,
        FEM_BRANCH, None, None, str(entry), None, f"{ENTRY_MODULE}:main",
        str(interpreter), False, False,
        "configured dependency present but uninspected" if installed
        else "configured dependency is not installed")


def inspect_fem_backend(
    root: str | Path | None = None, *, probe_import: bool = False,
) -> FEMBackendReceipt:
    root = Path(root or os.environ.get("FRACTURE_FEM_ROOT", DEFAULT_FEM_ROOT)).resolve()
    entry = root / ENTRY_RELATIVE
    try:
        observed_commit = _run(root, "git", "rev-parse", "HEAD")
        observed_branch = _run(root, "git", "branch", "--show-current")
        clean = not bool(_run(root, "git", "--no-optional-locks", "status", "--porcelain"))
    except (OSError, RuntimeError) as exc:
        return FEMBackendReceipt(
            "fem-backend-receipt/1", FEM_REPOSITORY, str(root), FEM_COMMIT, None,
            FEM_BRANCH, None, None, str(entry), None, f"{ENTRY_MODULE}:main",
            sys.executable, False, False, f"repository unavailable: {exc}")
    problems = []
    interpreter = Path(os.environ.get("FRACTURE_FEM_PYTHON", DEFAULT_FEM_PYTHON)).resolve()
    if observed_commit != FEM_COMMIT:
        problems.append(f"commit mismatch: {observed_commit}")
    if observed_branch != FEM_BRANCH:
        problems.append(f"branch mismatch: {observed_branch}")
    if not clean:
        problems.append("checkout has an undeclared dirty patch")
    if not entry.is_file():
        problems.append("mapped theta0 entry file is absent")
    if not interpreter.is_file():
        problems.append(f"configured FEM interpreter is absent: {interpreter}")
    elif entry.is_file() and probe_import:
        environment = dict(os.environ)
        environment.update({"PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0"})
        probe = subprocess.run(
            [str(interpreter), "-c",
             f"import {ENTRY_MODULE} as backend; assert callable(backend.main)"],
            cwd=root, env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if probe.returncode:
            problems.append(f"FEM constructor import failed: {probe.stderr.strip()}")
    available = not problems
    return FEMBackendReceipt(
        "fem-backend-receipt/1", FEM_REPOSITORY, str(root), FEM_COMMIT,
        observed_commit, FEM_BRANCH, observed_branch, clean, str(entry),
        _sha256(entry) if entry.is_file() else None, f"{ENTRY_MODULE}:main",
        str(interpreter), available, False,
        "available and source-verified; not yet executed" if available else "; ".join(problems))


def inspect_generic_fem_backend(
    root: str | Path | None = None, *, probe_import: bool = False,
) -> FEMBackendReceipt:
    """Verify the current generic companion while preserving untracked evidence.

    Only tracked changes can alter calculation source.  Untracked qualification
    outputs are deliberately retained and therefore are not treated as a source
    mutation.
    """
    root = Path(root or os.environ.get("FRACTURE_GENERIC_FEM_ROOT", GENERIC_FEM_ROOT)).resolve()
    entry = root / GENERIC_ENTRY_RELATIVE
    interpreter = Path(os.environ.get("FRACTURE_FEM_PYTHON", DEFAULT_FEM_PYTHON)).resolve()
    try:
        observed_commit = _run(root, "git", "rev-parse", "HEAD")
        observed_branch = _run(root, "git", "branch", "--show-current")
        tracked_clean = not bool(_run(
            root, "git", "--no-optional-locks", "status", "--porcelain",
            "--untracked-files=no",
        ))
    except (OSError, RuntimeError) as exc:
        return FEMBackendReceipt(
            "fem-backend-receipt/2", str(root), str(root), GENERIC_FEM_COMMIT,
            None, GENERIC_FEM_BRANCH, None, None, str(entry), None,
            f"{GENERIC_ENTRY_MODULE}:main", str(interpreter), False, False,
            f"repository unavailable: {exc}",
        )
    problems = []
    if observed_commit != GENERIC_FEM_COMMIT:
        problems.append(f"commit mismatch: {observed_commit}")
    if observed_branch != GENERIC_FEM_BRANCH:
        problems.append(f"branch mismatch: {observed_branch}")
    if not tracked_clean:
        problems.append("checkout has an undeclared tracked patch")
    if not entry.is_file():
        problems.append("mapped generic current entry file is absent")
    if not interpreter.is_file():
        problems.append(f"configured FEM interpreter is absent: {interpreter}")
    elif entry.is_file() and probe_import:
        environment = dict(os.environ)
        environment.update({"PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0"})
        probe = subprocess.run(
            [str(interpreter), "-c",
             f"import {GENERIC_ENTRY_MODULE} as backend; assert callable(backend.main)"],
            cwd=root, env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if probe.returncode:
            problems.append(f"FEM constructor import failed: {probe.stderr.strip()}")
    available = not problems
    return FEMBackendReceipt(
        "fem-backend-receipt/2", str(root), str(root), GENERIC_FEM_COMMIT,
        observed_commit, GENERIC_FEM_BRANCH, observed_branch, tracked_clean,
        str(entry), _sha256(entry) if entry.is_file() else None,
        f"{GENERIC_ENTRY_MODULE}:main", str(interpreter), available, False,
        "available; tracked source verified and preserved untracked evidence allowed; not yet executed"
        if available else "; ".join(problems),
    )

def require_fem_backend(root: str | Path | None = None) -> FEMBackendReceipt:
    receipt = inspect_fem_backend(root, probe_import=True)
    if not receipt.available:
        raise RuntimeError(f"full FEM/CZM backend unavailable: {receipt.reason}")
    return receipt


def require_generic_fem_backend(root: str | Path | None = None) -> FEMBackendReceipt:
    receipt = inspect_generic_fem_backend(root, probe_import=True)
    if not receipt.available:
        raise RuntimeError(f"generic full FEM/CZM backend unavailable: {receipt.reason}")
    return receipt

__all__ = ["DEFAULT_FEM_ROOT", "DEFAULT_FEM_PYTHON", "ENTRY_MODULE", "ENTRY_RELATIVE",
           "GENERIC_FEM_ROOT", "GENERIC_ENTRY_MODULE", "GENERIC_ENTRY_RELATIVE",
           "FEMBackendReceipt", "inspect_fem_backend", "inspect_generic_fem_backend",
           "require_fem_backend", "require_generic_fem_backend",
           "resolve_fem_backend_metadata"]
