#!/usr/bin/env python3
"""Run the same V11 audited entry in two clean worktrees and compare semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


VOLATILE = {"git_head", "package_import_path"}


def _run(root: Path, out: Path) -> dict:
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=root, text=True):
        raise RuntimeError(f"comparison worktree is dirty: {root}")
    command = [
        sys.executable, "-m", "arrhenius_fracture.sharp_front_v11_branching_audited",
        "--mechanistic-branching", "--mode", "2d", "--crack-backend", "sharp_wake",
        "--maximum-fronts", "16", "--audit-only", "--temperatures=700",
        "--theta-deg=45", "--hazard-seed=3621",
        "--material-option=v913_paper_weakT01_0129902_persistent_sites",
        "--out=" + str(out),
    ]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    return json.loads((out / "v11_branching_model_audit.json").read_text())


def _semantic(payload: dict) -> dict:
    value = {key: item for key, item in payload.items() if key not in VOLATILE}
    value["argv"] = [item for item in value["argv"] if not item.startswith("--out=")]
    return value


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 3:
        raise SystemExit("usage: compare_v11_pr58_pr59_clean.py PR58_ROOT PR59_ROOT OUT.json")
    left_root, right_root, output = map(Path, argv)
    with tempfile.TemporaryDirectory(prefix="v11-clean-compare-") as temp:
        base = Path(temp)
        left = _run(left_root.resolve(), base / "pr58")
        right = _run(right_root.resolve(), base / "pr59")
    left_semantic, right_semantic = _semantic(left), _semantic(right)
    payload = {
        "schema": "v12.v11-clean-worktree-comparison/1",
        "left_git_head": left["git_head"], "right_git_head": right["git_head"],
        "excluded_provenance_fields": sorted(VOLATILE | {"argv:--out"}),
        "left_semantic_sha256": hashlib.sha256(json.dumps(left_semantic, sort_keys=True).encode()).hexdigest(),
        "right_semantic_sha256": hashlib.sha256(json.dumps(right_semantic, sort_keys=True).encode()).hexdigest(),
        "semantic_exact_equal": left_semantic == right_semantic,
        "left_dirty_tree": left["dirty_tree"], "right_dirty_tree": right["dirty_tree"],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["semantic_exact_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
