"""Exact-commit adapter for the recovered OneD V3 stateful-voiding donor."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


MODEL_ID = "r2.recovered_oneD_v3_stateful_voiding/1"
DONOR_REPOSITORY = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/voiding/"
    "oneD-v3-stateful-voiding-reduction"
)
DONOR_COMMIT = "47a5a4bae048e5a8b1c6178efb6406325005f7ef"
DONOR_TREE = "bbd9a4b5432ab0c262a94fe8201bb810e6a9ecc3"
DONOR_ENTRY = "reduced_fracture_v3.autonomous_cli"


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(DONOR_REPOSITORY), *args), check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def donor_receipt() -> dict[str, Any]:
    commit = _git("rev-parse", DONOR_COMMIT)
    tree = _git("rev-parse", DONOR_COMMIT + "^{tree}")
    if commit != DONOR_COMMIT or tree != DONOR_TREE:
        raise RuntimeError("reduced void donor identity mismatch")
    source = subprocess.run(
        ("git", "-C", str(DONOR_REPOSITORY), "show",
         DONOR_COMMIT + ":reduced_fracture_v3/autonomous_cli.py"),
        check=True, stdout=subprocess.PIPE,
    ).stdout
    return {
        "schema": "r2-reduced-void-donor-receipt/1",
        "model_id": MODEL_ID,
        "repository": str(DONOR_REPOSITORY),
        "commit": commit,
        "tree": tree,
        "entry": DONOR_ENTRY,
        "entry_sha256": hashlib.sha256(source).hexdigest(),
        "source_worktree_dirty_ignored": bool(_git("status", "--porcelain")),
        "execution_source": "clean git archive of exact commit",
        "scope": "aligned monotonic functional control; not calibrated 2-D transfer",
        "birth_pathways": {
            "vacancy_diffusion": "independent vacancy action/hits/inventory",
            "direct_dislocation_interaction": "independent direct action/hits/free-volume",
            "cavity_surface_emission": "separate post-birth growth process",
        },
    }


def execute_reference(output: str | Path, *, steps: int = 1000, duration_s: float = 0.001) -> dict[str, Any]:
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = donor_receipt()
    with tempfile.TemporaryDirectory(prefix="r2_oned_v3_") as raw:
        root = Path(raw)
        archive = root / "donor.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ("git", "-C", str(DONOR_REPOSITORY), "archive", DONOR_COMMIT),
                check=True, stdout=stream,
            )
        source = root / "source"
        source.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(source, filter="data")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(source)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        run = subprocess.run(
            (
                sys.executable, "-m", DONOR_ENTRY, "--output", str(target),
                "--steps", str(int(steps)), "--duration-s", str(float(duration_s)),
            ),
            cwd=source, env=environment, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    result = json.loads(target.read_text())
    events = list(result.get("events", []))
    event_names = [str(item.get("event", "")) for item in events]
    history = list(result.get("history", []))
    summary = {
        **receipt,
        "result": str(target),
        "result_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "classification": result.get("classification"),
        "termination_reason": result.get("termination_reason"),
        "history_rows": len(history),
        "event_count": len(events),
        "event_types": sorted(set(event_names)),
        "vacancy_action_exercised": any(
            any(float(value) > 0.0 for value in row.get("vacancy_action", {}).values())
            for row in history
        ),
        "direct_action_exercised": any(
            any(float(value) > 0.0 for value in row.get("direct_action", {}).values())
            for row in history
        ),
        "resolved_or_subgrid_cavity_observed": any(
            any(value is not None for value in row.get("radius_m", {}).values())
            for row in history
        ),
        "capability_limits": result.get("capability_limits", []),
    }
    receipt_path = target.with_name(target.stem + ".r2_receipt.json")
    receipt_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


__all__ = [
    "DONOR_COMMIT", "DONOR_ENTRY", "DONOR_REPOSITORY", "DONOR_TREE",
    "MODEL_ID", "donor_receipt", "execute_reference",
]
