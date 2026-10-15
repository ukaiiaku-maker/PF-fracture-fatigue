#!/usr/bin/env python3
"""Build the compact, disabled readiness handoff without executing solvers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "common_framework_readiness"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.parameter_dossier import build_parameter_dossier
from arrhenius_fracture.prospective_campaign_v2 import build_cases_v2, summary
from arrhenius_fracture.qualification_manifest import manifest_payload
from arrhenius_fracture.readiness_fem_backend import inspect_current_companion


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> None:
    OUT.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    cases = build_cases_v2(ROOT)
    raw = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                   for row in cases)
    (OUT / "PROSPECTIVE_CAMPAIGN_V2.jsonl").write_bytes(raw)
    _write_json(OUT / "CAMPAIGN_V2_SUMMARY.json", summary(cases, raw))
    _write_json(OUT / "QUALIFICATION_ONLY_MANIFEST.json", manifest_payload(ROOT))
    _write_json(OUT / "PARAMETER_PARENT_DOSSIER_V2_NATIVE.json", build_parameter_dossier(ROOT))
    _write_json(OUT / "CURRENT_FEM_COMPANION_BINDING.json", inspect_current_companion().to_dict())

    companion = ROOT.parent / "fem-czm-current-pf-companion"
    continuation = Path("/Users/shen/.codex/attachments/5a96e08e-732a-4768-b29c-9f00dc80575a/Pasted text.txt")
    _write_json(OUT / "SOURCE_IDENTITIES.json", {
        "schema": "common-fracture-readiness-source-identities/1", "generated_utc": now,
        "pf": {"path": str(ROOT), "branch": _git(ROOT, "branch", "--show-current"),
               "commit": _git(ROOT, "rev-parse", "HEAD")},
        "fem": {"path": str(companion), "branch": _git(companion, "branch", "--show-current"),
                "commit": _git(companion, "rev-parse", "HEAD")},
        "external_evidence_manifest": {"path": "../common_framework_release/EXTERNAL_EVIDENCE_MANIFEST.json",
                                       "sha256": _sha(ROOT / "common_framework_release/EXTERNAL_EVIDENCE_MANIFEST.json")},
        "controlling_continuation": {"sha256": _sha(continuation), "date": "2026-10-12"},
    })
    _write_json(OUT / "FRAC_GOALS_1.0_REFERENCE.json", {
        "charter_id": "FRAC-GOALS-1.0", "immutable": True, "modified": False,
        "availability": "NOT_PRESENT_AS_A_STANDALONE_SOURCE_FILE_IN_THE_PROVIDED_WORKSPACE_OR_ATTACHMENTS",
        "handling": "No charter text was reconstructed or edited. Goal IDs in v2 are cross-references from the controlling continuation.",
        "controlling_continuation_sha256": _sha(continuation),
    })
    goals = []
    for i in range(1, 17):
        goal = f"FG-{i:02d}"
        status = "OPEN_SCIENTIFIC_PRODUCTION_NOT_REQUESTED_ON_THIS_MACHINE"
        if goal in {"FG-02", "FG-03", "FG-05", "FG-15"}:
            status = "PARAMETER_TOOLING_QUALIFIED_PARENT_DOMAIN_REMAINS_OPEN"
        elif goal in {"FG-08", "FG-09", "FG-10"}:
            status = "CONTRACT_QUALIFIED_PATCH_AND_CRACKED_BODY_EXECUTION_OPEN"
        elif goal == "FG-11":
            status = "NATIVE_INTERFACE_IMPLEMENTED_MINIMAL_CONSTITUTIVE_QUALIFICATION_COMPLETE_SPECIMEN_OPEN"
        elif goal == "FG-12":
            status = "FEM_INTERNAL_MATCHED_OBSERVER_AND_ACTIVE_RESTART_QUALIFIED_PF_FEM_THREE_LEVEL_OPEN"
        elif goal in {"FG-13", "FG-14"}:
            status = "TRANSACTION_CONTRACT_READY_SPATIAL_OR_RECURSIVE_NATIVE_ADAPTER_QUALIFICATION_OPEN"
        elif goal == "FG-16":
            status = "FEM_ACTIVE_TOPOLOGY_RESTART_QUALIFIED_BRANCH_AND_CAVITY_TRANSITIONS_OPEN"
        goals.append({"goal_id": goal, "status": status, "scientific_evidence": "DEFERRED_NOT_REQUESTED_ON_THIS_MACHINE"})
    _write_json(OUT / "FRAC_GOALS_1.0_LIVE_PROGRESS.json", {
        "schema": "frac-goals-live-progress/1", "charter_modified": False,
        "production_execution_authorized": False, "manuscript_modified": False,
        "goals": goals})

    (OUT / "RESTORE_AND_STORAGE.md").write_text(f"""# Restore and storage boundary

Generated {now}. Production execution is not authorized.

- PF checkout: `{ROOT}` at `{_git(ROOT, 'rev-parse', 'HEAD')}` (recorded parent identity; apply the readiness commits listed in the final receipt).
- Independent FEM/CZM checkout: `{companion}` at `{_git(companion, 'rev-parse', 'HEAD')}`.
- Python environments: `arrhenius-sharp-front-v10-codex` and `arrhenius-fem-czm`.
- External evidence remains referenced by the release manifest and is not duplicated here. Restore only the dependency subset required by a qualification.
- Capacity is admitted from free bytes against **peak transient + reserve**, one substantial worker at a time. See `QUALIFICATION_ONLY_MANIFEST.json` for the measured snapshot and per-job estimates.
- The earlier 28.35 GiB figure is a sum of retained-output estimates, not a peak-space proof. Current free space admits the preliminary peak estimates, but Q-PF-FEM-3LEVEL and Q-BRANCH-RECURSIVE remain unexecuted because their named adapters are unresolved—not because storage was falsely declared inadequate.
- Retain configuration/source identities, scalar histories, residual and convergence tables, selected accepted states and restart inputs. Do not retain movies, dense field sequences, per-iteration vectors or duplicate external evidence by default.
""")
    (OUT / "README.md").write_text("""# Common fracture framework — readiness handoff

This is an engineering-readiness package, not a scientific release. The v1 campaign is preserved in `../common_framework_release`; v2 is separately versioned and disabled. `production_execution_authorized=false` and `manuscript_modified=false` are invariant.

Numerical evidence marked `DEFERRED_RESOURCE` has not run. Production and all scientific surveys are **not requested on this machine**.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-dossier", action="store_true",
                        help="regenerate only the native v2 parameter dossier")
    args = parser.parse_args()
    if args.parameter_dossier:
        _write_json(
            OUT / "PARAMETER_PARENT_DOSSIER_V2_NATIVE.json",
            build_parameter_dossier(ROOT),
        )
    else:
        build()
