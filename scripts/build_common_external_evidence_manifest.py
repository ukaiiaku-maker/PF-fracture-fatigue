#!/usr/bin/env python3
"""Build the content-addressed raw-evidence manifest for the framework release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.external_evidence import fingerprint_path


DEFAULT_ARCHIVE = Path("/Users/shen/Fracture_Backups/fracture-completion-20261010-d1d15bf/final_release_20261011/final-fracture-evidence-20261011.tar")
SUPPLEMENT_ARCHIVE = Path("/Users/shen/Fracture_Backups/common-framework-20261012/common-framework-supplemental-evidence-20261012.tar")
RELEASE = ROOT / "common_framework_release"
RESTORE_ROOTS = (
    "integration_evidence/final_fracture_campaign",
    "integration_evidence/current_science_pilots_6abbf10",
    "integration_evidence/branch_cavity_bb0c21d9",
    "integration_evidence/v36_active_restart_340727f7",
)
SUPPLEMENT_ROOTS = (
    "integration_evidence/active_fracture_comparisons_0a5d55e",
    "integration_evidence/native_restart_transport_a30a78c",
    "integration_evidence/native_restart_transport_f4ba118",
    "integration_evidence/native_restart_transport_f766285",
    "integration_evidence/positive_transport_854d32d",
    "integration_evidence/targeted_completion",
    "integration_evidence/option_completion_v2",
)


def _item(identifier: str, source: Path, archive: Path, producer: str) -> dict:
    relative = source.relative_to(ROOT).as_posix()
    return {
        "evidence_id": identifier,
        "path": f"../{relative}",
        "mandatory": True,
        "producer": producer,
        "dependencies": ["qualified Python environment", "matching committed source identity"],
        "permitted_locations": [relative, str(source.resolve())],
        "restore_command": f"tar -xf {archive} -C <clean-checkout-parent> {relative}",
        "backup_archive": str(archive),
        "fingerprint": fingerprint_path(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RELEASE / "EXTERNAL_EVIDENCE_MANIFEST.json")
    parser.add_argument("--primary-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--supplement-archive", type=Path, default=SUPPLEMENT_ARCHIVE)
    args = parser.parse_args()
    items = []
    for relative in RESTORE_ROOTS:
        items.append(_item(relative.replace("/", "__"), ROOT / relative, args.primary_archive,
                           "retained original PF/FEM-CZM campaign or qualification producer"))
    for relative in SUPPLEMENT_ROOTS:
        items.append(_item(relative.replace("/", "__"), ROOT / relative, args.supplement_archive,
                           "post-release native integration/active-state producer"))
    for identifier, archive in (("independent_primary_archive", args.primary_archive),
                                ("independent_supplement_archive", args.supplement_archive)):
        items.append({
            "evidence_id": identifier, "path": str(archive.resolve()), "mandatory": True,
            "producer": "append-only independent-device backup",
            "dependencies": ["POSIX tar"], "permitted_locations": [str(archive.resolve())],
            "restore_command": f"tar -tf {archive}", "fingerprint": fingerprint_path(archive),
        })
    payload = {
        "schema": "common-framework-external-evidence-manifest/1",
        "policy": {
            "portable_suite_requires_items": False,
            "full_qualification_requires_all_mandatory_items": True,
            "missing_mandatory_status": "INCOMPLETE",
            "raw_evidence_tracked_in_git": False,
            "local_raw_data_deleted": False,
        },
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.out), "item_count": len(items),
                      "bytes": sum(item["fingerprint"]["bytes"] for item in items)}, indent=2))


if __name__ == "__main__":
    main()
