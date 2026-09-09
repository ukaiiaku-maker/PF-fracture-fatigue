#!/usr/bin/env python3
"""Build deterministic provenance for the compact V5.4.1 review packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


BASE = "e2aff736afe0e1d2d1b600c25743de317a71c7ba"
EXECUTION = "fb6516bb6a9f770fee899d382a902cf0bdf1e701"
V54_RECORD = "f9d6d115110bb0f915d6403f1a83671698a1ab3d"
V541_CODE = "ae9a06d8c42287428e917baef143b0c1142cefd8"
BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def inventory(root: Path, paths) -> list[dict]:
    records = []
    for path in sorted(paths):
        records.append({
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    packet = args.packet.resolve()
    required = (
        "pf_branching_v5_4_source_integration.patch",
        "pf_branching_v5_4_record_delta.patch",
        "pf_branching_production_restore_only_sentinel_v5_4_1.json",
        "pf_branching_exact_prefix_runtime_weights_v5_4_1.csv",
        "pf_branching_exact_prefix_runtime_weights_v5_4_1.json",
        "pf_branching_figure_identity_audit_v5_4_1.json",
        "pf_branching_completion_preflight_v5_4_1.json",
        "run_pf_current_source_branching_completion_pair_v5_4_1.py",
        "plot_pf_branching_archived_fields_v5_4_1.py",
    )
    for name in required:
        if not (packet / name).is_file():
            raise RuntimeError(f"missing V5.4.1 product: {name}")
    figures = list((packet / "figures").glob("pf_branching_*v5_4_1.*"))
    triplet_stems = (
        "pf_branching_final_crack_structure_v5_4_1",
        "pf_branching_final_damage_and_process_fields_v5_4_1",
        "pf_branching_model_native_local_K_vs_front_progress_v5_4_1",
    )
    for stem in triplet_stems:
        for suffix in ("png", "pdf", "svg"):
            if not (packet / "figures" / f"{stem}.{suffix}").is_file():
                raise RuntimeError(f"missing corrected figure: {stem}.{suffix}")
    source_paths = [path for path in (packet / "source_review/source").rglob("*") if path.is_file()]
    if not source_paths:
        raise RuntimeError("source review packet is empty")

    commits = {}
    for label, commit in (("corrected_base", BASE), ("qualified_execution", EXECUTION),
                          ("v5_4_record", V54_RECORD), ("v5_4_1_code", V541_CODE)):
        commits[label] = {"commit": commit, "tree": git(repo, "rev-parse", commit + "^{tree}")}
    ancestry = {}
    for left, right in ((BASE, EXECUTION), (EXECUTION, V54_RECORD), (V54_RECORD, V541_CODE)):
        result = subprocess.run(("git", "-C", str(repo), "merge-base", "--is-ancestor", left, right))
        ancestry[f"{left[:7]}_ancestor_of_{right[:7]}"] = result.returncode == 0
    if not all(ancestry.values()):
        raise RuntimeError("V5.4.1 ancestry chain is not linear")

    sentinel = json.loads((packet / required[2]).read_text())
    weights = json.loads((packet / required[4]).read_text())
    figure_audit = json.loads((packet / required[5]).read_text())
    preflight = json.loads((packet / required[6]).read_text())
    if any(item.get("qualification") != "PASS" for item in (sentinel, weights, figure_audit, preflight)):
        raise RuntimeError("one or more V5.4.1 qualification products did not pass")

    products = [
        path for path in packet.rglob("*") if path.is_file()
        and path.name not in {
            "pf_branching_v5_4_1_source_provenance.json",
            "PF_CURRENT_SOURCE_BRANCHING_COMPLETION_SEAL_V5_4_1.md",
        }
    ]
    payload = {
        "schema": "pf_branching_v5_4_1_source_provenance/1",
        "qualification": "PASS",
        "boundary": BOUNDARY,
        "commits": commits,
        "ancestry": ancestry,
        "record_generation_head": git(repo, "rev-parse", "HEAD"),
        "record_generation_tree": git(repo, "rev-parse", "HEAD^{tree}"),
        "producer_code_commit": V541_CODE,
        "qualified_execution_commit": EXECUTION,
        "v5_4_review_record_commit": V54_RECORD,
        "source_deltas": {
            "base_to_qualified_execution": "pf_branching_v5_4_source_integration.patch",
            "qualified_execution_to_v5_4_record": "pf_branching_v5_4_record_delta.patch",
            "execution_physics_file_changed_between_fb6516b_and_f9d6d11": False,
            "fb6516b_to_f9d6d11_scope": "reports, qualification publisher, launcher, figures, and audit records",
            "v5_4_1_production_file_change": "exact accepted-state restore plus default-off pre-solve sentinel",
            "v5_4_1_physics_or_parameter_change": False,
        },
        "pinned_inputs": {
            "source_family_sha256": "b109a2fd6fc393fc986b1f15d6edd7c37366d84c111710ea70bcaba75f426847",
            "target_family_sha256": "423bc3232326b8ccc3ffcca0aa6b5363c67bad2c64debee49925bf1e6413e8cb",
            "target_family_physics_fingerprint": "e0bc48eb8c3f5526877a21f8551e500046a614df8081693f340084fb73400302",
            "mechanical_configuration_sha256": "3ccc690c103f96da8eca7ee4bc67272653bd888b402bb0878c1b1f6eb9dee6f9",
        },
        "production_restore": {
            "qualification": sentinel["qualification"],
            "migrated_package_immutable": sentinel["migrated_package_immutable"],
            "pf_workers_started": sentinel["pf_workers_started"],
            "mechanics_solves_performed": sentinel["mechanics_solves_performed"],
            "provider_lookups_performed": sentinel["provider_lookups_performed"],
        },
        "resolver": {
            "qualification": weights["qualification"],
            "envelope_relative_tolerance": weights["envelope_relative_tolerance"],
            "interpolation_error_metadata": weights["interpolation_error_metadata"],
            "metadata_separated": weights[
                "envelope_tolerance_and_interpolation_error_separately_represented"
            ],
        },
        "figure_identity": {
            "qualification": figure_audit["qualification"],
            "ownership_checks": figure_audit["ownership_checks"],
            "postbirth_counts": figure_audit["postbirth_counts"],
            "invalid_contour_points_connected_as_local_contour": figure_audit[
                "invalid_contour_points_connected_as_local_contour"
            ],
        },
        "completion": {
            "authorized": preflight["completion_authorized"],
            "executed": preflight["completion_executed"],
            "thousand_um_extension_authorized": preflight["thousand_um_extension_authorized"],
            "predictive_branching_physics_validated": preflight[
                "predictive_branching_physics_validated"
            ],
        },
        "source_review_inventory": inventory(packet, source_paths),
        "product_inventory": inventory(packet, products),
    }
    target = packet / "pf_branching_v5_4_1_source_provenance.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
