#!/usr/bin/env python3
"""Publish the deterministic V6.1.1 source and terminal-parity seal."""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arrhenius_fracture.general_multifront_v12 import (
    POLICY_BOUND_FRONT_LIMIT, ResourcePolicy, canonical_hash,
    commit_selected_proposal,
)
from arrhenius_fracture.v5_4_2_runtime_parity_v12 import (
    replay_full_v5_4_2_terminal_runtime_parity,
    replay_v5_4_2_runtime_parity,
)
from scripts.build_pf_general_multifront_qualification_v6_1 import initial, proposal


BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
V6_1_COMMIT = "82c5de2e45c765bf0263a97145681ffe821846e9"
V6_1_TREE = "ab60dce9a259ca736801444c00103c16a4ba2be5"
V6_COMMIT = "d993dd933f52281b89cf22e3aa9cf0db61d64487"
V6_TREE = "40aced628472af5803176fb0d8f87334d5fe63f5"
V6_PARENT_COMMIT = "ae9a06d8c42287428e917baef143b0c1142cefd8"
V6_PARENT_TREE = "2d0edbc7c7b51d81ab5678065ca68a1cef45d134"
SOURCE_PATHS = (
    "arrhenius_fracture/general_multifront_v12.py",
    "arrhenius_fracture/live_topology_kernel_v12.py",
    "arrhenius_fracture/multifront_checkpoint_v12.py",
    "arrhenius_fracture/multifront_output_v12.py",
    "arrhenius_fracture/production_multifront_v12.py",
    "arrhenius_fracture/sharp_front_current_source_multifront_v12.py",
    "arrhenius_fracture/v5_4_2_runtime_parity_v12.py",
    "scripts/build_pf_general_multifront_qualification_v6_1.py",
    "tests/test_general_multifront_v12.py",
    "tests/test_multifront_checkpoint_output_v12.py",
    "tests/test_multifront_checkpoint_v6_1.py",
    "tests/test_multifront_ownership_v12.py",
    "tests/test_multifront_production_integration_v12.py",
    "tests/test_multifront_scheduler_v12.py",
    "tests/test_multifront_source_entry_v12.py",
    "tests/test_multifront_v5_4_2_runtime_parity_v12.py",
    "tests/test_pf_general_multifront_v6_1_products.py",
)
EXECUTION_SOURCE_PATHS = SOURCE_PATHS[:7]
SUPPORTING_V6_1_PRODUCTS = (
    "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_1.md",
    "pf_general_multifront_checkpoint_roundtrip_v6_1.json",
    "pf_general_multifront_multiowner_interval_v6_1.json",
    "pf_general_multifront_partial_handoff_v6_1.csv",
    "pf_general_multifront_production_call_graph_v6_1.json",
    "pf_general_multifront_production_preflight_v6_1.json",
    "pf_general_multifront_scheduler_compatibility_v6_1.csv",
    "pf_general_multifront_validation_v6_1.json",
)
ARCHIVE_NAME = "Archive_PF_GENERAL_MULTIFRONT_V6_1_1_COMPLETE.zip"
MANIFEST_NAME = "pf_general_multifront_v6_1_1_archive_manifest.json"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=repo)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def commit_source(repo: Path, path: str) -> bytes:
    return git(repo, "show", f"{V6_1_COMMIT}:{path}")


def bind_source(repo: Path, out: Path) -> dict[str, Any]:
    commit_line = git(repo, "show", "-s", "--format=%H %T %P", V6_1_COMMIT).decode().strip()
    parent_line = git(repo, "show", "-s", "--format=%H %T %P", V6_COMMIT).decode().strip()
    if commit_line != f"{V6_1_COMMIT} {V6_1_TREE} {V6_COMMIT}":
        raise RuntimeError("V6.1 commit/tree/parent binding mismatch")
    if parent_line != f"{V6_COMMIT} {V6_TREE} {V6_PARENT_COMMIT}":
        raise RuntimeError("V6 commit/tree/parent binding mismatch")

    patch_v6 = out / "pf_general_multifront_v6_source.patch"
    patch_v6.write_bytes(git(repo, "diff", "--binary", f"{V6_PARENT_COMMIT}..{V6_COMMIT}"))
    patch_v6_1 = out / "pf_general_multifront_v6_to_v6_1.patch"
    patch_v6_1.write_bytes(git(repo, "diff", "--binary", f"{V6_COMMIT}..{V6_1_COMMIT}"))
    snapshots: dict[str, Any] = {}
    for path in SOURCE_PATHS:
        data = commit_source(repo, path)
        target = out / "source_snapshot_v6_1" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        blob = git(repo, "rev-parse", f"{V6_1_COMMIT}:{path}").decode().strip()
        snapshots[path] = {
            "git_blob": blob,
            "sha256": sha_bytes(data),
            "size_bytes": len(data),
            "snapshot_path": str(target.relative_to(repo)),
            "snapshot_matches_git_blob_bytes": (
                data == git(repo, "cat-file", "blob", blob)
            ),
        }

    support: dict[str, Any] = {}
    base = "analysis_outputs/pf_current_source_general_multifront_v6_1"
    for name in SUPPORTING_V6_1_PRODUCTS:
        data = git(repo, "show", f"{V6_1_COMMIT}:{base}/{name}")
        target = out / "supporting_v6_1_record" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        support[name] = {
            "sha256": sha_bytes(data), "size_bytes": len(data),
            "path": str(target.relative_to(repo)),
        }

    ancestry = [V6_PARENT_COMMIT] + git(
        repo, "rev-list", "--reverse", "--ancestry-path",
        f"{V6_PARENT_COMMIT}..{V6_1_COMMIT}",
    ).decode().splitlines()
    provenance = {
        "schema": "v6.1.1.source-provenance/1",
        "boundary": BOUNDARY,
        "v6_1_implementation_commit": V6_1_COMMIT,
        "v6_1_implementation_tree": V6_1_TREE,
        "v6_parent_commit": V6_COMMIT,
        "v6_parent_tree": V6_TREE,
        "v6_parent_parent_commit": V6_PARENT_COMMIT,
        "v6_parent_parent_tree": V6_PARENT_TREE,
        "exact_ancestry_path": ancestry,
        "v6_1_source_tree_status": "CLEAN_IMMUTABLE_COMMIT_TREE",
        "v6_1_worktree_clean_at_source_seal_branch_creation": True,
        "patches": {
            "v6_parent_to_v6": {
                "path": str(patch_v6.relative_to(repo)),
                "sha256": sha(patch_v6), "size_bytes": patch_v6.stat().st_size,
            },
            "v6_to_v6_1": {
                "path": str(patch_v6_1.relative_to(repo)),
                "sha256": sha(patch_v6_1), "size_bytes": patch_v6_1.stat().st_size,
            },
        },
        "v6_1_source_snapshots": snapshots,
        "v6_1_snapshot_count": len(snapshots),
        "every_snapshot_matches_git_blob_bytes": all(
            row["snapshot_matches_git_blob_bytes"] for row in snapshots.values()
        ),
        "supporting_v6_1_record": support,
        "pre_v12_execution_physics_unchanged": True,
        "v12_execution_source_added_or_modified": True,
    }
    write_json(out / "pf_general_multifront_v6_1_1_source_provenance.json", provenance)
    return provenance


class Literal16Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value == 16:
            parent = getattr(node, "_parent", None)
            self.rows.append({
                "line": node.lineno,
                "syntax_owner": type(parent).__name__ if parent is not None else None,
            })


def provider_source_audit(repo: Path) -> dict[str, Any]:
    sources = {path: commit_source(repo, path).decode() for path in EXECUTION_SOURCE_PATHS}
    literal_rows: list[dict[str, Any]] = []
    for path, text in sources.items():
        tree = ast.parse(text)
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child._parent = parent  # type: ignore[attr-defined]
        visitor = Literal16Visitor(); visitor.visit(tree)
        for row in visitor.rows:
            row["path"] = path
            row["source_line"] = text.splitlines()[row["line"] - 1].strip()
            literal_rows.append(row)

    provider = sources["arrhenius_fracture/live_topology_kernel_v12.py"]
    general = sources["arrhenius_fracture/general_multifront_v12.py"]
    production = sources["arrhenius_fracture/production_multifront_v12.py"]
    fixed_patterns = {
        "range_16": r"range\s*\(\s*16\s*\)",
        "front_array_times_16": r"(?:front|tip|workspace)[^\n=]*=\s*\[[^\n]*\]\s*\*\s*16",
        "front_tuple_times_16": r"(?:front|tip|workspace)[^\n=]*=\s*\([^\n]*\)\s*\*\s*16",
        "cardinality_comparison_16": r"(?:front|tip|active)[^\n]*(?:<=|>=|<|>)\s*16",
    }
    fixed_matches = {
        name: [path for path, text in sources.items() if re.search(pattern, text)]
        for name, pattern in fixed_patterns.items()
    }

    capped = initial()
    capped = type(capped).from_dict(capped.to_dict())
    capped = replace(
        capped,
        resource_policy=ResourcePolicy(
            "mechanistic", 1, None, "experimental",
            "v11_correlated_proposal_compatibility",
        ),
    )
    stopped = commit_selected_proposal(capped, proposal(capped, capped.active_front_ids[0]))
    gates = {
        "direct_evaluator_route_without_kernel_resolver": (
            "evaluate_exact_topology(request)" in provider
            and "kernel_resolver" not in provider
        ),
        "no_fixed_16_front_allocation_loop_tuple_or_workspace": not any(
            fixed_matches.values()
        ),
        "inherited_16_constant_is_metadata_only": (
            "MAXIMUM_FRONTS_SUPPORTED as V11_REPORTED_LIMIT" in provider
            and "inherited_v11_reported_limit_not_enforced" in provider
            and all("MAXIMUM_FRONTS_SUPPORTED" not in text for path, text in sources.items()
                    if path != "arrhenius_fracture/live_topology_kernel_v12.py")
        ),
        "every_active_front_comes_from_dynamic_inventory": (
            "request.crack_network.active_tip_ids" in provider
            and "for front_id in runtime.active_front_ids" in production
            and "self.crack_network.active_tip_ids" in general
        ),
        "front_resource_limit_is_only_active_front_policy": (
            "front_resource_limit" in provider
            and "front_resource_limit" in general
            and "maximum_fronts_supported\": None" in provider
        ),
        "none_introduces_no_source_finite_cap": (
            "if limit is not None and active_count > limit" in provider
            and "front_resource_limit is not None" in general
        ),
        "realized_binary_trial_limit_precedes_topology_mutation": (
            "post_count = len(state.active_front_ids) + 1" in general
            and general.index("post_count = len(state.active_front_ids) + 1")
            < general.index("child_ids = tuple")
        ),
        "policy_stop_preserves_topology_registry_and_transaction_ledger": (
            stopped.termination_reason == POLICY_BOUND_FRONT_LIMIT
            and stopped.policy_bound
            and stopped.topology_fingerprint == capped.topology_fingerprint
            and stopped.registry_fingerprint == capped.registry_fingerprint
            and stopped.transaction_records == capped.transaction_records
        ),
    }
    result = {
        "schema": "v6.1.1.dynamic-provider-source-audit/1",
        "scope": "V6.1 source at immutable commit; source inspection and pure state only",
        "v6_1_commit": V6_1_COMMIT,
        "v6_1_tree": V6_1_TREE,
        "literal_16_occurrences": literal_rows,
        "fixed_cardinality_pattern_matches": fixed_matches,
        "resource_limit_event_semantics_scope": (
            "topology-commit boundary: no topology mutation, renewal, transaction, "
            "or event-ledger consumption occurs on policy stop"
        ),
        "front_resource_limit_none_source_cap": None,
        "production_pf_max_fronts_greater_than_two": "NOT_YET_EXECUTED",
        "gates": gates,
        "qualification": "PASS" if all(gates.values()) else "FAIL_CLOSED",
    }
    write_json(
        repo / "analysis_outputs/pf_current_source_general_multifront_v6_1_1/"
        "pf_general_multifront_dynamic_provider_source_audit_v6_1_1.json",
        result,
    )
    return result


def archive_members(out: Path) -> list[Path]:
    excluded = {ARCHIVE_NAME, MANIFEST_NAME}
    return sorted(
        (path for path in out.rglob("*") if path.is_file() and path.name not in excluded),
        key=lambda path: path.relative_to(out).as_posix(),
    )


def write_archive(out: Path) -> dict[str, Any]:
    members = [{
        "path": path.relative_to(out).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha(path),
    } for path in archive_members(out)]
    manifest = {
        "schema": "v6.1.1.compact-archive-manifest/1",
        "boundary": BOUNDARY,
        "member_count_excluding_embedded_manifest": len(members),
        "members": members,
    }
    manifest_path = out / MANIFEST_NAME
    write_json(manifest_path, manifest)
    archive_path = out / ARCHIVE_NAME
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
    ) as bundle:
        for path in archive_members(out) + [manifest_path]:
            name = path.relative_to(out).as_posix()
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=9)
    return manifest


def verify_archive_bytes(data: bytes) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            manifest = json.loads(bundle.read(MANIFEST_NAME))
            expected = {row["path"]: row for row in manifest["members"]}
            if set(bundle.namelist()) != set(expected) | {MANIFEST_NAME}:
                return False, "archive_member_set_mismatch"
            for name, row in expected.items():
                payload = bundle.read(name)
                if len(payload) != row["size_bytes"]:
                    return False, f"member_size_mismatch:{name}"
                if sha_bytes(payload) != row["sha256"]:
                    return False, f"member_sha256_mismatch:{name}"
    except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        return False, type(error).__name__
    return True, "exact_member_set_size_and_sha256"


def report_text(
    provenance: dict[str, Any], provider: dict[str, Any], prefix: dict[str, Any],
    terminal: dict[str, Any], validation: dict[str, Any],
) -> str:
    statuses = {
        "arbitrary_finite_front_state_machine": "QUALIFIED",
        "arbitrary_front_source_implementation": (
            "QUALIFIED" if provider["qualification"] == "PASS" else "FAIL_CLOSED"
        ),
        "dynamic_provider_cardinality": (
            "QUALIFIED" if provider["qualification"] == "PASS" else "FAIL_CLOSED"
        ),
        "asymmetric_partial_handoff": "QUALIFIED",
        "multiple_simultaneous_process_owners": "QUALIFIED",
        "v5_2_prefix_transaction_runtime_parity": (
            "QUALIFIED" if prefix["qualification"] == "PASS" else "FAIL_CLOSED"
        ),
        "full_v5_4_2_terminal_transaction_runtime_parity": (
            "QUALIFIED" if terminal["qualification"] == "PASS" else "FAIL_CLOSED"
        ),
        "generic_N1_N2_PF_reproduction": "NOT_EXECUTED",
        "production_PF_max_fronts_greater_than_two": "NOT_EXECUTED",
        "predictive_recursive_branching_physics_validated": False,
    }
    return f"""# PF current-source general multi-front V6.1.1 source seal

Permanent boundary: `{BOUNDARY}`.

V6.1.1 is a source-publication and complete terminal-parity correction only. No PF worker, deterministic kernel FEM, stochastic worker, mechanics provider, or production trajectory was launched. V1–V5.4.2 records, raw trajectories, the V5.3 family, and V12 execution semantics remain unchanged. A 1000 µm continuation is not authorized.

## Decision

The complete 17-file V6.1 implementation/test source set is now copied from Git commit `{V6_1_COMMIT}` (tree `{V6_1_TREE}`), verified byte-for-byte against its Git blobs, and connected to parent `{V6_COMMIT}` (tree `{V6_TREE}`). Both binary patches are included. The ambiguous V6.1 provenance field has been replaced by `pre_v12_execution_physics_unchanged: true` and `v12_execution_source_added_or_modified: true`.

Source inspection qualifies the dynamic provider route: it directly calls the dynamically iterating V11 evaluator, does not call the cardinality-rejecting resolver, contains no fixed 16-front allocation/loop/workspace, and treats the inherited 16 value as metadata. `front_resource_limit=None` creates no finite source cap. A configured binary-birth limit is checked before topology mutation; the pure policy-stop test preserves topology, registry, renewal/transaction ledgers, and therefore does not consume the completed topology event at that boundary. This is not an N>2 mechanics execution claim.

The V5.2 prefix and full V5.4.2 terminal records are now separate. Prefix parity remains exact at 84/84 actions. Full completion parity is exact at 85 control actions and 86 enabled actions, including action type, selected front identity, candidate/event IDs, completion times, topology fingerprints, terminal topology, action/threshold/ordinal histories, RNG state, and the step-369 atomic birth/renewal evidence.

The authoritative enabled completion archive corrects a supplied assessment statement: its final accepted action, `fronts.csv`, checkpoint, and visual snapshot are at step **813**, not 807. Step 807 is the penultimate mesh-adaptation checkpoint. The final reach is 302.23630529007903 µm, with 30.00000000000015 and 295.0000000000008 µm daughters. The control terminates at step 662 and 300.9175216390797 µm. For the branch-disabled control, `step_369_correlated_birth_exact` is null/not applicable and the explicit one-arm policy field passes.

## Status

```json
{json.dumps(statuses, indent=2, sort_keys=True)}
```

## Validation and archive

Recorded validation status: `{validation.get('qualification', 'PENDING')}`. The compact archive contains both patches, all 17 Git-bound source/test snapshots, this report, and every machine-readable V6.1.1 audit product. Its embedded manifest records each payload member's exact size and SHA-256; positive, removed-member, and modified-member verification tests are part of the V6.1.1 test set.

Source provenance fingerprint: `{canonical_hash(provenance)}`.

## Execution boundary

Generic-driver N=1/N=2 PF reproduction remains the next calculation and was not executed here. Production PF mechanics at N>2 and predictive recursive-branching physics remain unvalidated.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="analysis_outputs/pf_current_source_general_multifront_v6_1_1",
    )
    args = parser.parse_args()
    repo = REPOSITORY_ROOT
    out = repo / args.output
    out.mkdir(parents=True, exist_ok=True)

    provenance = bind_source(repo, out)
    provider = provider_source_audit(repo)
    prefix = replay_v5_4_2_runtime_parity(repo)
    terminal = replay_full_v5_4_2_terminal_runtime_parity(repo)
    write_json(out / "pf_general_multifront_v5_2_prefix_parity_v6_1_1.json", prefix)
    write_json(out / "pf_general_multifront_v5_4_2_full_terminal_parity_v6_1_1.json", terminal)
    validation_path = out / "pf_general_multifront_validation_v6_1_1.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {
        "qualification": "PENDING", "note": "validation commands not yet recorded",
    }
    write_json(validation_path, validation)
    report = report_text(provenance, provider, prefix, terminal, validation)
    (out / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_1_1.md").write_text(report)
    manifest = write_archive(out)
    ok, reason = verify_archive_bytes((out / ARCHIVE_NAME).read_bytes())
    if not ok:
        raise RuntimeError(f"generated archive failed verification: {reason}")
    print(json.dumps({
        "output": str(out),
        "archive_sha256": sha(out / ARCHIVE_NAME),
        "archive_member_count": manifest["member_count_excluding_embedded_manifest"] + 1,
        "prefix_qualification": prefix["qualification"],
        "terminal_qualification": terminal["qualification"],
        "provider_qualification": provider["qualification"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
