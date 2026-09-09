#!/usr/bin/env python3
"""Publish the compact terminal audit for the V12 cap-six trajectory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.general_multifront_v12 import canonical_hash
from arrhenius_fracture.multifront_checkpoint_v12 import (
    load_accepted_boundary_checkpoint_v12,
)
from arrhenius_fracture.network_metrics_v11 import crack_growth_metrics


LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
A0_M = 500.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tree_manifest(root: Path) -> tuple[list[dict], str]:
    rows = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        file_sha = sha256(path)
        row = {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha}
        rows.append(row)
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        )
    return rows, digest.hexdigest()


def plot_network(ax, network: dict, title: str) -> None:
    active = set(network.get("active_tip_ids", []))
    for branch in network.get("branches", []):
        points = branch.get("path_m", [])
        if not points:
            continue
        x = [(point[0] - A0_M) * 1.0e6 for point in points]
        y = [point[1] * 1.0e6 for point in points]
        is_active = branch["branch_id"] in active
        ax.plot(x, y, lw=1.8 if is_active else 1.0, alpha=1.0 if is_active else 0.7)
        if is_active:
            ax.scatter(x[-1:], y[-1:], s=20, zorder=3)
    ax.axvline(0.0, color="0.65", lw=0.7, ls="--")
    ax.set_title(title)
    ax.set_xlabel("forward coordinate from initial tip (µm)")
    ax.set_ylabel("transverse coordinate (µm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)


def terminal_classification(result: dict) -> str:
    births = int(result["cumulative_binary_branch_births"])
    if result.get("target_1000um_reached") is True:
        return f"V12_1000UM_COMPLETED_WITH_{births}_CUMULATIVE_BRANCH_BIRTHS"
    if result.get("front_resource_limit_bound") is True:
        return "V12_1000UM_RUN_STOPPED_POLICY_BOUND_AT_SIX_ACTIVE_FRONTS"
    if result.get("termination") == "owner_local_kernel_coordinate_outside_qualified_family_domain":
        return "V12_1000UM_RUN_STOPPED_FAIL_CLOSED_OWNER_KERNEL_ENVELOPE"
    return str(result.get("termination", "UNKNOWN_TERMINATION"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--migration-audit", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_run.expanduser().resolve()
    family = args.family.expanduser().resolve()
    migration_path = args.migration_audit.expanduser().resolve()
    out = args.outroot.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite final compact record: {out}")
    result = json.loads((raw / "v12_run_complete.json").read_text())
    if result.get("predictive_recursive_branching_physics") != "NOT_VALIDATED":
        raise RuntimeError("permanent capability boundary is absent")
    checkpoint_path = raw / "checkpoint" / "latest.v12.pkl"
    checkpoint = load_accepted_boundary_checkpoint_v12(checkpoint_path)
    runtime = checkpoint.runtime
    runtime.validate()
    out.mkdir(parents=True)

    progress = []
    progress_path = raw / "v12_progress.jsonl"
    if progress_path.is_file():
        progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    transaction_rows = []
    for item in runtime.transaction_records:
        row = {
            "transaction_id": item.transaction_id,
            "action_type": item.action_type,
            "proposal_id": item.proposal_id,
            "selected_front_id": item.selected_front_id,
            "selected_owner_id": item.selected_owner_id,
            "created_front_ids": "|".join(item.created_front_ids),
            "retired_front_ids": "|".join(item.retired_front_ids),
            "created_junction_id": item.created_junction_id or "",
            "pre_active_front_count": item.pre_active_front_count,
            "post_active_front_count": item.post_active_front_count,
            "realized_lengths_um": "|".join(f"{x * 1e6:.17g}" for x in item.realized_lengths_m),
            "renewal_owner_id": item.renewal_owner_id or "",
            "renewal_distance_um": item.renewal_distance_m * 1e6,
            "stored_energy_release_J_per_m": item.stored_energy_release_J_per_m,
            "stored_energy_cost_J_per_m": item.stored_energy_cost_J_per_m,
            "post_topology_fingerprint": item.post_topology_fingerprint,
            "transaction_fingerprint": item.transaction_fingerprint,
        }
        transaction_rows.append(row)
    write_csv(out / "topology_transaction_history.csv", transaction_rows)

    network = runtime.crack_network.to_dict()
    branch_rows = []
    active = set(runtime.active_front_ids)
    for branch in network["branches"]:
        path = branch.get("path_m", [])
        branch_rows.append({
            "branch_id": branch["branch_id"],
            "parent_branch_id": branch.get("parent_branch_id") or "",
            "active_front": branch["branch_id"] in active,
            "initiation_event": branch.get("initiation_event") or "",
            "point_count": len(path),
            "tip_x_m": path[-1][0], "tip_y_m": path[-1][1],
            "physical_path_length_m": branch.get("physical_path_length_m", 0.0),
            "candidate_id": branch.get("local_state", {}).get("candidate_id", ""),
        })
    write_csv(out / "branch_junction_tree.csv", branch_rows)
    write_json(out / "final_crack_network.json", network)

    compact_progress = []
    for row in progress:
        compact_progress.append({
            "step": row["step"], "disposition": row["disposition"],
            "accepted_duration_s": row.get("accepted_duration_s"),
            "physical_time_s": row.get("physical_time_s"),
            "accepted_opening_m": row.get("accepted_opening_m"),
            "maximum_network_forward_reach_um": row["maximum_network_forward_reach_um"],
            "max_root_to_tip_path_extension_um": row.get("physical_extension_um"),
            "active_front_count": row["active_front_count"],
            "cumulative_binary_branch_births": row["branch_birth_count"],
            "owner_local_coordinates_um": json.dumps(
                row.get("owner_local_coordinates_um", {}), sort_keys=True
            ),
            "topology_fingerprint": row.get("topology_fingerprint"),
            "registry_fingerprint": row.get("registry_fingerprint"),
        })
    write_csv(out / "front_birth_owner_history.csv", compact_progress)

    milestone_records = []
    milestone_networks = []
    for value in (250, 500, 750, 1000):
        source = raw / "milestone_morphology" / f"reach_{value:04d}.json"
        if not source.is_file():
            milestone_records.append({"milestone_um": value, "available": False})
            continue
        payload = json.loads(source.read_text())
        destination = out / source.name
        shutil.copyfile(source, destination)
        milestone_records.append({
            "milestone_um": value, "available": True,
            "step": payload["step"],
            "actual_maximum_network_forward_reach_um": payload[
                "actual_maximum_network_forward_reach_um"
            ],
            "topology_fingerprint": payload["topology_fingerprint"],
        })
        milestone_networks.append((value, payload["crack_network"]))
    write_csv(out / "milestone_inventory.csv", milestone_records)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    plot_network(ax, network, "Final V12 crack network")
    fig.tight_layout()
    fig.savefig(out / "final_crack_morphology.png", dpi=220)
    fig.savefig(out / "final_crack_morphology.svg")
    plt.close(fig)
    if milestone_networks:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, (value, item) in zip(axes.flat, milestone_networks):
            plot_network(ax, item, f"{value} µm milestone")
        for ax in axes.flat[len(milestone_networks):]:
            ax.axis("off")
        fig.savefig(out / "milestone_crack_morphology.png", dpi=200)
        fig.savefig(out / "milestone_crack_morphology.svg")
        plt.close(fig)
    if progress:
        steps = [row["step"] for row in progress]
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
        axes[0].plot(steps, [row["maximum_network_forward_reach_um"] for row in progress])
        axes[0].axhline(1000.0, color="k", ls="--", lw=0.8)
        axes[0].set_ylabel("maximum forward reach (µm)")
        axes[1].step(steps, [row["active_front_count"] for row in progress], where="post", label="active fronts")
        axes[1].step(steps, [row["branch_birth_count"] for row in progress], where="post", label="cumulative births")
        axes[1].set(xlabel="accepted step", ylabel="count")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "forward_reach_front_birth_history.png", dpi=220)
        fig.savefig(out / "forward_reach_front_birth_history.svg")
        plt.close(fig)

    growth = crack_growth_metrics(runtime.crack_network, initial_crack_length_m=A0_M)
    classification = terminal_classification(result)
    closure = {
        "schema": "v12.1000um-terminal-closure/1",
        "claim_label": LABEL,
        "terminal_classification": classification,
        "target_1000um_reached": bool(result["target_1000um_reached"]),
        "cumulative_binary_branch_births": runtime.cumulative_branch_births,
        "maximum_concurrent_active_fronts": int(result["maximum_concurrent_active_fronts"]),
        "at_least_four_cumulative_births_observed": bool(
            runtime.cumulative_branch_births >= 4
        ),
        "front_resource_limit_bound": bool(result["front_resource_limit_bound"]),
        "predictive_recursive_branching_physics_validated": False,
        "final_growth_metrics_um": growth.to_dict_um(),
        "active_front_count": len(runtime.active_front_ids),
        "active_front_accounting_expected": (
            1 + runtime.cumulative_branch_births - runtime.cumulative_coalescences
            - runtime.cumulative_retirements
        ),
        "active_front_accounting_exact": (
            len(runtime.active_front_ids)
            == 1 + runtime.cumulative_branch_births - runtime.cumulative_coalescences
            - runtime.cumulative_retirements
        ),
        "topology_fingerprint": runtime.topology_fingerprint,
        "runtime_registry_fingerprint": runtime.registry_fingerprint,
        "scheduler_sha256": canonical_hash(runtime.scheduler.to_dict()),
        "front_competition_sha256": canonical_hash({
            key: value.competition_state for key, value in runtime.front_runtimes.items()
        }),
        "front_rng_sha256": canonical_hash({
            key: value.lineage_rng_state for key, value in runtime.front_runtimes.items()
        }),
        "active_and_reservoir_ledgers": runtime.total_conserved_ledgers(),
        "active_and_reservoir_signed_ledgers": runtime.total_signed_system_ledgers(),
        "transaction_count": len(runtime.transaction_records),
        "physical_time_s": checkpoint.physical_time_s,
        "accepted_opening_m": checkpoint.accepted_opening_m,
        "step_count": checkpoint.step_count,
    }
    write_json(out / "terminal_closure.json", closure)

    raw_manifest, raw_tree = tree_manifest(raw)
    write_csv(out / "durable_raw_tree_manifest.csv", raw_manifest)
    provenance = {
        "raw_output_path": str(raw), "raw_output_tree_fingerprint": raw_tree,
        "raw_file_count": len(raw_manifest),
        "family": str(family), "family_sha256": sha256(family),
        "migration_audit": str(migration_path),
        "migration_audit_sha256": sha256(migration_path),
        "final_checkpoint": str(checkpoint_path),
        "final_checkpoint_sha256": sha256(checkpoint_path),
        "run_complete_sha256": sha256(raw / "v12_run_complete.json"),
        "launch_manifest_sha256": sha256(raw / "v12_1000um_launch_manifest.json"),
    }
    write_json(out / "provenance.json", provenance)
    migration = json.loads(migration_path.read_text())
    write_json(out / "migrated_checkpoint_audit.json", migration)

    report = f"""# V12 current-source multifront 1000 µm cap-six record

Permanent boundary: `{LABEL}`.

## Terminal decision

`{classification}`

- target_1000um_reached: `{str(closure['target_1000um_reached']).lower()}`
- cumulative_binary_branch_births: `{closure['cumulative_binary_branch_births']}`
- maximum_concurrent_active_fronts: `{closure['maximum_concurrent_active_fronts']}`
- at_least_four_cumulative_births_observed: `{str(closure['at_least_four_cumulative_births_observed']).lower()}`
- front_resource_limit_bound: `{str(closure['front_resource_limit_bound']).lower()}`
- predictive_recursive_branching_physics_validated: `false`

The branch-birth count is observational and is not a success gate. The six-front
limit is an operational resource policy, not a material or branching-physics law.

## Provenance and closure

The trajectory continued the sealed three-front/two-birth V12 boundary after a
one-shot append-only family migration. The migration audit preserves the accepted
FEM state, stress, time, opening, step, topology, scheduler, clocks, RNG/thresholds,
process state, ownership, wake state, ledgers, and transaction history. Only the
family object and explicit family/cache identity changed.

Final maximum forward reach is
`{growth.max_forward_projected_extension_m * 1e6:.12g} µm`; final maximum
root-to-tip path extension is `{growth.max_root_to_tip_path_extension_m * 1e6:.12g} µm`.
Active-front accounting closure is `{closure['active_front_accounting_exact']}`.

Durable raw output: `{raw}`

Raw tree fingerprint: `{raw_tree}`

This is a branching capability demonstration. It does not validate predictive
recursive-branching physics.
"""
    (out / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V12_1000UM_CAP6.md").write_text(report)

    archive = out.with_name("Archive_" + out.name.upper() + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(item for item in out.rglob("*") if item.is_file()):
            bundle.write(path, path.relative_to(out.parent))
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n"
    )
    print(json.dumps({**closure, **provenance, "archive": str(archive)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
