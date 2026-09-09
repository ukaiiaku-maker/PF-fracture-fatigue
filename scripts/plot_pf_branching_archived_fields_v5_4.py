#!/usr/bin/env python3
"""Plot final archived V5.2 fields and event-resolved model-native toughness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(path: Path):
    manifest_path = path / "checkpoint/latest.json"
    manifest = json.loads(manifest_path.read_text())
    state_path = manifest_path.parent / manifest["state_file"]
    checkpoint = pickle.loads(state_path.read_bytes())
    return checkpoint, manifest_path, state_path


def crack_paths(checkpoint):
    return [np.asarray(branch.path, dtype=float) for branch in checkpoint.state.crack_network.branches]


def plot_network(ax, checkpoint, *, label_prefix=""):
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(crack_paths(checkpoint)), 2)))
    for index, (branch, path) in enumerate(zip(checkpoint.state.crack_network.branches, crack_paths(checkpoint))):
        ax.plot(path[:, 0] * 1e6, path[:, 1] * 1e6, color=colors[index], lw=2.2,
                label=f"{label_prefix}{branch.branch_id} ({branch.status})")
        ax.plot(path[-1, 0] * 1e6, path[-1, 1] * 1e6, "o", color=colors[index], ms=4)


def damage_background(ax, checkpoint, field, title, cmap="magma", levels=24):
    state = checkpoint.state
    tri = mtri.Triangulation(state.mesh.nodes[:, 0] * 1e6, state.mesh.nodes[:, 1] * 1e6, state.mesh.elems)
    contour = ax.tricontourf(tri, field, levels=levels, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("laboratory x (µm)"); ax.set_ylabel("laboratory y (µm)")
    ax.set_aspect("equal", adjustable="box")
    return contour


def event_curve(case: Path):
    fronts: dict[int, float] = {}
    with (case / "fronts.csv").open() as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            projected = max(float(row["tip_x_m"]) - 5e-4, 0.0)
            fronts[step] = max(fronts.get(step, 0.0), projected)
    rows = []
    for line in (case / "branch_action_trials.jsonl").read_text().splitlines():
        item = json.loads(line)
        if not item.get("accepted"):
            continue
        step = int(item["step"])
        for tip, candidate, value in zip(
            item["participating_front_ids"], item["candidate_ids"],
            item["directional_K_Pa_sqrt_m"],
        ):
            rows.append({
                "step": step, "projected_extension_um": fronts.get(step, np.nan) * 1e6,
                "tip_id": tip, "candidate_id": candidate,
                "K_local_directional_MPa_sqrt_m": float(value) / 1e6,
                "local_J_valid": bool(item["local_J_valid"][0]) if item.get("local_J_valid") else False,
                "provider_identity": item["provider_identity"],
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    cases = {
        "control max-fronts=1": args.raw_root / "theta40_corrected_control_max1_seed3621",
        "branching max-fronts=2": args.raw_root / "theta40_corrected_enabled_max2_seed3621",
    }
    loaded = {label: load_case(path) for label, path in cases.items()}

    # Final crack structure over the archived nodal-damage field.
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)
    for ax, (label, (checkpoint, _, _)) in zip(axes, loaded.items()):
        contour = damage_background(ax, checkpoint, checkpoint.state.damage,
                                    f"{label}\nfinal archived nodal damage", cmap="viridis")
        plot_network(ax, checkpoint)
        ax.set_xlim(450, 850); ax.set_ylim(-230, 230)
        ax.legend(fontsize=7, loc="upper left")
        fig.colorbar(contour, ax=ax, shrink=0.82, label="damage d")
    fig.suptitle("Corrected V5.2 crack topology at the signed-kernel envelope stop\n"
                 "archive ends at 420 µm physical path extension; completion not inferred")
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"pf_branching_final_crack_structure_v5_4.{suffix}", dpi=220)
    plt.close(fig)

    # Various archived fields: nodal damage, displacement, and tip-relative MPZ states.
    checkpoint = loaded["branching max-fronts=2"][0]; state = checkpoint.state
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    c0 = damage_background(axes[0, 0], checkpoint, state.damage, "nodal damage d", cmap="viridis")
    plot_network(axes[0, 0], checkpoint); axes[0, 0].set_xlim(450, 850); axes[0, 0].set_ylim(-230, 230)
    fig.colorbar(c0, ax=axes[0, 0], label="d")
    displacement = np.asarray(state.displacement)
    if displacement.ndim == 1:
        displacement = displacement.reshape((-1, 2))
    displacement_um = np.linalg.norm(displacement, axis=1) * 1e6
    c1 = damage_background(axes[0, 1], checkpoint, displacement_um,
                           "displacement magnitude", cmap="magma")
    plot_network(axes[0, 1], checkpoint); axes[0, 1].set_xlim(450, 850); axes[0, 1].set_ylim(-230, 230)
    fig.colorbar(c1, ax=axes[0, 1], label="|u| (µm)")
    mpz = checkpoint.shared_process_state["mpz_fields"]
    x = np.asarray(mpz["x"]) * 1e6
    for system in range(int(mpz["n_systems"])):
        axes[1, 0].plot(x, np.asarray(mpz["mobile"])[system], lw=1.8, label=f"mobile s{system}")
        axes[1, 0].plot(x, np.asarray(mpz["retained"])[system], "--", lw=1.8, label=f"retained s{system}")
        axes[1, 1].plot(x, np.asarray(mpz["accumulated_slip"])[system], lw=1.8, label=f"system {system}")
    axes[1, 0].set(title="tip-relative MPZ line populations", xlabel="distance behind tip-grid origin (µm)", ylabel="signed line content")
    axes[1, 1].set(title="tip-relative accumulated slip state", xlabel="distance behind tip-grid origin (µm)", ylabel="accumulated slip")
    axes[1, 0].legend(); axes[1, 1].legend()
    fig.suptitle("Branching-enabled final archived fields (V5.2, no reconstructed time sequence)")
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"pf_branching_final_damage_and_process_fields_v5_4.{suffix}", dpi=220)
    plt.close(fig)

    # Event-resolved local directional K; this is not remote applied K.
    all_rows = []; archived_projected_endpoints = []
    fig, ax = plt.subplots(figsize=(8.7, 5.7), constrained_layout=True)
    for label, case in cases.items():
        rows = event_curve(case)
        for row in rows: row["case"] = label
        all_rows.extend(rows)
        archived_projected_endpoints.append(max(r["projected_extension_um"] for r in rows))
        ax.plot([r["projected_extension_um"] for r in rows],
                [r["K_local_directional_MPa_sqrt_m"] for r in rows],
                "o-", ms=3.2, lw=1.1, alpha=0.82, label=label)
    observed_max = max(archived_projected_endpoints)
    ax.axvspan(observed_max, observed_max + 18, color="0.85", alpha=0.7,
               label="no archived projected-growth states")
    for endpoint in archived_projected_endpoints:
        ax.axvline(endpoint, color="0.35", ls="--", lw=0.8)
    ax.set(xlabel="maximum projected crack extension (µm)",
           ylabel=r"accepted-event local directional $K$ (MPa√m)",
           title="Model-native event resistance versus archived projected crack growth")
    ax.set_xlim(-5, observed_max + 18)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.text(0.01, 0.01, "Accepted-event local directional K; not applied remote K.\n"
            "Archive stops at 420 µm physical path growth (≈295–297 µm projected).",
            transform=ax.transAxes, fontsize=8, va="bottom")
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"pf_branching_fracture_toughness_vs_crack_length_v5_4.{suffix}", dpi=220)
    plt.close(fig)
    table = args.out / "pf_branching_fracture_toughness_vs_crack_length_v5_4.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)

    sources = []
    for label, (_, manifest, state_path) in loaded.items():
        sources.extend([
            {"case": label, "path": str(manifest), "sha256": sha256(manifest)},
            {"case": label, "path": str(state_path), "sha256": sha256(state_path)},
            {"case": label, "path": str(cases[label] / "branch_action_trials.jsonl"),
             "sha256": sha256(cases[label] / "branch_action_trials.jsonl")},
            {"case": label, "path": str(cases[label] / "fronts.csv"),
             "sha256": sha256(cases[label] / "fronts.csv")},
        ])
    outputs = []
    for path in sorted(args.out.glob("pf_branching_*v5_4.*")):
        if path.name.endswith("manifest_v5_4.json"): continue
        outputs.append({"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    manifest = {
        "schema": "pf_branching_archived_field_figures_v5_4/1",
        "qualification": "ARCHIVE_DERIVED_NO_NEW_SOLVE",
        "sources": sources, "outputs": outputs,
        "field_scope": "final corrected V5.2 checkpoint only",
        "temporal_sequence_reconstructed": False,
        "completion_beyond_archive_inferred": False,
        "toughness_quantity": "accepted-event local directional K",
        "toughness_is_applied_remote_K": False,
        "pf_workers_started": 0, "mechanics_solve_performed": False,
    }
    (args.out / "pf_branching_archived_field_figure_manifest_v5_4.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
