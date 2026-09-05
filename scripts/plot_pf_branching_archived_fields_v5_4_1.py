#!/usr/bin/env python3
"""Publish corrected archive-only V5.4.1 branching figures and event identities."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


A0_M = 5.0e-4
SHORT_PLANE = "cleavage:(010)"
LONG_PLANE = "cleavage:(100)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(path: Path):
    manifest_path = path / "checkpoint/latest.json"
    manifest = json.loads(manifest_path.read_text())
    state_path = manifest_path.parent / manifest["state_file"]
    return pickle.loads(state_path.read_bytes()), manifest_path, state_path


def crack_paths(checkpoint):
    return [np.asarray(branch.path, dtype=float) for branch in checkpoint.state.crack_network.branches]


def plot_network(ax, checkpoint):
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(crack_paths(checkpoint)), 2)))
    for index, (branch, path) in enumerate(zip(checkpoint.state.crack_network.branches, crack_paths(checkpoint))):
        ax.plot(path[:, 0] * 1e6, path[:, 1] * 1e6, color=colors[index], lw=2.2,
                label=f"{branch.branch_id} ({branch.status})")
        ax.plot(path[-1, 0] * 1e6, path[-1, 1] * 1e6, "o", color=colors[index], ms=4)


def field_background(ax, checkpoint, field, title, cmap="magma", levels=24):
    state = checkpoint.state
    tri = mtri.Triangulation(
        state.mesh.nodes[:, 0] * 1e6,
        state.mesh.nodes[:, 1] * 1e6,
        state.mesh.elems,
    )
    contour = ax.tricontourf(tri, field, levels=levels, cmap=cmap)
    ax.set(title=title, xlabel="laboratory x (µm)", ylabel="laboratory y (µm)")
    ax.set_aspect("equal", adjustable="box")
    return contour


def read_fronts(case: Path):
    by_step = {}
    with (case / "fronts.csv").open() as stream:
        for row in csv.DictReader(stream):
            record = {
                "front_id": row["front_id"],
                "tip_x_m": float(row["tip_x_m"]),
                "tip_y_m": float(row["tip_y_m"]),
                "arclength_m": float(row["arclength_m"]),
            }
            by_step.setdefault(int(row["step"]), {})[row["front_id"]] = record
    return by_step


def branch_identity(case: Path):
    path = case / "branch_events.csv"
    if not path.is_file():
        return {}, {}, None
    candidate_to_front = {}
    birth_step = None
    junction_x = None
    with path.open() as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            birth_step = step if birth_step is None else min(birth_step, step)
            junction_x = float(json.loads(row["branch_junction"])[0])
            for front_id, event_id in zip(
                json.loads(row["arm_front_ids"]), json.loads(row["event_ids_consumed"])
            ):
                candidate = event_id.split("#event:", 1)[0]
                candidate_to_front[candidate] = front_id
    return candidate_to_front, {front: junction_x for front in candidate_to_front.values()}, birth_step


def event_curve(case: Path, case_label: str):
    fronts = read_fronts(case)
    candidate_birth_owner, daughter_origin, birth_step = branch_identity(case)
    selected_by_step = {}
    with (case / "directional_rates.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("selected_event_candidate_id") is not None:
                selected_by_step[int(row["step"])] = row["selected_event_tip_id"]

    rows = []
    with (case / "branch_action_trials.jsonl").open() as stream:
        for line in stream:
            action = json.loads(line)
            if not action.get("accepted"):
                continue
            step = int(action["step"])
            selected_tip = selected_by_step.get(step)
            if selected_tip is None:
                raise RuntimeError(f"accepted event at step {step} lacks selected-event owner")
            step_fronts = fronts[step]
            if selected_tip not in step_fronts:
                raise RuntimeError(f"selected-event owner {selected_tip} absent at step {step}")
            selected_front = step_fronts[selected_tip]
            max_forward = max(item["tip_x_m"] - A0_M for item in step_fronts.values())
            for index, candidate in enumerate(action["candidate_ids"]):
                candidate_owner = (
                    candidate_birth_owner[candidate]
                    if birth_step is not None and step == birth_step
                    else selected_tip
                )
                if candidate_owner not in step_fronts:
                    raise RuntimeError(f"candidate owner {candidate_owner} absent at step {step}")
                owner_front = step_fronts[candidate_owner]
                origin = daughter_origin.get(selected_tip, A0_M)
                selected_projected = selected_front["tip_x_m"] - origin
                daughter_length = (
                    owner_front["arclength_m"] if candidate_owner in daughter_origin else 0.0
                )
                valid = bool(action["local_J_valid"][index])
                local_j = float(action["J_local_signed_J_per_m2"][index])
                marginal = action["G_marginal_J_per_m2"][index]
                marginal = None if marginal is None else float(marginal)
                kinetic_j = float(action["J_kin_used_J_per_m2"][index])
                if valid and kinetic_j != local_j:
                    raise RuntimeError(f"valid local drive mismatch at step {step}")
                if not valid and (marginal is None or kinetic_j != marginal):
                    raise RuntimeError(f"invalid-contour fallback mismatch at step {step}")
                rows.append({
                    "step": step,
                    "selected_event_tip_id": selected_tip,
                    "candidate_owner_front_id": candidate_owner,
                    "candidate_id": candidate,
                    "selected_event_front_projected_extension_um": selected_projected * 1e6,
                    "daughter_length_from_birth_um": daughter_length * 1e6,
                    "maximum_network_forward_reach_um": max_forward * 1e6,
                    "local_signed_J_J_per_m2": local_j,
                    "local_J_valid": valid,
                    "local_J_invalid_reason": action["local_J_invalid_reason"][index],
                    "marginal_G_J_per_m2": marginal,
                    "kinetic_J_used_J_per_m2": kinetic_j,
                    "kinetic_K_used_MPa_sqrt_m": float(action["directional_K_Pa_sqrt_m"][index]) / 1e6,
                    "drive_source": "LOCAL_SIGNED_J" if valid else "MARGINAL_ENERGY_FALLBACK",
                    "case": case_label,
                })
    return rows, candidate_birth_owner, birth_step


def save_triplet(fig, stem: Path) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=220)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {
        "control max-fronts=1": args.raw_root / "theta40_corrected_control_max1_seed3621",
        "branching max-fronts=2": args.raw_root / "theta40_corrected_enabled_max2_seed3621",
    }
    loaded = {label: load_case(path) for label, path in cases.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)
    for ax, (label, (checkpoint, _, _)) in zip(axes, loaded.items()):
        contour = field_background(ax, checkpoint, checkpoint.state.damage,
                                   f"{label}\nfinal archived nodal damage", cmap="viridis")
        plot_network(ax, checkpoint)
        ax.set_xlim(450, 850); ax.set_ylim(-230, 230); ax.legend(fontsize=7, loc="upper left")
        fig.colorbar(contour, ax=ax, shrink=0.82, label="damage d")
    fig.suptitle("Corrected V5.2 crack topology at the 420 µm archived path stop\n"
                 "capability demonstration; later completion is not inferred")
    save_triplet(fig, args.out / "pf_branching_final_crack_structure_v5_4_1")
    plt.close(fig)

    checkpoint = loaded["branching max-fronts=2"][0]
    state = checkpoint.state
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    c0 = field_background(axes[0, 0], checkpoint, state.damage, "nodal damage d", cmap="viridis")
    plot_network(axes[0, 0], checkpoint); axes[0, 0].set_xlim(450, 850); axes[0, 0].set_ylim(-230, 230)
    fig.colorbar(c0, ax=axes[0, 0], label="d")
    displacement = np.asarray(state.displacement)
    if displacement.ndim == 1:
        displacement = displacement.reshape((-1, 2))
    c1 = field_background(axes[0, 1], checkpoint, np.linalg.norm(displacement, axis=1) * 1e6,
                          "displacement magnitude", cmap="magma")
    plot_network(axes[0, 1], checkpoint); axes[0, 1].set_xlim(450, 850); axes[0, 1].set_ylim(-230, 230)
    fig.colorbar(c1, ax=axes[0, 1], label="|u| (µm)")
    mpz = checkpoint.shared_process_state["mpz_fields"]
    x = np.asarray(mpz["x"]) * 1e6
    for system in range(int(mpz["n_systems"])):
        axes[1, 0].plot(x, np.asarray(mpz["mobile"])[system], lw=1.8, label=f"mobile s{system}")
        axes[1, 0].plot(x, np.asarray(mpz["retained"])[system], "--", lw=1.8, label=f"retained s{system}")
        axes[1, 1].plot(x, np.asarray(mpz["accumulated_slip"])[system], lw=1.8, label=f"system {system}")
    axes[1, 0].set(title="tip-relative MPZ line populations", xlabel="MPZ coordinate (µm)", ylabel="signed line content")
    axes[1, 1].set(title="tip-relative accumulated slip", xlabel="MPZ coordinate (µm)", ylabel="accumulated slip")
    axes[1, 0].legend(); axes[1, 1].legend()
    fig.suptitle("Branching-enabled final archived fields (no reconstructed time sequence)")
    save_triplet(fig, args.out / "pf_branching_final_damage_and_process_fields_v5_4_1")
    plt.close(fig)

    all_rows = []
    identities = {}
    for label, case in cases.items():
        rows, mapping, birth_step = event_curve(case, label)
        all_rows.extend(rows)
        identities[label] = {"candidate_to_front_after_birth": mapping, "birth_step": birth_step}
    table = args.out / "pf_branching_model_native_local_K_vs_front_progress_v5_4_1.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)

    fig, ax = plt.subplots(figsize=(9.2, 5.9), constrained_layout=True)
    colors = {"control max-fronts=1": "#4c78a8", "branching max-fronts=2": "#f58518"}
    for label in cases:
        subset = [row for row in all_rows if row["case"] == label]
        valid = [row for row in subset if row["local_J_valid"]]
        fallback = [row for row in subset if not row["local_J_valid"]]
        ax.scatter([row["maximum_network_forward_reach_um"] for row in valid],
                   [row["kinetic_K_used_MPa_sqrt_m"] for row in valid],
                   s=18, marker="o", color=colors[label], alpha=0.78,
                   label=f"{label}: valid local contour")
        ax.scatter([row["maximum_network_forward_reach_um"] for row in fallback],
                   [row["kinetic_K_used_MPa_sqrt_m"] for row in fallback],
                   s=28, marker="x", color=colors[label], alpha=0.85,
                   label=f"{label}: marginal-energy fallback")
    ax.set(xlabel="maximum network forward reach (µm)",
           ylabel=r"kinetic $K$ used (MPa√m)",
           title="Model-native event drive versus archived front progress")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
    ax.text(0.01, 0.01,
            "Markers are accepted-event kinetic drive. Crosses use marginal-energy fallback;\n"
            "they are not connected or represented as qualified local-contour values.",
            transform=ax.transAxes, fontsize=8, va="bottom")
    save_triplet(fig, args.out / "pf_branching_model_native_local_K_vs_front_progress_v5_4_1")
    plt.close(fig)

    enabled = [row for row in all_rows if row["case"] == "branching max-fronts=2"]
    postbirth_010 = [row for row in enabled if row["step"] > 369 and SHORT_PLANE in row["candidate_id"]]
    postbirth_100 = [row for row in enabled if row["step"] > 369 and LONG_PLANE in row["candidate_id"]]
    short_id = identities["branching max-fronts=2"]["candidate_to_front_after_birth"][
        next(row["candidate_id"] for row in enabled if SHORT_PLANE in row["candidate_id"])
    ]
    long_id = identities["branching max-fronts=2"]["candidate_to_front_after_birth"][
        next(row["candidate_id"] for row in enabled if LONG_PLANE in row["candidate_id"])
    ]
    ownership_checks = {
        "plane_010_maps_to_short_minus50_front_b0fa": short_id.startswith("b0fa"),
        "plane_100_maps_to_long_plus40_front_b7d": long_id.startswith("b7d"),
        "five_postbirth_010_events_owned_by_b0fa": (
            len(postbirth_010) == 5 and all(row["selected_event_tip_id"] == short_id for row in postbirth_010)
        ),
        "fifty_six_postbirth_100_events_owned_by_b7d": (
            len(postbirth_100) == 56 and all(row["selected_event_tip_id"] == long_id for row in postbirth_100)
        ),
    }
    if not all(ownership_checks.values()):
        raise RuntimeError(f"event ownership regression failed: {ownership_checks}")

    sources = []
    for label, (_, manifest, state_path) in loaded.items():
        for path in (manifest, state_path, cases[label] / "branch_action_trials.jsonl",
                     cases[label] / "directional_rates.jsonl", cases[label] / "fronts.csv"):
            sources.append({"case": label, "path": str(path.resolve()), "sha256": sha256(path)})
    outputs = [
        {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(args.out.glob("pf_branching_*v5_4_1.*"))
        if path.suffix in {".csv", ".png", ".pdf", ".svg"}
    ]
    audit = {
        "schema": "pf_branching_figure_identity_audit_v5_4_1/1",
        "qualification": "PASS",
        "boundary": "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS",
        "sources": sources,
        "outputs": outputs,
        "event_owner_source": "directional_rates.selected_event_tip_id",
        "candidate_owner_source": "branch_events.event_ids_consumed paired with arm_front_ids at birth; selected_event_tip_id thereafter",
        "identities": identities,
        "postbirth_counts": {"cleavage_010": len(postbirth_010), "cleavage_100": len(postbirth_100)},
        "ownership_checks": ownership_checks,
        "invalid_contour_points_connected_as_local_contour": False,
        "completion_beyond_archive_inferred": False,
        "temporal_sequence_reconstructed": False,
        "workers_started": 0,
        "mechanics_solve_performed": False,
    }
    (args.out / "pf_branching_figure_identity_audit_v5_4_1.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
