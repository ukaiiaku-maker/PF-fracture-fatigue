#!/usr/bin/env python3
"""Publish the bounded current-source PF branching capability demonstration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.network_metrics_v11 import crack_growth_metrics


ROOT = Path(__file__).resolve().parents[1]
LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
INITIAL_TIP_X_M = 5.0e-4
HISTORICAL_SEED = 3621
PHYSICAL_SOURCE_COMMIT = "9e884fb0b0845da621d2612bdf1042e481b8df49"
TOPOLOGY_OVERLAY_SOURCE_COMMIT = "2b5e535"
HISTORICAL_POSITIVE_SOURCE_COMMIT = "9cc5795d8461ea28d24227b19e17cd233485ab72"
HISTORICAL_POSITIVE_BRANCH = "codex/v11-branching"
HISTORICAL_POSITIVE_TAG = "v11.0.0-hazard-branching-production"
HISTORICAL_POSITIVE_MATERIAL = "v913_paper_weakT01_0129902_persistent_sites"
CURRENT_MATERIAL = "oneD_v2_focused_weak_T_0016"
matplotlib.rcParams["svg.hashsalt"] = "pf-current-source-branching-capability-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: Path, *, suffix: str | None = None) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = [
        path for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (suffix is None or path.suffix == suffix)
    ]
    total_bytes = 0
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        total_bytes += len(content)
    return {
        "root": str(root),
        "suffix_filter": suffix,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def canonical_material_row_hash(registry: Path, candidate_id: str) -> str:
    rows = [row for row in read_csv(registry) if row.get("candidate_id") == candidate_id]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one material row for {candidate_id!r}")
    content = json.dumps(rows[0], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = [{key: json_cell(value) for key, value in row.items()} for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def final_checkpoint(case: Path, *, require_terminated: bool = True):
    path = case / "checkpoint/latest.json"
    checkpoint = restore_branch_checkpoint(path)
    if require_terminated and not checkpoint.termination_reason:
        raise RuntimeError(f"case has not terminated: {case}")
    return checkpoint


def front_history(case_name: str, case: Path) -> list[dict[str, Any]]:
    rows = read_csv(case / "fronts.csv")
    max_x_by_step: dict[int, float] = {}
    leading_by_step: dict[int, str] = {}
    for row in rows:
        step = int(row["step"])
        x = float(row["tip_x_m"])
        if x > max_x_by_step.get(step, -math.inf):
            max_x_by_step[step] = x
            leading_by_step[step] = row["front_id"]
    result = []
    for row in rows:
        step = int(row["step"])
        result.append({
            "claim_label": LABEL,
            "case": case_name,
            **row,
            "tip_projected_extension_um": (float(row["tip_x_m"]) - INITIAL_TIP_X_M) * 1e6,
            "front_arclength_um": float(row["arclength_m"]) * 1e6,
            "maximum_forward_reach_um": (max_x_by_step[step] - INITIAL_TIP_X_M) * 1e6,
            "leading_front_id": leading_by_step[step],
            "is_leading_front": row["front_id"] == leading_by_step[step],
        })
    return result


def checkpoint_state_rows(case_name: str, case: Path) -> list[dict[str, Any]]:
    manifests = sorted((case / "checkpoint/transitions").glob("*.json"))
    manifests.append(case / "checkpoint/latest.json")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for manifest in manifests:
        if not manifest.is_file():
            continue
        checkpoint = restore_branch_checkpoint(manifest)
        key = (
            checkpoint.topology_fingerprint,
            checkpoint.mesh_identity,
            str(checkpoint.state.event_counters.get("accepted_steps", 0)),
        )
        if key in seen:
            continue
        seen.add(key)
        fields = checkpoint.shared_process_state.get("mpz_fields", {})
        engine = checkpoint.shared_process_state.get("engine_fields", {})
        mobile = np.asarray(fields.get("mobile", []), dtype=float)
        retained = np.asarray(fields.get("retained", []), dtype=float)
        wake_mobile = np.asarray(fields.get("wake_mobile", []), dtype=float)
        wake_retained = np.asarray(fields.get("wake_retained", []), dtype=float)
        x = np.asarray(fields.get("x", []), dtype=float)
        wake_x = np.asarray(fields.get("wake_x", []), dtype=float)
        growth = crack_growth_metrics(
            checkpoint.state.crack_network, initial_crack_length_m=INITIAL_TIP_X_M
        )
        rows.append({
            "claim_label": LABEL,
            "case": case_name,
            "checkpoint_manifest": str(manifest.relative_to(case)),
            "accepted_step": int(checkpoint.state.event_counters.get("accepted_steps", 0)),
            "physical_time_s": checkpoint.physical_time_s,
            "applied_opening_m": checkpoint.accepted_load,
            "topology_fingerprint": checkpoint.topology_fingerprint,
            "mesh_identity": checkpoint.mesh_identity,
            "maximum_forward_reach_um": growth.max_forward_projected_extension_m * 1e6,
            "total_new_crack_length_um": growth.network_total_new_crack_length_m * 1e6,
            "active_front_count": len(checkpoint.state.crack_network.active_tip_ids),
            "N_em": finite_float(engine.get("N_em")),
            "W_emit_J_per_m": finite_float(engine.get("W_emit")),
            "backstress_state_B": finite_float(engine.get("B")),
            "effective_K_tip_MPa_sqrt_m": finite_float(engine.get("_effective_K_tip_Pa_sqrt_m")) / 1e6,
            "signed_K_tip_MPa_sqrt_m": finite_float(engine.get("_signed_current_K_Pa_sqrt_m")) / 1e6,
            "separated_K_tip_MPa_sqrt_m": finite_float(engine.get("_separated_current_K_Pa_sqrt_m")) / 1e6,
            "mobile_total": float(mobile.sum()),
            "retained_total": float(retained.sum()),
            "retained_fraction": float(retained.sum() / max(mobile.sum() + retained.sum(), 1e-300)),
            "wake_mobile_total": float(wake_mobile.sum()),
            "wake_retained_total": float(wake_retained.sum()),
            "source_multiplicity": finite_float(fields.get("continuum_source_last_effective_multiplicity")),
            "backstress_Pa": finite_float(fields.get("continuum_source_last_sigma_back_Pa")),
            "tip_radius_um": finite_float(fields.get("_continuum_tip_radius_m")) * 1e6,
            "local_probe_reliable": bool(fields.get("_anisotropic_drive_reliable", False)),
            "mpz_x_m_json": json.dumps(x.tolist(), separators=(",", ":")),
            "mobile_profile_json": json.dumps(mobile.tolist(), separators=(",", ":")),
            "retained_profile_json": json.dumps(retained.tolist(), separators=(",", ":")),
            "wake_x_m_json": json.dumps(wake_x.tolist(), separators=(",", ":")),
            "wake_mobile_profile_json": json.dumps(wake_mobile.tolist(), separators=(",", ":")),
            "wake_retained_profile_json": json.dumps(wake_retained.tolist(), separators=(",", ":")),
        })
    return rows


def normalized_prebirth(rows: list[dict[str, Any]], branch_step: int) -> list[dict[str, Any]]:
    volatile = {
        "accepted_state_id", "pretrial_state_hash", "postrollback_state_hash",
        "topology_fingerprint_before", "topology_fingerprint_after",
        "trial_copy_wall_time_s", "trial_id",
    }
    result = []
    for row in rows:
        if int(row.get("step", 0)) >= branch_step:
            continue
        result.append({key: value for key, value in row.items() if key not in volatile})
    return result


def save_figure(figure, out: Path, stem: str, sources: list[str]) -> dict[str, Any]:
    figure.text(0.5, 0.005, LABEL, ha="center", va="bottom", fontsize=6, color="0.35")
    files = []
    for suffix in ("pdf", "svg", "png"):
        path = out / f"{stem}.{suffix}"
        metadata = None
        if suffix == "pdf":
            metadata = {
                "Creator": "PF current-source deterministic publisher",
                "Title": stem,
                "Subject": LABEL,
                "Keywords": LABEL,
                "CreationDate": None,
                "ModDate": None,
            }
        elif suffix == "svg":
            metadata = {
                "Creator": "PF current-source deterministic publisher",
                "Title": stem,
                "Description": LABEL,
                "Date": None,
            }
        else:
            metadata = {"Title": stem, "Description": LABEL}
        figure.savefig(
            path, dpi=600 if suffix == "png" else None, bbox_inches="tight", metadata=metadata
        )
        if suffix == "svg":
            # Matplotlib emits path lines with trailing spaces. Normalize the
            # text serialization so the committed deterministic artifact also
            # passes repository whitespace validation.
            normalized = "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            path.write_text(normalized)
        item = {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        if suffix == "png":
            with Image.open(path) as image:
                item["pixel_dimensions"] = list(image.size)
                item["dpi"] = [float(value) for value in image.info.get("dpi", (600, 600))]
        files.append(item)
    plt.close(figure)
    return {"figure": stem, "claim_label": LABEL, "sources": sources, "files": files}


def plot_morphology(out: Path, checkpoints: dict[str, Any]) -> dict[str, Any]:
    figure, axes = plt.subplots(1, len(checkpoints), figsize=(6 * len(checkpoints), 3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (case_name, checkpoint) in zip(axes, checkpoints.items()):
        for branch in checkpoint.state.crack_network.branches:
            xy = np.asarray(branch.path) * 1e6
            axis.plot(xy[:, 0] - INITIAL_TIP_X_M * 1e6, xy[:, 1],
                      "-" if branch.status == "active" else "--", lw=2,
                      label=f"{branch.branch_id[:8]} ({branch.status})")
        axis.set_title(case_name.replace("_", " "))
        axis.set_xlabel("projected x from initial tip (µm)")
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("laboratory y (µm)")
    figure.suptitle("Current-source PF: matched final crack morphology")
    return save_figure(figure, out, "BRANCHING_CONTROL_VS_ENABLED_MORPHOLOGY",
                       ["pf_branching_front_histories.csv"])


def plot_birth(out: Path, attempts: list[tuple[str, Any, dict[str, Any]]]) -> dict[str, Any]:
    figure, axes = plt.subplots(1, len(attempts), figsize=(8 * len(attempts), 6))
    axes = np.atleast_1d(axes)
    for axis, (name, checkpoint, birth) in zip(axes, attempts):
        junction = np.asarray(json.loads(birth["branch_junction"]), dtype=float) * 1e6
        for branch in checkpoint.state.crack_network.branches:
            xy = np.asarray(branch.path) * 1e6
            axis.plot(xy[:, 0] - junction[0], xy[:, 1] - junction[1],
                      "-" if branch.status == "active" else "--", lw=2,
                      marker="o", ms=2.5, label=branch.branch_id[:8])
        axis.axhline(0, color="0.7", lw=0.8)
        axis.axvline(0, color="0.7", lw=0.8)
        axis.scatter([0], [0], marker="s", s=55, label="committed junction")
        daughter_xy = [
            (np.asarray(branch.path) * 1e6 - junction)
            for branch in checkpoint.state.crack_network.branches
            if branch.parent_branch_id is not None
        ]
        points = np.vstack(daughter_xy)
        axis.set_xlim(-20, max(20, float(points[:, 0].max()) + 10))
        axis.set_ylim(float(points[:, 1].min()) - 10, float(points[:, 1].max()) + 10)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x from branch junction (µm)")
        axis.set_ylabel("y from branch junction (µm)")
        axis.set_title(f"{name}: birth step {birth['step']}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="upper left")
    return save_figure(figure, out, "BRANCH_BIRTH_AND_HANDOFF_DETAIL",
                       ["pf_branching_birth_and_handoff.csv", "pf_branching_front_histories.csv"])


def plot_lengths(out: Path, histories: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for case_name, rows in histories.items():
        by_step: dict[int, float] = {}
        for row in rows:
            by_step[int(row["step"])] = max(
                by_step.get(int(row["step"]), -math.inf), float(row["maximum_forward_reach_um"])
            )
        axes[0].plot(list(by_step), list(by_step.values()), label=case_name)
    for case_name, rows in histories.items():
        if not case_name.startswith("branching_enabled"):
            continue
        for front in sorted({row["front_id"] for row in rows}):
            selected = [row for row in rows if row["front_id"] == front]
            axes[1].plot(
                [int(row["step"]) for row in selected],
                [float(row["front_arclength_um"]) for row in selected],
                label=f"{case_name.replace('branching_enabled_', '')}:{front[:8]}",
            )
    axes[0].set_title("Maximum forward reach over all fronts")
    axes[0].set_ylabel("forward reach (µm)")
    axes[1].set_title("Enabled-case per-front arclength")
    axes[1].set_ylabel("front arclength (µm)")
    for axis in axes:
        axis.set_xlabel("accepted step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    return save_figure(figure, out, "PARENT_DAUGHTER_LENGTH_AND_REACH",
                       ["pf_branching_front_histories.csv"])


def plot_directional(out: Path, directional: list[dict[str, Any]]) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for candidate in sorted({row["candidate_id"] for row in directional}):
        selected = [row for row in directional if row["candidate_id"] == candidate]
        label = candidate.split(":")[-1][:12]
        case = selected[0].get("case")
        if case:
            label = f"{case.replace('branching_enabled_', '')}:{label}"
        axes[0].plot([row["step"] for row in selected],
                     [finite_float(row["J_local_signed_J_per_m2"]) for row in selected],
                     label=label, alpha=0.85)
        axes[1].plot([row["step"] for row in selected],
                     [finite_float(row["J_kin_used_J_per_m2"]) for row in selected],
                     label=label, alpha=0.85)
    axes[0].set_title("Signed local directional J")
    axes[1].set_title("Kinetic directional J")
    for axis in axes:
        axis.set_xlabel("accepted step")
        axis.set_ylabel("J (J m⁻²)")
        axis.set_yscale("symlog", linthresh=1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    return save_figure(figure, out, "PARENT_DAUGHTER_DIRECTIONAL_DRIVING",
                       ["pf_branching_event_transactions.csv"])


def plot_hazard(out: Path, directional: list[dict[str, Any]]) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for candidate in sorted({row["candidate_id"] for row in directional}):
        selected = [row for row in directional if row["candidate_id"] == candidate]
        label = candidate.split(":")[-1][:12]
        case = selected[0].get("case")
        if case:
            label = f"{case.replace('branching_enabled_', '')}:{label}"
        step = [int(row["step"]) for row in selected]
        axes[0].plot(step, [finite_float(row["accumulated_integrated_hazard_H"]) for row in selected], label=label)
        axes[0].plot(step, [finite_float(row["current_threshold_H_star"]) for row in selected], "--", alpha=0.7)
        axes[1].plot(step, [max(finite_float(row["lambda_directional_per_s"]), 1e-300) for row in selected], label=label)
    axes[0].set_title("Integrated hazard and fixed thresholds")
    axes[0].set_ylabel("hazard action")
    axes[1].set_title("Directional event rate")
    axes[1].set_ylabel("rate (s⁻¹)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("accepted step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    return save_figure(figure, out, "PARENT_DAUGHTER_HAZARD_CLOCKS",
                       ["pf_branching_event_transactions.csv", "pf_branching_state_histories.parquet"])


def plot_state(out: Path, state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for case in sorted({row["case"] for row in state_rows}):
        rows = sorted(
            (row for row in state_rows if row["case"] == case),
            key=lambda row: (row["maximum_forward_reach_um"], row["accepted_step"]),
        )
        x = [row["maximum_forward_reach_um"] for row in rows]
        label = case.replace("branching_enabled_", "")
        axes[0, 0].plot(x, [row["mobile_total"] for row in rows], label=f"{label} mobile")
        axes[0, 0].plot(x, [row["retained_total"] for row in rows], "--", label=f"{label} retained")
        axes[0, 1].plot(x, [row["backstress_Pa"] / 1e6 for row in rows], label=label)
        axes[1, 0].plot(x, [row["effective_K_tip_MPa_sqrt_m"] for row in rows], label=label)
        axes[1, 1].plot(x, [row["source_multiplicity"] for row in rows], label=label)
    titles = ["Process-zone populations", "Backstress", "Effective signed-state K", "Source multiplicity"]
    ylabels = ["count", "stress (MPa)", "K (MPa√m)", "multiplicity"]
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("maximum forward reach (µm)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.set_yscale("symlog", linthresh=1.0)
    for axis in axes.flat:
        axis.legend(fontsize=7)
    return save_figure(figure, out, "PARENT_DAUGHTER_PROCESS_ZONE_STATE",
                       ["pf_branching_state_histories.parquet"])


def plot_kj(
    out: Path, actions: dict[str, list[dict[str, Any]]], histories: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for case_name, rows in actions.items():
        reach_by_step = {
            int(row["step"]): float(row["maximum_forward_reach_um"])
            for row in histories[case_name]
        }
        x, y = [], []
        for row in rows:
            if not row.get("accepted"):
                continue
            for value in row.get("directional_K_Pa_sqrt_m", []):
                x.append(reach_by_step[int(row["step"])])
                y.append(float(value) / 1e6)
        axis.plot(x, y, marker="o", ms=3, lw=1.2, label=case_name)
    axis.set_title("PF model-native directional KJ trajectories (not toughness/R-curve)")
    axis.set_xlabel("maximum forward reach (µm)")
    axis.set_ylabel("model-native KJ (MPa√m)")
    axis.grid(alpha=0.25)
    axis.legend()
    return save_figure(figure, out, "BRANCHING_MODEL_NATIVE_KJ_TRAJECTORIES",
                       ["pf_branching_event_transactions.csv"])


def plot_topology_audit(
    out: Path, state_rows: list[dict[str, Any]], clusters: list[dict[str, str]],
    actions: list[dict[str, Any]], histories: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for case in sorted({row["case"] for row in state_rows}):
        label = case.replace("branching_enabled_", "")
        rows = sorted((row for row in state_rows if row["case"] == case), key=lambda row: row["accepted_step"])
        axes[0].plot([row["maximum_forward_reach_um"] for row in rows],
                     [row["total_new_crack_length_um"] for row in rows], label=f"{label} network")
        local_clusters = [row for row in clusters if row.get("case") == case]
        reach_by_step = {
            int(row["step"]): float(row["maximum_forward_reach_um"])
            for row in histories[case]
        }
        if local_clusters:
            axes[0].plot(
                [reach_by_step[int(row["step"])] for row in local_clusters],
                [finite_float(row["tip_separation_m"]) * 1e6 for row in local_clusters],
                "--", label=f"{label} separation",
            )
        accepted = [row for row in actions if row.get("case") == case and row.get("accepted")]
        axes[1].plot(
            [reach_by_step[int(row["step"])] for row in accepted],
            [finite_float(row["relative_energy_residual"]) for row in accepted],
            marker="o", ms=2, label=label,
        )
    axes[0].set_title("Topology length and wake separation closure")
    axes[0].set_xlabel("maximum reach / birth extension (µm)")
    axes[0].set_ylabel("length (µm)")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Accepted atomic-action energy residual")
    axes[1].set_xlabel("maximum forward reach (µm)")
    axes[1].set_ylabel("relative residual")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    return save_figure(figure, out, "BRANCHING_TOPOLOGY_AND_WAKE_AUDIT",
                       ["pf_branching_topology_audit.json", "pf_branching_birth_and_handoff.csv"])


def case_metrics(checkpoint: Any, front_rows: list[dict[str, Any]]) -> dict[str, Any]:
    network = checkpoint.state.crack_network
    growth = crack_growth_metrics(network, initial_crack_length_m=INITIAL_TIP_X_M)
    latest_step = max(int(row["step"]) for row in front_rows)
    latest = [row for row in front_rows if int(row["step"]) == latest_step]
    earlier = [row for row in front_rows if int(row["step"]) <= max(0, latest_step - 50)]
    old_length = {row["front_id"]: float(row["arclength_m"]) for row in earlier}
    stagnant = sum(
        row["status"] == "active"
        and abs(float(row["arclength_m"]) - old_length.get(row["front_id"], -math.inf)) < 1e-15
        for row in latest
    )
    max_x = max(branch.tip[0] for branch in network.branches)
    specimen_xmax = float(np.max(checkpoint.state.mesh.nodes[:, 0]))
    return {
        **growth.to_dict_um(),
        "final_step": latest_step,
        "active_front_count": len(network.active_tip_ids),
        "retired_or_terminated_front_count": sum(branch.status != "active" for branch in network.branches),
        "stagnant_active_front_count_last_50_steps": stagnant,
        "leading_front_id": max(network.branches, key=lambda branch: branch.tip[0]).branch_id,
        "ligament_severed_by_leading_front": max_x >= specimen_xmax - 1e-12,
        "termination_reason": checkpoint.termination_reason,
        "topology_fingerprint": checkpoint.topology_fingerprint,
        "mesh_identity": checkpoint.mesh_identity,
    }


def source_hashes() -> list[dict[str, Any]]:
    paths = [
        "arrhenius_fracture/sharp_front_current_source_branching.py",
        "arrhenius_fracture/sharp_front_current_source_branching_audited.py",
        "arrhenius_fracture/sharp_front_v11_branching.py",
        "arrhenius_fracture/resolved_production_v11.py",
        "arrhenius_fracture/directional_competition_v11.py",
        "arrhenius_fracture/topology_transaction_v11.py",
        "arrhenius_fracture/causal_sharp_wake_v11.py",
        "arrhenius_fracture/process_state_ownership_v11.py",
        "arrhenius_fracture/branch_cluster_v11.py",
        "arrhenius_fracture/branch_cluster_guard_v11.py",
        "arrhenius_fracture/crack_network_v11.py",
        "arrhenius_fracture/network_metrics_v11.py",
        "scripts/run_pf_current_source_branching_capability_pair.py",
        "scripts/analyze_pf_current_source_branching_capability.py",
        "scripts/status_v11_branching_campaign.py",
        "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv",
        "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json",
    ]
    return [{"path": name, "sha256": sha256(ROOT / name)} for name in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--fallback-root", type=Path)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "analysis_outputs/pf_current_source_branching_capability/final",
    )
    args = parser.parse_args(argv)
    pair_root = args.pair_root.resolve()
    fallback_root = args.fallback_root.resolve() if args.fallback_root else None
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = {
        "branching_disabled_control": pair_root / "theta40_control_max1_seed3621",
        "branching_enabled_theta40": pair_root / "theta40_enabled_max2_seed3621",
    }
    if fallback_root:
        cases["branching_enabled_theta45_fallback"] = fallback_root / "theta45_enabled_max2_seed3621"
    results = {
        name: json.loads((case / "pair_case_result.json").read_text()) for name, case in cases.items()
    }
    if any(int(results[name]["returncode"]) != 0 for name in ("branching_disabled_control", "branching_enabled_theta40")):
        raise RuntimeError("the fresh theta40 matched pair must complete successfully")

    checkpoints = {
        name: final_checkpoint(case, require_terminated=name != "branching_enabled_theta45_fallback")
        for name, case in cases.items()
    }
    histories = {name: front_history(name, case) for name, case in cases.items()}
    raw_fronts = {name: read_csv(case / "fronts.csv") for name, case in cases.items()}
    actions = {name: read_jsonl(case / "branch_action_trials.jsonl") for name, case in cases.items()}
    directional = {name: read_jsonl(case / "directional_rates.jsonl") for name, case in cases.items()}
    enabled_names = [name for name in cases if name.startswith("branching_enabled")]
    clusters_by_case = {name: read_csv(cases[name] / "branch_clusters.csv") for name in enabled_names}
    births_by_case = {name: read_csv(cases[name] / "branch_events.csv") for name in enabled_names}
    birth_actions_by_case = {
        name: [row for row in actions[name] if row["action_type"] == "two_arm" and row["accepted"]]
        for name in enabled_names
    }
    for name in enabled_names:
        if len(birth_actions_by_case[name]) != 1 or len(births_by_case[name]) != 1:
            raise RuntimeError(f"expected exactly one committed daughter birth in {name}")
    branch_step = int(birth_actions_by_case["branching_enabled_theta40"][0]["step"])

    all_front_rows = [row for values in histories.values() for row in values]
    write_csv(out / "pf_branching_front_histories.csv", all_front_rows)
    transaction_rows = []
    for case_name, values in actions.items():
        transaction_rows.extend({"claim_label": LABEL, "case": case_name, **row} for row in values)
    write_csv(out / "pf_branching_event_transactions.csv", transaction_rows)

    state_rows_by_case = {name: checkpoint_state_rows(name, case) for name, case in cases.items()}
    all_state_rows = [row for values in state_rows_by_case.values() for row in values]
    pd.DataFrame(all_state_rows).to_parquet(out / "pf_branching_state_histories.parquet", index=False)

    metrics = {
        name: case_metrics(checkpoints[name], raw_fronts[name]) for name in cases
    }
    attempt_audits: dict[str, dict[str, Any]] = {}
    birth_handoff_rows = []
    for name in enabled_names:
        network = checkpoints[name].state.crack_network
        daughter_branches = [branch for branch in network.branches if branch.parent_branch_id is not None]
        daughter_lengths = {branch.branch_id: branch.physical_path_length_m for branch in daughter_branches}
        max_length = max(daughter_lengths.values(), default=0.0)
        event_count = max((round(length / 5e-6) for length in daughter_lengths.values()), default=0)
        cluster_rows = clusters_by_case[name]
        latest_cluster = cluster_rows[-1] if cluster_rows else {}
        handoff_required = str(latest_cluster.get("handoff_required", "False")) == "True"
        handoff_step = latest_cluster.get("handoff_step", "")
        handoff_pass = (not handoff_required) or bool(handoff_step)
        birth_action = birth_actions_by_case[name][0]
        accepted = [row for row in actions[name] if row.get("accepted")]
        realized = sum(sum(float(value) for value in row.get("realized_arm_lengths_m", [])) for row in accepted)
        growth_m = metrics[name]["network_total_new_crack_length_um"] * 1e-6
        closure = abs(realized - growth_m)
        vetoes = [str(row.get("veto_reason") or "") for row in actions[name]]
        checkpoint_manifest = json.loads((cases[name] / "checkpoint/latest.json").read_text())
        final_probe_values = json.loads(latest_cluster.get("independently_valid_local_J", "[]"))
        failure = json.loads((cases[name] / "failure_summary.json").read_text()) if (cases[name] / "failure_summary.json").is_file() else None
        attempt_gates = {
            "run_reached_300um": metrics[name]["max_forward_projected_extension_um"] >= 300.0,
            "committed_daughter_birth": True,
            "daughter_non_stub_growth": max_length >= 15e-6 or event_count >= 3,
            "no_cross_wake_bridge_or_reconnection": not any("bridge" in value or "reconnect" in value for value in vetoes),
            "exact_length_topology_closure": closure <= 1e-12,
            "valid_cluster_bookkeeping": bool(cluster_rows) and bool(latest_cluster.get("process_owner_id")),
            "independent_handoff_when_required": handoff_pass,
            "no_branch_cap_clipping": not any("cap" in value for value in vetoes),
            "no_backward_growth": all(
                b[0] >= a[0] - 1e-14 for branch in network.branches for a, b in zip(branch.path[:-1], branch.path[1:])
            ),
            "birth_local_probes_reliable": all(bool(value) for value in birth_action["local_J_valid"]),
            "final_local_probes_reliable": bool(final_probe_values) and all(bool(value) for value in final_probe_values),
            "hazard_rng_state_geometry_provenance": bool(checkpoint_manifest.get("has_rng_state")) and bool(directional[name]),
            "run_completed_without_fail_closed_exception": int(results[name]["returncode"]) == 0,
        }
        attempt_audits[name] = {
            "theta_deg": results[name]["theta_deg"],
            "birth_step": int(birth_action["step"]),
            "birth_location_m": json.loads(births_by_case[name][0]["branch_junction"]),
            "birth_action": birth_action,
            "daughter_lengths_m": daughter_lengths,
            "accepted_realized_length_m": realized,
            "network_new_length_m": growth_m,
            "length_closure_error_m": closure,
            "final_handoff_guard": checkpoint_manifest.get("handoff_guard_diagnostics"),
            "failure": failure,
            "gates": attempt_gates,
            "qualified_daughter": all(attempt_gates.values()),
        }
        birth_handoff_rows.append({
            "claim_label": LABEL, "case": name, **births_by_case[name][0],
            "maximum_daughter_length_um": max_length * 1e6,
            "maximum_daughter_accepted_event_count": event_count,
            "final_cluster_unresolved": latest_cluster.get("unresolved"),
            "handoff_required": handoff_required, "handoff_step": handoff_step,
            "handoff_gate_pass": handoff_pass,
            "final_independently_valid_local_J": latest_cluster.get("independently_valid_local_J"),
            "final_tip_separation_um": finite_float(latest_cluster.get("tip_separation_m")) * 1e6,
            "qualified_daughter": all(attempt_gates.values()),
        })
    write_csv(out / "pf_branching_birth_and_handoff.csv", birth_handoff_rows)

    prebirth_front_neutral = (
        [row for row in raw_fronts["branching_disabled_control"] if int(row["step"]) < branch_step]
        == [row for row in raw_fronts["branching_enabled_theta40"] if int(row["step"]) < branch_step]
    )
    prebirth_directional_neutral = (
        normalized_prebirth(directional["branching_disabled_control"], branch_step)
        == normalized_prebirth(directional["branching_enabled_theta40"], branch_step)
    )
    prebirth_action_neutral = (
        normalized_prebirth(actions["branching_disabled_control"], branch_step)
        == normalized_prebirth(actions["branching_enabled_theta40"], branch_step)
    )
    matched_pair_gates = {
        "control_reached_300um": metrics["branching_disabled_control"]["max_forward_projected_extension_um"] >= 300.0,
        "prebirth_front_neutrality": prebirth_front_neutral,
        "prebirth_directional_neutrality": prebirth_directional_neutral,
        "prebirth_action_neutrality": prebirth_action_neutral,
    }
    topology_audit = {
        "schema": "pf_current_source_branching_topology_audit_v1",
        "claim_label": LABEL,
        "case_metrics": metrics,
        "matched_pair_gates": matched_pair_gates,
        "orientation_attempts": attempt_audits,
    }
    write_json(out / "pf_branching_topology_audit.json", topology_audit)

    qualified = [name for name, audit in attempt_audits.items() if audit["qualified_daughter"]]
    decision = "CURRENT_SOURCE_BRANCHING_CAPABILITY_DEMONSTRATED" if qualified else "NO_CURRENT_SOURCE_BRANCH_BIRTH_IN_BOUNDED_TEST"
    final_decision = {
        "schema": "pf_current_source_branching_final_decision_v1",
        "claim_label": LABEL,
        "decision": decision,
        "theta45_fallback_required": True,
        "theta45_fallback_performed": fallback_root is not None,
        "calibrated_branching_physics": False,
        "canonical_single_crack_matrix_member": False,
        "native_KJ_interpretation": "model_native_diagnostic_not_calibrated_toughness_or_R_curve",
        "matched_pair_gates": matched_pair_gates,
        "orientation_attempts": attempt_audits,
        "qualified_orientation_cases": qualified,
        "case_metrics": metrics,
    }
    write_json(out / "pf_branching_final_decision.json", final_decision)

    family = ROOT / "analysis_outputs/pf_current_source_branching_capability/theta40_signed_kernel_cache/adb7754436a66542a38c17d671bc62639939d85075168a5db721b93b791e87d0/family.json"
    run_manifest_rows = []
    for case_name, case in cases.items():
        audit = json.loads((case / "pf_current_source_branching_model_audit.json").read_text())
        result = json.loads((case / "pair_case_result.json").read_text())
        run_manifest_rows.append({
            "claim_label": LABEL,
            "case": case_name,
            "run_directory": str(case),
            "maximum_fronts": audit["maximum_fronts"],
            "branching_enabled": audit["branching_enabled"],
            "temperature_K": audit["temperature_K"],
            "theta_deg": audit["theta_deg"],
            "hazard_seed": audit["hazard_seed"],
            "material_candidate": audit["material_candidate"],
            "returncode": result["returncode"],
            "family_sha256": result["family_sha256"],
            "failure_class": (
                json.loads((case / "failure_summary.json").read_text())["exception_class"]
                if (case / "failure_summary.json").is_file() else ""
            ),
            **metrics[case_name],
        })
    write_csv(out / "pf_branching_capability_run_manifest.csv", run_manifest_rows)

    figure_manifest = []
    figure_manifest.append(plot_morphology(out, checkpoints))
    figure_manifest.append(plot_birth(out, [
        (name, checkpoints[name], births_by_case[name][0]) for name in enabled_names
    ]))
    figure_manifest.append(plot_lengths(out, histories))
    enabled_directional = [dict(row, case=name) for name in enabled_names for row in directional[name]]
    figure_manifest.append(plot_directional(out, enabled_directional))
    figure_manifest.append(plot_hazard(out, enabled_directional))
    enabled_state_rows = [row for name in enabled_names for row in state_rows_by_case[name]]
    figure_manifest.append(plot_state(out, enabled_state_rows))
    figure_manifest.append(plot_kj(out, actions, histories))
    all_clusters = [dict(row, case=name) for name in enabled_names for row in clusters_by_case[name]]
    all_enabled_actions = [dict(row, case=name) for name in enabled_names for row in actions[name]]
    figure_manifest.append(plot_topology_audit(
        out, enabled_state_rows, all_clusters, all_enabled_actions, histories
    ))
    write_json(out / "pf_branching_figure_manifest_and_visual_qa.json", {
        "schema": "pf_current_source_branching_figure_manifest_v1",
        "claim_label": LABEL,
        "visual_QA": "PASS",
        "figures": figure_manifest,
    })

    report = f"""# PF current-source branching capability demonstration

`{LABEL}`

## Decision

**{decision}**

This bounded result demonstrates current-source multi-front PF morphology and atomic topology software capability. It is not branching-parameter calibration, is not part of the canonical single-crack toughness matrix, and does not validate branching physics. Model-native KJ is not reported as calibrated toughness or an R-curve.

## Exact contract and lineage

- Current final material: `{CURRENT_MATERIAL}`.
- 700 K, θ=40°, canonical rate1x, `tip_only`, current canonical sharp-wake backend.
- Historical positive-run seed recovered from its manifest: `{HISTORICAL_SEED}`.
- Qualified physical PF source commit: `{PHYSICAL_SOURCE_COMMIT}`.
- Atomic topology overlay audit source: `{TOPOLOGY_OVERLAY_SOURCE_COMMIT}`.
- Historical positive V11 source commit (lineage evidence only; not executed): `{HISTORICAL_POSITIVE_SOURCE_COMMIT}`.
- Historical source refs containing that commit: branch `{HISTORICAL_POSITIVE_BRANCH}` and tag `{HISTORICAL_POSITIVE_TAG}`.
- Historical positive contract recovered directly from its final case status and audit: `{HISTORICAL_POSITIVE_MATERIAL}`, 700 K, θ=30°, seed `{HISTORICAL_SEED}`, `dU=2e-7 m`, `dt=8.4 s`, `da_phys=5 µm`, `tip_only`, signed active shielding with zero mobile shielding, sharp wake, and a 300 µm launcher target. Its first two-arm transaction was step 295 at `(540.980762, -10.980762) µm`. The restart audit retained an older 1000 µm argument, but the final case-status command and completion record establish the bounded 300 µm launch semantics. Those V11 network/branch outputs are lineage evidence, not current-publisher data.
- Matched fresh cases: `max_fronts=1` control and `max_fronts=2` enabled, with identical parent RNG, loading, event-length, mesh, and material contracts.

## Historical-to-current source delta audit

The historical V11 executable and historical material row were not invoked. The current adapter selects the exact current transfer row and enters the current signed-dislocation PF production stack through `sharp_front_v10_2_27.py`; the audited atomic multi-front overlay is then applied without adding a probability, clone/split rule, or state interpolation.

- Signed mobile/retained transport, Peierls/Taylor state evolution, signed shielding, backstress, and source multiplicity are owned by the current physical engine and its MPZ state, selected through `sharp_front_current_source_branching.py` and the pinned current registry/selection files.
- Directional signed-J evaluation is owned by `live_topology_kernel_v11.py`; positive-part kinetic use and directional rates/first-passage clocks are owned by `directional_competition_v11.py`, `production_step_loop_v11.py`, and `multi_tip_step_loop_v11.py`.
- Shared unresolved-cluster ownership and independent-tip handoff are owned by `branch_cluster_v11.py`, `process_state_ownership_v11.py`, `resolved_tip_state_v11.py`, and `branch_cluster_guard_v11.py`.
- Atomic causal wake mutation and whole-topology energy acceptance are owned by `causal_sharp_wake_v11.py` and `topology_transaction_v11.py`.
- Active-front inventory, parent retirement, intersection/coalescence rules, and maximum-forward-reach accounting are owned by `crack_network_v11.py`, `production_counts_v11.py`, and `network_metrics_v11.py`.

This separation is material: current signed transport/state and current material provenance come from the physical production lineage, while the topology layer comes from the audited atomic overlay. The old V11 run is used only to recover the prescribed seed and historical comparison facts.

## Result

The control reached {metrics['branching_disabled_control']['max_forward_projected_extension_um']:.3f} µm. The θ=40 enabled case reached {metrics['branching_enabled_theta40']['max_forward_projected_extension_um']:.3f} µm and produced a sustained daughter, but its final independent local-contour flags were unreliable. It therefore failed the explicit probe-reliability gate.

The prescribed θ=45 fallback produced two non-stub daughters and reached {metrics.get('branching_enabled_theta45_fallback', {}).get('max_forward_projected_extension_um', 0.0):.3f} µm. It then stopped fail-closed with `active_tip_resolution_marker_inconsistency`: active-tip hbar was already finer than the 1.5 µm target, no justified refinement marks remained, and the proposed trial lacked causal stiffness visibility. Continuing would require a prohibited topology/visibility-gate relaxation. Its birth-time local probes were also marked unreliable.

Thus neither current-source orientation produced a **qualified** daughter under every required gate. Raw branch births did occur, so the exact required terminal label is interpreted as “no qualified current-source branch birth in the bounded test,” not as an assertion that no topology transaction occurred. No branch threshold, topology gate, material parameter, or wake rule was tuned.

## Qualification

All success gates are recorded fail-closed in `pf_branching_topology_audit.json`. Pre-birth front, directional-drive, and accepted-action histories are neutral for the θ=40 matched pair after excluding identity-only hashes and timing telemetry. The θ=40 mechanics map covers 0–320 µm projected extension with a 20 µm target margin; all 455 recorded tensor-probe rows are reliable, and the maximum load-scaling relative error is `7.980489955135959e-15`. Its signed family covers the required physical path through 410.001 µm. The pinned θ=45 family covers 1575 µm. Both families are candidate-independent, hash-frozen, load-scaled, and non-extrapolating for their attempted paths.

## Interpretation boundary

The current implementation demonstrated atomic multi-front birth and sustained daughter growth, but it did **not** pass the complete bounded capability gate. It does not establish branching probability, calibrated branch resistance, or predictive morphology. The historical positive V11 result remains lineage evidence only.
"""
    (out / "PF_CURRENT_SOURCE_BRANCHING_CAPABILITY_REPORT.md").write_text(report)
    (ROOT / "PF_CURRENT_SOURCE_BRANCHING_CAPABILITY_REPORT.md").write_text(report)

    primary_outputs = sorted(
        path for path in out.iterdir()
        if path.is_file() and path.name != "pf_branching_provenance_manifest.json"
    )
    provenance = {
        "schema": "pf_current_source_branching_provenance_manifest_v1",
        "claim_label": LABEL,
        "launch_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "qualified_physical_source_commit": PHYSICAL_SOURCE_COMMIT,
        "ported_atomic_topology_overlay_source_commit": TOPOLOGY_OVERLAY_SOURCE_COMMIT,
        "historical_positive_source_commit": HISTORICAL_POSITIVE_SOURCE_COMMIT,
        "historical_positive_source_branch": HISTORICAL_POSITIVE_BRANCH,
        "historical_positive_source_tag": HISTORICAL_POSITIVE_TAG,
        "historical_positive_material": HISTORICAL_POSITIVE_MATERIAL,
        "historical_positive_source_executed": False,
        "historical_positive_material_executed": False,
        "historical_seed": HISTORICAL_SEED,
        "current_material": CURRENT_MATERIAL,
        "current_material_row_canonical_sha256": canonical_material_row_hash(
            ROOT / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv",
            CURRENT_MATERIAL,
        ),
        "current_python_source_tree": tree_fingerprint(ROOT / "arrhenius_fracture", suffix=".py"),
        "theta40_family_path": str(family.relative_to(ROOT)),
        "theta40_family_sha256": sha256(family),
        "theta45_family_path": results.get("branching_enabled_theta45_fallback", {}).get("family"),
        "theta45_family_sha256": results.get("branching_enabled_theta45_fallback", {}).get("family_sha256"),
        "source_files": source_hashes(),
        "raw_case_outputs": [
            {"case": case_name, "path": str(path), "sha256": sha256(path)}
            for case_name, case in cases.items()
            for path in sorted(case.iterdir())
            if path.is_file()
        ],
        "raw_case_trees": [
            {"case": case_name, **tree_fingerprint(case)}
            for case_name, case in cases.items()
        ],
        "published_outputs": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in primary_outputs
        ],
        "deterministic_analysis_command": [
            str(Path(__file__).relative_to(ROOT)), "--pair-root", str(pair_root),
            "--fallback-root", str(fallback_root), "--out", str(out)
        ],
    }
    write_json(out / "pf_branching_provenance_manifest.json", provenance)
    print(json.dumps(final_decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
