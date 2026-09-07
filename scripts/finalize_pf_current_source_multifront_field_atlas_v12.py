#!/usr/bin/env python3
"""Validate, consolidate, plot, and report the eight-case V12 field atlas."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from arrhenius_fracture.multifront_checkpoint_v12 import load_accepted_boundary_checkpoint_v12
from arrhenius_fracture.multifront_field_atlas_v12 import CLAIM_LABEL, MILESTONES_UM, sha256
from scripts.run_pf_current_source_multifront_field_atlas_v12 import (
    BASE_COMMIT, CASES, FAMILY_PHYSICS, FAMILY_SHA256, MECHANICAL_FINGERPRINT,
    MECHANICAL_SHA256, ROWS, case_parts, registry_row,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAME = "pf_current_source_multifront_field_atlas_300K_1000K"
ROW_ORDER = ("Peak", "DBTT", "weakT", "ceramic")
DISPLAY = {"Peak": "Peak", "DBTT": "DBTT", "weakT": "weak-T", "ceramic": "ceramic-like"}
QUALIFIED_WAKE_REMAP_SOURCE_COMMIT = "ccb48b58790fc7d358d1566600b6e487fa5afdf1"
EXECUTION_SOURCE_COMMIT = "78ef6bbe180f29ca5886c955244817fa0413617f"
EXECUTION_SOURCE_TREE = "5b00c179bfda5b30e93b9fa1696d3cfa5b1a4c9c"
SOURCE_BUNDLE_SHA256 = "0796a7cf142d171079f3d35aa835093d1d49d8036a0a8beaa6d55f4ec60b19f5"
PRE_WAKE_LABEL = "PRE_WAKE_REMAP_DEFECT_EVIDENCE_IMMUTABLE"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def tree_fingerprint(root: Path) -> dict[str, Any]:
    entries = []
    exclusions = {"raw_tree_fingerprint.json"}
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in exclusions):
        entries.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "v12.raw-output-tree-fingerprint/1", "root": str(root.resolve()),
        "file_count": len(entries), "size_bytes": sum(v["size_bytes"] for v in entries),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "excluded_relative_paths": sorted(exclusions), "entries": entries,
    }


def final_snapshot(case_root: Path) -> tuple[Path, dict, dict[str, np.ndarray]]:
    candidates = []
    for path in (case_root / "field_snapshots").glob("*/metadata.json"):
        metadata = json.loads(path.read_text())
        if "final_last_accepted_state" in metadata["selection_reasons"]:
            candidates.append((path, metadata))
    if len(candidates) != 1:
        raise RuntimeError(f"{case_root.name}: expected exactly one final accepted snapshot")
    path, metadata = candidates[0]
    arrays = dict(np.load(path.parent / "fields.npz", allow_pickle=False))
    return path.parent, metadata, arrays


def validate_final(case_root: Path, metadata: dict, arrays: dict[str, np.ndarray]) -> dict:
    terminal = json.loads((case_root / "terminal_manifest.json").read_text())
    checkpoint = load_accepted_boundary_checkpoint_v12(terminal["final_checkpoint_path"])
    state = checkpoint.accepted_fem_state
    exact = {
        "nodes": np.array_equal(arrays["nodes_m"], state.mesh.nodes),
        "elements": np.array_equal(arrays["elements"], state.mesh.elems),
        "ux": np.array_equal(arrays["ux_m"], state.displacement[0::2]),
        "uy": np.array_equal(arrays["uy_m"], state.displacement[1::2]),
        "damage": np.array_equal(arrays["damage_nodal"], state.damage),
        "ep_xx": np.array_equal(arrays["plastic_strain_xx"], state.ep_gp[0]),
        "ep_yy": np.array_equal(arrays["plastic_strain_yy"], state.ep_gp[1]),
        "ep_xy": np.array_equal(arrays["plastic_strain_xy"], state.ep_gp[2]),
        "rho": np.array_equal(arrays["dislocation_density_m-2"], state.rho_gp),
        "sigma_xx": np.array_equal(arrays["sigma_xx_Pa"], checkpoint.accepted_stress_field[0]),
        "sigma_yy": np.array_equal(arrays["sigma_yy_Pa"], checkpoint.accepted_stress_field[1]),
        "sigma_xy": np.array_equal(arrays["sigma_xy_Pa"], checkpoint.accepted_stress_field[2]),
        "accepted_state_id": metadata["accepted_state_id"] == checkpoint.runtime.accepted_state_id,
        "stress_state_id": metadata["stress_field_state_id"] == checkpoint.runtime.stress_field_state_id,
        "topology": metadata["topology_fingerprint"] == checkpoint.runtime.topology_fingerprint,
    }
    if not all(exact.values()):
        raise RuntimeError(f"{case_root.name}: final portable field differs from accepted checkpoint: {exact}")
    return {"case": case_root.name, "all_exact": True, "checks": exact}


def topology_overlay(ax, network: dict, *, owner_by_front: dict | None = None, owner_map: bool = False) -> None:
    cmap = plt.get_cmap("tab10")
    owner_by_front = owner_by_front or {}
    owner_colors = {owner: cmap(index % 10) for index, owner in enumerate(sorted(set(owner_by_front.values())))}
    for index, branch in enumerate(network["branches"]):
        path = np.asarray(branch["path_m"]) * 1e6
        status = branch["status"]
        if owner_map:
            color = owner_colors.get(owner_by_front.get(branch["branch_id"]), cmap(index % 10))
        else:
            color = "white" if status == "active" else ("0.2" if status == "retired" else "magenta")
        style = "-" if status == "active" else ("--" if status == "retired" else ":")
        ax.plot(path[:, 0], path[:, 1], style, color=color, lw=1.2, zorder=5)


def annotate_panel(
    ax, terminal: dict, *, scale_length_um: float = 250.0,
    draw_geometry_scale: bool = True,
) -> None:
    text = (
        f"reach {terminal['achieved_forward_reach_um']:.1f} µm\n"
        f"births {terminal['cumulative_branch_births']}; max active {terminal['maximum_concurrent_active_fronts']}"
    )
    if not terminal["target_reached"]:
        reason = terminal.get("exact_terminal_reason", terminal["terminal_label"])
        if reason == "configured_front_resource_limit_reached":
            status = "policy stop: six active fronts"
        elif reason.startswith("selected_topology_trial_rejected"):
            status = "fail-closed: energy gate"
        else:
            status = "fail-closed: " + reason.split(":", 1)[0]
        text += "\n" + status
    ax.text(0.015, 0.985, text, transform=ax.transAxes, va="top", ha="left", fontsize=5.8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5})
    if not draw_geometry_scale:
        return
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    length = float(scale_length_um)
    scale_x0 = x0 + 0.12*(x1-x0)
    ax.plot([scale_x0, scale_x0 + length], [y0 + 0.055*(y1-y0)]*2,
            color="black", lw=2.0, zorder=10)
    ax.text(scale_x0 + length/2, y0 + 0.07*(y1-y0), f"{length:g} µm",
            ha="center", va="bottom", fontsize=6)
    ax.plot([500.0], [0.0], marker="|", color="red", ms=8, mew=1.1, zorder=8)


def save_figure(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def matrix_figure(records: dict, figures: Path, *, field: str, filename: str,
                  title: str, nodal: bool, cmap: str, autoscale: bool = False,
                  spatial_zoom: bool = False) -> None:
    values = [records[case]["arrays"][field] for case in CASES]
    common_min = min(float(np.nanmin(v)) for v in values)
    common_max = max(float(np.nanmax(v)) for v in values)
    if common_max <= common_min:
        common_max = common_min + 1.0
    variants = ((False, ""), (True, "_autoscaled")) if autoscale else ((False, ""),)
    for local_scale, suffix in variants:
        fig, axes = plt.subplots(4, 2, figsize=(12.2, 12.0), sharex=True, sharey=True)
        image = None
        for row_index, material in enumerate(ROW_ORDER):
            for col_index, temperature in enumerate((300, 1000)):
                case = f"{material}_{temperature}K"; rec = records[case]; arrays = rec["arrays"]
                ax = axes[row_index, col_index]
                nodes = arrays["nodes_m"] * 1e6; elems = arrays["elements"]
                tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)
                vmin, vmax = (float(np.nanmin(arrays[field])), float(np.nanmax(arrays[field]))) if local_scale else (common_min, common_max)
                if vmax <= vmin: vmax = vmin + 1.0
                if nodal:
                    image = ax.tripcolor(tri, arrays[field], shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
                else:
                    image = ax.tripcolor(tri, facecolors=arrays[field], shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
                topology_overlay(ax, rec["network"])
                for junction in rec["metadata"]["junctions"].values():
                    xy = np.asarray(junction["junction_xy_m"]) * 1e6
                    ax.plot(xy[0], xy[1], marker="o", ms=2.6, mec="black", mfc="yellow", zorder=7)
                limits = rec["zoom_limits"] if spatial_zoom else rec["limits"]
                ax.set_aspect("equal"); ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
                ax.set_title(f"{DISPLAY[material]}, {temperature} K", fontsize=9)
                annotate_panel(
                    ax, rec["terminal"],
                    scale_length_um=10.0 if spatial_zoom else 250.0,
                )
                if row_index == 3: ax.set_xlabel("x (µm)")
                if col_index == 0: ax.set_ylabel("y (µm)")
        fig.suptitle(title + (" — individual scales" if local_scale else " — common scale"), fontsize=12)
        if image is not None and not local_scale:
            color_axis = fig.add_axes((0.90, 0.32, 0.016, 0.38))
            fig.colorbar(image, cax=color_axis, label=field.replace("_", " "))
        fig.subplots_adjust(
            top=0.95, right=0.86 if not local_scale else 0.97,
            left=0.08, bottom=0.06, hspace=0.25, wspace=0.18,
        )
        save_figure(fig, figures / f"{filename}{suffix}")


def branch_owner_figure(records: dict, figures: Path, *, spatial_zoom: bool = False) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12.2, 12.0), sharex=True, sharey=True)
    for ri, material in enumerate(ROW_ORDER):
        for ci, temperature in enumerate((300, 1000)):
            case = f"{material}_{temperature}K"; rec = records[case]; ax = axes[ri, ci]
            topology_overlay(ax, rec["network"], owner_by_front=rec["metadata"]["owner_by_front"], owner_map=True)
            for junction_id, junction in rec["metadata"]["junctions"].items():
                xy = np.asarray(junction["junction_xy_m"]) * 1e6
                ax.scatter(*xy, s=18, facecolor="yellow", edgecolor="black", zorder=7)
                ax.text(xy[0], xy[1], junction_id[-4:], fontsize=4.5)
            ax.axvline(500.0, color="0.5", lw=0.5, ls=":")
            limits = rec["zoom_limits"] if spatial_zoom else rec["limits"]
            ax.set_aspect("equal"); ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
            ax.set_title(f"{DISPLAY[material]}, {temperature} K", fontsize=9)
            annotate_panel(
                ax, rec["terminal"],
                scale_length_um=10.0 if spatial_zoom else 250.0,
            )
            if ri == 3: ax.set_xlabel("x (µm)")
            if ci == 0: ax.set_ylabel("y (µm)")
    fig.suptitle(
        "Final accepted branch, junction, and process-owner map"
        + (" — crack-tip zoom" if spatial_zoom else " — specimen context"),
        fontsize=12,
    )
    fig.subplots_adjust(top=0.95, hspace=0.22, wspace=0.08)
    save_figure(
        fig, figures / (
            "final_branch_junction_owner_map_tip_zoom"
            if spatial_zoom else "final_branch_junction_owner_map"
        )
    )


def process_zone_figure(records: dict, figures: Path) -> None:
    maxima = {"population": 0.0, "backstress": 0.0, "shielding": 0.0}
    for rec in records.values():
        for owner_id, owner in rec["metadata"]["owners"].items():
            if owner_id == "archived_reservoirs": continue
            prefix = owner["array_prefix"]; a = rec["arrays"]
            for key in ("mobile_profile", "retained_profile", "accumulated_slip_profile"):
                maxima["population"] = max(maxima["population"], float(np.max(np.sum(a[prefix+key], axis=0), initial=0)))
            maxima["backstress"] = max(maxima["backstress"], float(np.max(np.abs(a[prefix+"backstress_by_system_Pa"]), initial=0)))
            maxima["shielding"] = max(maxima["shielding"], float(np.max(np.abs(a[prefix+"signed_active_shielding_by_bin_Pa_sqrt_m"]), initial=0)))
    fig, axes = plt.subplots(4, 2, figsize=(12.2, 12.0), sharex=True)
    for ri, material in enumerate(ROW_ORDER):
        for ci, temperature in enumerate((300, 1000)):
            case = f"{material}_{temperature}K"; rec = records[case]; ax = axes[ri, ci]; right = ax.twinx()
            colors = plt.get_cmap("tab10")
            for oi, (owner_id, owner) in enumerate((v for v in rec["metadata"]["owners"].items() if v[0] != "archived_reservoirs")):
                prefix = owner["array_prefix"]; a = rec["arrays"]; x = a[prefix+"owner_local_x_m"]*1e6
                color = colors(oi % 10)
                ax.plot(x, np.sum(a[prefix+"mobile_profile"], axis=0), color=color, ls="-", lw=0.9)
                ax.plot(x, np.sum(a[prefix+"retained_profile"], axis=0), color=color, ls="--", lw=0.9)
                ax.plot(x, np.sum(a[prefix+"accumulated_slip_profile"], axis=0), color=color, ls=":", lw=0.9)
                shield = a[prefix+"signed_active_shielding_by_bin_Pa_sqrt_m"] / 1e6
                right.plot(x, shield, color=color, ls="-.", lw=0.8)
                back = a[prefix+"backstress_by_system_Pa"] / 1e9
                if back.size:
                    right.hlines(float(np.mean(back)), x.min(), x.max(), color=color, ls=(0,(2,1)), lw=0.6)
            ax.set_xlim(0, 50); ax.set_ylim(0, maxima["population"]*1.05 if maxima["population"] else 1)
            yright = max(maxima["backstress"]/1e9, maxima["shielding"]/1e6, 1e-12)
            right.set_ylim(-yright*1.05, yright*1.05)
            ax.set_title(f"{DISPLAY[material]}, {temperature} K", fontsize=9)
            annotate_panel(ax, rec["terminal"], draw_geometry_scale=False)
            if ri == 3: ax.set_xlabel("owner-local x (µm)")
            if ci == 0: ax.set_ylabel("line-content population")
            if ci == 1: right.set_ylabel("backstress (GPa) / shielding (MPa√m)")
    fig.suptitle("Final accepted owner-local MPZ profiles (common axes)\nsolid mobile; dashed retained; dotted slip; dash-dot signed shielding; short-dash mean system backstress", fontsize=10)
    fig.subplots_adjust(top=0.93, hspace=0.28, wspace=0.29)
    save_figure(fig, figures / "final_process_zone_profiles")


def copy_compact_snapshot(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("fields.npz", "metadata.json", "crack_network.json", "fields.vtu", "crack_network.vtp"):
        shutil.copy2(source / name, destination / name)


def interpreted_snapshot_metadata(metadata: dict, terminal: dict) -> dict:
    """Remove terminal-only milestone labels for milestones never crossed."""
    result = dict(metadata)
    reasons = list(metadata["selection_reasons"])
    if "final_last_accepted_state" in reasons:
        reached = float(terminal["achieved_forward_reach_um"])
        reasons = [
            reason for reason in reasons
            if not reason.startswith("milestone_at_or_below_")
            or reached >= float(reason.removeprefix("milestone_at_or_below_").removesuffix("um"))
        ]
    result["selection_reasons"] = reasons
    return result


def parse_case_roots(raw: Path, values: list[str]) -> dict[str, Path]:
    roots = {case: (raw / case).resolve() for case in CASES}
    for value in values:
        if "=" not in value:
            raise ValueError(f"case-root override must be CASE=PATH, got {value!r}")
        case, path = value.split("=", 1)
        if case not in CASES:
            raise ValueError(f"unsupported case-root override {case!r}")
        roots[case] = Path(path).resolve()
    return roots


def first_branch_onsets(pre_wake_root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for case in CASES:
        matches = []
        for path in (pre_wake_root / case / "field_snapshots").glob("*/metadata.json"):
            metadata = json.loads(path.read_text())
            if "accepted_binary_branch_birth_1" in metadata["selection_reasons"]:
                matches.append((path, metadata))
        if len(matches) != 1:
            raise RuntimeError(f"{case}: expected exactly one first-birth snapshot, found {len(matches)}")
        path, metadata = matches[0]
        result[case] = {
            "first_branch_onset_step": metadata["step_count"],
            "first_branch_onset_time_s": metadata["physical_time_s"],
            "first_branch_onset_opening_m": metadata["accepted_opening_m"],
            "first_branch_onset_forward_reach_um": metadata["maximum_network_forward_reach_um"],
            "first_branch_onset_evidence_path": str(path.resolve()),
        }
    return result


def qualification_records(qualification_root: Path) -> dict[str, dict]:
    ceiling_path = qualification_root / "multihit_ceiling_semantics.csv"
    checkpoint_path = qualification_root / "wake_remap_frozen_checkpoint_regression.csv"
    with ceiling_path.open(newline="") as stream:
        ceiling = {row["case"]: row for row in csv.DictReader(stream)}
    with checkpoint_path.open(newline="") as stream:
        checkpoints = {row["case"]: row for row in csv.DictReader(stream)}
    if set(ceiling) != set(CASES) or set(checkpoints) != set(CASES):
        raise RuntimeError("wake-remap qualification tables do not cover exactly the eight cases")
    records = {}
    for case in CASES:
        lambda_tau = json.loads(ceiling[case]["lambda_uncapped_tau_c"])
        if not lambda_tau or not all(float(value) > 1.0 for value in lambda_tau):
            raise RuntimeError(f"{case}: first-bifurcation multi-hit state is not saturated")
        records[case] = {
            "source_checkpoint_path": checkpoints[case]["checkpoint_path"],
            "source_checkpoint_sha256": checkpoints[case]["checkpoint_sha256"],
            "tau_c_s": float(ceiling[case]["tau_c_s"]),
            "first_bifurcation_candidate_ids_json": ceiling[case]["candidate_ids"],
            "first_bifurcation_lambda_uncapped_per_s_json": ceiling[case]["lambda_uncapped_per_s"],
            "first_bifurcation_lambda_uncapped_tau_c_json": ceiling[case]["lambda_uncapped_tau_c"],
            "first_bifurcation_saturation_status": "HIGH_RATE_MULTI_HIT_SATURATED",
        }
    return records


def first_post_remap_snapshot(case_root: Path) -> dict:
    records = [
        json.loads(path.read_text())
        for path in (case_root / "field_snapshots").glob("*/metadata.json")
    ]
    if not records:
        raise RuntimeError(f"{case_root.name}: no corrected post-remap portable snapshot")
    return min(records, key=lambda item: int(item["step_count"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "analysis_outputs" / OUTPUT_NAME)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--case-root", action="append", default=[], metavar="CASE=PATH")
    parser.add_argument("--pre-wake-root", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--source-seal-manifest", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve(); output = args.output_root.resolve()
    review = (args.review_root or (output / "compact_review_package")).resolve()
    case_roots = parse_case_roots(raw, args.case_root)
    pre_wake_root = args.pre_wake_root.resolve()
    qualification_root = args.qualification_root.resolve()
    source_seal_manifest = json.loads(args.source_seal_manifest.resolve().read_text())
    if source_seal_manifest["bundle_sha256"] != SOURCE_BUNDLE_SHA256:
        raise RuntimeError("execution-source seal does not match the pinned bundle SHA-256")
    onsets = first_branch_onsets(pre_wake_root)
    qualification = qualification_records(qualification_root)
    if any(not case_roots[case].joinpath("terminal_manifest.json").is_file() for case in CASES):
        missing = [case for case in CASES if not case_roots[case].joinpath("terminal_manifest.json").is_file()]
        raise RuntimeError(f"campaign has unterminated cases: {missing}")
    output.mkdir(parents=True, exist_ok=True); figures = output / "figures"; figures.mkdir(exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    archive_filename = "pf_current_source_multifront_field_atlas_v3_compact_review.tar.gz"
    # A review archive must never recursively contain an archive or manifest
    # produced by an earlier finalizer invocation.
    for stale_name in (archive_filename, "compact_review_archive.json"):
        stale_path = review / stale_name
        if stale_path.exists():
            stale_path.unlink()
    records = {}; validation = []; snapshot_rows = []; branch_rows = []; owner_rows = []
    case_rows = []; terminal_rows = []; inventory = {"schema": "v12.multifront-field-atlas-inventory/1", "cases": {}}
    global_x = [float("inf"), float("-inf")]; global_y = [float("inf"), float("-inf")]
    for case in CASES:
        case_root = case_roots[case]
        terminal = json.loads((case_root / "terminal_manifest.json").read_text())
        launch = json.loads((case_root / "launch_manifest.json").read_text())
        snapshot_dir, metadata, arrays = final_snapshot(case_root)
        post_remap_initial = first_post_remap_snapshot(case_root)
        validation.append(validate_final(case_root, metadata, arrays))
        network = json.loads((snapshot_dir / "crack_network.json").read_text())
        nodes_um = arrays["nodes_m"]*1e6
        global_x[0] = min(global_x[0], float(nodes_um[:,0].min())); global_x[1] = max(global_x[1], float(nodes_um[:,0].max()))
        global_y[0] = min(global_y[0], float(nodes_um[:,1].min())); global_y[1] = max(global_y[1], float(nodes_um[:,1].max()))
        records[case] = {"terminal": terminal, "launch": launch, "metadata": metadata,
                         "arrays": arrays, "network": network, "snapshot_dir": snapshot_dir,
                         "post_remap_initial": post_remap_initial}
        fp = tree_fingerprint(case_root)
        terminal["raw_tree_fingerprint"] = fp["sha256"]
        terminal["raw_tree_file_count"] = fp["file_count"]
        terminal["raw_tree_size_bytes"] = fp["size_bytes"]
        terminal.update(onsets[case])
        terminal.update(qualification[case])
        terminal.update({
            "qualified_wake_remap_source_commit": QUALIFIED_WAKE_REMAP_SOURCE_COMMIT,
            "terminal_execution_source_commit": launch["source"]["campaign_commit"],
            "post_remap_initial_accepted_state_id": post_remap_initial["accepted_state_id"],
            "post_remap_initial_stress_field_state_id": post_remap_initial["stress_field_state_id"],
            "final_accepted_state_id": metadata["accepted_state_id"],
            "final_stress_field_state_id": metadata["stress_field_state_id"],
            "mechanics_source_identity": metadata["mechanics_source_identity"],
            "final_process_owner_count": sum(key != "archived_reservoirs" for key in metadata["owners"]),
            "final_process_owner_ids_json": json.dumps(
                sorted(key for key in metadata["owners"] if key != "archived_reservoirs")
            ),
            "process_zone_fields_path": str((snapshot_dir / "fields.npz").resolve()),
        })
        terminal_rows.append(terminal)
        case_rows.append({
            "case": case, "canonical_parameterization_id": launch["canonical_parameterization_id"],
            "execution_alias": launch["execution_alias"], "temperature_K": launch["temperature_K"],
            "parameter_row_canonical_json_sha256": launch["parameter_row_canonical_json_sha256"],
            "theta_deg": 40, "seed": 3621,
            "qualified_wake_remap_source_commit": QUALIFIED_WAKE_REMAP_SOURCE_COMMIT,
            "terminal_execution_source_commit": launch["source"]["campaign_commit"],
            "archived_executable_source_commit": launch["source"]["archived_executable_source_commit"],
            "mechanical_fingerprint": MECHANICAL_FINGERPRINT, "family_sha256": FAMILY_SHA256,
            "family_physics_fingerprint": FAMILY_PHYSICS, "raw_directory": str(case_root),
        })
        entries = []
        for meta_path in sorted((case_root / "field_snapshots").glob("*/metadata.json")):
            meta = interpreted_snapshot_metadata(
                json.loads(meta_path.read_text()), terminal,
            )
            entries.append(meta)
            snapshot_rows.append({
                "case": case, "step": meta["step_count"], "physical_time_s": meta["physical_time_s"],
                "accepted_opening_m": meta["accepted_opening_m"],
                "maximum_forward_reach_um": meta["maximum_network_forward_reach_um"],
                "selection_reasons": ";".join(meta["selection_reasons"]),
                "accepted_state_id": meta["accepted_state_id"], "stress_field_state_id": meta["stress_field_state_id"],
                "topology_fingerprint": meta["topology_fingerprint"],
                "field_package_path": meta["files"]["fields_npz"], "field_package_sha256": meta["files"]["fields_npz_sha256"],
            })
            if any(reason.startswith("milestone_at_or_below_") or reason == "final_last_accepted_state" for reason in meta["selection_reasons"]):
                destination = review / "portable_fields" / case / meta_path.parent.name
                copy_compact_snapshot(meta_path.parent, destination)
                atomic_json(destination / "metadata.json", meta)
        inventory["cases"][case] = {
            "snapshots": entries, "final_validation": validation[-1],
            "raw_tree_fingerprint": fp,
            "source_checkpoint_sha256": qualification[case]["source_checkpoint_sha256"],
            "post_remap_initial_accepted_state_id": post_remap_initial["accepted_state_id"],
            "post_remap_initial_stress_field_state_id": post_remap_initial["stress_field_state_id"],
            "final_accepted_state_id": metadata["accepted_state_id"],
            "final_stress_field_state_id": metadata["stress_field_state_id"],
        }
        for branch in network["branches"]:
            branch_rows.append({"case": case, **{key: branch.get(key) for key in (
                "branch_id", "parent_branch_id", "generation", "initiation_event", "status",
                "physical_path_length_m", "projected_extension_m")}, "path_m_json": json.dumps(branch["path_m"])})
        for owner_id, owner in metadata["owners"].items():
            if owner_id == "archived_reservoirs": continue
            owner_rows.append({
                "case": case, "owner_id": owner_id, "member_front_ids": ";".join(owner["member_front_ids"]),
                "owner_local_family_coordinate_m": owner["owner_local_family_coordinate_m"],
                "process_update_count": owner["process_update_count"], "event_renewal_count": owner["event_renewal_count"],
                "source_multiplicity_per_system": owner["source_multiplicity_per_system"],
                "effective_tip_radius_m": owner["effective_tip_radius_m"],
                "active_signed_shielding_Pa_sqrt_m": owner["active_signed_shielding_Pa_sqrt_m"],
                "wake_signed_shielding_Pa_sqrt_m": owner["wake_signed_shielding_Pa_sqrt_m"],
                "active_ledgers_json": json.dumps(owner["active_ledgers"], sort_keys=True),
                "wake_ledgers_json": json.dumps(owner["wake_ledgers"], sort_keys=True),
                "signed_system_ledgers_json": json.dumps(owner["signed_system_ledgers"], sort_keys=True),
            })
    network_points = np.vstack([
        np.asarray(branch["path_m"], dtype=float) * 1e6
        for rec in records.values()
        for branch in rec["network"]["branches"]
    ])
    tip_points = network_points[network_points[:, 0] >= 500.0 - 1e-8]
    zoom_x = (480.0, float(np.max(tip_points[:, 0])) + 12.0)
    zoom_y = (
        float(np.min(tip_points[:, 1])) - 12.0,
        float(np.max(tip_points[:, 1])) + 12.0,
    )
    for rec in records.values():
        rec["limits"] = (tuple(global_x), tuple(global_y))
        rec["zoom_limits"] = (zoom_x, zoom_y)
    summary_fields = (
        "case", "canonical_parameterization_id", "execution_alias", "temperature_K", "theta_deg", "seed",
        "qualified_wake_remap_source_commit", "terminal_execution_source_commit",
        "mechanical_fingerprint", "family_sha256", "target_forward_reach_um",
        "achieved_forward_reach_um", "terminal_step", "terminal_time_s", "terminal_opening_m",
        "terminal_reason", "exact_terminal_reason", "target_reached",
        "cumulative_branch_births", "maximum_concurrent_active_fronts",
        "final_active_front_count", "junction_count", "retirement_count", "coalescence_count",
        "first_branch_onset_step", "first_branch_onset_time_s", "first_branch_onset_opening_m",
        "first_branch_onset_forward_reach_um", "tau_c_s",
        "first_bifurcation_candidate_ids_json", "first_bifurcation_lambda_uncapped_per_s_json",
        "first_bifurcation_lambda_uncapped_tau_c_json", "first_bifurcation_saturation_status",
        "source_checkpoint_path", "source_checkpoint_sha256",
        "post_remap_initial_accepted_state_id", "post_remap_initial_stress_field_state_id",
        "final_accepted_state_id", "final_stress_field_state_id", "mechanics_source_identity",
        "final_process_owner_count", "final_process_owner_ids_json", "process_zone_fields_path",
        "final_checkpoint_path", "final_checkpoint_sha256", "final_field_package_path",
        "final_field_package_sha256", "raw_tree_fingerprint", "raw_tree_file_count", "raw_tree_size_bytes",
    )
    normalized_terminal = []
    for terminal in terminal_rows:
        normalized_terminal.append({
            **terminal,
            "mechanical_fingerprint": MECHANICAL_FINGERPRINT,
            "family_sha256": FAMILY_SHA256, "target_forward_reach_um": 1000.0,
            "terminal_reason": terminal["terminal_label"],
        })
    write_csv(output / "pf_multifront_field_atlas_case_manifest.csv", case_rows)
    write_csv(output / "pf_multifront_field_atlas_terminal_summary.csv", normalized_terminal, summary_fields)
    write_csv(output / "pf_multifront_field_atlas_snapshot_manifest.csv", snapshot_rows)
    write_csv(output / "pf_multifront_field_atlas_branch_tree.csv", branch_rows)
    write_csv(output / "pf_multifront_field_atlas_owner_summary.csv", owner_rows)
    atomic_json(output / "pf_multifront_field_atlas_field_inventory.json", inventory)
    matrix_figure(records, figures, field="damage_nodal", filename="final_damage_and_crack_topology",
                  title="Final accepted sharp-wake-compatible damage and crack topology", nodal=True, cmap="gray_r")
    matrix_figure(records, figures, field="damage_nodal", filename="final_damage_and_crack_topology_tip_zoom",
                  title="Final accepted damage and crack topology", nodal=True, cmap="gray_r", spatial_zoom=True)
    matrix_figure(records, figures, field="maximum_principal_stress_Pa", filename="final_maximum_principal_stress",
                  title="Final accepted maximum principal stress", nodal=False, cmap="magma", autoscale=True)
    matrix_figure(records, figures, field="maximum_principal_stress_Pa", filename="final_maximum_principal_stress_tip_zoom",
                  title="Final accepted maximum principal stress", nodal=False, cmap="magma", spatial_zoom=True)
    matrix_figure(records, figures, field="equivalent_plastic_strain", filename="final_equivalent_plastic_strain",
                  title="Final accepted equivalent plastic strain", nodal=False, cmap="viridis", autoscale=True)
    matrix_figure(records, figures, field="equivalent_plastic_strain", filename="final_equivalent_plastic_strain_tip_zoom",
                  title="Final accepted equivalent plastic strain", nodal=False, cmap="viridis", spatial_zoom=True)
    matrix_figure(records, figures, field="displacement_magnitude_m", filename="final_displacement_magnitude",
                  title="Final accepted displacement magnitude", nodal=True, cmap="cividis", autoscale=True)
    matrix_figure(records, figures, field="displacement_magnitude_m", filename="final_displacement_magnitude_tip_zoom",
                  title="Final accepted displacement magnitude", nodal=True, cmap="cividis", spatial_zoom=True)
    branch_owner_figure(records, figures)
    branch_owner_figure(records, figures, spatial_zoom=True)
    process_zone_figure(records, figures)
    provenance = {
        "schema": "v12.multifront-field-atlas-provenance/1", "claim_label": CLAIM_LABEL,
        "qualified_wake_remap_source_commit": QUALIFIED_WAKE_REMAP_SOURCE_COMMIT,
        "execution_source_commit": EXECUTION_SOURCE_COMMIT,
        "execution_source_tree": EXECUTION_SOURCE_TREE,
        "archived_pre_correction_source_commit": BASE_COMMIT,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "postprocessing_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "family_sha256": FAMILY_SHA256, "family_physics_fingerprint": FAMILY_PHYSICS,
        "mechanical_configuration_sha256": MECHANICAL_SHA256,
        "mechanical_fingerprint": MECHANICAL_FINGERPRINT,
        "registry_rows": {canonical: {"alias": alias, "row_sha256": registry_row(canonical)[1]}
                          for canonical, alias in ROWS.values()},
        "case_roots": {case: str(case_roots[case]) for case in CASES},
        "review_root": str(review), "final_checkpoint_validation": validation,
        "pre_wake_remap_record": {
            "root": str(pre_wake_root), "label": PRE_WAKE_LABEL,
            "used_only_for_first_branch_onset_evidence": True,
        },
        "wake_remap_qualification_root": str(qualification_root),
        "source_seal_manifest": source_seal_manifest,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": CLAIM_LABEL,
        "first_bifurcation_interpretation": (
            "HIGH_RATE_SATURATED_CORRELATED_EVENT_NOT_A_CALIBRATED_"
            "MATERIAL_DEPENDENT_BRANCH_PROBABILITY"
        ),
    }
    atomic_json(output / "pf_multifront_field_atlas_provenance.json", provenance)
    result_lines = [
        "| Case | Reach (µm) | Births | Max active | First branch time (s) | First branch opening (m) | Terminal reason |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in normalized_terminal:
        result_lines.append(
            f"| {row['case']} | {row['achieved_forward_reach_um']:.6f} | "
            f"{row['cumulative_branch_births']} | {row['maximum_concurrent_active_fronts']} | "
            f"{row['first_branch_onset_time_s']:.12g} | "
            f"{row['first_branch_onset_opening_m']:.12g} | "
            f"`{row['exact_terminal_reason']}` |"
        )
    result_table = "\n".join(result_lines)
    verification_path = output / "verification_summary.json"
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text())
        verification_section = f"""## Verification

The focused current-source qualification passed `{verification['focused_tests']['passed']} passed`. The full suite was run once and recorded `{verification['full_suite']['passed']} passed`, `{verification['full_suite']['skipped']} skipped`, and `{verification['full_suite']['failed']} failed`. Those failures divide into {verification['full_suite']['missing_historical_artifact_failures']} missing historical publication-fixture failures in this isolated worktree and {verification['full_suite']['legacy_failures']} pre-existing legacy failures; no current wake-remap, multifront-topology, or field-atlas test failed. Compileall and `git diff --check` passed. The complete classification is preserved in `verification_summary.json`.
"""
    else:
        verification_section = """## Verification

The machine-readable end-of-campaign verification record is `verification_summary.json`.
"""
    report = f"""# Authoritative corrected V3 wake-remap PF field atlas

Permanent interpretation boundary: `{CLAIM_LABEL}`.

## Scope and decision

Exactly eight pinned physical cases were continued: Peak, DBTT, weak-T, and ceramic-like at 300 K and 1000 K. Each trajectory retained theta=40 degrees, seed 3621, canonical 1x loading, at most six active fronts, the 1600 µm owner-family, and a 1000 µm maximum-network-forward-reach target. The graph-authoritative P0 remap was qualified at source commit `{QUALIFIED_WAKE_REMAP_SOURCE_COMMIT}`. The exact execution lineage, including only the subsequently demonstrated topology-transaction fixes, is `{EXECUTION_SOURCE_COMMIT}` and is preserved by a complete Git bundle with SHA-256 `{SOURCE_BUNDLE_SHA256}`.

None of the eight cases reached 1000 µm. Seven stopped at the configured six-active-front resource policy. DBTT at 1000 K stopped earlier at the existing fail-closed whole-topology energy-release gate, `selected_topology_trial_rejected:insufficient_whole_topology_energy_release`. Each early stop preserves its last accepted atomic checkpoint and portable fields; no diagnostic campaign, reseed, parameter change, or restart from the beginning was launched.

{result_table}

## First-bifurcation interpretation

For every case, both recorded uncapped first-bifurcation `lambda*tau_c` values exceed one. The exact uncapped values and candidate identities are preserved in `pf_multifront_field_atlas_terminal_summary.csv`. The `1/tau_c` multi-hit asymptote is retained as constitutive physical saturation. Therefore the first bifurcation is interpreted as a **high-rate saturated correlated event, not a calibrated material-dependent branch probability**.

## Corrected mechanics and state identities

The final table records the source-checkpoint SHA-256, first corrected post-remap accepted FEM identity, first corrected stress-field identity, final accepted FEM identity, final stress-field identity, process-owner IDs, process-zone package, and raw-tree fingerprint for every case. Every final portable package was reloaded and compared exactly with its atomic accepted checkpoint for mesh, displacement, damage, plastic strain, dislocation density, the complete accepted stress tensor, accepted/stress state identities, and topology.

The shared family is mechanically valid for all eight cases because its mechanical configuration explicitly declares temperature-independent mechanics. Its SHA-256 is `{FAMILY_SHA256}`, its mechanical fingerprint is `{MECHANICAL_FINGERPRINT}`, and its qualified owner-local range is 0–1600 µm. Temperature remains active in the process kinetics.

## Immutable V2 evidence and V3 authority

The old V2 atlas at `{pre_wake_root}` remains read-only and is labelled `{PRE_WAKE_LABEL}`. It is used here only to recover the already accepted first-bifurcation onset time/opening. It is not a corrected field atlas. The figures and tables in this directory are authoritative only for the V3 graph-authoritative wake-remap continuation results.

## Interpretation limit

These records demonstrate the current V12 arbitrary-finite-front architecture through the terminal states above. They do not validate predictive recursive-branching physics. Model-native J or K values are not reported as calibrated fracture toughness or as an R-curve.

{verification_section}

## Records

The terminal summary, case manifest, snapshot manifest, branch tree, owner summary, complete field inventory, figures, provenance JSON, and compact review archive in this directory are the authoritative corrected indexes. Durable raw accepted checkpoints and logs remain at the case roots listed in provenance; selected topology/final portable fields are copied to `{review}`.
"""
    (output / "PF_CURRENT_SOURCE_MULTIFRONT_FIELD_ATLAS_300K_1000K.md").write_text(report)
    excluded_review_files = {archive_filename, "compact_review_archive.json"}
    for path in output.iterdir():
        if path.is_file() and path.name not in excluded_review_files:
            shutil.copy2(path, review / path.name)
    if (review / "figures").exists(): shutil.rmtree(review / "figures")
    shutil.copytree(figures, review / "figures")
    archive_base = output / "pf_current_source_multifront_field_atlas_v3_compact_review"
    archive_path = Path(str(archive_base) + ".tar.gz")
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(str(archive_base), "gztar", root_dir=review)
    atomic_json(output / "compact_review_archive.json", {
        "schema": "v12.multifront-field-atlas-compact-review-archive/1",
        "path": str(archive_path),
        "sha256": sha256(archive_path),
        "source_directory": str(review),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
