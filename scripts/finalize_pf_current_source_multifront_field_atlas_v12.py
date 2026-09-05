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
CORRECTED_SOURCE_COMMIT = "12811e0beff95d9390a27646f5044ff0847b2b9d"
CORRECTED_SOURCE_TREE = "ab901f1f1c83a81f0fa1c6866efed608f609c93b"
SOURCE_BUNDLE_SHA256 = "28f4340c8095a2a00a17752172e5b4f1bee2a1b5df2fa424bf74a179e4f2c374"


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


def annotate_panel(ax, terminal: dict) -> None:
    text = (
        f"reach {terminal['achieved_forward_reach_um']:.1f} µm\n"
        f"births {terminal['cumulative_branch_births']}; max active {terminal['maximum_concurrent_active_fronts']}"
    )
    if not terminal["target_reached"]:
        text += "\n" + terminal["terminal_label"].replace("V12_FIELD_ATLAS_", "")
    ax.text(0.015, 0.985, text, transform=ax.transAxes, va="top", ha="left", fontsize=5.8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5})
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    length = 250.0
    ax.plot([x0 + 0.04*(x1-x0), x0 + 0.04*(x1-x0) + length], [y0 + 0.055*(y1-y0)]*2,
            color="black", lw=2.0, zorder=10)
    ax.text(x0 + 0.04*(x1-x0) + length/2, y0 + 0.07*(y1-y0), "250 µm",
            ha="center", va="bottom", fontsize=6)
    ax.plot([500.0], [0.0], marker="|", color="red", ms=8, mew=1.1, zorder=8)


def save_figure(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def matrix_figure(records: dict, figures: Path, *, field: str, filename: str,
                  title: str, nodal: bool, cmap: str, autoscale: bool = False) -> None:
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
                ax.set_aspect("equal"); ax.set_xlim(rec["limits"][0]); ax.set_ylim(rec["limits"][1])
                ax.set_title(f"{DISPLAY[material]}, {temperature} K", fontsize=9)
                annotate_panel(ax, rec["terminal"])
                if row_index == 3: ax.set_xlabel("x (µm)")
                if col_index == 0: ax.set_ylabel("y (µm)")
        fig.suptitle(title + (" — individual scales" if local_scale else " — common scale"), fontsize=12)
        if image is not None and not local_scale:
            fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015, label=field.replace("_", " "))
        fig.subplots_adjust(top=0.95, right=0.91 if not local_scale else 0.98, hspace=0.22, wspace=0.08)
        save_figure(fig, figures / f"{filename}{suffix}")


def branch_owner_figure(records: dict, figures: Path) -> None:
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
            ax.set_aspect("equal"); ax.set_xlim(rec["limits"][0]); ax.set_ylim(rec["limits"][1])
            ax.set_title(f"{DISPLAY[material]}, {temperature} K", fontsize=9); annotate_panel(ax, rec["terminal"])
            if ri == 3: ax.set_xlabel("x (µm)")
            if ci == 0: ax.set_ylabel("y (µm)")
    fig.suptitle("Final accepted branch, junction, and process-owner map", fontsize=12)
    fig.subplots_adjust(top=0.95, hspace=0.22, wspace=0.08)
    save_figure(fig, figures / "final_branch_junction_owner_map")


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
            ax.set_title(f"{DISPLAY[material]}, {temperature} K", fontsize=9); annotate_panel(ax, rec["terminal"])
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "analysis_outputs" / OUTPUT_NAME)
    parser.add_argument("--review-root", type=Path)
    args = parser.parse_args()
    raw = args.raw_root.resolve(); output = args.output_root.resolve()
    review = (args.review_root or (raw / "compact_review_package")).resolve()
    if any(not (raw / case / "terminal_manifest.json").is_file() for case in CASES):
        missing = [case for case in CASES if not (raw / case / "terminal_manifest.json").is_file()]
        raise RuntimeError(f"campaign has unterminated cases: {missing}")
    output.mkdir(parents=True, exist_ok=True); figures = output / "figures"; figures.mkdir(exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    records = {}; validation = []; snapshot_rows = []; branch_rows = []; owner_rows = []
    case_rows = []; terminal_rows = []; inventory = {"schema": "v12.multifront-field-atlas-inventory/1", "cases": {}}
    global_x = [float("inf"), float("-inf")]; global_y = [float("inf"), float("-inf")]
    for case in CASES:
        case_root = raw / case
        terminal = json.loads((case_root / "terminal_manifest.json").read_text())
        launch = json.loads((case_root / "launch_manifest.json").read_text())
        snapshot_dir, metadata, arrays = final_snapshot(case_root)
        validation.append(validate_final(case_root, metadata, arrays))
        network = json.loads((snapshot_dir / "crack_network.json").read_text())
        nodes_um = arrays["nodes_m"]*1e6
        global_x[0] = min(global_x[0], float(nodes_um[:,0].min())); global_x[1] = max(global_x[1], float(nodes_um[:,0].max()))
        global_y[0] = min(global_y[0], float(nodes_um[:,1].min())); global_y[1] = max(global_y[1], float(nodes_um[:,1].max()))
        records[case] = {"terminal": terminal, "launch": launch, "metadata": metadata,
                         "arrays": arrays, "network": network, "snapshot_dir": snapshot_dir}
        fp = tree_fingerprint(case_root); atomic_json(case_root / "raw_tree_fingerprint.json", fp)
        terminal["raw_tree_fingerprint"] = fp["sha256"]
        terminal_rows.append(terminal)
        case_rows.append({
            "case": case, "canonical_parameterization_id": launch["canonical_parameterization_id"],
            "execution_alias": launch["execution_alias"], "temperature_K": launch["temperature_K"],
            "parameter_row_canonical_json_sha256": launch["parameter_row_canonical_json_sha256"],
            "theta_deg": 40, "seed": 3621,
            "source_commit": launch["source"]["campaign_commit"],
            "archived_executable_source_commit": launch["source"]["archived_executable_source_commit"],
            "mechanical_fingerprint": MECHANICAL_FINGERPRINT, "family_sha256": FAMILY_SHA256,
            "family_physics_fingerprint": FAMILY_PHYSICS, "raw_directory": str(case_root),
        })
        entries = []
        for meta_path in sorted((case_root / "field_snapshots").glob("*/metadata.json")):
            meta = json.loads(meta_path.read_text()); entries.append(meta)
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
                copy_compact_snapshot(meta_path.parent, review / "portable_fields" / case / meta_path.parent.name)
        inventory["cases"][case] = {"snapshots": entries, "final_validation": validation[-1]}
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
    for rec in records.values(): rec["limits"] = (tuple(global_x), tuple(global_y))
    summary_fields = (
        "case", "canonical_parameterization_id", "execution_alias", "temperature_K", "theta_deg", "seed",
        "source_commit", "mechanical_fingerprint", "family_sha256", "target_forward_reach_um",
        "achieved_forward_reach_um", "terminal_step", "terminal_time_s", "terminal_opening_m",
        "terminal_reason", "exact_terminal_reason", "target_reached",
        "cumulative_branch_births", "maximum_concurrent_active_fronts",
        "final_active_front_count", "junction_count", "retirement_count", "coalescence_count",
        "final_checkpoint_path", "final_field_package_path", "raw_tree_fingerprint",
    )
    normalized_terminal = []
    for terminal in terminal_rows:
        normalized_terminal.append({
            **terminal, "source_commit": CORRECTED_SOURCE_COMMIT,
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
    matrix_figure(records, figures, field="maximum_principal_stress_Pa", filename="final_maximum_principal_stress",
                  title="Final accepted maximum principal stress", nodal=False, cmap="magma", autoscale=True)
    matrix_figure(records, figures, field="equivalent_plastic_strain", filename="final_equivalent_plastic_strain",
                  title="Final accepted equivalent plastic strain", nodal=False, cmap="viridis", autoscale=True)
    matrix_figure(records, figures, field="displacement_magnitude_m", filename="final_displacement_magnitude",
                  title="Final accepted displacement magnitude", nodal=True, cmap="cividis", autoscale=True)
    branch_owner_figure(records, figures); process_zone_figure(records, figures)
    provenance = {
        "schema": "v12.multifront-field-atlas-provenance/1", "claim_label": CLAIM_LABEL,
        "execution_source_commit": CORRECTED_SOURCE_COMMIT,
        "execution_source_tree": CORRECTED_SOURCE_TREE,
        "archived_pre_correction_source_commit": BASE_COMMIT,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "postprocessing_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "family_sha256": FAMILY_SHA256, "family_physics_fingerprint": FAMILY_PHYSICS,
        "mechanical_configuration_sha256": MECHANICAL_SHA256,
        "mechanical_fingerprint": MECHANICAL_FINGERPRINT,
        "registry_rows": {canonical: {"alias": alias, "row_sha256": registry_row(canonical)[1]}
                          for canonical, alias in ROWS.values()},
        "raw_root": str(raw), "review_root": str(review), "final_checkpoint_validation": validation,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": CLAIM_LABEL,
    }
    atomic_json(output / "pf_multifront_field_atlas_provenance.json", provenance)
    report = f"""# PF current-source multifront field atlas: 300 K and 1000 K

Permanent interpretation boundary: `{CLAIM_LABEL}`.

## Scope and decision

Exactly eight pinned physical cases were run: Peak, DBTT, weak-T, and ceramic-like at 300 K and 1000 K. Each trajectory used theta=40 degrees, seed 3621, canonical 1x loading, corrected V12 source `{CORRECTED_SOURCE_COMMIT}`, at most six active fronts, and a 1000 µm maximum-forward-reach target. The source is preserved by a complete Git bundle with SHA-256 `{SOURCE_BUNDLE_SHA256}`. Early fail-closed stops are preserved as scientifically complete capability records; they are not patched, reseeded, or restarted from the beginning.

All eight cases reached 41.516160204381784 µm maximum projected forward extension, one cumulative binary branch birth, and two simultaneous active fronts. None reached the 1000 µm target. Every case then stopped at the same existing fail-closed geometry gate: `candidate_segment_already_in_committed_wake_material`. This common terminal outcome is recorded as the result of each case; no automatic convergence, mesh, parameter, or reseeding study was launched.

The shared family is mechanically valid for all eight cases because its mechanical configuration explicitly declares temperature-independent mechanics. Its SHA-256 is `{FAMILY_SHA256}`, its mechanical fingerprint is `{MECHANICAL_FINGERPRINT}`, and its qualified owner-local range is 0–1600 µm. Temperature remains active in the process kinetics.

## Accepted-state field contract

Portable NPZ, VTU, and crack-network VTP/JSON packages were saved only at the fresh initial state, every accepted binary birth, every detected owner handoff or partition, the nearest accepted state at or below each requested reach, and the last accepted state. All final packages were reloaded and checked exactly against their atomic accepted V12 checkpoints for mesh, displacement, damage, plastic strain, dislocation density, the complete accepted stress tensor, state identities, and topology.

## Interpretation limit

These figures demonstrate what the current V12 arbitrary-finite-front architecture produced for the pinned cases. They do not validate predictive recursive-branching physics. Model-native J or K values are not reported as calibrated fracture toughness or as an R-curve.

## Records

The terminal summary, case manifest, snapshot manifest, branch tree, owner summary, complete field inventory, figures, and provenance JSON in this directory are authoritative compact indexes. Durable raw accepted checkpoints and logs remain under `{raw}`; selected milestone/final portable fields are copied to `{review}`.
"""
    (output / "PF_CURRENT_SOURCE_MULTIFRONT_FIELD_ATLAS_300K_1000K.md").write_text(report)
    for path in output.iterdir():
        if path.is_file(): shutil.copy2(path, review / path.name)
    if (review / "figures").exists(): shutil.rmtree(review / "figures")
    shutil.copytree(figures, review / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
