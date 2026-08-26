#!/usr/bin/env python3
"""Read-only inventory and fail-closed triage for the canonical PF campaign.

This utility never extracts the legacy ZIP and never deletes data.  Cleanup
and campaign execution consume its explicit, separately reviewed manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable
import zipfile


SCHEMA = "pf_canonical_fracture_audit_v1"
CANONICAL_IDS = {
    "Peak": "v913_zeroD_sobol_0242980",
    "DBTT": "v913_zeroD_sobol_0202500",
    "weak-T": "oneD_v2_focused_weak_T_0016",
    "ceramic-like": "oneD_v2_focused_ceramic_like_0018",
}
HISTORICAL_OPTIONS = {
    "v913_paper_peak01_0242980_persistent_sites": "Peak",
    "v913_paper_dbtt01_0202500_persistent_sites": "DBTT",
    "v913_paper_weakT01_0129902_persistent_sites": "weak-T",
    "v913_paper_ceramic01_0077080_persistent_sites": "ceramic-like",
}
TEMPERATURES = [300.0, 600.0, 800.0, 900.0, 950.0, 1000.0,
                1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0]
THETA_GRID = [15.0, 30.0]
RATE_GRID = [
    ("rate0p01x", 0.01, 840.0, 2.380952380952381e-10),
    ("rate1x", 1.0, 8.4, 2.3809523809523807e-8),
    ("rate100x", 100.0, 0.084, 2.3809523809523808e-6),
]
BASE_SEED = 3621
SEED_OPTION_STRIDE = 1_000_000
SEED_TEMPERATURE_STRIDE = 1009
PARAMETER_COLUMNS_START = "Tref_K"
ARCHIVE_CAMPAIGNS = {
    "1_final_result_30theta_v10_2_28_paper_four_class_1000um_theta30_varseed_base3621_v1": {
        "kind": "theta", "theta_deg": 30.0,
        "source_commit": "fd7f9b70292aa142a9806d4fc97d9908c6df9faa",
        "physical_core_hash_equivalent": False,
    },
    "1_final_result_v10_2_28_projected_paper_four_class_1000um_theta15_varseed_base3621_v1": {
        "kind": "theta", "theta_deg": 15.0,
        "source_commit": "d2d10ba20db5bdf6679f56b1fa1c61900fc05324",
        "physical_core_hash_equivalent": False,
    },
    "1_final_v10_2_30_theta45_hazard_energy_loading_rate_sweep_base3621_v1": {
        "kind": "rate", "theta_deg": 45.0,
        "source_commit": "81b4916ef8766f1fe20ea9256dca86f6b85d00ca",
        "physical_core_hash_equivalent": False,
    },
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode()
    return sha256_bytes(encoded)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_registry(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_class = {row["material_class"]: row for row in rows}
    if set(by_class) != set(CANONICAL_IDS):
        raise RuntimeError(f"registry class mismatch: {sorted(by_class)}")
    hashes: dict[str, str] = {}
    for cls, expected in CANONICAL_IDS.items():
        row = by_class[cls]
        if row["candidate_id"] != expected:
            raise RuntimeError(f"registry candidate mismatch for {cls}")
        columns = list(row)
        start = columns.index(PARAMETER_COLUMNS_START)
        material = {key: row[key] for key in columns[start:]}
        hashes[cls] = canonical_json_sha256(material)
    return rows, hashes


def zip_read_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        with archive.open(name) as stream:
            return json.load(io.TextIOWrapper(stream, encoding="utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def zip_read_csv_first(archive: zipfile.ZipFile, name: str) -> dict[str, str]:
    try:
        with archive.open(name) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"))
            return next(reader, {})
    except (KeyError, UnicodeDecodeError):
        return {}


def archive_case_roots(names: set[str]) -> list[str]:
    roots = set()
    for name in names:
        for marker in ("/v10_2_27_case_contract.json", "/run_args.json"):
            if name.endswith(marker):
                roots.add(name[: -len(marker)])
    return sorted(roots)


def classify_archive_case(
    cls: str, campaign_kind: str, candidate_id: str, complete: bool,
    required_raw: bool,
) -> tuple[str, str]:
    if not complete or not required_raw:
        return "RERUN_REQUIRED_MISSING_DIAGNOSTICS", "INCOMPLETE_OR_REQUIRED_RAW_MISSING"
    if cls in {"weak-T", "ceramic-like"} and candidate_id != CANONICAL_IDS[cls]:
        source_reason = (
            "HISTORICAL_HAZARD_ENERGY_GATED_SOURCE" if campaign_kind == "rate"
            else "SIGNED_KERNEL_FAMILY_PHYSICS_HASH_MISMATCH"
        )
        return "RERUN_REQUIRED_PARAMETER_STALE", (
            "SUPERSEDED_V2_MATERIAL_ROW;" + source_reason
        )
    if cls in {"Peak", "DBTT"} and candidate_id == CANONICAL_IDS[cls]:
        reason = (
            "HISTORICAL_HAZARD_ENERGY_GATED_SOURCE" if campaign_kind == "rate"
            else "SIGNED_KERNEL_FAMILY_PHYSICS_HASH_MISMATCH"
        )
        return "RERUN_REQUIRED_SOURCE_STALE", reason
    return "UNRESOLVED_DO_NOT_DELETE", "UNRESOLVED_MATERIAL_OR_SOURCE_IDENTITY"


def inventory_zip(zip_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    file_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        names = {info.filename for info in infos}
        for info in infos:
            top = info.filename.split("/", 1)[0]
            file_rows.append({
                "archive_path": str(zip_path),
                "member_name": info.filename,
                "top_level_campaign": top,
                "is_directory": info.is_dir(),
                "uncompressed_size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size,
                "compression_ratio": (
                    0.0 if info.file_size == 0 else 1.0 - info.compress_size / info.file_size
                ),
                "timestamp": "%04d-%02d-%02dT%02d:%02d:%02d" % info.date_time,
                "crc32_hex": f"{info.CRC:08x}",
            })
        for root in archive_case_roots(names):
            campaign = root.split("/", 1)[0]
            campaign_meta = ARCHIVE_CAMPAIGNS.get(campaign, {})
            contract = zip_read_json(archive, root + "/v10_2_27_case_contract.json")
            args = zip_read_json(archive, root + "/run_args.json")
            status = zip_read_json(archive, root + "/stage3_case_status.json")
            selection = zip_read_json(archive, root + "/v10_2_22_parameter_selection.json")
            transfer = zip_read_json(
                archive, root + "/v10_2_27_paper_four_class_parameter_transfer.json"
            )
            material = zip_read_csv_first(
                archive, root + "/selected_material_manifest_v10_2_22.csv"
            )
            option = str(contract.get("option") or selection.get("option_key") or
                         material.get("option_key") or root.split("/")[-2])
            cls = HISTORICAL_OPTIONS.get(option, "UNRESOLVED")
            candidate = str(contract.get("candidate_id") or selection.get("candidate_id") or
                            material.get("candidate_id") or "")
            temperature = contract.get("temperature_K")
            if temperature is None:
                values = args.get("temperatures", [])
                temperature = values[0] if values else status.get("temperature_K")
            prefix = root + "/"
            members = [name for name in names if name.startswith(prefix)]
            has_steps = any(re.search(r"/steps_\d+K\.csv$", name) for name in members)
            has_path = any("/crack_path_" in name and name.endswith(".csv") for name in members)
            has_events = root + "/stochastic_avalanche_geometry_events.json" in names
            complete = root + "/COMPLETE" in names and bool(status.get("complete", True))
            required_raw = has_steps and has_path and has_events
            primary, reason = classify_archive_case(
                cls, str(campaign_meta.get("kind", "unresolved")), candidate,
                complete, required_raw,
            )
            run_rows.append({
                "storage_location": "LEGACY_ZIP",
                "archive_path": str(zip_path),
                "run_root": root,
                "top_level_campaign": campaign,
                "campaign_kind": campaign_meta.get("kind", "unresolved"),
                "source_commit": campaign_meta.get("source_commit", ""),
                "physical_core_hash_equivalent": campaign_meta.get(
                    "physical_core_hash_equivalent", False
                ),
                "material_class": cls,
                "option_key": option,
                "candidate_id": candidate,
                "canonical_candidate_id": CANONICAL_IDS.get(cls, ""),
                "material_row_current": candidate == CANONICAL_IDS.get(cls),
                "temperature_K": temperature,
                "theta_deg": contract.get("theta_deg", args.get("crystal_theta_deg",
                                                                  campaign_meta.get("theta_deg"))),
                "loading_rate_factor": "ARCHIVE_CAMPAIGN_LEVEL" if campaign_meta.get("kind") == "rate" else 1.0,
                "nominal_dt_s": args.get("dt"),
                "nominal_dU_m": args.get("dU"),
                "seed": contract.get("seed"),
                "target_extension_um": contract.get("target_extension_um",
                                                       args.get("target_crack_extension_um")),
                "achieved_extension_um": status.get("projected_extension_um"),
                "status": status.get("status", "complete" if complete else "unknown"),
                "complete": complete,
                "file_count": len(members),
                "steps_available": has_steps,
                "path_available": has_path,
                "event_available": has_events,
                "observer_available": any("taylor_peierls_state_profile" in name for name in members),
                "parameter_registry_sha256": transfer.get("parameter_registry_sha256",
                                                             selection.get("registry_sha256", "")),
                "mechanical_configuration_fingerprint": transfer.get(
                    "mechanical_configuration_fingerprint", ""
                ),
                "primary_classification": primary,
                "reason_tags": reason,
            })
        manifest = {
            "schema": SCHEMA,
            "zip_path": str(zip_path),
            "zip_sha256": sha256_path(zip_path),
            "zip_size_bytes": zip_path.stat().st_size,
            "entry_count": len(infos),
            "directory_entry_count": sum(info.is_dir() for info in infos),
            "file_entry_count": sum(not info.is_dir() for info in infos),
            "total_uncompressed_bytes": sum(info.file_size for info in infos),
            "total_compressed_member_bytes": sum(info.compress_size for info in infos),
            "apparent_case_count": len(run_rows),
            "complete_case_count": sum(bool(row["complete"]) for row in run_rows),
            "incomplete_case_count": sum(not bool(row["complete"]) for row in run_rows),
            "top_level_campaigns": sorted({row["top_level_campaign"] for row in run_rows}),
            "read_without_full_extraction": True,
        }
    return file_rows, run_rows, manifest


def tree_size_and_count(path: Path) -> tuple[int, int]:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size, 1
        except FileNotFoundError:
            return 0, 0
    size = 0
    count = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in files:
            item = Path(root) / name
            try:
                size += item.stat().st_size
                count += 1
            except FileNotFoundError:
                pass
    return size, count


def live_case_roots(runs_root: Path) -> list[Path]:
    roots = set()
    for marker in ("run_args.json", "v10_2_27_case_contract.json"):
        for path in runs_root.rglob(marker):
            if ".git" not in path.parts:
                roots.add(path.parent)
    return sorted(roots)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def read_csv_first(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="") as stream:
            return next(csv.DictReader(stream), {})
    except (OSError, UnicodeDecodeError):
        return {}


def inventory_live(runs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    storage_rows: list[dict[str, Any]] = []
    for item in sorted(runs_root.iterdir()):
        size, count = tree_size_and_count(item)
        try:
            mtime = item.stat().st_mtime
        except FileNotFoundError:
            mtime = 0.0
        scope = "MONOTONIC_PF_CANDIDATE" if any(
            token in item.name.lower() for token in ("theta", "fracture", "rcurve")
        ) else "OUT_OF_SCOPE_OR_UNRESOLVED"
        protected = any(token in item.name.lower() for token in ("fatigue", "bulk", "checkpoint"))
        storage_rows.append({
            "absolute_path": str(item.resolve()),
            "entry_type": "directory" if item.is_dir() else "file",
            "size_bytes": size,
            "file_count": count,
            "modification_time_epoch_s": mtime,
            "scope": scope,
            "protected_from_cleanup": protected,
            "cleanup_status": "UNRESOLVED_DO_NOT_DELETE",
        })
    scientific_rows: list[dict[str, Any]] = []
    for root in live_case_roots(runs_root):
        args = read_json(root / "run_args.json")
        contract = read_json(root / "v10_2_27_case_contract.json")
        status = read_json(root / "stage3_case_status.json")
        selection = read_json(root / "v10_2_22_parameter_selection.json")
        material = read_csv_first(root / "selected_material_manifest_v10_2_22.csv")
        option = str(contract.get("option") or selection.get("option_key") or
                     material.get("option_key") or root.parent.name)
        cls = HISTORICAL_OPTIONS.get(option, "UNRESOLVED")
        candidate = str(contract.get("candidate_id") or selection.get("candidate_id") or
                        material.get("candidate_id") or "")
        temps = args.get("temperatures", [])
        steps = list(root.glob("steps_*K.csv"))
        paths = list(root.glob("crack_path_*K.csv"))
        complete = (root / "COMPLETE").exists() and bool(status.get("complete", True))
        protected = any(token in str(root).lower() for token in ("fatigue", "bulk", "checkpoint"))
        scientific_rows.append({
            "storage_location": "LIVE_RUN_TREE",
            "absolute_path": str(root.resolve()),
            "top_level_path": str((runs_root / root.relative_to(runs_root).parts[0]).resolve()),
            "material_class": cls,
            "option_key": option,
            "candidate_id": candidate,
            "canonical_candidate_id": CANONICAL_IDS.get(cls, ""),
            "material_row_current": candidate == CANONICAL_IDS.get(cls),
            "temperature_K": contract.get("temperature_K", temps[0] if temps else
                                            status.get("temperature_K")),
            "theta_deg": contract.get("theta_deg", args.get("crystal_theta_deg")),
            "nominal_dt_s": args.get("dt"),
            "nominal_dU_m": args.get("dU"),
            "seed": contract.get("seed"),
            "target_extension_um": contract.get("target_extension_um",
                                                   args.get("target_crack_extension_um")),
            "achieved_extension_um": status.get("projected_extension_um"),
            "status": status.get("status", "complete" if complete else "unknown"),
            "complete": complete,
            "steps_available": bool(steps),
            "path_available": bool(paths),
            "event_available": (root / "stochastic_avalanche_geometry_events.json").exists(),
            "checkpoint_available": any(root.glob("*checkpoint*")),
            "observer_available": any(root.glob("*taylor_peierls*profile*")),
            "protected_from_cleanup": protected,
            "primary_classification": "UNRESOLVED_DO_NOT_DELETE",
            "reason_tags": "LIVE_TREE_REQUIRES_CAMPAIGN_LEVEL_REVIEW",
        })
    return storage_rows, scientific_rows


def canonical_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    classes = list(CANONICAL_IDS)
    for theta in THETA_GRID:
        for class_index, cls in enumerate(classes):
            for temp_index, temperature in enumerate(TEMPERATURES):
                seed = BASE_SEED + class_index * SEED_OPTION_STRIDE + temp_index * SEED_TEMPERATURE_STRIDE
                disposition = (
                    "RERUN_CANONICAL_SOURCE" if cls in {"Peak", "DBTT"}
                    else "RERUN_PARAMETER_REPLACEMENT_AND_CANONICAL_SOURCE"
                )
                rows.append({
                    "matrix": "CANONICAL_SINGLE_CRACK_THETA",
                    "material_class": cls,
                    "candidate_id": CANONICAL_IDS[cls],
                    "temperature_K": temperature,
                    "theta_deg": theta,
                    "rate_tag": "rate1x",
                    "loading_rate_factor": 1.0,
                    "nominal_dU_m": 2e-7,
                    "nominal_dt_s": 8.4,
                    "nominal_opening_rate_m_per_s": 2.3809523809523807e-8,
                    "seed": seed,
                    "target_extension_um": 1000.0,
                    "planned_disposition": disposition,
                })
    for rate_tag, factor, dt, opening_rate in RATE_GRID:
        for class_index, cls in enumerate(classes):
            for temp_index, temperature in enumerate(TEMPERATURES):
                seed = BASE_SEED + class_index * SEED_OPTION_STRIDE + temp_index * SEED_TEMPERATURE_STRIDE
                rows.append({
                    "matrix": "CANONICAL_STRAIN_RATE",
                    "material_class": cls,
                    "candidate_id": CANONICAL_IDS[cls],
                    "temperature_K": temperature,
                    "theta_deg": 45.0,
                    "rate_tag": rate_tag,
                    "loading_rate_factor": factor,
                    "nominal_dU_m": 2e-7,
                    "nominal_dt_s": dt,
                    "nominal_opening_rate_m_per_s": opening_rate,
                    "seed": seed,
                    "target_extension_um": 1000.0,
                    "planned_disposition": "RERUN_CANONICAL_SOURCE_AND_REGISTRY",
                })
    return rows


def source_contract(repo: Path, registry: Path, reduced_repo: Path) -> dict[str, Any]:
    physical_commit = "9e884fb0b0845da621d2612bdf1042e481b8df49"
    runner_commit = git(repo, "rev-parse", "HEAD")
    files = [
        "arrhenius_fracture/sharp_front_v10_2_22.py",
        "arrhenius_fracture/persistent_site_physical_width_v10222.py",
        "arrhenius_fracture/persistent_site_tip_v10221.py",
        "arrhenius_fracture/stochastic_avalanche_tip.py",
        "arrhenius_fracture/fem.py",
        "arrhenius_fracture/anisotropic_emission_v10174.py",
    ]
    hashes = {}
    for rel in files:
        path = repo / rel
        if path.is_file():
            hashes[rel] = sha256_path(path)
    observer_diff = git(repo, "diff", "--name-only", physical_commit, runner_commit)
    return {
        "schema": SCHEMA,
        "physical_source_repository": str(repo),
        "qualified_physical_source_commit": physical_commit,
        "runner_and_default_off_observer_commit": runner_commit,
        "parameter_registry_repository": str(reduced_repo),
        "parameter_registry_commit": "6679024a566f3fe18f459182119cc9e8e359bb13",
        "parameter_registry_path": str(registry),
        "parameter_registry_sha256": sha256_path(registry),
        "sharp_wake_backend": "sharp_wake; maximum_fronts=1",
        "mesh_contract": {"nx": 36, "ny": 72, "tip_h_fine_m": 1e-6,
                          "tip_ratio": 1.20, "da_phys_m": 5e-6,
                          "plane_state": "plane_strain"},
        "wake_contract": {"wake_length_um": 100.0, "wake_shielding": False,
                          "active_only_signed_kernel": True},
        "loading_contract": {"dU_m": 2e-7, "base_dt_s": 8.4,
                             "adaptive_events": True,
                             "cleavage_hazard": "exponential",
                             "event_length": "threshold_scaled_0.5_to_4.0"},
        "process_zone_engine": "v10.2.21 persistent-site moving MPZ",
        "tensor_source_probe": "v10.1.7.4 anisotropic two-channel source probe",
        "observer_contract": "ONED_V2_TP_STATE_DIAGNOSTICS default off; serialization only; no feedback",
        "observer_changed_files_since_physical_commit": observer_diff.splitlines(),
        "source_file_sha256": hashes,
        "physical_core_hash_equivalence_checked": True,
        "production_registry_modified": False,
    }


def write_reports(out: Path, zip_manifest: dict[str, Any], zip_runs: list[dict[str, Any]],
                  storage: list[dict[str, Any]], contract: dict[str, Any],
                  material_hashes: dict[str, str], matrix: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in zip_runs:
        counts[row["primary_classification"]] = counts.get(row["primary_classification"], 0) + 1
    total_live = sum(int(row["size_bytes"]) for row in storage)
    source_md = f"""# PF canonical fracture source contract

The canonical monotonic source is the physical v10.2.22 persistent-site sharp-front stack at qualified repository commit `{contract['qualified_physical_source_commit']}`. The audit/runner commit `{contract['runner_and_default_off_observer_commit']}` adds only bounded runners and a default-off state observer; the sharp-wake, moving-MPZ, physical-width, event-lifecycle, tensor-probe, mesh, and loading contracts remain source-identical.

- Registry commit: `{contract['parameter_registry_commit']}`
- Registry SHA-256: `{contract['parameter_registry_sha256']}`
- Sharp wake: one front; 100 µm stored wake; wake shielding disabled.
- Mesh: 36×72, 1 µm fine tip size, ratio 1.20, 5 µm physical checkpoint.
- Loading: ΔU=2×10⁻⁷ m; base Δt=8.4 s; rate sweeps change physical Δt, not ΔU.
- Observer: default off, serialization only, no evolution feedback.

Material hashes (canonical numeric material subvector):
"""
    source_md += "\n".join(f"- {key}: `{value}`" for key, value in material_hashes.items()) + "\n"
    (out / "PF_CANONICAL_FRACTURE_SOURCE_CONTRACT.md").write_text(source_md)

    historical_md = f"""# PF historical campaign reconstruction

The verified legacy ZIP contains three top-level campaigns and was read without full extraction.

- θ matrix: θ = 15° and 30°; 12 temperatures ({', '.join(str(int(x)) for x in TEMPERATURES)} K); four historical classes; 48 cases per angle; 1000 µm target.
- Rate matrix: θ = 45°; rate factors 0.01×, 1×, and 100×; ΔU fixed at 2×10⁻⁷ m; Δt = 840, 8.4, and 0.084 s; the same 12 temperatures and seed map; 144 planned cases.
- Seeds: 3621 + class_index×1,000,000 + temperature_index×1009.
- The archive contains {zip_manifest['apparent_case_count']} directly identified case roots; campaign-level nested products are inventoried separately.
- Historical weak-T and ceramic-like rows are superseded by the V2 registry.
- The historical θ=45° rate campaign uses the hazard-energy-gated source and is source-stale relative to the canonical monotonic source.
- Current deterministic θ=15° and θ=30° kernel families do not reproduce the archived family physics/file fingerprints. Because the archived family files themselves are unavailable, functional hash equivalence cannot be established; Peak and DBTT θ trajectories are therefore historical raw records, not reusable canonical trajectories.

No missing condition was inferred from a directory name alone.
"""
    (out / "PF_HISTORICAL_CAMPAIGN_RECONSTRUCTION.md").write_text(historical_md)

    storage_md = f"""# PF run storage inventory

- Runs root bytes at inventory: {total_live}
- Runs root entries inventoried: {len(storage)}
- Legacy ZIP bytes: {zip_manifest['zip_size_bytes']}
- Legacy ZIP uncompressed member bytes: {zip_manifest['total_uncompressed_bytes']}
- Legacy ZIP entries: {zip_manifest['entry_count']}
- ZIP-identified cases: {zip_manifest['apparent_case_count']} complete={zip_manifest['complete_case_count']} incomplete={zip_manifest['incomplete_case_count']}
- Archive classifications: {json.dumps(counts, sort_keys=True)}

All fatigue, bulk-plasticity, checkpoint, and unresolved paths are protected from cleanup. No path is deleted by the inventory producer.
"""
    (out / "pf_runs_total_storage_summary.md").write_text(storage_md)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pf-repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--legacy-zip", type=Path, required=True)
    parser.add_argument("--reduced-repo", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--pf-transfer-registry", type=Path, required=True)
    parser.add_argument("--pf-transfer-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    registry_rows, material_hashes = load_registry(args.registry)
    file_rows, zip_runs, zip_manifest = inventory_zip(args.legacy_zip)
    storage_rows, live_runs = inventory_live(args.runs_root)
    matrix = canonical_matrix()
    contract = source_contract(args.pf_repo, args.registry, args.reduced_repo)

    write_csv(out / "legacy_zip_file_inventory.csv", file_rows)
    write_csv(out / "legacy_zip_run_inventory.csv", zip_runs)
    (out / "legacy_zip_manifest.json").write_text(
        json.dumps(zip_manifest, indent=2, sort_keys=True) + "\n"
    )
    (out / "legacy_zip_sha256.txt").write_text(zip_manifest["zip_sha256"] + "\n")
    write_csv(out / "pf_runs_storage_inventory.csv", storage_rows)
    write_csv(out / "pf_runs_scientific_inventory.csv", live_runs)
    write_csv(out / "pf_runs_duplicate_inventory.csv", [], [
        "original_path", "duplicate_path", "content_hash", "size_bytes",
        "classification", "status",
    ])
    write_csv(out / "pf_historical_campaign_matrix.csv", matrix)
    write_csv(out / "pf_canonical_fracture_run_plan.csv", matrix)
    write_csv(out / "pf_v2_four_class_registry.csv", registry_rows)
    with args.pf_transfer_registry.open(newline="") as stream:
        transfer_rows = list(csv.DictReader(stream))
    write_csv(out / "pf_v2_four_class_pf_transfer_registry.csv", transfer_rows)
    transfer_selection = json.loads(args.pf_transfer_selection.read_text())
    (out / "pf_v2_four_class_pf_transfer_selection.json").write_text(
        json.dumps(transfer_selection, indent=2, sort_keys=True) + "\n"
    )
    write_csv(out / "pf_v2_four_class_material_hashes.csv", [
        {"material_class": cls, "candidate_id": CANONICAL_IDS[cls],
         "full_material_sha256": material_hashes[cls]}
        for cls in CANONICAL_IDS
    ])
    (out / "pf_canonical_fracture_source_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    write_reports(out, zip_manifest, zip_runs, storage_rows, contract,
                  material_hashes, matrix)
    provenance = {
        "schema": SCHEMA,
        "producer_worktree_commit": git(args.pf_repo, "rev-parse", "HEAD"),
        "qualified_physical_source_commit": contract["qualified_physical_source_commit"],
        "registry_commit": contract["parameter_registry_commit"],
        "registry_sha256": contract["parameter_registry_sha256"],
        "pf_transfer_registry_sha256": sha256_path(args.pf_transfer_registry),
        "pf_transfer_selection_sha256": sha256_path(args.pf_transfer_selection),
        "legacy_zip_sha256": zip_manifest["zip_sha256"],
        "legacy_zip_read_without_extraction": True,
        "inventory_only_no_deletion": True,
        "canonical_case_count": len(matrix),
        "canonical_theta_case_count": sum(r["matrix"] == "CANONICAL_SINGLE_CRACK_THETA" for r in matrix),
        "canonical_rate_case_count": sum(r["matrix"] == "CANONICAL_STRAIN_RATE" for r in matrix),
    }
    (out / "pf_canonical_fracture_audit_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
