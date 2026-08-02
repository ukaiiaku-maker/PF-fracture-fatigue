#!/usr/bin/env python3
"""Export bounded local evidence for the v10.4.2 external audit.

The simulation trees live under /Volumes/Data and are intentionally not stored
in Git.  This script packages the audit-critical CSV/JSON/text files, all reuse
audits, directory inventories, Git metadata, process information, and bounded
log excerpts into one ZIP with SHA-256 hashes.

The exporter never modifies a run directory.  By default, large logs are stored
as head/tail excerpts while their full-file hashes and sizes are recorded.  Pass
``--include-full-logs`` only when the resulting archive size is acceptable.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable
import zipfile


DEFAULT_WORKTREE = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_4_2_plastic_flow_terminal"
)
DEFAULT_RUNS = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs"
)
DEFAULT_SMOKE_CASE = DEFAULT_RUNS / (
    "v10_4_2_DBTT_1000K_positiveJ_20um_smoke_seed1008666_v2/"
    "v913_paper_dbtt01_0202500_persistent_sites/"
    "T1000K_th0_seed2008666"
)
DEFAULT_SOURCE_CAMPAIGN = DEFAULT_RUNS / (
    "v10_4_1_theta0_rate1x_bulk_PT_four_class_1000um_"
    "selective_reuse_base3621_v1"
)
DEFAULT_PRODUCTION_ROOT = DEFAULT_RUNS / (
    "v10_4_2_theta0_rate1x_bulk_PT_positiveJ_plastic_terminal_"
    "four_class_1000um_reuse17_base3621_v1"
)
DEFAULT_QUARANTINE_ROOT = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/quarantine/"
    "v10_4_2_pre_reuse_scheduler_fix_20260802_074842"
)

SMOKE_FILES = (
    "fronts_1000K.csv",
    "steps_1000K.csv",
    "stage3_case_status.json",
    "summary.json",
    "stochastic_avalanche_geometry_events.json",
    "v10_2_30_hazard_energy_gate_audit.json",
    "v10_2_27_energy_ledger_output_audit.json",
    "v10_4_bulk_peierls_taylor_coupling_audit.json",
    "v10_4_bulk_coupled_model_audit.json",
    "v10_4_1_bulk_detailed_balance_audit.json",
    "command.sh",
    "run.log",
    "COMPLETE",
    "PLASTIC_FLOW",
    "RUN_FAILED",
)

SOURCE_FILES = (
    "v10_4_2_positive_directional_J_compatibility_report.json",
)

PRODUCTION_ROOT_NAMES = (
    "v10_4_2_materialized_reuse_manifest.json",
    "v10_4_2_bulk_plastic_flow_campaign_lock.json",
    "v10_2_28_campaign_kernel_lock.json",
    "v10_2_28_kernel_resolution.json",
    "v10_2_27_case_seed_map.csv",
    "v10_2_27_campaign_manifest.json",
)

QUARANTINE_CASE_NAMES = (
    "command.sh",
    "run.log",
    "stage3_case_status.json",
    "summary.json",
    "RUN_FAILED",
    "COMPLETE",
    "PLASTIC_FLOW",
)

LOG_SUFFIXES = (".log",)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=None if cwd is None else str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        return {
            "args": args,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "args": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        try:
            return str(path.relative_to(root))
        except ValueError:
            return path.name


def archive_name(section: str, path: Path, root: Path) -> str:
    relative = safe_relative(path, root).replace(os.sep, "/")
    return f"{section}/{relative}"


def log_excerpt(path: Path, head_lines: int, tail_lines: int) -> bytes:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return (text + ("\n" if text and not text.endswith("\n") else "")).encode()
    omitted = len(lines) - head_lines - tail_lines
    output = [
        *lines[:head_lines],
        "",
        f"--- {omitted} lines omitted by audit exporter ---",
        "",
        *lines[-tail_lines:],
        "",
    ]
    return "\n".join(output).encode()


def is_log(path: Path) -> bool:
    return path.name.endswith(LOG_SUFFIXES) or path.name in {"run.log"}


def add_path(
    zf: zipfile.ZipFile,
    *,
    source: Path,
    arcname: str,
    manifest: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    include_full_logs: bool,
    head_lines: int,
    tail_lines: int,
    required: bool = False,
) -> None:
    entry: dict[str, Any] = {
        "source": str(source),
        "archive_name": arcname,
        "required": required,
    }

    if not source.exists() and not source.is_symlink():
        entry["status"] = "missing"
        missing.append(entry)
        return

    entry["is_symlink"] = source.is_symlink()
    if source.is_symlink():
        try:
            entry["symlink_target"] = os.readlink(source)
        except OSError as exc:
            entry["symlink_target_error"] = f"{type(exc).__name__}: {exc}"

    if source.is_dir():
        entry["status"] = "directory_not_embedded"
        manifest.append(entry)
        return

    try:
        size = source.stat().st_size
        digest = sha256_file(source)
    except OSError as exc:
        entry["status"] = "unreadable"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        missing.append(entry)
        return

    entry["size_bytes"] = size
    entry["sha256"] = digest

    if is_log(source) and not include_full_logs:
        data = log_excerpt(source, head_lines, tail_lines)
        excerpt_arcname = arcname + ".excerpt.txt"
        zf.writestr(excerpt_arcname, data)
        entry["status"] = "excerpt_embedded"
        entry["archive_name"] = excerpt_arcname
        entry["excerpt_sha256"] = sha256_bytes(data)
        entry["full_file_embedded"] = False
    else:
        zf.write(source, arcname)
        entry["status"] = "full_file_embedded"
        entry["full_file_embedded"] = True

    manifest.append(entry)


def inventory(root: Path, max_entries: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": str(root),
        "exists": root.exists(),
        "entries": [],
        "truncated": False,
    }
    if not root.exists():
        return result

    entries: list[dict[str, Any]] = []
    for index, path in enumerate(sorted(root.rglob("*"))):
        if index >= max_entries:
            result["truncated"] = True
            break
        item: dict[str, Any] = {
            "path": safe_relative(path, root),
            "is_symlink": path.is_symlink(),
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
        }
        if path.is_symlink():
            try:
                item["symlink_target"] = os.readlink(path)
            except OSError as exc:
                item["symlink_target_error"] = f"{type(exc).__name__}: {exc}"
        if path.is_file():
            try:
                item["size_bytes"] = path.stat().st_size
            except OSError:
                pass
        entries.append(item)
    result["entries"] = entries
    result["entry_count_recorded"] = len(entries)
    return result


def iter_existing(root: Path, names: Iterable[str]) -> Iterable[Path]:
    for name in names:
        path = root / name
        if path.exists() or path.is_symlink():
            yield path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--worktree", type=Path, default=DEFAULT_WORKTREE)
    result.add_argument("--smoke-case", type=Path, default=DEFAULT_SMOKE_CASE)
    result.add_argument(
        "--source-campaign", type=Path, default=DEFAULT_SOURCE_CAMPAIGN
    )
    result.add_argument(
        "--production-root", type=Path, default=DEFAULT_PRODUCTION_ROOT
    )
    result.add_argument(
        "--quarantine-root", type=Path, default=DEFAULT_QUARANTINE_ROOT
    )
    result.add_argument("--output", type=Path)
    result.add_argument("--include-full-logs", action="store_true")
    result.add_argument("--head-lines", type=int, default=250)
    result.add_argument("--tail-lines", type=int, default=500)
    result.add_argument("--max-inventory-entries", type=int, default=20000)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output
    if output is None:
        output = Path.cwd() / f"v10_4_2_audit_bundle_{timestamp}.zip"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    worktree = args.worktree.expanduser().resolve()
    smoke = args.smoke_case.expanduser().resolve()
    source_campaign = args.source_campaign.expanduser().resolve()
    production = args.production_root.expanduser().resolve()
    quarantine = args.quarantine_root.expanduser().resolve()

    manifest: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    metadata: dict[str, Any] = {
        "schema": "v10.4.2_external_audit_bundle_v1",
        "created_local": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "include_full_logs": bool(args.include_full_logs),
        "roots": {
            "worktree": str(worktree),
            "smoke_case": str(smoke),
            "source_campaign": str(source_campaign),
            "production_root": str(production),
            "quarantine_root": str(quarantine),
        },
        "git": {
            "head": run_command(["git", "rev-parse", "HEAD"], worktree),
            "branch": run_command(
                ["git", "branch", "--show-current"], worktree
            ),
            "status": run_command(
                ["git", "status", "--short"], worktree
            ),
            "remote": run_command(
                ["git", "remote", "-v"], worktree
            ),
        },
        "processes": run_command(["ps", "-axo", "pid=,ppid=,etime=,command="]),
    }

    inventories = {
        "smoke_case": inventory(smoke, args.max_inventory_entries),
        "source_campaign": inventory(
            source_campaign, args.max_inventory_entries
        ),
        "production_root": inventory(production, args.max_inventory_entries),
        "quarantine_root": inventory(quarantine, args.max_inventory_entries),
    }

    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        # Corrected smoke evidence.
        for path in iter_existing(smoke, SMOKE_FILES):
            add_path(
                zf,
                source=path,
                arcname=archive_name("smoke_case", path, smoke),
                manifest=manifest,
                missing=missing,
                include_full_logs=args.include_full_logs,
                head_lines=args.head_lines,
                tail_lines=args.tail_lines,
                required=path.name in {
                    "fronts_1000K.csv",
                    "stage3_case_status.json",
                    "v10_2_30_hazard_energy_gate_audit.json",
                },
            )

        # Compatibility report.
        for path in iter_existing(source_campaign, SOURCE_FILES):
            add_path(
                zf,
                source=path,
                arcname=archive_name(
                    "source_campaign", path, source_campaign
                ),
                manifest=manifest,
                missing=missing,
                include_full_logs=args.include_full_logs,
                head_lines=args.head_lines,
                tail_lines=args.tail_lines,
                required=True,
            )

        # Production-root manifests, logs, PID files, and all per-case reuse audits.
        production_candidates: set[Path] = set(
            iter_existing(production, PRODUCTION_ROOT_NAMES)
        )
        if production.exists():
            production_candidates.update(production.glob("*.log"))
            production_candidates.update(production.glob("*.pid"))
            production_candidates.update(
                production.rglob("v10_4_2_reuse_audit.json")
            )
            production_candidates.update(
                production.rglob("stage3_case_status.json")
            )
            production_candidates.update(production.rglob("RUN_FAILED"))
            production_candidates.update(production.rglob("COMPLETE"))
            production_candidates.update(production.rglob("PLASTIC_FLOW"))

        # The launcher logs are siblings of OUTROOT, not children.
        production_candidates.update(production.parent.glob(production.name + "*.log"))
        production_candidates.update(production.parent.glob(production.name + "*.pid"))

        for path in sorted(production_candidates):
            root = production if path == production or production in path.parents else production.parent
            add_path(
                zf,
                source=path,
                arcname=archive_name("production", path, root),
                manifest=manifest,
                missing=missing,
                include_full_logs=args.include_full_logs,
                head_lines=args.head_lines,
                tail_lines=args.tail_lines,
                required=path.name == "v10_4_2_materialized_reuse_manifest.json",
            )

        # Partial runs moved out of the production tree.
        quarantine_candidates: set[Path] = set()
        if quarantine.exists():
            for name in QUARANTINE_CASE_NAMES:
                quarantine_candidates.update(quarantine.rglob(name))
        for path in sorted(quarantine_candidates):
            add_path(
                zf,
                source=path,
                arcname=archive_name("quarantine", path, quarantine),
                manifest=manifest,
                missing=missing,
                include_full_logs=args.include_full_logs,
                head_lines=args.head_lines,
                tail_lines=args.tail_lines,
            )

        zf.writestr(
            "metadata/session_metadata.json",
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )
        zf.writestr(
            "metadata/directory_inventories.json",
            json.dumps(inventories, indent=2, sort_keys=True) + "\n",
        )
        zf.writestr(
            "metadata/file_manifest.json",
            json.dumps(
                {
                    "schema": "v10.4.2_external_audit_file_manifest_v1",
                    "files": manifest,
                    "missing_or_unreadable": missing,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    archive_digest = sha256_file(output)
    print(json.dumps({
        "archive": str(output),
        "sha256": archive_digest,
        "embedded_records": len(manifest),
        "missing_or_unreadable": len(missing),
        "include_full_logs": bool(args.include_full_logs),
    }, indent=2))

    required_missing = [item for item in missing if item.get("required")]
    if required_missing:
        print(
            "WARNING: required audit files were missing; inspect "
            "metadata/file_manifest.json",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
