#!/usr/bin/env python3
"""Build the relocatable original-opening campaign profile from frozen V3."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO.parents[1]
SOURCE = REPO / "common_framework_readiness/PROSPECTIVE_CAMPAIGN_V3.jsonl"
OUT = Path(__file__).resolve().parent
KERNEL = (
    PACKAGE / "state/preservation/pre-final-changes-20261017T231804Z/"
    "external_dependencies/signed_kernel_4fa015d77f1aadf0/family.json"
)
OLD_PF = "/Volumes/Data/working-papers/Fracture/shared-fracture-fatigue-options-v1"
OLD_FEM = "/Volumes/Data/working-papers/Fracture/fem-czm-current-pf-companion"
OLD_KERNEL = (
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/"
    "v10_2_28_kernel_cache/"
    "4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/"
    "family.json"
)
PHASE = {
    "P1": (1, "native_initiation_and_matched_controls"),
    "P3": (2, "state_history_controls"),
    "P4": (2, "orientation_constraint_and_source_support"),
    "P5": (2, "rate_geometry_and_bulk_source_comparisons"),
    "P2": (3, "developed_propagation"),
    "P8": (4, "selective_independent_fem"),
    "P6": (5, "native_branching_and_matched_controls"),
    "P7": (5, "native_voiding_and_matched_controls"),
    "P9": (6, "original_opening_same_material_controls"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if not isinstance(value, str):
        return value
    if value == OLD_KERNEL:
        return "{PACKAGE_ROOT}/" + str(KERNEL.relative_to(PACKAGE))
    if value.startswith(OLD_PF + "/"):
        return "{PF_ROOT}/" + value.removeprefix(OLD_PF + "/")
    if value.startswith(OLD_FEM + "/"):
        return "{FEM_ROOT}/" + value.removeprefix(OLD_FEM + "/")
    return value


def disposition(row: dict) -> tuple[str, list[str]]:
    if row["cohesive_law"]["kind"] != "original_activated_opening":
        return "DEFERRED_OPTIONAL_CZM", ["closed_alternative_czm_fitting_scope"]
    readiness = row["readiness"]
    reasons = []
    if readiness["material_domain"].startswith("BLOCKED_"):
        reasons.append(readiness["material_domain"])
    if readiness["input_availability"].startswith(("PENDING_", "BLOCKED_")):
        reasons.append(readiness["input_availability"])
    if readiness["input_availability"] == "AVAILABLE_QUALIFICATION_INPUT_PRODUCTION_SELECTION_PENDING":
        reasons.append("PRODUCTION_MATERIAL_SELECTION_PENDING_QUALIFICATION_ROUTE_ONLY")
    if reasons:
        return "CONDITIONAL_OR_BLOCKED", reasons
    return "READY_AFTER_TARGET_RESOURCE_ADMISSION", []


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    released = []
    deferred = []
    for original in rows:
        status, reasons = disposition(original)
        row = portable(original)
        row["release_disposition"] = status
        row["release_blocking_reasons"] = reasons
        order, name = PHASE[row["block"]]
        row["release_phase_order"] = order
        row["release_phase"] = name
        shared = row.get("shared_options", {})
        direct_fem = (
            row.get("backend") == "full_fem_czm"
            and (
                row.get("source_location") in {"bulk_uniform", "tip_plus_bulk"}
                or any(key in shared for key in ("plane_state", "in_plane_mode", "grip_mode_angle_deg"))
            )
        )
        row["release_execution"] = {
            "default": "dry_run",
            "launcher_kind": "direct_fem_native" if direct_fem else "shared_source_attesting_worker",
            "worker": (
                "python -m arrhenius_fracture.generic_current_pf_companion {native_args} --out {case_output}"
                if direct_fem else
                "python -m arrhenius_fracture.shared_execution_worker {case_request_json}"
            ),
            "target_output": "{OUTPUT_ROOT}/cases/" + row["case_id"],
            "fem_root": "{FEM_ROOT}",
            "package_root": "{PACKAGE_ROOT}",
        }
        if status == "DEFERRED_OPTIONAL_CZM":
            deferred.append(row)
        else:
            released.append(row)

    released.sort(key=lambda row: (row["release_phase_order"], row["block"], row["case_id"]))
    raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in released)
    (OUT / "MAIN_SEQUENCE.jsonl").write_text(raw)
    (OUT / "DEFERRED_OPTIONAL_CZM.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in deferred)
    )
    fields = [
        "release_phase_order", "release_phase", "block", "case_id", "release_disposition",
        "family", "temperature_K", "rate_factor", "arm", "backend", "study",
        "source_lifecycle", "source_location", "plasticity_scope", "crack_path",
        "cavity_mode", "orientation_deg", "seed", "parameter_registry", "parameter_option",
        "native_entrypoint", "output_profile", "target_extension_um", "eligibility",
        "material_domain", "input_availability", "native_qualification",
        "numerical_convergence", "peak_transient_GiB", "retained_output_GiB",
        "blocking_reasons",
    ]
    with (OUT / "MAIN_SEQUENCE_CROSSWALK.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in released:
            writer.writerow({
                **{key: row.get(key) for key in fields},
                "material_domain": row["readiness"]["material_domain"],
                "input_availability": row["readiness"]["input_availability"],
                "native_qualification": row["readiness"]["native_qualification"],
                "numerical_convergence": row["readiness"]["numerical_convergence"],
                "peak_transient_GiB": row["resource_estimate"]["peak_transient_GiB"],
                "retained_output_GiB": row["resource_estimate"]["retained_output_GiB"],
                "blocking_reasons": ";".join(row["release_blocking_reasons"]),
            })
    summary = {
        "schema": "native-fracture-main-sequence/1",
        "source": str(SOURCE.relative_to(REPO)),
        "source_sha256": sha256(SOURCE),
        "source_case_count": len(rows),
        "main_original_opening_case_count": len(released),
        "deferred_optional_czm_case_count": len(deferred),
        "main_by_block": dict(sorted(Counter(row["block"] for row in released).items())),
        "main_by_release_disposition": dict(sorted(Counter(row["release_disposition"] for row in released).items())),
        "main_by_launcher_kind": dict(sorted(Counter(row["release_execution"]["launcher_kind"] for row in released).items())),
        "deferred_by_opening_kind": dict(sorted(Counter(row["cohesive_law"]["kind"] for row in deferred).items())),
        "protected_axes": {
            key: sorted({str(row.get(key)) for row in released})
            for key in ("family", "temperature_K", "rate_factor", "orientation_deg", "backend", "source_location", "source_lifecycle", "crack_path", "cavity_mode")
        },
        "main_sequence_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "production_execution_authorized_by_profile": False,
    }
    (OUT / "MAIN_SEQUENCE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
